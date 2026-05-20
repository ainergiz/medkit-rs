"""Launch CXR Modal benchmark rows concurrently and collect local artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOCAL_REPO_ROOT = next(
    (
        parent
        for parent in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
        if (parent / "Cargo.toml").exists()
        and "[workspace]" in (parent / "Cargo.toml").read_text()
    ),
    Path.cwd(),
)
MODAL_SCRIPT = "crates/medkit-benchmarks/scripts/modal_cxr_classification.py"
SOURCE_REPORT_ROOT = LOCAL_REPO_ROOT / "target" / "reports" / "cxr"
CURRENT_TOOLS_ROOT = LOCAL_REPO_ROOT / "target" / "reports" / "cxr-current-tools"
CACHE_IMAGE_PSS_MIN_MB = 1.0
CACHE_IMAGE_PSS_NEAR_ZERO_MB = 1.0


@dataclass
class Row:
    name: str
    baseline: str
    cache_dtype: str
    read_mode: str
    purpose: str


@dataclass
class RunningRow:
    row: Row
    run_id: str
    command: list[str]
    process: subprocess.Popen[str]
    output_path: Path
    started_at: float


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch CXR Modal benchmark rows.")
    parser.add_argument("--batch-id", default="")
    parser.add_argument("--splits", default="")
    parser.add_argument(
        "--baselines",
        default="pytorch_raw,monai_raw,torchxrayvision",
        help="Comma-separated baselines to launch as separate rows.",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--cache-dtype", choices=("float32", "float16", "uint8"), default="float32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=6000)
    parser.add_argument("--max-train", type=int, default=4096)
    parser.add_argument("--max-val", type=int, default=1024)
    parser.add_argument("--max-test", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--loader-batches", type=int, default=64)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--profile-batches", type=int, default=0)
    parser.add_argument("--drop-last-train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--prefetch-depth", type=int, default=1)
    parser.add_argument("--prefetch-read-workers", type=int, default=1)
    parser.add_argument("--read-mode", choices=("mmap", "stream"), default="mmap")
    parser.add_argument(
        "--read-modes",
        default="",
        help="Optional comma-separated read modes; overrides --read-mode for matrix rows.",
    )
    parser.add_argument("--include-metadata", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--cache-dtypes",
        default="",
        help="Optional comma-separated cache dtypes; overrides --cache-dtype for matrix rows.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--force-rematerialize", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--modal-gpu",
        default="",
        help="Optional Modal GPU selector, passed as MEDKIT_MODAL_GPU to row commands.",
    )
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M")
    batch_id = args.batch_id or (
        f"nih-cxr14-current-tools-parallel-size{args.image_size}-"
        f"b{args.batch_size}-{timestamp}"
    )
    batch_dir = CURRENT_TOOLS_ROOT / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    baselines = parse_csv(args.baselines)
    cache_dtypes = parse_csv(args.cache_dtypes) or [args.cache_dtype]
    read_modes = parse_csv(args.read_modes) or [args.read_mode]
    validate_choices("cache dtype", cache_dtypes, {"float32", "float16", "uint8"})
    validate_choices("read mode", read_modes, {"mmap", "stream"})
    rows = []
    for baseline in baselines:
        settings = matrix_settings_for_baseline(baseline, cache_dtypes, read_modes)
        for cache_dtype, read_mode in settings:
            rows.append(
                Row(
                    name=row_name(baseline, cache_dtype, read_mode),
                    baseline=baseline,
                    cache_dtype=cache_dtype,
                    read_mode=read_mode,
                    purpose=row_purpose(baseline, cache_dtype, read_mode),
                )
            )
    if not rows:
        raise ValueError("No baselines provided")

    pending = list(rows)
    running: list[RunningRow] = []
    completed: list[dict[str, Any]] = []
    batch_started = time.perf_counter()

    write_json(
        batch_dir / "batch-config.json",
        {
            "batch_id": batch_id,
            "rows": [row.__dict__ for row in rows],
            "settings": vars(args),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )

    if args.dry_run:
        for row in rows:
            run_id = run_id_for(batch_id, row)
            command = build_command(args, run_id=run_id, row=row)
            print(" ".join(command_with_env(args, command)))
        return 0

    while pending or running:
        while pending and len(running) < max(args.concurrency, 1):
            row = pending.pop(0)
            run_id = run_id_for(batch_id, row)
            row_dir = batch_dir / run_id
            if row_dir.exists():
                shutil.rmtree(row_dir)
            row_dir.mkdir(parents=True, exist_ok=True)
            source_dir = SOURCE_REPORT_ROOT / run_id
            if source_dir.exists():
                shutil.rmtree(source_dir)
            output_path = row_dir / "modal-output.log"
            command = build_command(args, run_id=run_id, row=row)
            (row_dir / "launcher-command.txt").write_text(
                " ".join(command_with_env(args, command)) + "\n"
            )
            pain_diary_path = row_dir / "pain-diary.md"
            pain_diary_path.write_text(
                initial_pain_diary(row, run_id, command_with_env(args, command))
            )
            output_handle = output_path.open("w")
            env = row_environment(args)
            process = subprocess.Popen(
                command,
                cwd=LOCAL_REPO_ROOT,
                env=env,
                text=True,
                stdout=output_handle,
                stderr=subprocess.STDOUT,
            )
            output_handle.close()
            running.append(
                RunningRow(
                    row=row,
                    run_id=run_id,
                    command=command,
                    process=process,
                    output_path=output_path,
                    started_at=time.perf_counter(),
                )
            )
            print(f"started {run_id}: {row.baseline}")

        time.sleep(5)
        still_running: list[RunningRow] = []
        for active in running:
            returncode = active.process.poll()
            if returncode is None:
                still_running.append(active)
                continue
            result = collect_row(active, batch_dir, returncode)
            completed.append(result)
            print(
                f"finished {active.run_id}: returncode={returncode} "
                f"status={result.get('status')}"
            )
        running = still_running
        write_batch_summary(batch_dir, batch_id, completed, running, pending, batch_started)

    write_batch_summary(batch_dir, batch_id, completed, running, pending, batch_started)
    failures = [row for row in completed if row.get("status") != "ok"]
    return 1 if failures else 0


def build_command(args: argparse.Namespace, *, run_id: str, row: Row) -> list[str]:
    command = [
        "modal",
        "run",
        MODAL_SCRIPT,
        "--run-id",
        run_id,
        "--max-samples",
        str(args.max_samples),
        "--max-train",
        str(args.max_train),
        "--max-val",
        str(args.max_val),
        "--max-test",
        str(args.max_test),
        "--image-size",
        str(args.image_size),
        "--cache-dtype",
        row.cache_dtype,
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--epochs",
        str(args.epochs),
        "--loader-batches",
        str(args.loader_batches),
        "--warmup-batches",
        str(args.warmup_batches),
        "--profile-batches",
        str(args.profile_batches),
        "--drop-last-train" if args.drop_last_train else "--no-drop-last-train",
        "--prefetch-depth",
        str(args.prefetch_depth),
        "--prefetch-read-workers",
        str(args.prefetch_read_workers),
        "--read-mode",
        row.read_mode,
        "--include-metadata" if args.include_metadata else "--no-include-metadata",
        "--baselines",
        row.baseline,
    ]
    if args.max_train_batches:
        command.extend(["--max-train-batches", str(args.max_train_batches)])
    if args.max_eval_batches:
        command.extend(["--max-eval-batches", str(args.max_eval_batches)])
    if args.splits:
        command.extend(["--splits", args.splits])
    if args.smoke:
        command.append("--smoke")
    if args.force_cache:
        command.append("--force-cache")
    if args.force_rematerialize:
        command.append("--force-rematerialize")
    return command


def row_environment(args: argparse.Namespace) -> dict[str, str] | None:
    overrides = modal_environment_overrides(args)
    if not overrides:
        return None
    env = os.environ.copy()
    env.update(overrides)
    return env


def command_with_env(args: argparse.Namespace, command: list[str]) -> list[str]:
    overrides = modal_environment_overrides(args)
    if not overrides:
        return command
    return [*(f"{key}={value}" for key, value in overrides.items()), *command]


def modal_environment_overrides(args: argparse.Namespace) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if args.modal_gpu:
        overrides["MEDKIT_MODAL_GPU"] = args.modal_gpu
    for key in ("MEDKIT_MODAL_USE_PYPI", "MEDKIT_MODAL_MEDKIT_PACKAGE"):
        value = os.environ.get(key)
        if value:
            overrides[key] = value
    return overrides


def collect_row(active: RunningRow, batch_dir: Path, returncode: int) -> dict[str, Any]:
    row_dir = batch_dir / active.run_id
    source_dir = SOURCE_REPORT_ROOT / active.run_id
    if source_dir.exists():
        copy_report_artifacts(source_dir, row_dir)
    modal_result = load_json_if_exists(row_dir / "modal-result.json")
    run_summary = load_json_if_exists(row_dir / "run-summary.json")
    loader = load_json_if_exists(row_dir / "loader-throughput.json")
    gpu = load_json_if_exists(row_dir / "gpu-throughput.json")
    profile = load_json_if_exists(row_dir / "step-profile.json")
    quality = load_json_if_exists(row_dir / "model-quality.json")
    environment = load_json_if_exists(row_dir / "environment.json")
    elapsed = time.perf_counter() - active.started_at
    validation_errors = validate_row_artifacts(
        active=active,
        returncode=returncode,
        modal_result=modal_result,
        run_summary=run_summary,
        loader=loader,
        gpu=gpu,
        profile=profile,
        quality=quality,
        environment=environment,
    )
    status = "ok" if not validation_errors else "failed"
    result = {
        "run_id": active.run_id,
        "baseline": active.row.baseline,
        "cache_dtype": active.row.cache_dtype,
        "read_mode": active.row.read_mode,
        "purpose": active.row.purpose,
        "returncode": returncode,
        "status": status,
        "elapsed_seconds": elapsed,
        "report_dir": str(row_dir),
        "source_report_dir": str(source_dir),
        "run_summary": run_summary,
        "loader": loader,
        "gpu": gpu,
        "profile": profile,
        "quality": quality,
        "environment": environment,
        "modal_status": modal_result.get("status") if modal_result else None,
        "validation_errors": validation_errors,
    }
    append_pain_diary(row_dir / "pain-diary.md", result)
    write_json(row_dir / "row-summary.json", result)
    return result


def copy_report_artifacts(source_dir: Path, row_dir: Path) -> None:
    for path in source_dir.iterdir():
        target = row_dir / path.name
        if path.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)


def write_batch_summary(
    batch_dir: Path,
    batch_id: str,
    completed: list[dict[str, Any]],
    running: list[RunningRow],
    pending: list[Row],
    batch_started: float,
) -> None:
    summary = {
        "batch_id": batch_id,
        "status": "running" if running or pending else "ok",
        "elapsed_seconds": time.perf_counter() - batch_started,
        "completed": completed,
        "running": [
            {
                "run_id": active.run_id,
                "baseline": active.row.baseline,
                "cache_dtype": active.row.cache_dtype,
                "read_mode": active.row.read_mode,
                "elapsed_seconds": time.perf_counter() - active.started_at,
            }
            for active in running
        ],
        "pending": [row.__dict__ for row in pending],
    }
    if any(row.get("status") != "ok" for row in completed):
        summary["status"] = "failed" if not running else "running_with_failures"
    write_json(batch_dir / "batch-summary.json", summary)


def validate_row_artifacts(
    *,
    active: RunningRow,
    returncode: int,
    modal_result: dict[str, Any],
    run_summary: dict[str, Any],
    loader: dict[str, Any],
    gpu: dict[str, Any],
    profile: dict[str, Any],
    quality: dict[str, Any],
    environment: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    baseline = active.row.baseline

    if returncode != 0:
        errors.append(f"modal command returned {returncode}")
    if not modal_result:
        errors.append("missing modal-result.json")
    if not run_summary:
        errors.append("missing run-summary.json")
    if not loader:
        errors.append("missing loader-throughput.json")
    if not gpu:
        errors.append("missing gpu-throughput.json")
    if not environment:
        errors.append("missing environment.json")
    if modal_result and modal_result.get("status") != "ok":
        errors.append(f"modal-result status is {modal_result.get('status')!r}")

    run_id = run_summary.get("run_id")
    if run_summary and not run_id:
        errors.append("run-summary missing run_id")
    elif run_id and run_id != active.run_id:
        errors.append(f"run-summary run_id {run_id!r} != expected {active.run_id!r}")

    modal_run_summary = (modal_result.get("artifacts") or {}).get("run-summary.json")
    if isinstance(modal_run_summary, dict):
        modal_run_id = modal_run_summary.get("run_id")
        if modal_run_id and modal_run_id != active.run_id:
            errors.append(
                f"modal-result run-summary run_id {modal_run_id!r} "
                f"!= expected {active.run_id!r}"
            )

    loader_row = loader.get(baseline)
    gpu_row = gpu.get(baseline)
    quality_row = quality.get(baseline) if quality else None
    if not isinstance(loader_row, dict):
        errors.append(f"loader-throughput missing baseline {baseline!r}")
        loader_row = {}
    if not isinstance(gpu_row, dict):
        errors.append(f"gpu-throughput missing baseline {baseline!r}")
        gpu_row = {}
    if quality and not isinstance(quality_row, dict):
        errors.append(f"model-quality missing baseline {baseline!r}")

    if loader_row.get("status") and loader_row.get("status") != "ok":
        errors.append(f"loader status is {loader_row.get('status')!r}")
    if gpu_row.get("status") and gpu_row.get("status") != "ok":
        errors.append(f"gpu status is {gpu_row.get('status')!r}")
    compare_float(
        errors,
        "train samples/s",
        (run_summary.get("train_samples_per_second") or {}).get(baseline),
        gpu_row.get("samples_per_second"),
        tolerance=1e-3,
    )
    compare_float(
        errors,
        "loader samples/s",
        (run_summary.get("loader_samples_per_second") or {}).get(baseline),
        loader_row.get("samples_per_second"),
        tolerance=1e-3,
    )

    metadata = environment.get("run_metadata") or {}
    validate_profile_artifacts(
        errors=errors,
        context="profile",
        baseline=baseline,
        gpu_row=gpu_row,
        profile=profile,
        profile_required=profile_batches_requested(metadata) > 0,
    )
    if metadata:
        if metadata.get("run_id") and metadata.get("run_id") != active.run_id:
            errors.append(
                f"environment run_id {metadata.get('run_id')!r} "
                f"!= expected {active.run_id!r}"
            )
        if metadata.get("cache_dtype") != active.row.cache_dtype:
            errors.append(
                f"environment cache_dtype {metadata.get('cache_dtype')!r} "
                f"!= expected {active.row.cache_dtype!r}"
            )
        if metadata.get("read_mode") != active.row.read_mode:
            errors.append(
                f"environment read_mode {metadata.get('read_mode')!r} "
                f"!= expected {active.row.read_mode!r}"
            )

    for phase, row in [("loader", loader_row), ("gpu", gpu_row)]:
        pipeline = row.get("pipeline") or {}
        if not pipeline and active.row.baseline.startswith("medkit"):
            errors.append(f"{phase} pipeline metadata missing")
            continue
        if not pipeline:
            continue
        if pipeline.get("cache_dtype") != active.row.cache_dtype:
            errors.append(
                f"{phase} pipeline cache_dtype {pipeline.get('cache_dtype')!r} "
                f"!= expected {active.row.cache_dtype!r}"
            )
        if pipeline.get("read_mode") != active.row.read_mode:
            errors.append(
                f"{phase} pipeline read_mode {pipeline.get('read_mode')!r} "
                f"!= expected {active.row.read_mode!r}"
            )
        if active.row.baseline.startswith("medkit") and pipeline.get("include_metadata") is not False:
            errors.append(f"{phase} pipeline include_metadata is not false")
        validate_cache_pss_semantics(
            errors=errors,
            context=phase,
            baseline=active.row.baseline,
            read_mode=active.row.read_mode,
            memory=row.get("memory") or {},
        )

    return errors


def validate_profile_artifacts(
    *,
    errors: list[str],
    context: str,
    baseline: str,
    gpu_row: dict[str, Any],
    profile: dict[str, Any],
    profile_required: bool = False,
) -> None:
    if not profile_required and not profile_enabled(gpu_row):
        return
    if not profile:
        errors.append(f"{context} missing step-profile.json")
        return
    profile_row = profile.get(baseline)
    if not isinstance(profile_row, dict):
        errors.append(f"{context} missing baseline {baseline!r}")
        return
    if profile_row.get("status") != "ok":
        errors.append(f"{context} status is {profile_row.get('status')!r}")
    records = profile_row.get("records")
    if not isinstance(records, list):
        errors.append(f"{context} records missing")
        records = []
    summary = profile_row.get("summary")
    if not isinstance(summary, dict):
        errors.append(f"{context} summary missing")
        summary = {}
    validate_profile_records(errors, context, records, summary)
    for field in profile_summary_fields():
        if field in gpu_row and field in summary:
            compare_float(errors, f"{context} {field}", gpu_row[field], summary[field], tolerance=1e-6)


def profile_enabled(gpu_row: dict[str, Any]) -> bool:
    return "profile_artifact_path" in gpu_row or "profiled_batches" in gpu_row


def profile_batches_requested(metadata: dict[str, Any]) -> int:
    value = numeric_value(metadata.get("profile_batches"))
    return int(value) if value is not None and value > 0 else 0


def profile_summary_fields() -> tuple[str, ...]:
    fields = [
        "profiled_batches",
        "profiled_samples",
        "profile_data_wait_total_ms",
        "profile_total_step_ms",
        "profile_train_samples_per_s",
        "profile_end_to_end_ms",
        "profile_end_to_end_samples_per_s",
    ]
    for field in (
        "data_wait_ms",
        "h2d_ms",
        "forward_ms",
        "backward_ms",
        "optimizer_ms",
    ):
        fields.extend(
            [
                f"profile_{field}_mean",
                f"profile_{field}_p50",
                f"profile_{field}_p95",
            ]
        )
    return tuple(fields)


def validate_profile_records(
    errors: list[str],
    context: str,
    records: list[Any],
    summary: dict[str, Any],
) -> None:
    if len(records) < 20:
        errors.append(f"{context} profiled_batches {len(records)} < 20")
    profiled_samples = 0
    data_wait_total_ms = 0.0
    total_step_ms = 0.0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"{context} record {index} is not an object")
            continue
        samples = numeric_value(record.get("samples"))
        if samples is None or samples <= 0:
            errors.append(f"{context} record {index} samples invalid: {record.get('samples')!r}")
        else:
            profiled_samples += int(samples)
        for field in (
            "data_wait_ms",
            "h2d_ms",
            "forward_ms",
            "backward_ms",
            "optimizer_ms",
            "total_step_ms",
        ):
            value = numeric_value(record.get(field))
            value_valid = value is not None and math.isfinite(value) and value >= 0.0
            if not value_valid:
                errors.append(f"{context} record {index} {field} invalid: {record.get(field)!r}")
            if field == "data_wait_ms" and value_valid:
                data_wait_total_ms += value
            if field == "total_step_ms" and value is not None:
                if value <= 0.0:
                    errors.append(f"{context} record {index} total_step_ms <= 0")
                if value_valid:
                    total_step_ms += value
    compare_float(errors, f"{context} profiled_batches", summary.get("profiled_batches"), len(records), tolerance=0.0)
    compare_float(errors, f"{context} profiled_samples", summary.get("profiled_samples"), profiled_samples, tolerance=0.0)
    compare_float(errors, f"{context} profile_total_step_ms", summary.get("profile_total_step_ms"), total_step_ms, tolerance=1e-6)
    compare_float(
        errors,
        f"{context} profile_data_wait_total_ms",
        summary.get("profile_data_wait_total_ms"),
        data_wait_total_ms,
        tolerance=1e-6,
    )
    profile_end_to_end_ms = data_wait_total_ms + total_step_ms
    compare_float(
        errors,
        f"{context} profile_end_to_end_ms",
        summary.get("profile_end_to_end_ms"),
        profile_end_to_end_ms,
        tolerance=1e-6,
    )
    if total_step_ms > 0.0:
        expected_sps = 1000.0 * profiled_samples / total_step_ms
        compare_float(
            errors,
            f"{context} profile_train_samples_per_s",
            summary.get("profile_train_samples_per_s"),
            expected_sps,
            tolerance=max(1e-6, abs(expected_sps) * 1e-3),
        )
    if profile_end_to_end_ms > 0.0:
        expected_e2e_sps = 1000.0 * profiled_samples / profile_end_to_end_ms
        compare_float(
            errors,
            f"{context} profile_end_to_end_samples_per_s",
            summary.get("profile_end_to_end_samples_per_s"),
            expected_e2e_sps,
            tolerance=max(1e-6, abs(expected_e2e_sps) * 1e-3),
        )


def validate_cache_pss_semantics(
    *,
    errors: list[str],
    context: str,
    baseline: str,
    read_mode: str,
    memory: dict[str, Any],
) -> None:
    cache_pss = numeric_value(memory.get("smaps_pss_cache_images_mb"))
    file_pss = numeric_value(memory.get("smaps_pss_file_mb"))
    is_medkit = baseline.startswith("medkit")

    if is_medkit and read_mode == "mmap":
        if cache_pss is None:
            errors.append(f"{context} smaps_pss_cache_images_mb missing for medkit mmap row")
            return
        if cache_pss <= CACHE_IMAGE_PSS_MIN_MB:
            errors.append(
                f"{context} medkit mmap cache-image PSS {cache_pss:.3f} MB "
                f"<= {CACHE_IMAGE_PSS_MIN_MB:.3f} MB"
            )
        if file_pss is None:
            errors.append(f"{context} smaps_pss_file_mb missing for medkit mmap row")
        elif file_pss + 1e-6 < cache_pss:
            errors.append(
                f"{context} smaps_pss_file_mb {file_pss:.3f} MB "
                f"< cache-image PSS {cache_pss:.3f} MB"
            )
        return

    if is_medkit and read_mode == "stream":
        if cache_pss is None:
            errors.append(f"{context} smaps_pss_cache_images_mb missing for medkit stream row")
            return
        if cache_pss > CACHE_IMAGE_PSS_NEAR_ZERO_MB:
            errors.append(
                f"{context} medkit stream cache-image PSS {cache_pss:.3f} MB "
                f"> {CACHE_IMAGE_PSS_NEAR_ZERO_MB:.3f} MB"
            )
        return

    if not is_medkit and cache_pss is not None and cache_pss > CACHE_IMAGE_PSS_NEAR_ZERO_MB:
        errors.append(
            f"{context} raw/non-medkit cache-image PSS {cache_pss:.3f} MB "
            f"> {CACHE_IMAGE_PSS_NEAR_ZERO_MB:.3f} MB"
        )


def numeric_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compare_float(
    errors: list[str],
    label: str,
    left: Any,
    right: Any,
    *,
    tolerance: float,
) -> None:
    if left is None or right is None:
        return
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        errors.append(f"{label} is not numeric: {left!r} vs {right!r}")
        return
    if abs(left_value - right_value) > tolerance:
        errors.append(f"{label} mismatch: {left_value} != {right_value}")


def initial_pain_diary(row: Row, run_id: str, command: list[str]) -> str:
    return (
        f"# Pain Diary: {run_id}\n\n"
        f"Run id: `{run_id}`\n\n"
        f"Tool: `{row.baseline}`\n\n"
        f"Purpose: {row.purpose}\n\n"
        "Start state: launched by `modal_cxr_parallel_matrix.py`.\n\n"
        "Command:\n\n"
        "```bash\n"
        + " ".join(command)
        + "\n```\n\n"
        "Time to first batch: pending.\n\n"
        "Time to first completed training report: pending.\n\n"
        "Manual steps: pending.\n\n"
        "Errors: pending.\n\n"
        "Ambiguous choices: pending.\n\n"
        "Hidden assumptions: pending.\n\n"
        "What medkit should remove: pending.\n\n"
        "What medkit should copy: pending.\n"
    )


def append_pain_diary(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "\n## Launcher Result\n",
        f"- Status: `{result.get('status')}`",
        f"- Return code: `{result.get('returncode')}`",
        f"- Elapsed seconds: `{result.get('elapsed_seconds'):.3f}`",
    ]
    run_summary = result.get("run_summary") or {}
    loader = (result.get("loader") or {}).get(result.get("baseline"), {})
    gpu = (result.get("gpu") or {}).get(result.get("baseline"), {})
    quality = (result.get("quality") or {}).get(result.get("baseline"), {})
    if run_summary:
        lines.append(f"- Run summary exists: yes")
    else:
        lines.append("- Run summary exists: no")
    if loader:
        lines.append(f"- Loader samples/s: `{loader.get('samples_per_second')}`")
        lines.append(f"- Time to first batch ms: `{loader.get('time_to_first_batch_ms')}`")
    if gpu:
        lines.append(f"- Train samples/s: `{gpu.get('samples_per_second')}`")
        lines.append(f"- Data wait percent: `{gpu.get('data_wait_percent')}`")
    if quality:
        lines.append(f"- Macro AUROC: `{quality.get('macro_auroc')}`")
    if result.get("status") != "ok":
        lines.append("- Errors: inspect `modal-output.log` and `modal-result.json`.")
    lines.append("\n## Pain Notes\n")
    lines.append("- Manual steps: none beyond launcher command for this row.")
    lines.append("- Ambiguous choices: review after inspecting artifacts.")
    lines.append("- Hidden assumptions: review split, label, transform, and worker defaults.")
    lines.append("- What medkit should remove: fill after artifact review.")
    lines.append("- What medkit should copy: fill after artifact review.")
    with path.open("a") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def validate_choices(label: str, values: list[str], allowed: set[str]) -> None:
    invalid = [value for value in values if value not in allowed]
    if invalid:
        raise ValueError(f"Unsupported {label} values {invalid}; expected one of {sorted(allowed)}")


def baseline_to_name(baseline: str) -> str:
    return baseline.replace("_", "-")


def row_name(baseline: str, cache_dtype: str, read_mode: str) -> str:
    return f"{baseline_to_name(baseline)}-{cache_dtype}-{read_mode}"


def row_purpose(baseline: str, cache_dtype: str, read_mode: str) -> str:
    purposes = {
        "pytorch_raw": "Hand-rolled PyTorch control path.",
        "monai_raw": "MONAI medical-imaging framework path.",
        "torchxrayvision": "CXR-specific toolkit path.",
        "medkit_native_prefetch_pinned": "Reference medkit native-prefetch path.",
    }
    return (
        purposes.get(baseline, "Current-tool benchmark row.")
        + f" Matrix settings: cache_dtype={cache_dtype}, read_mode={read_mode}."
    )


def matrix_settings_for_baseline(
    baseline: str,
    cache_dtypes: list[str],
    read_modes: list[str],
) -> list[tuple[str, str]]:
    if not baseline.startswith("medkit"):
        return [("float32", "mmap")]
    return [(cache_dtype, read_mode) for cache_dtype in cache_dtypes for read_mode in read_modes]


def run_id_for(batch_id: str, row: Row) -> str:
    return f"{batch_id}-{row.name}"


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as error:
        return {"status": "parse_failed", "error": str(error)}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
