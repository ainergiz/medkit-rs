"""Launch CXR Modal benchmark rows concurrently and collect local artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


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
RAW_MEDKIT_BASELINES = "pytorch_raw,medkit_native_prefetch_pinned"
GATE_PRESETS: dict[str, dict[str, Any]] = {
    "h100-512-b32": {
        "baselines": RAW_MEDKIT_BASELINES,
        "image_size": 512,
        "cache_dtypes": "float32,uint8",
        "batch_size": 32,
        "workers": 8,
        "max_samples": 6000,
        "max_train": 4096,
        "max_val": 1024,
        "max_test": 1024,
        "epochs": 1,
        "loader_batches": 64,
        "warmup_batches": 4,
        "profile_batches": 128,
        "drop_last_train": True,
        "max_train_batches": 0,
        "max_eval_batches": 1,
        "prefetch_depth": 2,
        "prefetch_read_workers": 4,
        "shuffle_block_batches": 0,
        "gpu_prefetch_batches": 0,
        "read_modes": "stream",
        "include_metadata": False,
        "modal_gpu": "H100",
    },
    "l4-224-b64": {
        "baselines": RAW_MEDKIT_BASELINES,
        "image_size": 224,
        "cache_dtypes": "float32",
        "batch_size": 64,
        "workers": 8,
        "max_samples": 6000,
        "max_train": 4096,
        "max_val": 1024,
        "max_test": 1024,
        "epochs": 1,
        "loader_batches": 64,
        "warmup_batches": 4,
        "profile_batches": 64,
        "drop_last_train": True,
        "max_train_batches": 0,
        "max_eval_batches": 1,
        "prefetch_depth": 2,
        "prefetch_read_workers": 4,
        "shuffle_block_batches": 0,
        "gpu_prefetch_batches": 0,
        "read_modes": "stream",
        "include_metadata": False,
        "modal_gpu": "L4",
    },
}
GATE_OPTION_FLAGS: dict[str, tuple[str, ...]] = {
    "baselines": ("--baselines",),
    "image_size": ("--image-size",),
    "cache_dtypes": ("--cache-dtypes", "--cache-dtype"),
    "batch_size": ("--batch-size",),
    "workers": ("--workers",),
    "max_samples": ("--max-samples",),
    "max_train": ("--max-train",),
    "max_val": ("--max-val",),
    "max_test": ("--max-test",),
    "epochs": ("--epochs",),
    "loader_batches": ("--loader-batches",),
    "warmup_batches": ("--warmup-batches",),
    "profile_batches": ("--profile-batches",),
    "drop_last_train": ("--drop-last-train", "--no-drop-last-train"),
    "max_train_batches": ("--max-train-batches",),
    "max_eval_batches": ("--max-eval-batches",),
    "prefetch_depth": ("--prefetch-depth",),
    "prefetch_read_workers": ("--prefetch-read-workers",),
    "shuffle_block_batches": ("--shuffle-block-batches",),
    "gpu_prefetch_batches": ("--gpu-prefetch-batches",),
    "read_modes": ("--read-modes", "--read-mode"),
    "include_metadata": ("--include-metadata", "--no-include-metadata"),
    "modal_gpu": ("--modal-gpu",),
}


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch CXR Modal benchmark rows.")
    parser.add_argument(
        "--gate",
        choices=sorted(GATE_PRESETS),
        default="",
        help="Apply a repeatable raw+medkit gate preset before launching rows.",
    )
    parser.add_argument(
        "--list-gates",
        action="store_true",
        help="Print available gate presets as JSON and exit.",
    )
    parser.add_argument(
        "--audit-batch",
        type=Path,
        help="Re-validate an existing batch artifact directory without launching Modal.",
    )
    parser.add_argument("--batch-id", default="")
    parser.add_argument("--splits", default="")
    parser.add_argument(
        "--baselines",
        default=RAW_MEDKIT_BASELINES,
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
    parser.add_argument("--shuffle-block-batches", type=int, default=0)
    parser.add_argument("--gpu-prefetch-batches", type=int, default=0)
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
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    if args.gate:
        apply_gate_preset(args, raw_argv)
    return args


def apply_gate_preset(args: argparse.Namespace, argv: Sequence[str]) -> None:
    preset = GATE_PRESETS[args.gate]
    for field, value in preset.items():
        flags = GATE_OPTION_FLAGS[field]
        if not option_was_provided(argv, flags):
            setattr(args, field, value)


def option_was_provided(argv: Sequence[str], flags: tuple[str, ...]) -> bool:
    for item in argv:
        for flag in flags:
            if item == flag or item.startswith(f"{flag}="):
                return True
    return False


def gate_catalog() -> dict[str, dict[str, Any]]:
    return {
        name: {
            **preset,
            "rows": [row.__dict__ for row in rows_for_settings(preset)],
        }
        for name, preset in GATE_PRESETS.items()
    }


def main() -> int:
    args = parse_args()
    if args.list_gates:
        print(json.dumps(gate_catalog(), indent=2, sort_keys=True))
        return 0
    if args.audit_batch:
        return audit_batch(args.audit_batch)

    batch_id = batch_id_for_args(args)
    batch_dir = CURRENT_TOOLS_ROOT / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    rows = rows_for_args(args)

    pending = list(rows)
    running: list[RunningRow] = []
    completed: list[dict[str, Any]] = []
    batch_started = time.perf_counter()

    write_json(
        batch_dir / "batch-config.json",
        {
            "batch_id": batch_id,
            "gate": args.gate or None,
            "gate_preset": GATE_PRESETS.get(args.gate),
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
        *modal_cli_command(),
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
        "--shuffle-block-batches",
        str(args.shuffle_block_batches),
        "--gpu-prefetch-batches",
        str(args.gpu_prefetch_batches),
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


def modal_cli_command() -> list[str]:
    return shlex.split(os.environ.get("MEDKIT_MODAL_CLI", "modal")) or ["modal"]


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
    summary_consistency = load_json_if_exists(row_dir / "summary-consistency.json")
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
        summary_consistency=summary_consistency,
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
        "summary_consistency": summary_consistency,
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
    summary_consistency: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    baseline = active.row.baseline
    summary_consistency = summary_consistency or {}

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
    if not summary_consistency:
        errors.append("missing summary-consistency.json")
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
    compare_modal_artifact(
        errors,
        modal_result=modal_result,
        artifact_name="run-summary.json",
        local_artifact=run_summary,
    )
    compare_modal_artifact(
        errors,
        modal_result=modal_result,
        artifact_name="summary-consistency.json",
        local_artifact=summary_consistency,
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
    validate_summary_provenance_artifacts(
        errors=errors,
        active=active,
        run_summary=run_summary,
        environment=environment,
        summary_consistency=summary_consistency,
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


def compare_modal_artifact(
    errors: list[str],
    *,
    modal_result: dict[str, Any],
    artifact_name: str,
    local_artifact: dict[str, Any],
) -> None:
    artifacts = modal_result.get("artifacts") if isinstance(modal_result, dict) else None
    if not isinstance(artifacts, dict) or not local_artifact:
        return
    remote_artifact = artifacts.get(artifact_name)
    if isinstance(remote_artifact, dict) and remote_artifact != local_artifact:
        errors.append(f"modal-result {artifact_name} does not match local copy")


def validate_summary_provenance_artifacts(
    *,
    errors: list[str],
    active: RunningRow,
    run_summary: dict[str, Any],
    environment: dict[str, Any],
    summary_consistency: dict[str, Any],
) -> None:
    if summary_consistency:
        if summary_consistency.get("status") != "ok":
            errors.append(
                "summary-consistency status is "
                f"{summary_consistency.get('status')!r}: "
                + "; ".join(str(error) for error in summary_consistency.get("errors", []))
            )
        if summary_consistency.get("run_id") != active.run_id:
            errors.append(
                f"summary-consistency run_id {summary_consistency.get('run_id')!r} "
                f"!= expected {active.run_id!r}"
            )
    if not run_summary:
        return
    provenance = run_summary.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("run-summary provenance missing")
        return
    required_fields = (
        "run_id",
        "dataset_loaded",
        "samples",
        "targets",
        "baselines",
        "image_size",
        "cache_image_size",
        "cache_dtype",
        "batch_size",
        "drop_last_train",
        "workers",
        "prefetch_depth",
        "prefetch_read_workers",
        "shuffle_block_batches",
        "gpu_prefetch_batches",
        "read_mode",
        "include_metadata",
        "profile_batches",
        "seed",
        "cache",
        "artifacts",
    )
    for field in required_fields:
        if field not in provenance:
            errors.append(f"run-summary provenance missing {field}")
    if provenance.get("run_id") != active.run_id:
        errors.append(f"provenance run_id {provenance.get('run_id')!r} != expected {active.run_id!r}")
    if provenance.get("cache_dtype") != active.row.cache_dtype:
        errors.append(
            f"provenance cache_dtype {provenance.get('cache_dtype')!r} "
            f"!= expected {active.row.cache_dtype!r}"
        )
    if provenance.get("read_mode") != active.row.read_mode:
        errors.append(
            f"provenance read_mode {provenance.get('read_mode')!r} "
            f"!= expected {active.row.read_mode!r}"
        )
    if provenance.get("baselines") != [active.row.baseline]:
        errors.append(
            f"provenance baselines {provenance.get('baselines')!r} "
            f"!= expected {[active.row.baseline]!r}"
        )
    metadata = environment.get("run_metadata") if isinstance(environment, dict) else {}
    if isinstance(metadata, dict):
        for field in (
            "image_size",
            "cache_image_size",
            "batch_size",
            "workers",
            "prefetch_depth",
            "prefetch_read_workers",
            "shuffle_block_batches",
            "gpu_prefetch_batches",
            "profile_batches",
            "seed",
        ):
            if provenance.get(field) != metadata.get(field):
                errors.append(
                    f"provenance {field} {provenance.get(field)!r} "
                    f"!= environment {metadata.get(field)!r}"
                )
    artifacts = provenance.get("artifacts") or {}
    if artifacts.get("summary_consistency") != "summary-consistency.json":
        errors.append("provenance summary_consistency artifact path missing")
    if profile_batches_requested(provenance) > 0 and artifacts.get("step_profile") != "step-profile.json":
        errors.append("provenance step_profile artifact path missing")


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
        lines.append("- Run summary exists: yes")
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


def batch_id_for_args(args: argparse.Namespace) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M")
    if args.batch_id:
        return args.batch_id
    if args.gate:
        return f"cxr-gate-{args.gate}-{timestamp}"
    return f"nih-cxr14-current-tools-parallel-size{args.image_size}-b{args.batch_size}-{timestamp}"


def rows_for_args(args: argparse.Namespace) -> list[Row]:
    return rows_for_settings(
        {
            "baselines": args.baselines,
            "cache_dtypes": args.cache_dtypes,
            "cache_dtype": args.cache_dtype,
            "read_modes": args.read_modes,
            "read_mode": args.read_mode,
        }
    )


def rows_for_settings(settings: dict[str, Any]) -> list[Row]:
    baselines = parse_csv(str(settings.get("baselines", "")))
    cache_dtypes = parse_csv(str(settings.get("cache_dtypes", ""))) or [
        str(settings.get("cache_dtype", "float32"))
    ]
    read_modes = parse_csv(str(settings.get("read_modes", ""))) or [
        str(settings.get("read_mode", "mmap"))
    ]
    validate_choices("cache dtype", cache_dtypes, {"float32", "float16", "uint8"})
    validate_choices("read mode", read_modes, {"mmap", "stream"})
    rows = []
    for baseline in baselines:
        row_settings = matrix_settings_for_baseline(baseline, cache_dtypes, read_modes)
        for cache_dtype, read_mode in row_settings:
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
    return rows


def audit_batch(batch_dir: Path) -> int:
    config = load_json_if_exists(batch_dir / "batch-config.json")
    if not config:
        raise FileNotFoundError(f"Missing batch-config.json in {batch_dir}")
    batch_id = str(config.get("batch_id") or batch_dir.name)
    rows = [Row(**row) for row in config.get("rows", [])]
    if not rows:
        settings = config.get("settings") or {}
        rows = rows_for_settings(settings)
    completed: list[dict[str, Any]] = []
    started = time.perf_counter()
    for row in rows:
        run_id = run_id_for(batch_id, row)
        row_dir = batch_dir / run_id
        if not row_dir.exists():
            source_dir = SOURCE_REPORT_ROOT / run_id
            if source_dir.exists():
                row_dir.mkdir(parents=True, exist_ok=True)
                (row_dir / "launcher-command.txt").write_text(
                    "adopted existing source report during audit\n"
                )
            else:
                completed.append(
                    {
                        "run_id": run_id,
                        "baseline": row.baseline,
                        "cache_dtype": row.cache_dtype,
                        "read_mode": row.read_mode,
                        "purpose": row.purpose,
                        "returncode": 1,
                        "status": "failed",
                        "elapsed_seconds": 0.0,
                        "report_dir": str(row_dir),
                        "validation_errors": [f"missing row directory {row_dir}"],
                    }
                )
                continue
        modal_result = load_json_if_exists(row_dir / "modal-result.json")
        returncode = int(modal_result.get("returncode") or 0) if modal_result else 0
        completed.append(
            collect_existing_row(
                batch_dir=batch_dir,
                row=row,
                run_id=run_id,
                returncode=returncode,
            )
        )
    write_batch_summary(batch_dir, batch_id, completed, running=[], pending=[], batch_started=started)
    failures = [row for row in completed if row.get("status") != "ok"]
    print(json.dumps({"batch_id": batch_id, "status": "failed" if failures else "ok", "failures": failures}, indent=2, sort_keys=True))
    return 1 if failures else 0


def collect_existing_row(
    *,
    batch_dir: Path,
    row: Row,
    run_id: str,
    returncode: int,
) -> dict[str, Any]:
    active = type(
        "CompletedRow",
        (),
        {
            "row": row,
            "run_id": run_id,
            "command": [],
            "started_at": time.perf_counter(),
        },
    )()
    return collect_row(active, batch_dir, returncode)


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
