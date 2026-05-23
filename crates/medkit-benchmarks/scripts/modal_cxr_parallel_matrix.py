"""Launch CXR Modal benchmark rows concurrently and collect local artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


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
REMOTE_REPORT_ROOT = Path("/cache/results/cxr")
CACHE_IMAGE_PSS_MIN_MB = 1.0
CACHE_IMAGE_PSS_NEAR_ZERO_MB = 1.0
RAW_MEDKIT_BASELINES = "pytorch_raw,medkit_native_prefetch_pinned"
STABILITY_THRESHOLDS = {
    "H100": {
        "train_samples_per_second_cv_ok_percent": 3.0,
        "train_samples_per_second_cv_warn_percent": 5.0,
    },
    "L4": {
        "train_samples_per_second_cv_ok_percent": 3.0,
        "train_samples_per_second_cv_warn_percent": 5.0,
    },
}
PROFILE_PHASE_METRICS = (
    "profile_data_wait_ms",
    "profile_batch_prepare_ms",
    "profile_zero_grad_wall_ms",
    "profile_forward_ms",
    "profile_backward_ms",
    "profile_optimizer_ms",
    "profile_prefetch_maintenance_wall_ms",
    "profile_residual_step_ms_signed",
)
PROFILE_EVENT_METRICS = (
    "profile_h2d_ms",
    "profile_batch_prepare_wall_ms",
    "profile_accounted_step_ms",
    "profile_residual_step_ms",
    "profile_residual_step_percent",
    "profile_total_step_ms",
    "profile_step_accounted_percent",
    "profile_residual_step_signed_percent",
    "profile_step_reconciled_percent",
)
MEMORY_METRICS = (
    "gpu_pss_mb",
    "gpu_anon_pss_mb",
    "gpu_file_pss_mb",
    "gpu_private_dirty_mb",
    "gpu_pinned_estimated_mb",
    "cache_image_pss_mb",
    "loader_pss_mb",
    "loader_cache_image_pss_mb",
)
RUNTIME_METRICS = (
    "warmup_ms",
    "torch_compile_setup_ms",
    "cuda_peak_allocated_mb",
)
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
        "gpu_prefetch_reuse_buffers": False,
        "sync_every_step": True,
        "read_modes": "stream",
        "include_metadata": False,
        "modal_gpu": "H100",
        "repeats": 3,
        "fail_fast": True,
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
        "gpu_prefetch_reuse_buffers": False,
        "sync_every_step": True,
        "read_modes": "stream",
        "include_metadata": False,
        "modal_gpu": "L4",
        "repeats": 3,
        "fail_fast": True,
    },
    "l4-quality-224-b64": {
        "baselines": RAW_MEDKIT_BASELINES,
        "image_size": 224,
        "cache_dtypes": "float32",
        "batch_size": 64,
        "workers": 8,
        "max_samples": 6000,
        "max_train": 4096,
        "max_val": 1024,
        "max_test": 1024,
        "epochs": 2,
        "loader_batches": 16,
        "warmup_batches": 2,
        "profile_batches": 0,
        "drop_last_train": True,
        "max_train_batches": 0,
        "max_eval_batches": 0,
        "prefetch_depth": 2,
        "prefetch_read_workers": 4,
        "shuffle_block_batches": 0,
        "gpu_prefetch_batches": 0,
        "gpu_prefetch_reuse_buffers": False,
        "sync_every_step": True,
        "loss_pos_weight": "balanced",
        "loss_pos_weight_cap": 10.0,
        "quality_gate": True,
        "quality_min_eval_samples": 900,
        "quality_min_metric_targets": 5,
        "quality_min_macro_auroc": 0.55,
        "quality_min_macro_auprc": 0.10,
        "read_modes": "stream",
        "include_metadata": False,
        "modal_gpu": "L4",
        "repeats": 1,
        "fail_fast": True,
    },
}
GATE_OPTION_FLAGS: dict[str, tuple[str, ...]] = {
    "baselines": ("--baselines",),
    "image_size": ("--image-size",),
    "cache_dtypes": ("--cache-dtypes", "--cache-dtype"),
    "cache_splits": ("--cache-splits",),
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
    "gpu_prefetch_reuse_buffers": (
        "--gpu-prefetch-reuse-buffers",
        "--no-gpu-prefetch-reuse-buffers",
    ),
    "sync_every_step": ("--sync-every-step", "--no-sync-every-step"),
    "channels_last": ("--channels-last", "--no-channels-last"),
    "torch_compile": ("--torch-compile", "--no-torch-compile"),
    "torch_compile_mode": ("--torch-compile-mode",),
    "learning_rate": ("--learning-rate",),
    "amp_dtype": ("--amp-dtype",),
    "model_init": ("--model-init",),
    "loss_kind": ("--loss-kind",),
    "loss_pos_weight": ("--loss-pos-weight",),
    "loss_pos_weight_cap": ("--loss-pos-weight-cap",),
    "focal_gamma": ("--focal-gamma",),
    "focal_alpha": ("--focal-alpha",),
    "quality_gate": ("--quality-gate", "--no-quality-gate"),
    "quality_min_eval_samples": ("--quality-min-eval-samples",),
    "quality_min_metric_targets": ("--quality-min-metric-targets",),
    "quality_min_macro_auroc": ("--quality-min-macro-auroc",),
    "quality_min_macro_auprc": ("--quality-min-macro-auprc",),
    "skip_eval": ("--skip-eval", "--no-skip-eval"),
    "train_order_evidence": ("--train-order-evidence", "--no-train-order-evidence"),
    "paired_train_order": ("--paired-train-order", "--no-paired-train-order"),
    "read_modes": ("--read-modes", "--read-mode"),
    "include_metadata": ("--include-metadata", "--no-include-metadata"),
    "modal_gpu": ("--modal-gpu",),
    "repeats": ("--repeats",),
    "fail_fast": ("--fail-fast", "--no-fail-fast"),
}


@dataclass
class Row:
    name: str
    baseline: str
    cache_dtype: str
    read_mode: str
    purpose: str
    repeat_index: int = 0
    repeat_count: int = 1


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
    parser.add_argument(
        "--comparator-batch",
        type=Path,
        help="Optional stable comparator batch directory for promotion readiness checks.",
    )
    parser.add_argument("--batch-id", default="")
    parser.add_argument("--dataset", default="arudaev/chest-xray-14-320")
    parser.add_argument("--rsna-root", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--splits", default="")
    parser.add_argument(
        "--baselines",
        default=RAW_MEDKIT_BASELINES,
        help="Comma-separated baselines to launch as separate rows.",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--cache-dtype", choices=("float32", "float16", "uint8"), default="float32")
    parser.add_argument("--cache-build-workers", type=int, default=1)
    parser.add_argument("--cache-splits", default="train,val,test")
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
    parser.add_argument(
        "--gpu-prefetch-reuse-buffers",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--sync-every-step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-compile-mode", default="default")
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument(
        "--amp-dtype",
        choices=("auto", "float16", "bfloat16", "disabled"),
        default="auto",
    )
    parser.add_argument("--model-init", choices=("random", "imagenet"), default="random")
    parser.add_argument("--loss-kind", choices=("bce", "focal"), default="bce")
    parser.add_argument("--loss-pos-weight", choices=("none", "balanced"), default="none")
    parser.add_argument("--loss-pos-weight-cap", type=float, default=0.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.0)
    parser.add_argument("--quality-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quality-min-eval-samples", type=int, default=0)
    parser.add_argument("--quality-min-metric-targets", type=int, default=0)
    parser.add_argument("--quality-min-macro-auroc", type=float, default=0.0)
    parser.add_argument("--quality-min-macro-auprc", type=float, default=0.0)
    parser.add_argument("--skip-eval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-order-evidence", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--paired-train-order", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--read-mode", choices=("mmap", "stream"), default="mmap")
    parser.add_argument(
        "--read-modes",
        default="",
        help="Optional comma-separated read modes; overrides --read-mode for matrix rows.",
    )
    parser.add_argument("--include-metadata", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--shared-data",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Run one prepare-only job and pass its manifest/splits/cache inputs to matrix rows. "
            "Defaults to enabled for quality gates and disabled for speed gates."
        ),
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--cache-dtypes",
        default="",
        help="Optional comma-separated cache dtypes; overrides --cache-dtype for matrix rows.",
    )
    parser.add_argument(
        "--cache-key-mode",
        choices=("legacy", "content"),
        default="legacy",
        help="Cache key mode forwarded to the Modal CXR benchmark.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--force-rematerialize", action="store_true")
    parser.add_argument("--allow-destructive-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--modal-gpu",
        default="",
        help="Optional Modal GPU selector, passed as MEDKIT_MODAL_GPU to row commands.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Launch each logical matrix row this many times and aggregate repeat stats.",
    )
    parser.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Abort remaining rows after the first row validation failure.",
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


def serializable_settings(args: argparse.Namespace) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in vars(args).items():
        output[key] = str(value) if isinstance(value, Path) else value
    return output


def namespace_with(args: argparse.Namespace, **changes: Any) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(changes)
    return argparse.Namespace(**values)


def auto_shared_data_required(args: argparse.Namespace) -> bool:
    if args.manifest or args.splits:
        return False
    if args.shared_data is not None:
        return bool(args.shared_data)
    return bool(args.quality_gate)


def shared_data_paths(batch_id: str) -> dict[str, str]:
    run_id = f"{batch_id}-prepare-data"
    report_dir = REMOTE_REPORT_ROOT / run_id
    return {
        "run_id": run_id,
        "remote_report_dir": str(report_dir),
        "manifest": str(report_dir / "manifest.jsonl"),
        "splits": str(report_dir / "splits.json"),
    }


def build_prepare_command(args: argparse.Namespace, *, run_id: str) -> list[str]:
    cache_dtypes = parse_csv(args.cache_dtypes) or [args.cache_dtype]
    read_modes = parse_csv(args.read_modes) or [args.read_mode]
    prepare_args = namespace_with(
        args,
        baselines="pytorch_raw",
        quality_gate=False,
        train_order_evidence=False,
        paired_train_order=False,
        manifest="",
        splits="",
    )
    prepare_row = Row(
        name="prepare-data",
        baseline="pytorch_raw",
        cache_dtype=cache_dtypes[0],
        read_mode=read_modes[0],
        purpose="Shared manifest/split/cache preparation.",
    )
    command = build_command(prepare_args, run_id=run_id, row=prepare_row)
    command.append("--prepare-only")
    return command


def run_prepare_data(
    args: argparse.Namespace,
    *,
    batch_dir: Path,
    shared_data: dict[str, str],
) -> dict[str, Any]:
    run_id = shared_data["run_id"]
    row_dir = batch_dir / run_id
    if row_dir.exists():
        shutil.rmtree(row_dir)
    row_dir.mkdir(parents=True, exist_ok=True)
    source_dir = SOURCE_REPORT_ROOT / run_id
    if source_dir.exists():
        shutil.rmtree(source_dir)
    command = build_prepare_command(args, run_id=run_id)
    (row_dir / "launcher-command.txt").write_text(
        " ".join(command_with_env(args, command)) + "\n"
    )
    output_path = row_dir / "modal-output.log"
    started = time.perf_counter()
    with output_path.open("w") as output_handle:
        completed = subprocess.run(
            command,
            cwd=LOCAL_REPO_ROOT,
            env=row_environment(args),
            text=True,
            stdout=output_handle,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.perf_counter() - started
    if source_dir.exists():
        copy_report_artifacts(source_dir, row_dir)
    result = {
        **shared_data,
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "local_report_dir": str(row_dir),
    }
    write_json(row_dir / "shared-data-result.json", result)
    return result


def main() -> int:
    args = parse_args()
    if args.list_gates:
        print(json.dumps(gate_catalog(), indent=2, sort_keys=True))
        return 0
    if args.audit_batch:
        return audit_batch(args.audit_batch, comparator_batch=args.comparator_batch)

    batch_id = batch_id_for_args(args)
    batch_dir = CURRENT_TOOLS_ROOT / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    rows = rows_for_args(args)
    shared_data: dict[str, Any] | None = None
    prepare_command: list[str] | None = None
    if auto_shared_data_required(args):
        shared_data = {
            **shared_data_paths(batch_id),
            "mode": "auto_quality_gate",
        }
        prepare_command = build_prepare_command(args, run_id=str(shared_data["run_id"]))
        args.manifest = str(shared_data["manifest"])
        args.splits = str(shared_data["splits"])

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
            "settings": serializable_settings(args),
            "shared_data": shared_data,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )

    if args.dry_run:
        if prepare_command is not None:
            print(" ".join(command_with_env(args, prepare_command)))
        for row in rows:
            run_id = run_id_for(batch_id, row)
            command = build_command(args, run_id=run_id, row=row)
            print(" ".join(command_with_env(args, command)))
        return 0

    if shared_data is not None:
        prepare_result = run_prepare_data(args, batch_dir=batch_dir, shared_data=shared_data)
        shared_data.update(prepare_result)
        write_json(batch_dir / "shared-data.json", shared_data)
        if prepare_result.get("status") != "ok":
            write_batch_summary(
                batch_dir,
                batch_id,
                completed,
                running=[],
                pending=pending,
                batch_started=batch_started,
                modal_gpu=args.modal_gpu,
                comparator_batch=args.comparator_batch,
            )
            return 1

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
        should_abort = False
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
            if args.fail_fast and result.get("status") != "ok":
                should_abort = True
        running = still_running
        write_batch_summary(
            batch_dir,
            batch_id,
            completed,
            running,
            pending,
            batch_started,
            modal_gpu=args.modal_gpu,
            comparator_batch=args.comparator_batch,
        )
        if should_abort:
            terminate_running_rows(running)
            write_batch_summary(
                batch_dir,
                batch_id,
                completed,
                running=[],
                pending=[],
                batch_started=batch_started,
                fail_fast_aborted=True,
                modal_gpu=args.modal_gpu,
                comparator_batch=args.comparator_batch,
            )
            return 1

    write_batch_summary(
        batch_dir,
        batch_id,
        completed,
        running,
        pending,
        batch_started,
        modal_gpu=args.modal_gpu,
        comparator_batch=args.comparator_batch,
    )
    failures = [row for row in completed if row.get("status") != "ok"]
    repeat_summary = load_json_if_exists(batch_dir / "repeat-summary.json")
    repeat_errors = list(repeat_summary.get("train_order_pairing_errors") or [])
    return 1 if failures or repeat_errors else 0


def build_command(args: argparse.Namespace, *, run_id: str, row: Row) -> list[str]:
    command = [
        *modal_cli_command(),
        "run",
        MODAL_SCRIPT,
        "--run-id",
        run_id,
        "--dataset",
        args.dataset,
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
        "--cache-build-workers",
        str(args.cache_build_workers),
        "--cache-key-mode",
        args.cache_key_mode,
        "--cache-splits",
        args.cache_splits,
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
        "--gpu-prefetch-reuse-buffers"
        if args.gpu_prefetch_reuse_buffers
        else "--no-gpu-prefetch-reuse-buffers",
        "--sync-every-step" if args.sync_every_step else "--no-sync-every-step",
        "--channels-last" if args.channels_last else "--no-channels-last",
        "--torch-compile" if args.torch_compile else "--no-torch-compile",
        "--torch-compile-mode",
        str(args.torch_compile_mode),
        "--learning-rate",
        str(args.learning_rate),
        "--amp-dtype",
        str(args.amp_dtype),
        "--model-init",
        str(args.model_init),
        "--loss-kind",
        str(args.loss_kind),
        "--loss-pos-weight",
        str(args.loss_pos_weight),
        "--loss-pos-weight-cap",
        str(args.loss_pos_weight_cap),
        "--focal-gamma",
        str(args.focal_gamma),
        "--focal-alpha",
        str(args.focal_alpha),
        "--quality-gate" if args.quality_gate else "--no-quality-gate",
        "--quality-min-eval-samples",
        str(args.quality_min_eval_samples),
        "--quality-min-metric-targets",
        str(args.quality_min_metric_targets),
        "--quality-min-macro-auroc",
        str(args.quality_min_macro_auroc),
        "--quality-min-macro-auprc",
        str(args.quality_min_macro_auprc),
        "--skip-eval" if args.skip_eval else "--no-skip-eval",
        "--train-order-evidence"
        if (args.quality_gate if args.train_order_evidence is None else args.train_order_evidence)
        else "--no-train-order-evidence",
        "--paired-train-order"
        if (args.quality_gate if args.paired_train_order is None else args.paired_train_order)
        else "--no-paired-train-order",
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
    if args.manifest:
        command.extend(["--manifest", args.manifest])
    if args.splits:
        command.extend(["--splits", args.splits])
    if args.rsna_root:
        command.extend(["--rsna-root", args.rsna_root])
    if args.smoke:
        command.append("--smoke")
    if args.force_cache:
        command.append("--force-cache")
    if args.force_rematerialize:
        command.append("--force-rematerialize")
    if args.allow_destructive_cache:
        command.append("--allow-destructive-cache")
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
    quality_gate = load_json_if_exists(row_dir / "quality-gate.json")
    ground_truth = load_json_if_exists(row_dir / "training-ground-truth.json")
    predictions = load_json_if_exists(row_dir / "eval-predictions-summary.json")
    train_order = load_json_if_exists(row_dir / "train-order-summary.json")
    environment = load_json_if_exists(row_dir / "environment.json")
    summary_consistency = load_json_if_exists(row_dir / "summary-consistency.json")
    cache_report = load_json_if_exists(row_dir / "cache-report.json")
    cache_preflight = load_json_if_exists(row_dir / "cache-preflight.json")
    cache_registry_entry = load_json_if_exists(row_dir / "cache-registry-entry.json")
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
        quality_gate=quality_gate,
        predictions=predictions,
        train_order=train_order,
        environment=environment,
        summary_consistency=summary_consistency,
        cache_report=cache_report,
        cache_preflight=cache_preflight,
        artifact_dir=row_dir,
    )
    status = "ok" if not validation_errors else "failed"
    result = {
        "run_id": active.run_id,
        "baseline": active.row.baseline,
        "cache_dtype": active.row.cache_dtype,
        "read_mode": active.row.read_mode,
        "repeat_index": active.row.repeat_index,
        "repeat_count": active.row.repeat_count,
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
        "quality_gate": quality_gate,
        "ground_truth": ground_truth,
        "predictions": predictions,
        "train_order": train_order,
        "environment": environment,
        "summary_consistency": summary_consistency,
        "cache_report": cache_report,
        "cache_preflight": cache_preflight,
        "cache_registry_entry": cache_registry_entry,
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
    *,
    fail_fast_aborted: bool = False,
    modal_gpu: str = "",
    comparator_batch: Path | None = None,
) -> None:
    repeat_summary = summarize_repeats(completed, running, pending, modal_gpu=modal_gpu)
    cache_wait_summary = summarize_cache_wait(completed, running, pending)
    comparator = load_comparator_repeat_summary(comparator_batch) if comparator_batch else None
    promotion_readiness = promotion_readiness_report(
        repeat_summary,
        batch_id=batch_id,
        modal_gpu=modal_gpu,
        comparator_summary=comparator[1] if comparator else None,
        comparator_batch_id=comparator[0] if comparator else None,
    )
    summary = {
        "batch_id": batch_id,
        "status": "running" if running or pending else repeat_summary.get("status", "ok"),
        "elapsed_seconds": time.perf_counter() - batch_started,
        "fail_fast_aborted": fail_fast_aborted,
        "completed": completed,
        "running": [
            {
                "run_id": active.run_id,
                "baseline": active.row.baseline,
                "cache_dtype": active.row.cache_dtype,
                "read_mode": active.row.read_mode,
                "repeat_index": active.row.repeat_index,
                "repeat_count": active.row.repeat_count,
                "elapsed_seconds": time.perf_counter() - active.started_at,
            }
            for active in running
        ],
        "pending": [row.__dict__ for row in pending],
        "repeat_summary": repeat_summary,
        "cache_wait_summary": cache_wait_summary,
        "promotion_readiness": promotion_readiness,
    }
    if any(row.get("status") != "ok" for row in completed):
        summary["status"] = "failed" if not running else "running_with_failures"
    write_json(batch_dir / "batch-summary.json", summary)
    write_json(batch_dir / "repeat-summary.json", repeat_summary)
    write_json(batch_dir / "cache-wait-summary.json", cache_wait_summary)


def terminate_running_rows(running: list[RunningRow]) -> None:
    for active in running:
        if active.process.poll() is not None:
            continue
        active.process.terminate()
    deadline = time.monotonic() + 20.0
    for active in running:
        if active.process.poll() is not None:
            continue
        remaining = max(0.0, deadline - time.monotonic())
        try:
            active.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            active.process.kill()


def summarize_repeats(
    completed: list[dict[str, Any]],
    running: list[RunningRow],
    pending: list[Row],
    *,
    modal_gpu: str = "",
) -> dict[str, Any]:
    expected: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in completed:
        key = repeat_group_key(result)
        expected.setdefault(key, repeat_group_descriptor(result))
        groups.setdefault(key, []).append(result)
    for active in running:
        key = repeat_group_key(active.row)
        expected.setdefault(key, repeat_group_descriptor(active.row))
    for row in pending:
        key = repeat_group_key(row)
        expected.setdefault(key, repeat_group_descriptor(row))

    summaries: dict[str, Any] = {}
    for key, descriptor in sorted(expected.items()):
        results = groups.get(key, [])
        ok_results = [result for result in results if result.get("status") == "ok"]
        failed_results = [result for result in results if result.get("status") != "ok"]
        expected_repeats = max(
            [
                int(descriptor.get("repeat_count") or 1),
                *[int(result.get("repeat_count") or 1) for result in results],
            ]
        )
        metrics = repeat_metric_summary(ok_results)
        stability = repeat_group_stability(metrics, modal_gpu=modal_gpu)
        if failed_results:
            status = "failed"
        elif len(ok_results) >= expected_repeats:
            status = "ok"
        else:
            status = "running"
        summaries[key] = {
            **descriptor,
            "status": status,
            "expected_repeats": expected_repeats,
            "completed_repeats": len(results),
            "ok_repeats": len(ok_results),
            "failed_repeats": len(failed_results),
            "metrics": metrics,
            "diagnostics": repeat_group_diagnostics(metrics),
            "stability": stability,
            "runs": [
                {
                    "run_id": result.get("run_id"),
                    "status": result.get("status"),
                    "repeat_index": result.get("repeat_index", 0),
                    "validation_errors": result.get("validation_errors") or [],
                }
                for result in sorted(results, key=lambda item: int(item.get("repeat_index") or 0))
            ],
        }
    comparisons = summarize_repeat_comparisons(summaries)
    prediction_comparisons = summarize_repeat_prediction_comparisons(completed)
    train_order_comparisons = summarize_repeat_train_order_comparisons(completed)
    train_order_pairing_errors = repeat_train_order_pairing_errors(
        completed,
        train_order_comparisons,
    )
    return {
        "schema_version": 1,
        "stability_thresholds": STABILITY_THRESHOLDS,
        "modal_gpu": modal_gpu or None,
        "status": "failed"
        if any(group["status"] == "failed" for group in summaries.values())
        or train_order_pairing_errors
        else "ok"
        if summaries and all(group["status"] == "ok" for group in summaries.values())
        else "running",
        "groups": summaries,
        "comparisons": comparisons,
        "prediction_comparisons": prediction_comparisons,
        "train_order_comparisons": train_order_comparisons,
        "train_order_pairing_errors": train_order_pairing_errors,
    }


def summarize_cache_wait(
    completed: list[dict[str, Any]],
    running: list[RunningRow],
    pending: list[Row],
) -> dict[str, Any]:
    expected: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in completed:
        key = repeat_group_key(result)
        expected.setdefault(key, repeat_group_descriptor(result))
        groups.setdefault(key, []).append(cache_wait_row_summary(result))
    for active in running:
        key = repeat_group_key(active.row)
        expected.setdefault(key, repeat_group_descriptor(active.row))
    for row in pending:
        key = repeat_group_key(row)
        expected.setdefault(key, repeat_group_descriptor(row))

    group_summaries: dict[str, Any] = {}
    for key, descriptor in sorted(expected.items()):
        rows = groups.get(key, [])
        failed_rows = [row for row in rows if row.get("status") != "ok"]
        metrics = {
            "cache_stage_seconds": summarize_metric_values(
                numeric_value(row.get("cache_stage_seconds")) for row in rows
            ),
            "build_seconds": summarize_metric_values(
                numeric_value(row.get("build_seconds")) for row in rows
            ),
            "stage_minus_build_seconds": summarize_metric_values(
                numeric_value(row.get("stage_minus_build_seconds")) for row in rows
            ),
            "cache_size_bytes": summarize_metric_values(
                numeric_value(row.get("cache_size_bytes")) for row in rows
            ),
            "sample_payload_seconds": summarize_metric_values(
                numeric_value(row.get("split_phase_totals", {}).get("sample_payload_seconds"))
                for row in rows
            ),
            "hash_seconds": summarize_metric_values(
                numeric_value(row.get("split_phase_totals", {}).get("hash_seconds"))
                for row in rows
            ),
            "metadata_write_seconds": summarize_metric_values(
                numeric_value(row.get("split_phase_totals", {}).get("metadata_write_seconds"))
                for row in rows
            ),
            "image_flush_seconds": summarize_metric_values(
                numeric_value(row.get("split_phase_totals", {}).get("image_flush_seconds"))
                for row in rows
            ),
        }
        content_keys = sorted(
            {
                str(row.get("content_key"))
                for row in rows
                if row.get("content_key") is not None
            }
        )
        evidence_modes = count_values(str(row.get("evidence_mode") or "unknown") for row in rows)
        if failed_rows:
            status = "failed"
        elif rows and len(rows) >= int(descriptor.get("repeat_count") or 1):
            status = "ok"
        else:
            status = "running"
        group_summaries[key] = {
            **descriptor,
            "status": status,
            "rows": rows,
            "metrics": metrics,
            "diagnostics": {
                "evidence_modes": evidence_modes,
                "cache_reused_count": sum(1 for row in rows if row.get("cache_reused") is True),
                "cold_rebuild_count": sum(
                    1 for row in rows if row.get("evidence_mode") == "cold_rebuild"
                ),
                "content_keys": content_keys,
                "single_content_key": len(content_keys) <= 1 if rows else None,
                "cache_dirs": sorted(
                    {
                        str(row.get("cache_dir"))
                        for row in rows
                        if row.get("cache_dir") is not None
                    }
                ),
            },
        }

    errors = [
        f"{key}: {error}"
        for key, group in group_summaries.items()
        for row in group.get("rows", [])
        for error in row.get("errors", [])
    ]
    if any(group.get("status") == "failed" for group in group_summaries.values()):
        status = "failed"
    elif group_summaries and all(group.get("status") == "ok" for group in group_summaries.values()):
        status = "ok"
    else:
        status = "running"
    return {
        "schema_version": 1,
        "status": status,
        "classification": (
            "cache_wait_evidence_failed"
            if status == "failed"
            else "cache_wait_evidence_recorded"
            if status == "ok"
            else "cache_wait_evidence_pending"
        ),
        "requirements": {
            "cold_evidence": "cache_reused=false and preflight action rebuild/blocked_destructive_rebuild",
            "reuse_evidence": "cache_reused=true or preflight action reuse",
            "required_artifacts": [
                "cache-report.json",
                "cache-preflight.json",
                "run-summary.json",
                "summary-consistency.json",
            ],
            "required_identity_fields": [
                "source_manifest_checksum",
                "transform_fingerprint",
                "target_fingerprint",
                "content_key",
                "cache_dtype",
                "image_size",
            ],
        },
        "groups": group_summaries,
        "errors": errors,
    }


def cache_wait_row_summary(result: dict[str, Any]) -> dict[str, Any]:
    cache_report = result.get("cache_report") or {}
    cache_preflight = result.get("cache_preflight") or {}
    errors: list[str] = []
    if not cache_report:
        errors.append("missing cache-report.json")
        cache_report = {}
    if not cache_preflight:
        errors.append("missing cache-preflight.json")
        cache_preflight = {}
    if cache_report.get("cache_schema_version") not in (None, 1):
        errors.append(f"unsupported cache schema {cache_report.get('cache_schema_version')!r}")
    if cache_report.get("cache_reused") is None:
        errors.append("cache_reused missing")
    for field in ("build_seconds", "cache_stage_seconds", "cache_size_bytes"):
        if numeric_value(cache_report.get(field)) is None:
            errors.append(f"{field} missing or non-numeric")

    identity = cache_report.get("cache_identity") or {}
    if not isinstance(identity, dict):
        identity = {}
    for field in (
        "source_manifest_checksum",
        "transform_fingerprint",
        "target_fingerprint",
        "content_key",
        "cache_dtype",
        "image_size",
    ):
        if identity.get(field) is None:
            errors.append(f"cache identity {field} missing")

    preflight_action = cache_preflight.get("action")
    cache_reused = cache_report.get("cache_reused")
    if cache_reused is True or preflight_action == "reuse":
        evidence_mode = "reuse"
    elif cache_reused is False and preflight_action == "blocked_destructive_rebuild":
        evidence_mode = "blocked_destructive_rebuild"
    elif cache_reused is False:
        evidence_mode = "cold_rebuild"
    else:
        evidence_mode = "unknown"

    split_phase_totals = cache_split_phase_totals(cache_report)
    build_seconds = numeric_value(cache_report.get("build_seconds"))
    stage_seconds = numeric_value(cache_report.get("cache_stage_seconds"))
    stage_minus_build_seconds = None
    if evidence_mode == "cold_rebuild" and stage_seconds is not None and build_seconds is not None:
        stage_minus_build_seconds = stage_seconds - build_seconds
    return {
        "run_id": result.get("run_id"),
        "status": "failed" if errors else "ok",
        "errors": errors,
        "evidence_mode": evidence_mode,
        "cache_reused": cache_reused,
        "preflight_action": preflight_action,
        "preflight_reasons": cache_preflight.get("reasons") or [],
        "cache_dir": cache_report.get("cache_dir"),
        "cache_key_mode": cache_report.get("cache_key_mode"),
        "cache_kind": cache_report.get("cache_kind"),
        "cache_dtype": cache_report.get("dtype") or identity.get("cache_dtype"),
        "image_size": cache_report.get("image_size") or identity.get("image_size"),
        "cache_splits": list(cache_report.get("split_names") or []),
        "content_key": identity.get("content_key"),
        "source_manifest_checksum": (
            identity.get("source_manifest_checksum")
            or cache_report.get("source_manifest_checksum")
        ),
        "transform_fingerprint": (
            identity.get("transform_fingerprint")
            or cache_report.get("transform_fingerprint")
        ),
        "target_fingerprint": identity.get("target_fingerprint"),
        "build_seconds": build_seconds,
        "cache_stage_seconds": stage_seconds,
        "stage_minus_build_seconds": stage_minus_build_seconds,
        "cache_size_bytes": cache_report.get("cache_size_bytes"),
        "mean_std_seconds": cache_report.get("mean_std_seconds"),
        "split_phase_totals": split_phase_totals,
        "split_samples": {
            split: details.get("samples")
            for split, details in (cache_report.get("splits") or {}).items()
            if isinstance(details, dict)
        },
        "split_image_bytes": {
            split: details.get("image_bytes")
            for split, details in (cache_report.get("splits") or {}).items()
            if isinstance(details, dict)
        },
    }


def cache_split_phase_totals(cache_report: dict[str, Any]) -> dict[str, float]:
    totals: dict[str, float] = {}
    splits = cache_report.get("splits") or {}
    if not isinstance(splits, dict):
        return totals
    fields = (
        "build_seconds",
        "sample_payload_seconds",
        "metadata_write_seconds",
        "image_flush_seconds",
        "labels_masks_write_seconds",
        "hash_seconds",
        "image_bytes",
    )
    for details in splits.values():
        if not isinstance(details, dict):
            continue
        for field in fields:
            value = numeric_value(details.get(field))
            if value is not None:
                totals[field] = totals.get(field, 0.0) + value
    return totals


def count_values(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def repeat_metric_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        "train_samples_per_second": summarize_metric_values(
            extract_metric(result, "train_samples_per_second") for result in results
        ),
        "profile_end_to_end_samples_per_second": summarize_metric_values(
            extract_metric(result, "profile_end_to_end_samples_per_second") for result in results
        ),
        "loader_samples_per_second": summarize_metric_values(
            extract_metric(result, "loader_samples_per_second") for result in results
        ),
        "data_wait_percent": summarize_metric_values(
            extract_metric(result, "data_wait_percent") for result in results
        ),
        "train_native_prefetch_read_ms_per_batch": summarize_metric_values(
            extract_metric(result, "train_native_prefetch_read_ms_per_batch")
            for result in results
        ),
        "train_native_prefetch_scatter_ms_per_batch": summarize_metric_values(
            extract_metric(result, "train_native_prefetch_scatter_ms_per_batch")
            for result in results
        ),
        "train_native_prefetch_read_scatter_ms_per_batch": summarize_metric_values(
            extract_metric(result, "train_native_prefetch_read_scatter_ms_per_batch")
            for result in results
        ),
        "train_native_prefetch_read_scatter_percent": summarize_metric_values(
            extract_metric(result, "train_native_prefetch_read_scatter_percent")
            for result in results
        ),
        "train_native_prefetch_runs_per_batch": summarize_metric_values(
            extract_metric(result, "train_native_prefetch_runs_per_batch")
            for result in results
        ),
        "train_native_prefetch_slot_count": summarize_metric_values(
            extract_metric(result, "train_native_prefetch_slot_count") for result in results
        ),
        "train_native_prefetch_preallocated_batch_buffers": summarize_metric_values(
            extract_metric(result, "train_native_prefetch_preallocated_batch_buffers")
            for result in results
        ),
    }
    for metric in (*PROFILE_PHASE_METRICS, *PROFILE_EVENT_METRICS):
        metrics[metric] = summarize_metric_values(extract_metric(result, metric) for result in results)
    for metric in MEMORY_METRICS:
        metrics[metric] = summarize_metric_values(extract_metric(result, metric) for result in results)
    for metric in RUNTIME_METRICS:
        metrics[metric] = summarize_metric_values(extract_metric(result, metric) for result in results)
    return metrics


def summarize_repeat_comparisons(groups: dict[str, dict[str, Any]]) -> dict[str, Any]:
    raw_groups = [
        (key, group)
        for key, group in groups.items()
        if group.get("baseline") == "pytorch_raw"
        and group.get("status") == "ok"
        and metric_mean(group, "train_samples_per_second") is not None
    ]
    if not raw_groups:
        return {}
    raw_key, raw_group = raw_groups[0]
    comparisons: dict[str, Any] = {}
    for key, group in groups.items():
        if group.get("baseline") == "pytorch_raw":
            continue
        train_speedup = ratio_or_none(
            metric_mean(group, "train_samples_per_second"),
            metric_mean(raw_group, "train_samples_per_second"),
        )
        profile_speedup = ratio_or_none(
            metric_mean(group, "profile_end_to_end_samples_per_second"),
            metric_mean(raw_group, "profile_end_to_end_samples_per_second"),
        )
        comparisons[f"{key}:vs:{raw_key}"] = {
            "candidate": key,
            "baseline": raw_key,
            "status": "ok" if train_speedup is not None else "insufficient_data",
            "train_samples_per_second_speedup": train_speedup,
            "profile_end_to_end_speedup": profile_speedup,
            "phase_delta_ms_per_batch": metric_deltas(group, raw_group, PROFILE_PHASE_METRICS),
            "phase_ratio": metric_ratios(group, raw_group, PROFILE_PHASE_METRICS),
            "memory_delta_mb": metric_deltas(group, raw_group, MEMORY_METRICS),
            "train_samples_per_second_per_gpu_pss_gb_delta": delta_or_none(
                samples_per_second_per_gpu_pss_gb(group),
                samples_per_second_per_gpu_pss_gb(raw_group),
            ),
        }
    return comparisons


def summarize_repeat_prediction_comparisons(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok_results = [result for result in results if result.get("status") == "ok"]
    raw_results = [
        result
        for result in ok_results
        if result.get("baseline") == "pytorch_raw"
        and result_prediction_summary(result).get("status") == "ok"
    ]
    if not raw_results:
        return {}
    by_repeat = {int(result.get("repeat_index") or 0): result for result in raw_results}
    fallback_raw = raw_results[0]
    comparisons: dict[str, Any] = {}
    for result in ok_results:
        if result.get("baseline") == "pytorch_raw":
            continue
        candidate = result_prediction_summary(result)
        if candidate.get("status") != "ok":
            continue
        repeat_index = int(result.get("repeat_index") or 0)
        raw_result = by_repeat.get(repeat_index, fallback_raw)
        raw = result_prediction_summary(raw_result)
        key = (
            f"{repeat_group_key(result)}:r{repeat_index + 1:02d}:"
            f"vs:{repeat_group_key(raw_result)}:r{int(raw_result.get('repeat_index') or 0) + 1:02d}"
        )
        comparisons[key] = paired_prediction_summary(candidate=candidate, raw=raw)
    return comparisons


def summarize_repeat_train_order_comparisons(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok_results = [result for result in results if result.get("status") == "ok"]
    raw_results = [
        result
        for result in ok_results
        if result.get("baseline") == "pytorch_raw"
        and result_train_order_summary(result).get("status") == "ok"
    ]
    if not raw_results:
        return {}
    by_repeat = {int(result.get("repeat_index") or 0): result for result in raw_results}
    fallback_raw = raw_results[0]
    comparisons: dict[str, Any] = {}
    for result in ok_results:
        if result.get("baseline") == "pytorch_raw":
            continue
        candidate = result_train_order_summary(result)
        if candidate.get("status") != "ok":
            continue
        repeat_index = int(result.get("repeat_index") or 0)
        raw_result = by_repeat.get(repeat_index, fallback_raw)
        raw = result_train_order_summary(raw_result)
        key = (
            f"{repeat_group_key(result)}:r{repeat_index + 1:02d}:"
            f"vs:{repeat_group_key(raw_result)}:r{int(raw_result.get('repeat_index') or 0) + 1:02d}"
        )
        comparisons[key] = paired_train_order_summary(candidate=candidate, raw=raw)
    return comparisons


def repeat_train_order_pairing_errors(
    results: list[dict[str, Any]],
    comparisons: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    for result in results:
        if result.get("status") != "ok" or result.get("baseline") == "pytorch_raw":
            continue
        metadata = ((result.get("environment") or {}).get("run_metadata") or {})
        if not bool(metadata.get("paired_train_order")):
            continue
        repeat_index = int(result.get("repeat_index") or 0)
        prefix = f"{repeat_group_key(result)}:r{repeat_index + 1:02d}:"
        matched = [
            comparison
            for key, comparison in comparisons.items()
            if key.startswith(prefix)
        ]
        if not matched:
            errors.append(
                f"{repeat_group_key(result)} repeat {repeat_index + 1} has no paired "
                "train-order comparison against pytorch_raw"
            )
            continue
        if not any(isinstance(comparison, dict) and comparison.get("paired") is True for comparison in matched):
            errors.append(
                f"{repeat_group_key(result)} repeat {repeat_index + 1} train order is not paired"
            )
    return errors


def repeat_group_stability(metrics: dict[str, Any], *, modal_gpu: str = "") -> dict[str, Any]:
    thresholds = stability_thresholds_for(modal_gpu)
    train_metric = metrics.get("train_samples_per_second") or {}
    return {
        "metric": "train_samples_per_second",
        "classification": classify_metric_cv(train_metric, thresholds),
        "cv_percent": numeric_value(train_metric.get("cv_percent")),
        "count": int(train_metric.get("count") or 0),
        "thresholds": thresholds,
    }


def stability_thresholds_for(modal_gpu: str = "") -> dict[str, float]:
    gpu = (modal_gpu or "").upper()
    if gpu in STABILITY_THRESHOLDS:
        return dict(STABILITY_THRESHOLDS[gpu])
    return dict(STABILITY_THRESHOLDS["H100"])


def classify_metric_cv(metric: dict[str, Any], thresholds: dict[str, float]) -> str:
    count = int(metric.get("count") or 0)
    if count < 2:
        return "insufficient_repeats"
    cv = numeric_value(metric.get("cv_percent"))
    if cv is None:
        return "insufficient_data"
    if cv <= thresholds["train_samples_per_second_cv_ok_percent"]:
        return "ok"
    if cv <= thresholds["train_samples_per_second_cv_warn_percent"]:
        return "warn"
    return "reject"


def load_comparator_repeat_summary(path: Path | None) -> tuple[str, dict[str, Any]] | None:
    if path is None:
        return None
    batch_summary_path = path / "batch-summary.json"
    repeat_summary_path = path / "repeat-summary.json"
    if batch_summary_path.exists():
        summary = load_json_if_exists(batch_summary_path)
        return str(summary.get("batch_id") or path.name), summary.get("repeat_summary") or {}
    if repeat_summary_path.exists():
        return path.name, load_json_if_exists(repeat_summary_path)
    return None


def promotion_readiness_report(
    repeat_summary: dict[str, Any],
    *,
    batch_id: str,
    modal_gpu: str = "",
    comparator_summary: dict[str, Any] | None = None,
    comparator_batch_id: str | None = None,
) -> dict[str, Any]:
    groups = repeat_summary.get("groups") or {}
    raw_key, raw_group, raw_source = promotion_denominator(
        groups=groups,
        batch_id=batch_id,
        comparator_summary=comparator_summary,
        comparator_batch_id=comparator_batch_id,
    )
    raw_comparators = raw_comparator_readiness(groups, modal_gpu=modal_gpu)
    candidates: dict[str, Any] = {}
    for key, group in groups.items():
        if group.get("baseline") == "pytorch_raw":
            continue
        if group.get("status") != "ok":
            continue
        candidate_stability = group.get("stability") or repeat_group_stability(
            group.get("metrics") or {},
            modal_gpu=modal_gpu,
        )
        candidate_reasons: list[str] = []
        candidate_status = "eligible"
        if not raw_group:
            candidate_status = "rejected"
            candidate_reasons.append(
                "H100 medkit speed promotion requires a same-batch raw row or an external stable raw-control batch"
            )
        else:
            raw_stability = raw_group.get("stability") or repeat_group_stability(
                raw_group.get("metrics") or {},
                modal_gpu=modal_gpu,
            )
            raw_classification = raw_stability.get("classification")
            candidate_classification = candidate_stability.get("classification")
            if raw_classification == "reject":
                candidate_status = "rejected"
                candidate_reasons.append("raw comparator train_samples_per_second CV exceeds reject threshold")
            elif raw_classification == "warn":
                candidate_status = "warning"
                candidate_reasons.append("raw comparator train_samples_per_second CV is in warning range")
            elif raw_classification != "ok":
                candidate_status = "rejected"
                candidate_reasons.append(f"raw comparator stability is {raw_classification!r}")
            if candidate_classification == "reject":
                candidate_status = "rejected"
                candidate_reasons.append("candidate train_samples_per_second CV exceeds reject threshold")
            elif candidate_classification == "warn" and candidate_status == "eligible":
                candidate_status = "warning"
                candidate_reasons.append("candidate train_samples_per_second CV is in warning range")
            elif candidate_classification != "ok":
                candidate_status = "rejected"
                candidate_reasons.append(f"candidate stability is {candidate_classification!r}")
        candidates[key] = {
            "status": candidate_status,
            "reasons": candidate_reasons,
            "candidate_stability": candidate_stability,
            "speedup_denominator": (
                {
                    "batch_id": raw_source.get("batch_id"),
                    "group_key": raw_key,
                    "source": raw_source.get("source"),
                    "stability": raw_group.get("stability") if raw_group else None,
                }
                if raw_group
                else None
            ),
        }
    statuses = [candidate.get("status") for candidate in candidates.values()]
    if any(status == "eligible" for status in statuses):
        status = "eligible"
    elif any(status == "warning" for status in statuses):
        status = "warning"
    elif candidates:
        status = "rejected"
    elif any(row.get("status") == "eligible" for row in raw_comparators.values()):
        status = "comparator_ready"
    else:
        status = "no_speed_candidates"
    return {
        "schema_version": 1,
        "status": status,
        "modal_gpu": modal_gpu or None,
        "thresholds": STABILITY_THRESHOLDS,
        "batch_id": batch_id,
        "raw_comparators": raw_comparators,
        "candidates": candidates,
    }


def raw_comparator_readiness(groups: dict[str, Any], *, modal_gpu: str) -> dict[str, Any]:
    comparators: dict[str, Any] = {}
    for key, group in groups.items():
        if group.get("baseline") != "pytorch_raw" or group.get("status") != "ok":
            continue
        stability = group.get("stability") or repeat_group_stability(
            group.get("metrics") or {},
            modal_gpu=modal_gpu,
        )
        classification = stability.get("classification")
        if classification == "ok":
            status = "eligible"
        elif classification == "warn":
            status = "warning"
        else:
            status = "rejected"
        comparators[key] = {
            "status": status,
            "stability": stability,
            "role": "raw_speed_denominator",
        }
    return comparators


def promotion_denominator(
    *,
    groups: dict[str, Any],
    batch_id: str,
    comparator_summary: dict[str, Any] | None,
    comparator_batch_id: str | None,
) -> tuple[str | None, dict[str, Any] | None, dict[str, str]]:
    if comparator_summary:
        comparator_groups = comparator_summary.get("groups") or {}
        raw_groups = [
            (key, group)
            for key, group in comparator_groups.items()
            if group.get("baseline") == "pytorch_raw" and group.get("status") == "ok"
        ]
        if raw_groups:
            key, group = raw_groups[0]
            return key, group, {
                "source": "external_comparator_batch",
                "batch_id": comparator_batch_id or "external",
            }
    raw_groups = [
        (key, group)
        for key, group in groups.items()
        if group.get("baseline") == "pytorch_raw" and group.get("status") == "ok"
    ]
    if raw_groups:
        key, group = raw_groups[0]
        return key, group, {"source": "same_batch", "batch_id": batch_id}
    return None, None, {}


def result_prediction_summary(result: dict[str, Any]) -> dict[str, Any]:
    baseline = str(result.get("baseline") or "")
    predictions = result.get("predictions") or {}
    summary = (predictions.get("baselines") or {}).get(baseline)
    return summary if isinstance(summary, dict) else {}


def result_train_order_summary(result: dict[str, Any]) -> dict[str, Any]:
    baseline = str(result.get("baseline") or "")
    train_order = result.get("train_order") or {}
    summary = (train_order.get("baselines") or {}).get(baseline)
    return summary if isinstance(summary, dict) else {}


def paired_train_order_summary(
    *,
    candidate: dict[str, Any],
    raw: dict[str, Any],
) -> dict[str, Any]:
    candidate_hashes = candidate.get("hashes") or {}
    raw_hashes = raw.get("hashes") or {}
    return {
        "paired": (
            candidate_hashes.get("train_sample_order_hash")
            == raw_hashes.get("train_sample_order_hash")
            and candidate_hashes.get("train_sample_multiset_hash")
            == raw_hashes.get("train_sample_multiset_hash")
            and candidate_hashes.get("dropped_samples_by_epoch_hash")
            == raw_hashes.get("dropped_samples_by_epoch_hash")
        ),
        "identical_train_order": (
            candidate_hashes.get("train_sample_order_hash")
            == raw_hashes.get("train_sample_order_hash")
        ),
        "identical_train_sample_multiset": (
            candidate_hashes.get("train_sample_multiset_hash")
            == raw_hashes.get("train_sample_multiset_hash")
        ),
        "identical_dropped_samples_by_epoch": (
            candidate_hashes.get("dropped_samples_by_epoch_hash")
            == raw_hashes.get("dropped_samples_by_epoch_hash")
        ),
        "identical_batch_label_sums": (
            candidate_hashes.get("batch_label_sums_hash")
            == raw_hashes.get("batch_label_sums_hash")
        ),
        "candidate_same_train_order_each_epoch": candidate.get("same_train_order_each_epoch"),
        "raw_same_train_order_each_epoch": raw.get("same_train_order_each_epoch"),
        "candidate_train_batches": candidate.get("train_batches"),
        "raw_train_batches": raw.get("train_batches"),
        "candidate_train_samples": candidate.get("train_samples"),
        "raw_train_samples": raw.get("train_samples"),
        "epoch_deltas": paired_train_order_epoch_deltas(
            candidate_epochs=candidate.get("epoch_summaries") or [],
            raw_epochs=raw.get("epoch_summaries") or [],
        ),
    }


def paired_train_order_epoch_deltas(
    *,
    candidate_epochs: Sequence[dict[str, Any]],
    raw_epochs: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    raw_by_epoch = {
        int(row.get("epoch")): row for row in raw_epochs if row.get("epoch") is not None
    }
    deltas: dict[str, Any] = {}
    for candidate_row in candidate_epochs:
        if candidate_row.get("epoch") is None:
            continue
        epoch = int(candidate_row.get("epoch"))
        raw_row = raw_by_epoch.get(epoch, {})
        candidate_dropped = set(str(value) for value in candidate_row.get("dropped_sample_ids", []))
        raw_dropped = set(str(value) for value in raw_row.get("dropped_sample_ids", []))
        deltas[str(epoch)] = {
            "identical_order": candidate_row.get("sample_order_hash")
            == raw_row.get("sample_order_hash"),
            "candidate_dropped_sample_count": candidate_row.get("dropped_sample_count"),
            "raw_dropped_sample_count": raw_row.get("dropped_sample_count"),
            "shared_dropped_sample_count": len(candidate_dropped & raw_dropped),
            "candidate_only_dropped_sample_count": len(candidate_dropped - raw_dropped),
            "raw_only_dropped_sample_count": len(raw_dropped - candidate_dropped),
            "candidate_dropped_target_counts": candidate_row.get("dropped_target_counts"),
            "raw_dropped_target_counts": raw_row.get("dropped_target_counts"),
        }
    return deltas


def paired_prediction_summary(
    *,
    candidate: dict[str, Any],
    raw: dict[str, Any],
) -> dict[str, Any]:
    candidate_ids = [str(value) for value in candidate.get("sample_ids", [])]
    raw_ids = [str(value) for value in raw.get("sample_ids", [])]
    candidate_set = set(candidate_ids)
    raw_set = set(raw_ids)
    candidate_metrics = candidate.get("metric_recompute") or {}
    raw_metrics = raw.get("metric_recompute") or {}
    missing_from_raw = sorted(candidate_set - raw_set)
    missing_from_candidate = sorted(raw_set - candidate_set)
    label_mask_hash_match = (
        (candidate.get("hashes") or {}).get("label_mask_hash")
        == (raw.get("hashes") or {}).get("label_mask_hash")
    )
    identical_target_order = candidate.get("target_names") == raw.get("target_names")
    paired = (
        not missing_from_raw
        and not missing_from_candidate
        and candidate_ids == raw_ids
        and identical_target_order
        and label_mask_hash_match
    )
    return {
        "paired": paired,
        "matched_sample_count": len(candidate_set.intersection(raw_set)),
        "missing_from_raw_count": len(missing_from_raw),
        "missing_from_candidate_count": len(missing_from_candidate),
        "missing_from_medkit_count": len(missing_from_candidate),
        "identical_order": candidate_ids == raw_ids,
        "identical_target_order": identical_target_order,
        "label_mask_hash_match": label_mask_hash_match,
        "threshold_hash_match": (
            (candidate.get("hashes") or {}).get("threshold_hash")
            == (raw.get("hashes") or {}).get("threshold_hash")
        ),
        "macro_auroc_delta": delta_or_none(
            numeric_value(candidate_metrics.get("macro_auroc")),
            numeric_value(raw_metrics.get("macro_auroc")),
        ),
        "macro_auprc_delta": delta_or_none(
            numeric_value(candidate_metrics.get("macro_auprc")),
            numeric_value(raw_metrics.get("macro_auprc")),
        ),
        "target_metric_deltas": target_metric_deltas(candidate_metrics, raw_metrics),
    }


def target_metric_deltas(candidate_metrics: dict[str, Any], raw_metrics: dict[str, Any]) -> dict[str, Any]:
    candidate_targets = candidate_metrics.get("targets") or {}
    raw_targets = raw_metrics.get("targets") or {}
    deltas: dict[str, Any] = {}
    for target in sorted(set(candidate_targets).intersection(raw_targets)):
        candidate_row = candidate_targets.get(target) or {}
        raw_row = raw_targets.get(target) or {}
        deltas[target] = {
            "auroc_delta": delta_or_none(
                numeric_value(candidate_row.get("auroc")),
                numeric_value(raw_row.get("auroc")),
            ),
            "auprc_delta": delta_or_none(
                numeric_value(candidate_row.get("auprc")),
                numeric_value(raw_row.get("auprc")),
            ),
        }
    return deltas


def repeat_group_diagnostics(metrics: dict[str, Any]) -> dict[str, Any]:
    phase_means = {
        metric: numeric_value(metrics.get(metric, {}).get("mean"))
        for metric in PROFILE_PHASE_METRICS
    }
    phase_means = {metric: value for metric, value in phase_means.items() if value is not None}
    largest_phase = None
    if phase_means:
        largest_metric, largest_value = max(phase_means.items(), key=lambda item: item[1])
        largest_phase = {"metric": largest_metric, "mean_ms_per_batch": largest_value}
    return {
        "largest_profile_phase": largest_phase,
        "train_samples_per_second_per_gpu_pss_gb": samples_per_second_per_gpu_pss_gb(
            {"metrics": metrics}
        ),
        "cache_image_pss_mb": numeric_value(metrics.get("cache_image_pss_mb", {}).get("mean")),
        "gpu_pss_mb": numeric_value(metrics.get("gpu_pss_mb", {}).get("mean")),
        "profile_step_accounted_percent": numeric_value(
            metrics.get("profile_step_accounted_percent", {}).get("mean")
        ),
    }


def metric_mean(group: dict[str, Any], metric: str) -> float | None:
    value = ((group.get("metrics") or {}).get(metric) or {}).get("mean")
    return numeric_value(value)


def metric_deltas(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    metrics: Sequence[str],
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for metric in metrics:
        delta = delta_or_none(metric_mean(candidate, metric), metric_mean(baseline, metric))
        if delta is not None:
            deltas[metric] = delta
    return deltas


def metric_ratios(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    metrics: Sequence[str],
) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for metric in metrics:
        ratio = ratio_or_none(metric_mean(candidate, metric), metric_mean(baseline, metric))
        if ratio is not None:
            ratios[metric] = ratio
    return ratios


def delta_or_none(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


def samples_per_second_per_gpu_pss_gb(group: dict[str, Any]) -> float | None:
    samples_per_second = metric_mean(group, "train_samples_per_second")
    gpu_pss_mb = metric_mean(group, "gpu_pss_mb")
    if samples_per_second is None or gpu_pss_mb is None or gpu_pss_mb <= 0.0:
        return None
    return samples_per_second / (gpu_pss_mb / 1024.0)


def ratio_or_none(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def extract_metric(result: dict[str, Any], metric: str) -> float | None:
    baseline = str(result.get("baseline") or "")
    gpu_row = ((result.get("gpu") or {}).get(baseline) or {})
    loader_row = ((result.get("loader") or {}).get(baseline) or {})
    profile_row = ((result.get("profile") or {}).get(baseline) or {})
    profile_summary = profile_row.get("summary") or {}
    memory = gpu_row.get("memory") or {}
    loader_memory = loader_row.get("memory") or {}
    fields = {
        "train_samples_per_second": gpu_row.get("samples_per_second"),
        "profile_end_to_end_samples_per_second": profile_summary.get(
            "profile_end_to_end_samples_per_s"
        ),
        "loader_samples_per_second": loader_row.get("samples_per_second"),
        "data_wait_percent": gpu_row.get("data_wait_percent"),
        "warmup_ms": gpu_row.get("warmup_ms"),
        "torch_compile_setup_ms": gpu_row.get("torch_compile_setup_ms"),
        "cuda_peak_allocated_mb": gpu_row.get("cuda_peak_allocated_mb"),
        "train_native_prefetch_read_ms_per_batch": gpu_row.get(
            "train_native_prefetch_read_ms_per_batch"
        ),
        "train_native_prefetch_scatter_ms_per_batch": gpu_row.get(
            "train_native_prefetch_scatter_ms_per_batch"
        ),
        "train_native_prefetch_read_scatter_ms_per_batch": gpu_row.get(
            "train_native_prefetch_read_scatter_ms_per_batch"
        ),
        "train_native_prefetch_read_scatter_percent": gpu_row.get(
            "train_native_prefetch_read_scatter_percent"
        ),
        "train_native_prefetch_runs_per_batch": gpu_row.get(
            "train_native_prefetch_runs_per_batch"
        ),
        "train_native_prefetch_slot_count": gpu_row.get("train_native_prefetch_slot_count"),
        "train_native_prefetch_preallocated_batch_buffers": gpu_row.get(
            "train_native_prefetch_preallocated_batch_buffers"
        ),
        "gpu_pss_mb": memory.get("smaps_pss_mb"),
        "gpu_anon_pss_mb": memory.get("smaps_pss_anon_mb"),
        "gpu_file_pss_mb": memory.get("smaps_pss_file_mb"),
        "gpu_private_dirty_mb": memory.get("smaps_private_dirty_mb"),
        "gpu_pinned_estimated_mb": memory.get("estimated_pinned_memory_mb"),
        "cache_image_pss_mb": memory.get("smaps_pss_cache_images_mb"),
        "loader_pss_mb": loader_memory.get("smaps_pss_mb"),
        "loader_cache_image_pss_mb": loader_memory.get("smaps_pss_cache_images_mb"),
    }
    value = fields.get(metric)
    if value is None and metric.startswith("profile_"):
        value = profile_summary.get(f"{metric}_mean")
    if value is None and metric in (
        "profile_step_accounted_percent",
        "profile_residual_step_signed_percent",
        "profile_step_reconciled_percent",
    ):
        value = profile_summary.get(metric)
    return numeric_value(value)


def summarize_metric_values(values: Iterable[float | None]) -> dict[str, Any]:
    cleaned = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not cleaned:
        return {"count": 0, "values": []}
    mean = statistics.fmean(cleaned)
    stdev = statistics.stdev(cleaned) if len(cleaned) > 1 else 0.0
    return {
        "count": len(cleaned),
        "values": [round(value, 6) for value in cleaned],
        "mean": mean,
        "stdev": stdev,
        "cv_percent": 100.0 * stdev / mean if mean else 0.0,
        "min": min(cleaned),
        "max": max(cleaned),
    }


def repeat_group_key(row: Row | dict[str, Any]) -> str:
    baseline = row.baseline if isinstance(row, Row) else str(row.get("baseline"))
    cache_dtype = row.cache_dtype if isinstance(row, Row) else str(row.get("cache_dtype"))
    read_mode = row.read_mode if isinstance(row, Row) else str(row.get("read_mode"))
    return f"{baseline}:{cache_dtype}:{read_mode}"


def repeat_group_descriptor(row: Row | dict[str, Any]) -> dict[str, Any]:
    if isinstance(row, Row):
        return {
            "baseline": row.baseline,
            "cache_dtype": row.cache_dtype,
            "read_mode": row.read_mode,
            "repeat_count": row.repeat_count,
        }
    return {
        "baseline": row.get("baseline"),
        "cache_dtype": row.get("cache_dtype"),
        "read_mode": row.get("read_mode"),
        "repeat_count": int(row.get("repeat_count") or 1),
    }


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
    cache_report: dict[str, Any] | None = None,
    cache_preflight: dict[str, Any] | None = None,
    quality_gate: dict[str, Any] | None = None,
    predictions: dict[str, Any] | None = None,
    train_order: dict[str, Any] | None = None,
    artifact_dir: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    baseline = active.row.baseline
    quality_gate = quality_gate or {}
    summary_consistency = summary_consistency or {}
    metadata = environment.get("run_metadata") or {}
    predictions = predictions or {}
    train_order = train_order or {}
    cache_report = cache_report or {}
    cache_preflight = cache_preflight or {}

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
    if artifact_dir is not None:
        if not cache_report:
            errors.append("missing cache-report.json")
        if not cache_preflight:
            errors.append("missing cache-preflight.json")
    if cache_report or cache_preflight:
        cache_wait_errors = cache_wait_row_summary(
            {
                "run_id": active.run_id,
                "cache_report": cache_report,
                "cache_preflight": cache_preflight,
            }
        ).get("errors", [])
        errors.extend(f"cache wait: {error}" for error in cache_wait_errors)
    if metadata and "quality_gate" in metadata and not quality_gate:
        errors.append("missing quality-gate.json")
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
    compare_modal_artifact(
        errors,
        modal_result=modal_result,
        artifact_name="eval-predictions-summary.json",
        local_artifact=predictions,
    )
    compare_modal_artifact(
        errors,
        modal_result=modal_result,
        artifact_name="train-order-summary.json",
        local_artifact=train_order,
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
    if metadata.get("quality_gate") and quality_gate.get("status") != "ok":
        errors.append(
            f"quality-gate status is {quality_gate.get('status')!r}: "
            + "; ".join(str(error) for error in quality_gate.get("errors", []))
        )
    validate_prediction_artifacts(
        errors=errors,
        baseline=baseline,
        row_dir=artifact_dir,
        quality_row=quality_row if isinstance(quality_row, dict) else {},
        predictions=predictions,
        quality_gate_enabled=bool(metadata.get("quality_gate")),
    )
    validate_train_order_artifacts(
        errors=errors,
        baseline=baseline,
        row_dir=artifact_dir,
        gpu_row=gpu_row,
        train_order=train_order,
        train_order_required=bool(metadata.get("train_order_evidence")),
    )

    if loader_row.get("status") and loader_row.get("status") != "ok":
        errors.append(f"loader status is {loader_row.get('status')!r}")
    if gpu_row.get("status") and gpu_row.get("status") != "ok":
        errors.append(f"gpu status is {gpu_row.get('status')!r}")
    if metadata.get("channels_last"):
        if gpu_row.get("channels_last_active") is not True:
            errors.append("channels-last requested but gpu row did not report it active")
        if gpu_row.get("channels_last_all_checked_batches") is not True:
            errors.append("channels-last requested but not all checked train batches were channels-last")
    if metadata.get("torch_compile"):
        if gpu_row.get("torch_compile_status") != "active":
            errors.append("torch.compile requested but gpu row did not report active compilation")
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
        validate_memory_telemetry(
            errors=errors,
            context=phase,
            memory=row.get("memory") or {},
        )
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
        expected_include_metadata = bool(
            metadata.get("include_metadata") or metadata.get("train_order_evidence")
        )
        if (
            active.row.baseline.startswith("medkit")
            and pipeline.get("include_metadata") is not expected_include_metadata
        ):
            errors.append(
                f"{phase} pipeline include_metadata {pipeline.get('include_metadata')!r} "
                f"!= expected {expected_include_metadata!r}"
            )
        validate_pipeline_request_metadata(
            errors=errors,
            context=phase,
            pipeline=pipeline,
            metadata=metadata,
        )
        validate_cache_pss_semantics(
            errors=errors,
            context=phase,
            baseline=active.row.baseline,
            read_mode=active.row.read_mode,
            memory=row.get("memory") or {},
        )

    return errors


def validate_pipeline_request_metadata(
    *,
    errors: list[str],
    context: str,
    pipeline: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    if not metadata or not pipeline:
        return
    expected_shuffle_blocks = metadata.get("shuffle_block_batches")
    if (
        expected_shuffle_blocks is not None
        and pipeline.get("shuffle_block_batches") != expected_shuffle_blocks
    ):
        errors.append(
            f"{context} pipeline shuffle_block_batches "
            f"{pipeline.get('shuffle_block_batches')!r} != expected {expected_shuffle_blocks!r}"
        )
    if pipeline.get("native_prefetch"):
        for field in ("prefetch_depth", "prefetch_read_workers"):
            expected = metadata.get(field)
            if expected is not None and pipeline.get(field) != expected:
                errors.append(
                    f"{context} pipeline {field} {pipeline.get(field)!r} "
                    f"!= expected {expected!r}"
                )


def validate_prediction_artifacts(
    *,
    errors: list[str],
    baseline: str,
    row_dir: Path | None,
    quality_row: dict[str, Any],
    predictions: dict[str, Any],
    quality_gate_enabled: bool,
) -> None:
    if not quality_gate_enabled:
        return
    if not predictions:
        errors.append("missing eval-predictions-summary.json for quality-gated row")
        return
    baseline_summary = ((predictions.get("baselines") or {}).get(baseline) or {})
    if not isinstance(baseline_summary, dict) or not baseline_summary:
        errors.append(f"eval-predictions-summary missing baseline {baseline!r}")
        return
    if baseline_summary.get("enabled") is not True:
        errors.append(f"{baseline} prediction capture not enabled")
    if baseline_summary.get("status") != "ok":
        errors.append(f"{baseline} prediction summary status is {baseline_summary.get('status')!r}")
    artifact_name = baseline_summary.get("artifact_path")
    if not artifact_name:
        errors.append(f"{baseline} prediction artifact path missing")
    elif row_dir is not None and not (row_dir / str(artifact_name)).exists():
        errors.append(f"{baseline} prediction artifact missing: {artifact_name}")
    if baseline_summary.get("metric_recompute_matches_quality") is not True:
        errors.append(f"{baseline} prediction metrics did not match model-quality.json")
    if baseline_summary.get("metric_recompute_matches_artifact") is not True:
        errors.append(f"{baseline} prediction summary did not match artifact rows")
    quality_capture = quality_row.get("prediction_capture") or {}
    if quality_capture.get("enabled") is not True:
        errors.append(f"{baseline} model-quality prediction_capture not enabled")
    if quality_row.get("metric_recompute_matches_predictions") is not True:
        errors.append(f"{baseline} model-quality metric recompute flag is not true")


def validate_train_order_artifacts(
    *,
    errors: list[str],
    baseline: str,
    row_dir: Path | None,
    gpu_row: dict[str, Any],
    train_order: dict[str, Any],
    train_order_required: bool,
) -> None:
    if not train_order_required:
        return
    if not train_order:
        errors.append("missing train-order-summary.json for train-order-evidence row")
        return
    baseline_summary = ((train_order.get("baselines") or {}).get(baseline) or {})
    if not isinstance(baseline_summary, dict) or not baseline_summary:
        errors.append(f"train-order-summary missing baseline {baseline!r}")
        return
    if baseline_summary.get("enabled") is not True:
        errors.append(f"{baseline} train order evidence not enabled")
    if baseline_summary.get("status") != "ok":
        errors.append(f"{baseline} train order summary status is {baseline_summary.get('status')!r}")
    artifact_name = baseline_summary.get("artifact_path")
    if not artifact_name:
        errors.append(f"{baseline} train order artifact path missing")
    elif row_dir is not None and not (row_dir / str(artifact_name)).exists():
        errors.append(f"{baseline} train order artifact missing: {artifact_name}")
    if baseline_summary.get("artifact_recheck_matches_summary") is not True:
        errors.append(f"{baseline} train order artifact recheck did not match summary")
    capture = gpu_row.get("train_order_capture") or {}
    if capture.get("enabled") is not True:
        errors.append(f"{baseline} gpu train_order_capture not enabled")
    if capture.get("status") != "ok":
        errors.append(f"{baseline} gpu train_order_capture status is {capture.get('status')!r}")


def validate_memory_telemetry(
    *,
    errors: list[str],
    context: str,
    memory: dict[str, Any],
) -> None:
    if not isinstance(memory, dict) or not memory:
        errors.append(f"{context} memory telemetry missing")
        return
    required = (
        "psutil_pss_mb",
        "psutil_uss_mb",
        "smaps_pss_mb",
        "smaps_uss_mb",
        "smaps_pss_file_mb",
        "smaps_pss_anon_mb",
    )
    for field in required:
        value = numeric_value(memory.get(field))
        if value is None or not math.isfinite(value) or value < 0.0:
            errors.append(f"{context} memory {field} invalid: {memory.get(field)!r}")
    sources = memory.get("sources")
    if not isinstance(sources, list) or "/proc/self/smaps" not in sources:
        errors.append(f"{context} memory sources missing /proc/self/smaps")


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
        "gpu_prefetch_reuse_buffers",
        "sync_every_step",
        "channels_last",
        "torch_compile",
        "torch_compile_mode",
        "learning_rate",
        "amp_dtype",
        "model_init",
        "loss_kind",
        "loss_pos_weight",
        "loss_pos_weight_cap",
        "focal_gamma",
        "focal_alpha",
        "quality_gate",
        "quality_min_eval_samples",
        "quality_min_metric_targets",
        "quality_min_macro_auroc",
        "quality_min_macro_auprc",
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
            "gpu_prefetch_reuse_buffers",
            "sync_every_step",
            "channels_last",
            "torch_compile",
            "torch_compile_mode",
            "learning_rate",
            "amp_dtype",
            "model_init",
            "loss_kind",
            "loss_pos_weight",
            "loss_pos_weight_cap",
            "focal_gamma",
            "focal_alpha",
            "quality_gate",
            "quality_min_eval_samples",
            "quality_min_metric_targets",
            "quality_min_macro_auroc",
            "quality_min_macro_auprc",
            "eval_predictions",
            "train_order_evidence",
            "paired_train_order",
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
    ground_truth_artifact = artifacts.get("training_ground_truth")
    if ground_truth_artifact is not None and ground_truth_artifact != "training-ground-truth.json":
        errors.append("provenance training_ground_truth artifact path invalid")
    splits_artifact = artifacts.get("splits")
    if splits_artifact is not None and splits_artifact != "splits.json":
        errors.append("provenance splits artifact path invalid")
    predictions_artifact = artifacts.get("eval_predictions_summary")
    if predictions_artifact is not None and predictions_artifact != "eval-predictions-summary.json":
        errors.append("provenance eval_predictions_summary artifact path invalid")
    train_order_artifact = artifacts.get("train_order_summary")
    if train_order_artifact is not None and train_order_artifact != "train-order-summary.json":
        errors.append("provenance train_order_summary artifact path invalid")


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
        "profile_step_accounted_percent",
        "profile_residual_step_signed_total_ms",
        "profile_residual_step_signed_percent",
        "profile_step_reconciled_percent",
    ]
    for field in (
        "data_wait_ms",
        "h2d_ms",
        "batch_prepare_ms",
        "batch_prepare_wall_ms",
        "zero_grad_wall_ms",
        "forward_ms",
        "backward_ms",
        "optimizer_ms",
        "prefetch_maintenance_wall_ms",
        "accounted_step_ms",
        "residual_step_ms",
        "residual_step_ms_signed",
        "residual_step_percent",
        "total_step_ms",
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
    required_record_fields = {
        "data_wait_ms",
        "h2d_ms",
        "forward_ms",
        "backward_ms",
        "optimizer_ms",
        "total_step_ms",
    }
    optional_record_fields = {
        "batch_prepare_ms",
        "batch_prepare_wall_ms",
        "zero_grad_wall_ms",
        "prefetch_maintenance_wall_ms",
        "accounted_step_ms",
        "residual_step_ms",
        "residual_step_ms_signed",
        "residual_step_percent",
    }
    signed_record_fields = {"residual_step_ms_signed", "residual_step_percent"}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"{context} record {index} is not an object")
            continue
        samples = numeric_value(record.get("samples"))
        if samples is None or samples <= 0:
            errors.append(f"{context} record {index} samples invalid: {record.get('samples')!r}")
        else:
            profiled_samples += int(samples)
        for field in (*sorted(required_record_fields), *sorted(optional_record_fields)):
            if field in optional_record_fields and field not in record:
                continue
            value = numeric_value(record.get(field))
            value_valid = (
                value is not None
                and math.isfinite(value)
                and (value >= 0.0 or field in signed_record_fields)
            )
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
        f"Repeat: `{row.repeat_index + 1}/{row.repeat_count}`\n\n"
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
            "repeats": args.repeats,
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
    repeats = int(settings.get("repeats") or 1)
    if repeats <= 0:
        raise ValueError("repeats must be greater than zero")
    validate_choices("cache dtype", cache_dtypes, {"float32", "float16", "uint8"})
    validate_choices("read mode", read_modes, {"mmap", "stream"})
    rows = []
    for repeat_index in range(repeats):
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
                        repeat_index=repeat_index,
                        repeat_count=repeats,
                    )
                )
    if not rows:
        raise ValueError("No baselines provided")
    return rows


def audit_batch(batch_dir: Path, *, comparator_batch: Path | None = None) -> int:
    config = load_json_if_exists(batch_dir / "batch-config.json")
    if not config:
        raise FileNotFoundError(f"Missing batch-config.json in {batch_dir}")
    batch_id = str(config.get("batch_id") or batch_dir.name)
    settings = config.get("settings") or {}
    rows = [Row(**row) for row in config.get("rows", [])]
    if not rows:
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
    write_batch_summary(
        batch_dir,
        batch_id,
        completed,
        running=[],
        pending=[],
        batch_started=started,
        modal_gpu=str(settings.get("modal_gpu") or ""),
        comparator_batch=comparator_batch,
    )
    failures = [row for row in completed if row.get("status") != "ok"]
    repeat_summary = load_json_if_exists(batch_dir / "repeat-summary.json")
    repeat_errors = list(repeat_summary.get("train_order_pairing_errors") or [])
    status = "failed" if failures or repeat_errors else "ok"
    print(
        json.dumps(
            {
                "batch_id": batch_id,
                "status": status,
                "failures": failures,
                "repeat_errors": repeat_errors,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if failures or repeat_errors else 0


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
    if row.repeat_count > 1:
        return f"{batch_id}-r{row.repeat_index + 1:02d}-{row.name}"
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
