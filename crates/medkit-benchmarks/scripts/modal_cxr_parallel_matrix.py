"""Launch CXR Modal benchmark rows concurrently and collect local artifacts."""

from __future__ import annotations

import argparse
import json
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


@dataclass
class Row:
    name: str
    baseline: str
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=6000)
    parser.add_argument("--max-train", type=int, default=4096)
    parser.add_argument("--max-val", type=int, default=1024)
    parser.add_argument("--max-test", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--loader-batches", type=int, default=64)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--prefetch-depth", type=int, default=1)
    parser.add_argument("--prefetch-read-workers", type=int, default=1)
    parser.add_argument("--read-mode", choices=("mmap", "stream"), default="mmap")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--force-rematerialize", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M")
    batch_id = args.batch_id or (
        f"nih-cxr14-current-tools-parallel-size{args.image_size}-"
        f"b{args.batch_size}-{timestamp}"
    )
    batch_dir = CURRENT_TOOLS_ROOT / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        Row(name=baseline_to_name(baseline), baseline=baseline, purpose=row_purpose(baseline))
        for baseline in [item.strip() for item in args.baselines.split(",") if item.strip()]
    ]
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
            command = build_command(args, run_id=run_id, baseline=row.baseline)
            print(" ".join(command))
        return 0

    while pending or running:
        while pending and len(running) < max(args.concurrency, 1):
            row = pending.pop(0)
            run_id = run_id_for(batch_id, row)
            row_dir = batch_dir / run_id
            row_dir.mkdir(parents=True, exist_ok=True)
            output_path = row_dir / "modal-output.log"
            command = build_command(args, run_id=run_id, baseline=row.baseline)
            (row_dir / "launcher-command.txt").write_text(" ".join(command) + "\n")
            pain_diary_path = row_dir / "pain-diary.md"
            pain_diary_path.write_text(initial_pain_diary(row, run_id, command))
            output_handle = output_path.open("w")
            process = subprocess.Popen(
                command,
                cwd=LOCAL_REPO_ROOT,
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
        write_batch_summary(batch_dir, batch_id, completed, running, batch_started)

    write_batch_summary(batch_dir, batch_id, completed, running, batch_started)
    failures = [row for row in completed if row.get("status") != "ok"]
    return 1 if failures else 0


def build_command(args: argparse.Namespace, *, run_id: str, baseline: str) -> list[str]:
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
        "--prefetch-depth",
        str(args.prefetch_depth),
        "--prefetch-read-workers",
        str(args.prefetch_read_workers),
        "--read-mode",
        args.read_mode,
        "--baselines",
        baseline,
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


def collect_row(active: RunningRow, batch_dir: Path, returncode: int) -> dict[str, Any]:
    row_dir = batch_dir / active.run_id
    source_dir = SOURCE_REPORT_ROOT / active.run_id
    if source_dir.exists():
        copy_report_artifacts(source_dir, row_dir)
    modal_result = load_json_if_exists(row_dir / "modal-result.json")
    run_summary = load_json_if_exists(row_dir / "run-summary.json")
    loader = load_json_if_exists(row_dir / "loader-throughput.json")
    gpu = load_json_if_exists(row_dir / "gpu-throughput.json")
    quality = load_json_if_exists(row_dir / "model-quality.json")
    elapsed = time.perf_counter() - active.started_at
    status = "ok" if returncode == 0 and run_summary else "failed"
    result = {
        "run_id": active.run_id,
        "baseline": active.row.baseline,
        "purpose": active.row.purpose,
        "returncode": returncode,
        "status": status,
        "elapsed_seconds": elapsed,
        "report_dir": str(row_dir),
        "source_report_dir": str(source_dir),
        "run_summary": run_summary,
        "loader": loader,
        "gpu": gpu,
        "quality": quality,
        "modal_status": modal_result.get("status") if modal_result else None,
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
    batch_started: float,
) -> None:
    summary = {
        "batch_id": batch_id,
        "status": "running" if running else "ok",
        "elapsed_seconds": time.perf_counter() - batch_started,
        "completed": completed,
        "running": [
            {
                "run_id": active.run_id,
                "baseline": active.row.baseline,
                "elapsed_seconds": time.perf_counter() - active.started_at,
            }
            for active in running
        ],
    }
    if any(row.get("status") != "ok" for row in completed):
        summary["status"] = "failed" if not running else "running_with_failures"
    write_json(batch_dir / "batch-summary.json", summary)


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


def baseline_to_name(baseline: str) -> str:
    return baseline.replace("_", "-")


def row_purpose(baseline: str) -> str:
    purposes = {
        "pytorch_raw": "Hand-rolled PyTorch control path.",
        "monai_raw": "MONAI medical-imaging framework path.",
        "torchxrayvision": "CXR-specific toolkit path.",
        "medkit_native_prefetch_pinned": "Reference medkit native-prefetch path.",
    }
    return purposes.get(baseline, "Current-tool benchmark row.")


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
