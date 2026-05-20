"""Run the CXR classification benchmark on Modal GPU."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import modal


APP_ROOT = Path("/opt/medkit-rs")
VOLUME_ROOT = Path("/cache")
WORK_DIR = VOLUME_ROOT / "cxr"
REMOTE_REPORT_ROOT = VOLUME_ROOT / "results" / "cxr"
MODAL_GPU = os.environ.get("MEDKIT_MODAL_GPU", "L4")
MEDKIT_PACKAGE = os.environ.get("MEDKIT_MODAL_MEDKIT_PACKAGE", "medkit-rs==0.1.1")
USE_PUBLISHED_MEDKIT = os.environ.get("MEDKIT_MODAL_USE_PYPI", "1") != "0"
LOCAL_REPO_ROOT = next(
    (
        parent
        for parent in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
        if (parent / "Cargo.toml").exists()
        and "[workspace]" in (parent / "Cargo.toml").read_text()
    ),
    Path.cwd(),
)


def ignore_modal_source(path: Path) -> bool:
    parts = set(path.parts)
    if parts.intersection(
        {
            ".git",
            "target",
            "data",
            "references",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            ".mypy_cache",
            ".venv",
            "venv",
            "reports",
        }
    ):
        return True
    if path.suffix in {".pyc", ".profraw", ".profdata"}:
        return True
    return False


base_image = (
    modal.Image.debian_slim(python_version="3.11")
    if USE_PUBLISHED_MEDKIT
    else modal.Image.from_registry("rust:1.88-bookworm", add_python="3.11")
)

image = (
    base_image.apt_install("ca-certificates", "git")
    .pip_install(
        "uv",
        "torch",
        "torchvision",
        "torchxrayvision",
        "monai",
        "webdataset",
        "datasets",
        "pillow",
        "psutil",
        "scikit-learn",
        "numpy",
        *([MEDKIT_PACKAGE] if USE_PUBLISHED_MEDKIT else []),
    )
    .run_commands(
        "uv pip install --system --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda130"
    )
    .add_local_dir(LOCAL_REPO_ROOT, str(APP_ROOT), copy=True, ignore=ignore_modal_source)
    .workdir(str(APP_ROOT))
)

if not USE_PUBLISHED_MEDKIT:
    image = image.run_commands("rustc --version && cargo --version && uv pip install --system .")


app = modal.App("medkit-rs-cxr-classification")
volume = modal.Volume.from_name("medkit-rs-cxr", create_if_missing=True)


@app.function(
    image=image,
    gpu=MODAL_GPU,
    cpu=16,
    memory=65536,
    ephemeral_disk=524288,
    timeout=60 * 60 * 6,
    volumes={str(VOLUME_ROOT): volume},
)
def run_cxr_benchmark(
    run_id: str,
    max_samples: int = 6000,
    max_train: int = 4096,
    max_val: int = 1024,
    max_test: int = 1024,
    image_size: int = 224,
    cache_dtype: str = "float32",
    batch_size: int = 64,
    workers: int = 4,
    epochs: int = 1,
    loader_batches: int = 64,
    warmup_batches: int = 2,
    profile_batches: int = 0,
    drop_last_train: bool = False,
    prefetch_depth: int = 1,
    prefetch_read_workers: int = 1,
    read_mode: str = "mmap",
    shuffle_block_batches: int = 0,
    gpu_prefetch_batches: int = 0,
    gpu_prefetch_reuse_buffers: bool = False,
    sync_every_step: bool = True,
    loss_pos_weight: str = "none",
    quality_gate: bool = False,
    quality_min_eval_samples: int = 0,
    quality_min_metric_targets: int = 0,
    quality_min_macro_auroc: float = 0.0,
    quality_min_macro_auprc: float = 0.0,
    include_metadata: bool = False,
    max_train_batches: int = 0,
    max_eval_batches: int = 0,
    baselines: str = "pytorch_raw,monai_raw,medkit_cached_mmap,medkit_pinned_prefetch",
    splits: str = "",
    smoke: bool = False,
    force_rematerialize: bool = False,
    force_cache: bool = False,
) -> dict[str, Any]:
    os.chdir(APP_ROOT)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    REMOTE_REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "crates/medkit-benchmarks/scripts/cxr_classification_benchmark.py",
        "--work-dir",
        str(WORK_DIR),
        "--report-dir",
        str(REMOTE_REPORT_ROOT),
        "--run-id",
        run_id,
        "--max-samples",
        str(max_samples),
        "--max-train",
        str(max_train),
        "--max-val",
        str(max_val),
        "--max-test",
        str(max_test),
        "--image-size",
        str(image_size),
        "--cache-dtype",
        cache_dtype,
        "--batch-size",
        str(batch_size),
        "--workers",
        str(workers),
        "--epochs",
        str(epochs),
        "--loader-batches",
        str(loader_batches),
        "--warmup-batches",
        str(warmup_batches),
        "--profile-batches",
        str(profile_batches),
        "--drop-last-train" if drop_last_train else "--no-drop-last-train",
        "--prefetch-depth",
        str(prefetch_depth),
        "--prefetch-read-workers",
        str(prefetch_read_workers),
        "--read-mode",
        read_mode,
        "--shuffle-block-batches",
        str(shuffle_block_batches),
        "--gpu-prefetch-batches",
        str(gpu_prefetch_batches),
        "--gpu-prefetch-reuse-buffers"
        if gpu_prefetch_reuse_buffers
        else "--no-gpu-prefetch-reuse-buffers",
        "--sync-every-step" if sync_every_step else "--no-sync-every-step",
        "--loss-pos-weight",
        loss_pos_weight,
        "--quality-gate" if quality_gate else "--no-quality-gate",
        "--quality-min-eval-samples",
        str(quality_min_eval_samples),
        "--quality-min-metric-targets",
        str(quality_min_metric_targets),
        "--quality-min-macro-auroc",
        str(quality_min_macro_auroc),
        "--quality-min-macro-auprc",
        str(quality_min_macro_auprc),
        "--include-metadata" if include_metadata else "--no-include-metadata",
        "--baselines",
        baselines,
        "--device",
        "cuda:0",
    ]
    if splits:
        command.extend(["--splits", splits])
    if max_train_batches:
        command.extend(["--max-train-batches", str(max_train_batches)])
    if max_eval_batches:
        command.extend(["--max-eval-batches", str(max_eval_batches)])
    if smoke:
        command.append("--smoke")
    if force_rematerialize:
        command.append("--force-rematerialize")
    if force_cache:
        command.append("--force-cache")

    env = os.environ.copy()
    env["MEDKIT_BENCHMARK_USE_LOCAL_SOURCE"] = "0" if USE_PUBLISHED_MEDKIT else "1"
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=APP_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.perf_counter() - start
    report_dir = REMOTE_REPORT_ROOT / run_id
    artifacts = collect_report_artifacts(report_dir)
    volume.commit()
    if completed.returncode != 0:
        return {
            "status": "failed",
            "returncode": completed.returncode,
            "elapsed_seconds": elapsed,
            "command": command,
            "output": completed.stdout,
            "report_dir": str(report_dir),
            "artifacts": artifacts,
        }
    return {
        "status": "ok",
        "elapsed_seconds": elapsed,
        "command": command,
        "output": completed.stdout,
        "report_dir": str(report_dir),
        "artifacts": artifacts,
    }


@app.local_entrypoint()
def main(
    run_id: str = "",
    max_samples: int = 6000,
    max_train: int = 4096,
    max_val: int = 1024,
    max_test: int = 1024,
    image_size: int = 224,
    cache_dtype: str = "float32",
    batch_size: int = 64,
    workers: int = 4,
    epochs: int = 1,
    loader_batches: int = 64,
    warmup_batches: int = 2,
    profile_batches: int = 0,
    drop_last_train: bool = False,
    prefetch_depth: int = 1,
    prefetch_read_workers: int = 1,
    read_mode: str = "mmap",
    shuffle_block_batches: int = 0,
    gpu_prefetch_batches: int = 0,
    gpu_prefetch_reuse_buffers: bool = False,
    sync_every_step: bool = True,
    loss_pos_weight: str = "none",
    quality_gate: bool = False,
    quality_min_eval_samples: int = 0,
    quality_min_metric_targets: int = 0,
    quality_min_macro_auroc: float = 0.0,
    quality_min_macro_auprc: float = 0.0,
    include_metadata: bool = False,
    max_train_batches: int = 0,
    max_eval_batches: int = 0,
    baselines: str = "pytorch_raw,monai_raw,medkit_cached_mmap,medkit_pinned_prefetch",
    splits: str = "",
    smoke: bool = False,
    force_rematerialize: bool = False,
    force_cache: bool = False,
) -> None:
    if not run_id:
        gpu_label = MODAL_GPU.lower().replace(" ", "-").replace(":", "-")
        mode = "smoke" if smoke else gpu_label
        run_id = f"nih-cxr14-320-{mode}-size{image_size}-n{max_samples}-{time.strftime('%Y%m%d-%H%M%S')}"
    result = run_cxr_benchmark.remote(
        run_id=run_id,
        max_samples=max_samples,
        max_train=max_train,
        max_val=max_val,
        max_test=max_test,
        image_size=image_size,
        cache_dtype=cache_dtype,
        batch_size=batch_size,
        workers=workers,
        epochs=epochs,
        loader_batches=loader_batches,
        warmup_batches=warmup_batches,
        profile_batches=profile_batches,
        drop_last_train=drop_last_train,
        prefetch_depth=prefetch_depth,
        prefetch_read_workers=prefetch_read_workers,
        read_mode=read_mode,
        shuffle_block_batches=shuffle_block_batches,
        gpu_prefetch_batches=gpu_prefetch_batches,
        gpu_prefetch_reuse_buffers=gpu_prefetch_reuse_buffers,
        sync_every_step=sync_every_step,
        loss_pos_weight=loss_pos_weight,
        quality_gate=quality_gate,
        quality_min_eval_samples=quality_min_eval_samples,
        quality_min_metric_targets=quality_min_metric_targets,
        quality_min_macro_auroc=quality_min_macro_auroc,
        quality_min_macro_auprc=quality_min_macro_auprc,
        include_metadata=include_metadata,
        max_train_batches=max_train_batches,
        max_eval_batches=max_eval_batches,
        baselines=baselines,
        splits=splits,
        smoke=smoke,
        force_rematerialize=force_rematerialize,
        force_cache=force_cache,
    )
    local_report_dir = LOCAL_REPO_ROOT / "target" / "reports" / "cxr" / run_id
    local_report_dir.mkdir(parents=True, exist_ok=True)
    for name, value in result.get("artifacts", {}).items():
        suffix = Path(name).suffix
        path = local_report_dir / name
        if suffix == ".json":
            path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        else:
            path.write_text(str(value))
    (local_report_dir / "modal-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summarize_result(result), indent=2, sort_keys=True))


def collect_report_artifacts(report_dir: Path) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    if not report_dir.exists():
        return artifacts
    for path in sorted(report_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix == ".json":
            try:
                artifacts[path.name] = json.loads(path.read_text())
            except Exception as error:
                artifacts[path.name] = {"error": str(error), "raw": path.read_text()}
        elif path.suffix in {".md", ".txt"}:
            artifacts[path.name] = path.read_text()
    return artifacts


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    artifacts = result.get("artifacts", {})
    return {
        "status": result.get("status"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "report_dir": result.get("report_dir"),
        "command": result.get("command"),
        "output": result.get("output"),
        "run_summary": artifacts.get("run-summary.json"),
        "gpu_throughput": artifacts.get("gpu-throughput.json"),
        "loader_throughput": artifacts.get("loader-throughput.json"),
        "quality_gate": artifacts.get("quality-gate.json"),
    }


if __name__ == "__main__":
    raise SystemExit("Run with `modal run crates/medkit-benchmarks/scripts/modal_cxr_classification.py`.")
