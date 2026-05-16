"""Run the full MSD Spleen medkit/MONAI/GPU benchmark on Modal."""

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
WORK_DIR = VOLUME_ROOT / "msd-spleen"
RESULTS_DIR = VOLUME_ROOT / "results"
REPO_ROOT = next(
    (
        parent
        for parent in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
        if (parent / "Cargo.toml").exists()
        and "[workspace]" in (parent / "Cargo.toml").read_text()
    ),
    APP_ROOT,
)


def ignore_modal_source(path: Path) -> bool:
    parts = set(path.parts)
    if parts.intersection({".git", "target", "data", "references", "__pycache__", ".venv", "venv"}):
        return True
    if path.name.startswith("_native") and path.suffix in {".so", ".pyd"}:
        return True
    if path.suffix in {".pyc", ".profraw", ".profdata"}:
        return True
    return False


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "build-essential",
        "ca-certificates",
        "curl",
        "pkg-config",
        "libssl-dev",
    )
    .run_commands(
        "curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable"
    )
    .env({"PYO3_NO_PYTHON": "1"})
    .pip_install("torch", "monai", "nibabel", "numpy")
    .add_local_dir(REPO_ROOT, str(APP_ROOT), copy=True, ignore=ignore_modal_source)
    .workdir(str(APP_ROOT))
    .run_commands(
        "/root/.cargo/bin/cargo build -p medkit-cli --release",
        "/root/.cargo/bin/cargo build -p medkit-python --release --features extension-module",
        "cp target/release/libmedkit_rs_native.so python/medkit_rs/_native.abi3.so",
    )
)


app = modal.App("medkit-rs-msd-gpu")
volume = modal.Volume.from_name("medkit-rs-msd", create_if_missing=True)


@app.function(
    image=image,
    gpu="L40S",
    cpu=16,
    memory=65536,
    ephemeral_disk=524288,
    timeout=60 * 60 * 6,
    volumes={str(VOLUME_ROOT): volume},
)
def run_msd_gpu_benchmark(
    samples: int = 10000,
    batch_size: int = 16,
    cases: int = 0,
    patch: str = "96,96,96",
    cache_shape: str = "160,160,160",
    chunk: str = "96,96,96",
    spacing: str = "1.0,1.0,1.0",
    medkit_workers: int = 16,
    monai_workers: int = 0,
    prefetch_batches: int = 3,
    warmup_batches: int = 4,
    model_step: str = "forward",
    include_chunked: bool = True,
) -> dict[str, Any]:
    os.chdir(APP_ROOT)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{APP_ROOT / 'python'}:{env.get('PYTHONPATH', '')}"
    env["RAYON_NUM_THREADS"] = str(medkit_workers)

    import torch

    started = time.strftime("%Y%m%d-%H%M%S")
    prefix = f"msd-spleen-cases{cases}-samples{samples}-batch{batch_size}-{started}"
    workflow_out = RESULTS_DIR / f"{prefix}-workflow.json"

    workflow_cmd = [
        sys.executable,
        "crates/medkit-benchmarks/scripts/msd_spleen_workflow.py",
        "--work-dir",
        str(WORK_DIR),
        "--cases",
        str(cases),
        "--patch",
        patch,
        "--cache-shape",
        cache_shape,
        "--chunk",
        chunk,
        "--spacing",
        spacing,
        "--samples",
        str(samples),
        "--workers",
        str(medkit_workers),
        "--monai-workers",
        str(monai_workers),
        "--medkit-torch-workers",
        "0",
        "--batch-size",
        str(batch_size),
        "--medkit-torch-backend",
        "native-batch",
        "--medkit-bin",
        "target/release/medkit",
        "--python",
        sys.executable,
        "--out",
        str(workflow_out),
    ]
    workflow_stage = run_command("full_msd_workflow", workflow_cmd, env)
    workflow = json.loads(workflow_out.read_text())
    dataset_root = Path(workflow["dataset_root"])
    cache_dir = Path(workflow["reports"]["cache"])
    patches_path = dataset_root / "patches.jsonl"

    dataloader_reports: dict[str, Any] = {
        "resident": workflow.get("medkit_torch"),
    }
    if include_chunked:
        chunked_out = RESULTS_DIR / f"{prefix}-native-chunk-dataloader.json"
        chunked_stage = run_command(
            "native_chunk_dataloader",
            [
                sys.executable,
                "crates/medkit-benchmarks/scripts/medkit_torch_dataset_baseline.py",
                "--cache",
                str(cache_dir),
                "--patches",
                str(patches_path),
                "--samples",
                str(samples),
                "--workers",
                "0",
                "--batch-size",
                str(batch_size),
                "--backend",
                "native-chunk-batch",
                "--out",
                str(chunked_out),
            ],
            env,
        )
        dataloader_reports["chunked"] = json.loads(chunked_out.read_text())
    else:
        chunked_stage = None

    gpu_reports: dict[str, Any] = {}
    gpu_stages: dict[str, Any] = {}
    storages = ["resident", "chunked"] if include_chunked else ["resident"]
    for storage in storages:
        gpu_out = RESULTS_DIR / f"{prefix}-{storage}-gpu-{model_step}.json"
        gpu_stages[storage] = run_command(
            f"{storage}_gpu_loop",
            [
                sys.executable,
                "crates/medkit-benchmarks/scripts/gpu_training_loop.py",
                "--cache",
                str(cache_dir),
                "--patches",
                str(patches_path),
                "--samples",
                str(samples),
                "--batch-size",
                str(batch_size),
                "--storage",
                storage,
                "--prefetch-batches",
                str(prefetch_batches),
                "--warmup-batches",
                str(warmup_batches),
                "--model-step",
                model_step,
                "--pin-memory",
                "--out",
                str(gpu_out),
            ],
            env,
        )
        gpu_reports[storage] = json.loads(gpu_out.read_text())

    result = {
        "modal": {
            "gpu": str(torch.cuda.get_device_name(0)),
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_version": str(torch.__version__),
        },
        "params": {
            "samples": samples,
            "batch_size": batch_size,
            "cases": cases,
            "patch": patch,
            "cache_shape": cache_shape,
            "chunk": chunk,
            "spacing": spacing,
            "medkit_workers": medkit_workers,
            "monai_workers": monai_workers,
            "prefetch_batches": prefetch_batches,
            "warmup_batches": warmup_batches,
            "model_step": model_step,
            "include_chunked": include_chunked,
        },
        "workflow_stage": workflow_stage,
        "chunked_dataloader_stage": chunked_stage,
        "gpu_stages": gpu_stages,
        "workflow": workflow,
        "dataloader": dataloader_reports,
        "gpu_loop": gpu_reports,
        "reports": {
            "workflow": str(workflow_out),
            "results_dir": str(RESULTS_DIR),
        },
    }
    volume.commit()
    return result


@app.local_entrypoint()
def main(
    samples: int = 10000,
    batch_size: int = 16,
    cases: int = 0,
    medkit_workers: int = 16,
    monai_workers: int = 0,
    prefetch_batches: int = 3,
    warmup_batches: int = 4,
    model_step: str = "forward",
    include_chunked: bool = True,
) -> None:
    result = run_msd_gpu_benchmark.remote(
        samples=samples,
        batch_size=batch_size,
        cases=cases,
        medkit_workers=medkit_workers,
        monai_workers=monai_workers,
        prefetch_batches=prefetch_batches,
        warmup_batches=warmup_batches,
        model_step=model_step,
        include_chunked=include_chunked,
    )
    print(json.dumps(result, indent=2))


def run_command(name: str, command: list[str], env: dict[str, str]) -> dict[str, Any]:
    print(f"[{name}] {' '.join(command)}", flush=True)
    start = time.perf_counter()
    completed = subprocess.run(command, cwd=APP_ROOT, env=env, text=True)
    elapsed = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}")
    return {
        "name": name,
        "command": command,
        "elapsed_ms": elapsed * 1000.0,
        "returncode": completed.returncode,
    }
