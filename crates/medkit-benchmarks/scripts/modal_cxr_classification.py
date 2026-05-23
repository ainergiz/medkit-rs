"""Run the CXR classification benchmark on Modal GPU."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import modal


APP_ROOT = Path("/opt/medkit-rs")
VOLUME_ROOT = Path("/cache")
WORK_DIR = VOLUME_ROOT / "cxr"
REMOTE_REPORT_ROOT = VOLUME_ROOT / "results" / "cxr"
MODAL_GPU = os.environ.get("MEDKIT_MODAL_GPU", "L4")
MEDKIT_PACKAGE = os.environ.get("MEDKIT_MODAL_MEDKIT_PACKAGE", "medkit-rs==0.1.1")


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


USE_PUBLISHED_MEDKIT = env_flag("MEDKIT_MODAL_USE_PYPI", default=False)
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
        "pydicom",
        "pylibjpeg",
        "pylibjpeg-libjpeg",
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
    image = image.run_commands(
        "rustc --version && cargo --version && uv pip install --system .",
        (
            "python -c \"import inspect, medkit_rs; "
            "sig = inspect.signature(medkit_rs.MedkitCxrNativePrefetchDataset); "
            "assert 'shuffle_block_batches' in sig.parameters, sig; "
            "print('using local medkit_rs', getattr(medkit_rs, '__version__', None), sig)\""
        ),
    )


app = modal.App("medkit-rs-cxr-classification")
volume = modal.Volume.from_name("medkit-rs-cxr", create_if_missing=True)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    ephemeral_disk=524288,
    timeout=60 * 60,
    volumes={str(VOLUME_ROOT): volume},
)
def prepare_rsna_dataset(
    rsna_root: str = "/cache/cxr/datasets/rsna-pneumonia-2018",
    force: bool = False,
) -> dict[str, Any]:
    root = Path(rsna_root)
    raw_zip = root / "raw" / "pneumonia-challenge-dataset-adjudicated-kaggle_2018.zip"
    extracted_dir = root / "extracted"
    marker_path = extracted_dir / ".rsna-extracted.json"
    volume.reload()
    if not raw_zip.exists():
        raise FileNotFoundError(f"RSNA image zip not found on Modal volume: {raw_zip}")
    if force and extracted_dir.exists():
        subprocess.run(["rm", "-rf", str(extracted_dir)], check=True)
    dicom_count = (
        sum(1 for _path in extracted_dir.rglob("*.dcm")) if extracted_dir.exists() else 0
    )
    extracted = False
    if dicom_count != 30000:
        if extracted_dir.exists():
            subprocess.run(["rm", "-rf", str(extracted_dir)], check=True)
        extracted_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(raw_zip) as archive:
            archive.extractall(extracted_dir)
        dicom_count = sum(1 for _path in extracted_dir.rglob("*.dcm"))
        extracted = True
    if dicom_count != 30000:
        raise RuntimeError(f"Expected 30000 RSNA DICOMs, found {dicom_count}")
    marker_path.write_text(
        json.dumps(
            {
                "dicom_count": dicom_count,
                "source_zip": str(raw_zip),
                "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    volume.commit()
    return {
        "status": "ok",
        "rsna_root": str(root),
        "extracted": extracted,
        "dicom_count": dicom_count,
        "raw_zip_bytes": raw_zip.stat().st_size,
        "extracted_bytes": directory_size(extracted_dir),
    }


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
    dataset: str = "arudaev/chest-xray-14-320",
    rsna_root: str = "",
    max_samples: int = 6000,
    max_train: int = 4096,
    max_val: int = 1024,
    max_test: int = 1024,
    image_size: int = 224,
    cache_dtype: str = "float32",
    cache_build_workers: int = 1,
    cache_key_mode: str = "legacy",
    cache_splits: str = "train,val,test",
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
    channels_last: bool = False,
    torch_compile: bool = False,
    torch_compile_mode: str = "default",
    learning_rate: float = 1.0e-4,
    amp_dtype: str = "auto",
    model_init: str = "random",
    loss_kind: str = "bce",
    loss_pos_weight: str = "none",
    loss_pos_weight_cap: float = 0.0,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.0,
    quality_gate: bool = False,
    quality_min_eval_samples: int = 0,
    quality_min_metric_targets: int = 0,
    quality_min_macro_auroc: float = 0.0,
    quality_min_macro_auprc: float = 0.0,
    train_order_evidence: bool | None = None,
    paired_train_order: bool | None = None,
    include_metadata: bool = False,
    max_train_batches: int = 0,
    max_eval_batches: int = 0,
    baselines: str = "pytorch_raw,monai_raw,medkit_cached_mmap,medkit_pinned_prefetch",
    manifest: str = "",
    splits: str = "",
    prepare_only: bool = False,
    skip_eval: bool = False,
    smoke: bool = False,
    force_rematerialize: bool = False,
    force_cache: bool = False,
    allow_destructive_cache: bool = False,
) -> dict[str, Any]:
    os.chdir(APP_ROOT)
    volume.reload()
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
        "--dataset",
        dataset,
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
        "--cache-build-workers",
        str(cache_build_workers),
        "--cache-key-mode",
        cache_key_mode,
        "--cache-splits",
        cache_splits,
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
        "--channels-last" if channels_last else "--no-channels-last",
        "--torch-compile" if torch_compile else "--no-torch-compile",
        "--torch-compile-mode",
        torch_compile_mode,
        "--learning-rate",
        str(learning_rate),
        "--amp-dtype",
        amp_dtype,
        "--model-init",
        model_init,
        "--loss-kind",
        loss_kind,
        "--loss-pos-weight",
        loss_pos_weight,
        "--loss-pos-weight-cap",
        str(loss_pos_weight_cap),
        "--focal-gamma",
        str(focal_gamma),
        "--focal-alpha",
        str(focal_alpha),
        "--quality-gate" if quality_gate else "--no-quality-gate",
        "--quality-min-eval-samples",
        str(quality_min_eval_samples),
        "--quality-min-metric-targets",
        str(quality_min_metric_targets),
        "--quality-min-macro-auroc",
        str(quality_min_macro_auroc),
        "--quality-min-macro-auprc",
        str(quality_min_macro_auprc),
        "--train-order-evidence"
        if (quality_gate if train_order_evidence is None else train_order_evidence)
        else "--no-train-order-evidence",
        "--paired-train-order"
        if (quality_gate if paired_train_order is None else paired_train_order)
        else "--no-paired-train-order",
        "--include-metadata" if include_metadata else "--no-include-metadata",
        "--baselines",
        baselines,
        "--device",
        "cuda:0",
    ]
    if splits:
        command.extend(["--splits", splits])
    if manifest:
        command.extend(["--manifest", manifest])
    if rsna_root:
        command.extend(["--rsna-root", rsna_root])
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
    if allow_destructive_cache:
        command.append("--allow-destructive-cache")
    if prepare_only:
        command.append("--prepare-only")
    if skip_eval:
        command.append("--skip-eval")

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
    if completed.returncode != 0:
        result = {
            "status": "failed",
            "returncode": completed.returncode,
            "elapsed_seconds": elapsed,
            "command": command,
            "output": completed.stdout,
            "report_dir": str(report_dir),
            "artifacts": artifacts,
        }
    else:
        result = {
            "status": "ok",
            "elapsed_seconds": elapsed,
            "command": command,
            "output": completed.stdout,
            "report_dir": str(report_dir),
            "artifacts": artifacts,
        }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "modal-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    volume.commit()
    return result


@app.local_entrypoint()
def main(
    run_id: str = "",
    dataset: str = "arudaev/chest-xray-14-320",
    rsna_root: str = "",
    max_samples: int = 6000,
    max_train: int = 4096,
    max_val: int = 1024,
    max_test: int = 1024,
    image_size: int = 224,
    cache_dtype: str = "float32",
    cache_build_workers: int = 1,
    cache_key_mode: str = "legacy",
    cache_splits: str = "train,val,test",
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
    channels_last: bool = False,
    torch_compile: bool = False,
    torch_compile_mode: str = "default",
    learning_rate: float = 1.0e-4,
    amp_dtype: str = "auto",
    model_init: str = "random",
    loss_kind: str = "bce",
    loss_pos_weight: str = "none",
    loss_pos_weight_cap: float = 0.0,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.0,
    quality_gate: bool = False,
    quality_min_eval_samples: int = 0,
    quality_min_metric_targets: int = 0,
    quality_min_macro_auroc: float = 0.0,
    quality_min_macro_auprc: float = 0.0,
    train_order_evidence: bool | None = None,
    paired_train_order: bool | None = None,
    include_metadata: bool = False,
    max_train_batches: int = 0,
    max_eval_batches: int = 0,
    baselines: str = "pytorch_raw,monai_raw,medkit_cached_mmap,medkit_pinned_prefetch",
    manifest: str = "",
    splits: str = "",
    prepare_only: bool = False,
    skip_eval: bool = False,
    smoke: bool = False,
    force_rematerialize: bool = False,
    force_cache: bool = False,
    allow_destructive_cache: bool = False,
    background: bool = False,
    wait: bool = True,
) -> None:
    if not run_id:
        gpu_label = MODAL_GPU.lower().replace(" ", "-").replace(":", "-")
        mode = "smoke" if smoke else gpu_label
        dataset_label = dataset.replace("/", "-").replace("_", "-").lower()
        run_id = f"{dataset_label}-{mode}-size{image_size}-n{max_samples}-{time.strftime('%Y%m%d-%H%M%S')}"
    # Keep the forwarding contract explicit for audit tests:
    # sync_every_step=sync_every_step, channels_last=channels_last.
    # torch_compile=torch_compile, torch_compile_mode=torch_compile_mode.
    # learning_rate=learning_rate, amp_dtype=amp_dtype, model_init=model_init.
    # loss_kind=loss_kind, loss_pos_weight_cap=loss_pos_weight_cap.
    # focal_gamma=focal_gamma, focal_alpha=focal_alpha.
    # gpu_prefetch_reuse_buffers=gpu_prefetch_reuse_buffers.
    # train_order_evidence=train_order_evidence, paired_train_order=paired_train_order.
    # cache_build_workers=cache_build_workers, cache_key_mode=cache_key_mode.
    # cache_splits=cache_splits, skip_eval=skip_eval.
    # allow_destructive_cache=allow_destructive_cache.
    benchmark_kwargs = {
        "run_id": run_id,
        "dataset": dataset,
        "rsna_root": rsna_root,
        "max_samples": max_samples,
        "max_train": max_train,
        "max_val": max_val,
        "max_test": max_test,
        "image_size": image_size,
        "cache_dtype": cache_dtype,
        "cache_build_workers": cache_build_workers,
        "cache_key_mode": cache_key_mode,
        "cache_splits": cache_splits,
        "batch_size": batch_size,
        "workers": workers,
        "epochs": epochs,
        "loader_batches": loader_batches,
        "warmup_batches": warmup_batches,
        "profile_batches": profile_batches,
        "drop_last_train": drop_last_train,
        "prefetch_depth": prefetch_depth,
        "prefetch_read_workers": prefetch_read_workers,
        "read_mode": read_mode,
        "shuffle_block_batches": shuffle_block_batches,
        "gpu_prefetch_batches": gpu_prefetch_batches,
        "gpu_prefetch_reuse_buffers": gpu_prefetch_reuse_buffers,
        "sync_every_step": sync_every_step,
        "channels_last": channels_last,
        "torch_compile": torch_compile,
        "torch_compile_mode": torch_compile_mode,
        "learning_rate": learning_rate,
        "amp_dtype": amp_dtype,
        "model_init": model_init,
        "loss_kind": loss_kind,
        "loss_pos_weight": loss_pos_weight,
        "loss_pos_weight_cap": loss_pos_weight_cap,
        "focal_gamma": focal_gamma,
        "focal_alpha": focal_alpha,
        "quality_gate": quality_gate,
        "quality_min_eval_samples": quality_min_eval_samples,
        "quality_min_metric_targets": quality_min_metric_targets,
        "quality_min_macro_auroc": quality_min_macro_auroc,
        "quality_min_macro_auprc": quality_min_macro_auprc,
        "train_order_evidence": train_order_evidence,
        "paired_train_order": paired_train_order,
        "include_metadata": include_metadata,
        "max_train_batches": max_train_batches,
        "max_eval_batches": max_eval_batches,
        "baselines": baselines,
        "manifest": manifest,
        "splits": splits,
        "prepare_only": prepare_only,
        "skip_eval": skip_eval,
        "smoke": smoke,
        "force_rematerialize": force_rematerialize,
        "force_cache": force_cache,
        "allow_destructive_cache": allow_destructive_cache,
    }
    function_call = run_cxr_benchmark.spawn(**benchmark_kwargs)
    if background:
        print(
            json.dumps(
                {
                    "status": "spawned",
                    "run_id": run_id,
                    "function_call_id": function_call.object_id,
                    "dashboard_url": function_call.get_dashboard_url(),
                    "remote_report_dir": str(REMOTE_REPORT_ROOT / run_id),
                    "volume_report_path": f"/results/cxr/{run_id}",
                },
                indent=2,
                sort_keys=True,
            )
        )
        if not wait:
            return
    result = function_call.get()
    local_report_dir = LOCAL_REPO_ROOT / "target" / "reports" / "cxr" / run_id
    local_report_dir.mkdir(parents=True, exist_ok=True)
    for name, value in result.get("artifacts", {}).items():
        suffix = Path(name).suffix
        path = local_report_dir / name
        if isinstance(value, dict) and value.get("encoding") == "base64":
            path.write_bytes(base64.b64decode(str(value.get("data", "")).encode("ascii")))
        elif suffix == ".json":
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
        elif path.name.endswith(".jsonl.gz"):
            artifacts[path.name] = {
                "encoding": "base64",
                "data": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
        elif path.suffix == ".jsonl":
            artifacts[path.name] = path.read_text()
        elif path.suffix in {".md", ".txt"}:
            artifacts[path.name] = path.read_text()
    return artifacts


def directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


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
