"""Benchmark medkit-rs cache consumption inside a PyTorch DataLoader."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark medkit_rs.MedkitPatchDataset with torch DataLoader."
    )
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--patches", required=True, type=Path)
    parser.add_argument("--samples", default=1024, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument(
        "--backend",
        choices=["map", "ffi-batch", "native-batch", "native-chunk-batch", "view-batch"],
        default="map",
    )
    parser.add_argument("--ffi-lib", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = run(args)
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)
    return 0


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.samples <= 0:
        raise ValueError("--samples must be greater than zero")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")

    torch = import_torch()
    from medkit_rs import (
        MedkitFfiBatchIterableDataset,
        MedkitNativeBatchIterableDataset,
        MedkitPatchDataset,
        MedkitViewBatchIterableDataset,
    )

    init_start = time.perf_counter()
    if args.backend == "ffi-batch":
        dataset = MedkitFfiBatchIterableDataset(
            args.cache,
            args.patches,
            length=args.samples,
            batch_size=args.batch_size,
            library_path=args.ffi_lib,
        )
        loader_batch_size = None
    elif args.backend in {"native-batch", "native-chunk-batch"}:
        dataset = MedkitNativeBatchIterableDataset(
            args.cache,
            args.patches,
            length=args.samples,
            batch_size=args.batch_size,
            storage="chunked" if args.backend == "native-chunk-batch" else "resident",
        )
        loader_batch_size = None
    elif args.backend == "view-batch":
        dataset = MedkitViewBatchIterableDataset(
            args.cache,
            args.patches,
            length=args.samples,
            batch_size=args.batch_size,
        )
        loader_batch_size = None
    else:
        dataset = MedkitPatchDataset(args.cache, args.patches, length=args.samples)
        loader_batch_size = args.batch_size
    init_elapsed = time.perf_counter() - init_start

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=loader_batch_size,
        num_workers=args.workers,
        shuffle=False,
    )
    sample_start = time.perf_counter()
    checksum = 0
    samples_seen = 0
    for batch in loader:
        label = batch["label"]
        if isinstance(label, list):
            current = len(label)
            if "label_sum" in batch:
                checksum += int(batch["label_sum"])
            else:
                checksum += sum(int(item.sum().item()) for item in label)
        else:
            current = int(label.shape[0])
            checksum += int(label.sum().item())
        samples_seen += current
        if samples_seen >= args.samples:
            break
    sample_elapsed = time.perf_counter() - sample_start

    if args.backend in {"ffi-batch", "native-batch", "native-chunk-batch"}:
        patch = tuple(int(value) for value in dataset._patch)
        records = int(dataset._records)
    elif args.backend == "view-batch":
        _, _, _, _, sx, sy, sz = dataset.records[0]
        patch = (sx, sy, sz)
        records = len(dataset.records)
    else:
        first = dataset.records[0]
        patch = tuple(int(value) for value in first["patch_size"])
        records = len(dataset.records)
    bytes_per_patch = patch[0] * patch[1] * patch[2] * (4 + 2)
    return {
        "backend": f"medkit_rs {args.backend} + torch DataLoader",
        "adapter_backend": args.backend,
        "cache": str(args.cache),
        "patches": str(args.patches),
        "records": records,
        "patch": patch,
        "samples": samples_seen,
        "workers": args.workers,
        "batch_size": args.batch_size,
        "dataset_init_ms": init_elapsed * 1000.0,
        "sample_iter_ms": sample_elapsed * 1000.0,
        "samples_per_second": samples_seen / max(sample_elapsed, sys.float_info.epsilon),
        "mb_per_second": (samples_seen * bytes_per_patch / (1024.0 * 1024.0))
        / max(sample_elapsed, sys.float_info.epsilon),
        "checksum": checksum,
    }


def import_torch():
    try:
        import torch  # type: ignore
    except ImportError as error:
        raise RuntimeError("PyTorch is required for this benchmark") from error
    return torch


if __name__ == "__main__":
    raise SystemExit(main())
