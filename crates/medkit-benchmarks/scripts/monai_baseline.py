#!/usr/bin/env python3
"""MONAI baseline for medkit benchmark fixtures and real MSD workflows.

The script intentionally imports MONAI and PyTorch only when `run` is invoked so
`python -m py_compile` can validate syntax in environments without ML packages.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark a MONAI CacheDataset + RandCropByPosNegLabeld baseline."
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--patch", default="96,96,96")
    parser.add_argument("--samples", default=1024, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--spacing", default="1.0,1.0,1.0")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    try:
        report = run(args)
    except MissingDependency as error:
        print(error, file=sys.stderr)
        return 3

    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)
    return 0


def run(args: argparse.Namespace) -> dict[str, Any]:
    monai = import_monai()
    import_torch()
    patch = parse_int3(args.patch, "--patch")
    spacing = parse_float3(args.spacing, "--spacing")
    cases = discover_cases(args.data_root)
    if not cases:
        raise ValueError(f"no paired cases found under {args.data_root}")
    if args.samples <= 0:
        raise ValueError("--samples must be greater than zero")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")

    transforms = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image", "label"]),
            monai.transforms.EnsureChannelFirstd(keys=["image", "label"]),
            monai.transforms.Spacingd(
                keys=["image", "label"],
                pixdim=spacing,
                mode=("bilinear", "nearest"),
            ),
            monai.transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=-1000.0,
                a_max=1000.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            monai.transforms.CropForegroundd(
                keys=["image", "label"],
                source_key="label",
                margin=4,
            ),
            monai.transforms.SpatialPadd(keys=["image", "label"], spatial_size=patch),
            monai.transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch,
                pos=1.0,
                neg=1.0,
                num_samples=args.batch_size,
            ),
        ]
    )

    cache_start = time.perf_counter()
    dataset = monai.data.CacheDataset(
        data=cases,
        transform=transforms,
        cache_rate=1.0,
        num_workers=args.workers,
    )
    cache_elapsed = time.perf_counter() - cache_start

    loader = monai.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.workers,
        shuffle=False,
    )
    sample_start = time.perf_counter()
    checksum = 0
    samples_seen = 0
    while samples_seen < args.samples:
        for batch in loader:
            label = first_label_tensor(batch)
            current = min(int(label.shape[0]), args.samples - samples_seen)
            if current != int(label.shape[0]):
                label = label[:current]
            checksum += int(label.sum().item())
            samples_seen += current
            if samples_seen >= args.samples:
                break
    sample_elapsed = time.perf_counter() - sample_start

    bytes_per_patch = patch[0] * patch[1] * patch[2] * (4 + 2)
    return {
        "backend": "MONAI CacheDataset + RandCropByPosNegLabeld",
        "data_root": str(args.data_root),
        "cases": len(cases),
        "patch": patch,
        "samples": args.samples,
        "workers": args.workers,
        "batch_size": args.batch_size,
        "case_batch_size": 1,
        "crops_per_case": args.batch_size,
        "effective_patch_batch_size": args.batch_size,
        "cache_build_ms": cache_elapsed * 1000.0,
        "sample_iter_ms": sample_elapsed * 1000.0,
        "samples_per_second": args.samples / max(sample_elapsed, sys.float_info.epsilon),
        "mb_per_second": (args.samples * bytes_per_patch / (1024.0 * 1024.0))
        / max(sample_elapsed, sys.float_info.epsilon),
        "checksum": checksum,
    }


def import_monai():
    try:
        import monai  # type: ignore
    except ImportError as error:
        raise MissingDependency(
            "MONAI is required for this baseline. Install with: "
            "python -m pip install monai nibabel"
        ) from error
    return monai


def import_torch():
    try:
        import torch  # type: ignore
    except ImportError as error:
        raise MissingDependency(
            "PyTorch is required for this baseline. Install a torch build for your platform."
        ) from error
    return torch


def discover_cases(root: Path) -> list[dict[str, str]]:
    images = root / "imagesTr"
    labels = root / "labelsTr"
    cases: list[dict[str, str]] = []
    for image in sorted(images.glob("*.nii*")):
        case_id = case_id_from_image_name(image.name)
        label = labels / f"{case_id}.nii"
        gz_label = labels / f"{case_id}.nii.gz"
        if not label.exists() and gz_label.exists():
            label = gz_label
        if label.exists():
            cases.append({"image": str(image), "label": str(label)})
    return cases


def case_id_from_image_name(name: str) -> str:
    if name.endswith(".nii.gz"):
        stem = name[:-7]
    elif name.endswith(".nii"):
        stem = name[:-4]
    else:
        stem = Path(name).stem
    if stem.endswith("_0000"):
        stem = stem[:-5]
    return stem


def first_label_tensor(batch: Any) -> Any:
    if isinstance(batch, list):
        if not batch:
            raise ValueError("MONAI DataLoader returned an empty batch list")
        return first_label_tensor(batch[0])
    if isinstance(batch, tuple):
        if not batch:
            raise ValueError("MONAI DataLoader returned an empty batch tuple")
        return first_label_tensor(batch[0])
    if isinstance(batch, dict):
        return batch["label"]
    raise TypeError(f"unsupported MONAI batch type: {type(batch).__name__}")


def parse_int3(value: str, flag: str) -> tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise ValueError(f"{flag} must be formatted as x,y,z, got {value}")
    out = tuple(int(part) for part in parts)
    if any(part <= 0 for part in out):
        raise ValueError(f"{flag} values must be positive, got {value}")
    return out  # type: ignore[return-value]


def parse_float3(value: str, flag: str) -> tuple[float, float, float]:
    parts = value.split(",")
    if len(parts) != 3:
        raise ValueError(f"{flag} must be formatted as x,y,z, got {value}")
    out = tuple(float(part) for part in parts)
    if any(part <= 0.0 for part in out):
        raise ValueError(f"{flag} values must be positive, got {value}")
    return out  # type: ignore[return-value]


class MissingDependency(RuntimeError):
    """Raised when optional Python benchmark dependencies are missing."""


if __name__ == "__main__":
    raise SystemExit(main())
