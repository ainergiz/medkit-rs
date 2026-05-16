"""Build and consume a MONAI-style datalist from a medkit CXR cache.

This is an adapter example, not the fastest medkit runtime path. It is useful
when an existing MONAI training script expects a datalist plus dictionary
transforms and the prepared medkit cache should remain the source of split,
target-order, label-mask, and preprocessing truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--iterate", action="store_true")
    args = parser.parse_args()

    rows = medkit_cxr_to_monai_datalist(args.cache_dir, args.split)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
        print(f"wrote {len(rows)} MONAI rows to {args.out}")

    if args.iterate:
        for batch in make_monai_loader(rows, args.batch_size, args.workers):
            print(
                json.dumps(
                    {
                        "image_shape": list(batch["image"].shape),
                        "labels_shape": list(batch["labels"].shape),
                        "mask_shape": list(batch["mask"].shape),
                    },
                    sort_keys=True,
                )
            )
            break
    return 0


def medkit_cxr_to_monai_datalist(cache_dir: Path, split: str) -> list[dict[str, Any]]:
    """Return MONAI-compatible row dictionaries for one medkit cache split."""

    summary = read_cache_summary(cache_dir)
    split_summary = summary["splits"][split]
    targets = list(summary["targets"])
    samples = int(split_summary["samples"])
    image_shape = tuple(int(value) for value in split_summary["shape"])

    metadata_rows = read_jsonl(resolve_cache_path(cache_dir, split_summary["metadata_path"]))
    labels = np.memmap(
        resolve_cache_path(cache_dir, split_summary["labels_path"]),
        dtype="<f4",
        mode="r",
        shape=(samples, len(targets)),
    )
    masks = np.memmap(
        resolve_cache_path(cache_dir, split_summary["masks_path"]),
        dtype="<f4",
        mode="r",
        shape=(samples, len(targets)),
    )
    image_path = resolve_cache_path(cache_dir, split_summary["images_path"])

    rows: list[dict[str, Any]] = []
    for index, metadata in enumerate(metadata_rows):
        rows.append(
            {
                "image": str(image_path),
                "image_index": index,
                "image_shape": list(image_shape),
                "labels": labels[index].astype("float32").tolist(),
                "mask": masks[index].astype("float32").tolist(),
                "targets": targets,
                "cache_schema_version": summary.get("cache_schema_version", 0),
                "transform_fingerprint": summary.get("transform_fingerprint", ""),
                "sample_id": metadata.get("sample_id"),
                "patient_id": metadata.get("patient_id"),
                "study_id": metadata.get("study_id"),
                "image_id": metadata.get("image_id"),
                "view_position": metadata.get("view_position"),
            }
        )
    return rows


class LoadMedkitCxrCached:
    """MONAI dictionary transform that materializes one cached medkit row."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        import torch

        row = dict(data)
        shape = tuple(int(value) for value in row["image_shape"])
        images = np.memmap(row["image"], dtype="<f4", mode="r", shape=shape)
        image = np.array(images[int(row["image_index"])], dtype=np.float32, copy=True)
        row["image"] = torch.from_numpy(image)
        row["labels"] = torch.tensor(row["labels"], dtype=torch.float32)
        row["mask"] = torch.tensor(row["mask"], dtype=torch.float32)
        return row


def make_monai_loader(rows: list[dict[str, Any]], batch_size: int, workers: int):
    """Create a MONAI DataLoader over the medkit datalist rows."""

    try:
        from monai.data import DataLoader, Dataset, list_data_collate
        from monai.transforms import Compose, EnsureTyped
    except ImportError as error:
        raise SystemExit(
            "Install MONAI to run the loader path: "
            "uv run --with monai examples/cxr_monai_datalist_adapter.py ..."
        ) from error

    dataset = Dataset(
        data=rows,
        transform=Compose(
            [
                LoadMedkitCxrCached(),
                EnsureTyped(keys=["image", "labels", "mask"]),
            ]
        ),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=list_data_collate,
    )


def read_cache_summary(cache_dir: Path) -> dict[str, Any]:
    return json.loads((cache_dir / "cache-metadata.json").read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def resolve_cache_path(cache_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return cache_dir / path


if __name__ == "__main__":
    raise SystemExit(main())
