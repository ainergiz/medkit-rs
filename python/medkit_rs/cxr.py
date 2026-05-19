"""Drop-in Chest X-ray datasets and loaders for PyTorch workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .dataset import (
    MedkitCxrNativeBatchIterableDataset,
    MedkitCxrNativePrefetchDataset,
    _IterableDatasetBase,
    _torch,
)


class Dataset(_IterableDatasetBase):
    """Pre-batched CXR dataset backed by a medkit Rust-compatible cache.

    This facade is intentionally small: it exposes a normal PyTorch iterable
    dataset surface while hiding the lower-level native class names. Batches use
    the stable CXR contract:

    ``image``: ``[B, 1, H, W]`` float32 tensor
    ``labels``: ``[B, T]`` float32 tensor
    ``mask``: ``[B, T]`` float32 tensor
    """

    def __init__(
        self,
        cache_dir: str | Path,
        split: str = "train",
        *,
        length: int | None = None,
        batch_size: int = 32,
        shuffle: bool = False,
        seed: int = 0,
        pin_memory: bool = False,
        prefetch: bool = True,
        prefetch_depth: int = 1,
        read_workers: int = 1,
        read_mode: str = "mmap",
        include_metadata: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if prefetch_depth <= 0:
            raise ValueError("prefetch_depth must be greater than zero")
        if read_workers <= 0:
            raise ValueError("read_workers must be greater than zero")
        if read_mode not in {"mmap", "stream"}:
            raise ValueError("read_mode must be 'mmap' or 'stream'")
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.length = length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.pin_memory = pin_memory
        self.prefetch = prefetch
        self.prefetch_depth = prefetch_depth
        self.read_workers = read_workers
        self.read_mode = read_mode
        self.include_metadata = include_metadata
        self._inner: Any | None = None

    def __iter__(self):
        return iter(self._ensure_inner())

    def __len__(self) -> int:
        return self.num_batches

    @property
    def num_samples(self) -> int:
        return len(self._ensure_inner())

    @property
    def num_batches(self) -> int:
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    @property
    def targets(self) -> list[str]:
        return list(self._ensure_inner().targets)

    @property
    def image_shape(self) -> tuple[int, int, int, int]:
        return tuple(self._ensure_inner().image_shape)

    def report_metadata(self) -> dict[str, Any]:
        return {
            "dataset": "medkit_rs.cxr.Dataset",
            "cache_dir": str(self.cache_dir),
            "split": self.split,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "pin_memory": self.pin_memory,
            "prefetch": self.prefetch,
            "prefetch_depth": self.prefetch_depth if self.prefetch else 0,
            "read_workers": self.read_workers if self.prefetch else 0,
            "read_mode": self.read_mode,
            "include_metadata": self.include_metadata,
            "worker_mode": "rust_thread_prefetch" if self.prefetch else "single_process",
            "num_workers": 0,
            "num_samples": self.num_samples,
            "num_batches": self.num_batches,
            "targets": self.targets,
            "label_policy": self._cache_metadata().get(
                "label_policy",
                {
                    "uncertain": "ignore",
                    "missing": "ignore",
                    "loss_mask": "uncertain and missing labels are masked from loss",
                },
            ),
            "image_shape": list(self.image_shape),
            "cache_schema_version": self._cache_metadata().get("cache_schema_version"),
            "report_schema_version": self._cache_metadata().get("report_schema_version"),
            "transform_fingerprint": self._cache_metadata().get(
                "transform_fingerprint",
                self._cache_metadata().get("transform_plan_hash"),
            ),
            "source_manifest_checksum": self._cache_metadata().get(
                "source_manifest_checksum"
            ),
        }

    def with_options(self, **overrides: Any) -> "Dataset":
        options = {
            "cache_dir": self.cache_dir,
            "split": self.split,
            "length": self.length,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "pin_memory": self.pin_memory,
            "prefetch": self.prefetch,
            "prefetch_depth": self.prefetch_depth,
            "read_workers": self.read_workers,
            "read_mode": self.read_mode,
            "include_metadata": self.include_metadata,
        }
        options.update(
            {key: value for key, value in overrides.items() if value is not None}
        )
        return type(self)(**options)

    def _ensure_inner(self):
        if self._inner is None:
            self._inner = self._build_inner()
        return self._inner

    def _build_inner(self):
        kwargs = {
            "cache_dir": self.cache_dir,
            "split": self.split,
            "length": self.length,
            "batch_size": self.batch_size,
            "pin_memory": self.pin_memory,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "read_mode": self.read_mode,
            "include_metadata": self.include_metadata,
        }
        if self.prefetch:
            return MedkitCxrNativePrefetchDataset(
                **kwargs,
                prefetch_depth=self.prefetch_depth,
                read_workers=self.read_workers,
            )
        return MedkitCxrNativeBatchIterableDataset(**kwargs)

    def _cache_metadata(self) -> dict[str, Any]:
        path = self.cache_dir / "cache-metadata.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}


def DataLoader(
    dataset: Dataset,
    *,
    batch_size: int | None = None,
    shuffle: bool | None = None,
    seed: int | None = None,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    prefetch: bool | None = None,
    prefetch_depth: int | None = None,
    read_workers: int | None = None,
    read_mode: str | None = None,
    include_metadata: bool | None = None,
    drop_last: bool = False,
    persistent_workers: bool = False,
    **kwargs: Any,
):
    """Create a PyTorch DataLoader for a pre-batched medkit CXR dataset.

    Familiar PyTorch ``DataLoader`` arguments such as ``batch_size``,
    ``shuffle``, and ``pin_memory`` are accepted here and routed into the
    medkit dataset policy. The returned PyTorch DataLoader still uses
    ``batch_size=None`` and ``num_workers=0`` internally because the dataset
    yields complete native batches.
    """

    if not isinstance(dataset, Dataset):
        raise TypeError("medkit_rs.cxr.DataLoader expects a medkit_rs.cxr.Dataset")
    if num_workers != 0:
        raise ValueError(
            "medkit_rs.cxr.DataLoader must use num_workers=0 because native "
            "prefetch threads run inside the parent process"
        )
    if drop_last:
        raise ValueError("medkit_rs.cxr.DataLoader does not yet support drop_last=True")
    if persistent_workers:
        raise ValueError("persistent_workers requires num_workers > 0")

    requested_options = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "seed": seed,
        "pin_memory": pin_memory,
        "prefetch": prefetch,
        "prefetch_depth": prefetch_depth,
        "read_workers": read_workers,
        "read_mode": read_mode,
        "include_metadata": include_metadata,
    }
    overrides = {
        key: value
        for key, value in requested_options.items()
        if value is not None and getattr(dataset, key) != value
    }
    if overrides:
        dataset = dataset.with_options(**overrides)
    torch = _torch()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        **kwargs,
    )
    loader.report_metadata = dataset.report_metadata
    return loader


def datasets(
    cache_dir: str | Path,
    *,
    batch_size: int = 32,
    splits: Iterable[str] = ("train", "val", "test"),
    pin_memory: bool = False,
    prefetch: bool = True,
    prefetch_depth: int = 1,
    read_workers: int = 1,
    read_mode: str = "mmap",
    include_metadata: bool = False,
    seed: int = 0,
) -> dict[str, Dataset]:
    """Construct matching CXR datasets for common train/val/test splits."""

    return {
        split: Dataset(
            cache_dir=cache_dir,
            split=split,
            batch_size=batch_size,
            shuffle=(split == "train"),
            seed=seed,
            pin_memory=pin_memory,
            prefetch=prefetch,
            prefetch_depth=prefetch_depth,
            read_workers=read_workers,
            read_mode=read_mode,
            include_metadata=include_metadata,
        )
        for split in splits
    }


__all__ = ["Dataset", "DataLoader", "datasets"]
