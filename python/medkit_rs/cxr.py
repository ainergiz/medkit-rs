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


_DEFAULT_OPTIONS = {
    "pin_memory": False,
    "prefetch": True,
    "prefetch_depth": 1,
    "read_workers": 1,
    "read_mode": "mmap",
}

_PRESETS = {
    "memory": {
        "pin_memory": False,
        "prefetch": True,
        "prefetch_depth": 1,
        "read_workers": 1,
        "read_mode": "stream",
    },
    "speed": {
        "pin_memory": True,
        "prefetch": True,
        "prefetch_depth": 2,
        "read_workers": 4,
        "read_mode": "stream",
    },
}


def presets() -> dict[str, dict[str, Any]]:
    """Return named CXR loader policy presets."""

    return {name: dict(options) for name, options in _PRESETS.items()}


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
        pin_memory: bool | None = None,
        prefetch: bool | None = None,
        prefetch_depth: int | None = None,
        read_workers: int | None = None,
        read_mode: str | None = None,
        include_metadata: bool = False,
        drop_last: bool = False,
        shuffle_block_batches: int = 0,
        preset: str | None = None,
    ):
        options = _resolve_options(
            preset,
            {
                "pin_memory": pin_memory,
                "prefetch": prefetch,
                "prefetch_depth": prefetch_depth,
                "read_workers": read_workers,
                "read_mode": read_mode,
            },
        )
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if options["prefetch_depth"] <= 0:
            raise ValueError("prefetch_depth must be greater than zero")
        if options["read_workers"] <= 0:
            raise ValueError("read_workers must be greater than zero")
        if options["read_mode"] not in {"mmap", "stream"}:
            raise ValueError("read_mode must be 'mmap' or 'stream'")
        if shuffle_block_batches < 0:
            raise ValueError("shuffle_block_batches must be non-negative")
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.length = length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.pin_memory = bool(options["pin_memory"])
        self.prefetch = bool(options["prefetch"])
        self.prefetch_depth = int(options["prefetch_depth"])
        self.read_workers = int(options["read_workers"])
        self.read_mode = str(options["read_mode"])
        self.include_metadata = include_metadata
        self.drop_last = drop_last
        self.shuffle_block_batches = shuffle_block_batches
        self.preset = preset
        self._inner: Any | None = None
        self._metadata_cache: dict[str, Any] | None = None

    def __iter__(self):
        return iter(self._ensure_inner())

    def __len__(self) -> int:
        return self.num_batches

    @property
    def num_samples(self) -> int:
        return len(self._ensure_inner())

    @property
    def num_batches(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    @property
    def yielded_samples(self) -> int:
        if self.drop_last:
            return self.num_batches * self.batch_size
        return self.num_samples

    @property
    def targets(self) -> list[str]:
        return list(self._ensure_inner().targets)

    @property
    def image_shape(self) -> tuple[int, int, int, int]:
        return tuple(self._ensure_inner().image_shape)

    def report_metadata(self) -> dict[str, Any]:
        metadata = self._cache_metadata()
        return {
            "dataset": "medkit_rs.cxr.Dataset",
            "cache_dir": str(self.cache_dir),
            "split": self.split,
            "preset": self.preset,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "pin_memory": self.pin_memory,
            "prefetch": self.prefetch,
            "prefetch_depth": self.prefetch_depth if self.prefetch else 0,
            "read_workers": self.read_workers if self.prefetch else 0,
            "prefetch_read_workers": self.read_workers if self.prefetch else 0,
            "read_mode": self.read_mode,
            "include_metadata": self.include_metadata,
            "drop_last": self.drop_last,
            "shuffle_block_batches": self.shuffle_block_batches,
            "worker_mode": "rust_thread_prefetch" if self.prefetch else "single_process",
            "num_workers": 0,
            "num_samples": self.num_samples,
            "yielded_samples": self.yielded_samples,
            "dropped_samples": self.num_samples - self.yielded_samples,
            "num_batches": self.num_batches,
            "targets": self.targets,
            "label_policy": metadata.get(
                "label_policy",
                {
                    "uncertain": "ignore",
                    "missing": "ignore",
                    "loss_mask": "uncertain and missing labels are masked from loss",
                },
            ),
            "image_shape": list(self.image_shape),
            "cache_schema_version": metadata.get("cache_schema_version"),
            "report_schema_version": metadata.get("report_schema_version"),
            "transform_fingerprint": metadata.get(
                "transform_fingerprint",
                metadata.get("transform_plan_hash"),
            ),
            "source_manifest_checksum": metadata.get("source_manifest_checksum"),
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
            "drop_last": self.drop_last,
            "shuffle_block_batches": self.shuffle_block_batches,
            "preset": self.preset,
        }
        explicit = {key: value for key, value in overrides.items() if value is not None}
        if "preset" in explicit and explicit["preset"] != self.preset:
            for key in _DEFAULT_OPTIONS:
                if key not in explicit:
                    options[key] = None
        options.update(explicit)
        return type(self)(**options)

    def _ensure_inner(self):
        if self._inner is None:
            self._inner = self._build_inner()
        return self._inner

    def _build_inner(self):
        self._validate_cache_layout()
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
            "drop_last": self.drop_last,
            "shuffle_block_batches": self.shuffle_block_batches,
        }
        if self.prefetch:
            return MedkitCxrNativePrefetchDataset(
                **kwargs,
                prefetch_depth=self.prefetch_depth,
                read_workers=self.read_workers,
            )
        return MedkitCxrNativeBatchIterableDataset(**kwargs)

    def validate_cache(self) -> dict[str, Any]:
        """Validate that ``cache_dir`` looks like a medkit CXR cache for ``split``."""

        return dict(self._validate_cache_layout())

    def _validate_cache_layout(self) -> dict[str, Any]:
        metadata = self._read_cache_metadata(required=True)
        splits = metadata.get("splits")
        if not isinstance(splits, dict):
            raise ValueError(
                f"CXR cache metadata at {self.cache_dir / 'cache-metadata.json'} "
                "does not contain a 'splits' object"
            )
        if self.split not in splits:
            available = ", ".join(sorted(str(split) for split in splits)) or "none"
            raise ValueError(
                f"CXR cache split {self.split!r} was not found in "
                f"{self.cache_dir / 'cache-metadata.json'}; available splits: {available}"
            )
        return metadata

    def _cache_metadata(self) -> dict[str, Any]:
        return self._read_cache_metadata(required=False)

    def _read_cache_metadata(self, *, required: bool) -> dict[str, Any]:
        if self._metadata_cache is not None:
            return self._metadata_cache
        path = self.cache_dir / "cache-metadata.json"
        if not self.cache_dir.exists():
            if required:
                raise FileNotFoundError(
                    f"CXR cache directory does not exist: {self.cache_dir}. "
                    "Create one with `medkit cxr cache ...` or pass the directory "
                    "containing cache-metadata.json."
                )
            return {}
        if not path.exists():
            if required:
                raise FileNotFoundError(
                    f"CXR cache metadata not found: {path}. "
                    "Expected a medkit CXR cache directory created by "
                    "`medkit cxr cache ...`."
                )
            return {}
        try:
            metadata = json.loads(path.read_text())
        except OSError as error:
            if required:
                raise OSError(f"could not read CXR cache metadata at {path}: {error}") from error
            return {}
        except json.JSONDecodeError as error:
            if required:
                raise ValueError(f"CXR cache metadata is not valid JSON at {path}: {error}") from error
            return {}
        if not isinstance(metadata, dict):
            if required:
                raise ValueError(f"CXR cache metadata at {path} must be a JSON object")
            return {}
        self._metadata_cache = metadata
        return metadata


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
    drop_last: bool | None = None,
    shuffle_block_batches: int | None = None,
    preset: str | None = None,
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
    if persistent_workers:
        raise ValueError("persistent_workers requires num_workers > 0")

    requested_options = {
        "preset": preset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "seed": seed,
        "pin_memory": pin_memory,
        "prefetch": prefetch,
        "prefetch_depth": prefetch_depth,
        "read_workers": read_workers,
        "read_mode": read_mode,
        "include_metadata": include_metadata,
        "drop_last": drop_last,
        "shuffle_block_batches": shuffle_block_batches,
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
    pin_memory: bool | None = None,
    prefetch: bool | None = None,
    prefetch_depth: int | None = None,
    read_workers: int | None = None,
    read_mode: str | None = None,
    include_metadata: bool = False,
    drop_last: bool = False,
    shuffle_block_batches: int = 0,
    preset: str | None = None,
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
            drop_last=drop_last,
            shuffle_block_batches=shuffle_block_batches,
            preset=preset,
        )
        for split in splits
    }


def _resolve_options(
    preset: str | None,
    explicit: dict[str, Any | None],
) -> dict[str, Any]:
    if preset is not None and preset not in _PRESETS:
        names = "', '".join(sorted(_PRESETS))
        raise ValueError(f"preset must be one of '{names}'")
    options = dict(_DEFAULT_OPTIONS)
    if preset is not None:
        options.update(_PRESETS[preset])
    options.update({key: value for key, value in explicit.items() if value is not None})
    return options


__all__ = ["Dataset", "DataLoader", "datasets", "presets"]
