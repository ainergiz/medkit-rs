from __future__ import annotations

import importlib
import json
import sys
import types
import builtins
from pathlib import Path
from typing import Any

import numpy as np
import pytest


class FakeTensor:
    def __init__(
        self,
        data: Any | None = None,
        *,
        shape: tuple[int, ...] | None = None,
        dtype: Any = "float32",
        contiguous: bool = True,
    ):
        self.data = None if data is None else np.asarray(data)
        self.shape = tuple(shape if shape is not None else self.data.shape)
        self.dtype = dtype
        self._contiguous = contiguous

    def __getitem__(self, item: Any) -> "FakeTensor":
        if self.data is not None:
            return FakeTensor(self.data[item], dtype=self.dtype, contiguous=self._contiguous)
        shape = list(self.shape)
        if isinstance(item, slice):
            start, stop, step = item.indices(shape[0])
            shape[0] = max(0, (stop - start + (step - 1)) // step)
        elif isinstance(item, tuple):
            for axis, part in enumerate(item):
                if isinstance(part, slice):
                    start, stop, step = part.indices(shape[axis])
                    shape[axis] = max(0, (stop - start + (step - 1)) // step)
        return FakeTensor(shape=tuple(shape), dtype=self.dtype, contiguous=self._contiguous)

    def unsqueeze(self, dim: int) -> "FakeTensor":
        shape = list(self.shape)
        shape.insert(dim, 1)
        return FakeTensor(shape=tuple(shape), dtype=self.dtype, contiguous=self._contiguous)

    def is_contiguous(self) -> bool:
        return self._contiguous

    def data_ptr(self) -> int:
        return 123456


class FakeLoader:
    def __init__(self, dataset: Any, **kwargs: Any):
        self.dataset = dataset
        self.kwargs = kwargs
        self.report_metadata = None

    def __iter__(self):
        return iter(self.dataset)


def install_fake_torch(monkeypatch: pytest.MonkeyPatch, worker: Any | None = None):
    torch = types.ModuleType("torch")
    torch.float32 = "float32"
    torch.int64 = "int64"
    torch.empty = lambda shape, dtype=None: FakeTensor(shape=tuple(shape), dtype=dtype)
    torch.empty_like = lambda tensor: FakeTensor(shape=tensor.shape, dtype=tensor.dtype)
    torch.from_numpy = lambda array: FakeTensor(array, dtype=array.dtype)
    torch.tensor = lambda value, dtype=None: FakeTensor(np.asarray(value), dtype=dtype)
    torch.utils = types.SimpleNamespace(
        data=types.SimpleNamespace(
            Dataset=object,
            IterableDataset=object,
            DataLoader=FakeLoader,
            get_worker_info=lambda: worker,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def reload_dataset_module(monkeypatch: pytest.MonkeyPatch, worker: Any | None = None):
    install_fake_torch(monkeypatch, worker)
    sys.modules.pop("medkit_rs.dataset", None)
    sys.modules.pop("medkit_rs.cxr", None)
    sys.modules.pop("medkit_rs", None)
    return importlib.import_module("medkit_rs.dataset")


def write_patch_fixture(tmp_path: Path) -> tuple[Path, Path]:
    cache = tmp_path / "cache"
    cache.mkdir()
    image = np.arange(8, dtype="<f4").reshape((2, 2, 2))
    label = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype="<u2").reshape((2, 2, 2))
    foreground = (label > 0).astype("<u4")
    prefix = np.zeros((3, 3, 3), dtype="<u4")
    prefix[1:, 1:, 1:] = foreground.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
    image.tofile(cache / "image.raw")
    label.tofile(cache / "label.raw")
    prefix.tofile(cache / "prefix.raw")
    (cache / "cache_manifest.json").write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "case-a",
                        "shape": [2, 2, 2],
                        "image_cache_path": "image.raw",
                        "label_cache_path": "label.raw",
                        "foreground_prefix_path": "prefix.raw",
                    }
                ]
            }
        )
    )
    patches = tmp_path / "patches.jsonl"
    patches.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "case_id": "case-a",
                        "patch_start": [0, 0, 0],
                        "patch_size": [1, 2, 2],
                    }
                ),
                json.dumps(
                    {
                        "case_id": "case-a",
                        "patch_start": [1, 0, 0],
                        "patch_size": [1, 2, 2],
                    }
                ),
            ]
        )
        + "\n"
    )
    return cache, patches


def test_patch_dataset_reads_memmaps_metadata_and_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = reload_dataset_module(monkeypatch)
    cache, patches = write_patch_fixture(tmp_path)

    dataset = ds.MedkitPatchDataset(cache, patches, length=3, include_metadata=True)

    assert len(dataset) == 3
    sample = dataset[2]
    assert sample["image"].shape == (1, 2, 2, 1)
    assert sample["label"].shape == (1, 2, 2, 1)
    assert sample["case_id"] == "case-a"
    assert sample["patch_start"].dtype == "int64"
    assert dataset._resolve("image.raw") == cache / "image.raw"
    assert dataset._resolve(str(cache / "image.raw")) == cache / "image.raw"
    assert dataset._resolve("missing.raw") == Path("missing.raw")
    assert dataset.__getstate__()["_volumes"] == {}

    empty_plan = tmp_path / "empty.jsonl"
    empty_plan.write_text("\n")
    with pytest.raises(ValueError, match="patch plan contains no records"):
        ds.MedkitPatchDataset(cache, empty_plan)
    with pytest.raises(ValueError, match="length must be greater than zero"):
        ds.MedkitPatchDataset(cache, patches, length=0)


def test_patch_dataset_reads_multichannel_memmap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = reload_dataset_module(monkeypatch)
    cache, patches = write_patch_fixture(tmp_path)
    stacked = np.concatenate(
        [
            np.arange(8, dtype="<f4"),
            np.arange(8, dtype="<f4") + 100.0,
        ]
    )
    stacked.tofile(cache / "image.raw")
    manifest = json.loads((cache / "cache_manifest.json").read_text())
    manifest["cases"][0]["image_channel_count"] = 2
    (cache / "cache_manifest.json").write_text(json.dumps(manifest))

    sample = ds.MedkitPatchDataset(cache, patches)[0]

    assert sample["image"].shape == (2, 2, 2, 1)
    assert sample["label"].shape == (1, 2, 2, 1)
    assert sample["image"].data[1, 0, 0, 0] == 100.0


def test_patch_iterable_shards_by_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = reload_dataset_module(monkeypatch)
    cache, patches = write_patch_fixture(tmp_path)

    assert len(list(ds.MedkitPatchIterableDataset(cache, patches, length=2))) == 2

    worker = types.SimpleNamespace(id=1, num_workers=2)
    ds = reload_dataset_module(monkeypatch, worker)

    iterable = ds.MedkitPatchIterableDataset(cache, patches, length=4)
    samples = list(iterable)

    assert len(samples) == 2
    assert all(sample["image"].shape[0] == 1 for sample in samples)


class FakeFfi:
    records = 3
    channels = 1
    fill_return: int | None = None
    freed: list[int] = []

    def __init__(self, library_path: Path):
        self.library_path = library_path
        self.calls: list[tuple[int, int]] = []

    def open(self, cache_dir: Path, patches_path: Path) -> int:
        return 77

    def len(self, handle: int) -> int:
        return self.records

    def patch_x(self, handle: int) -> int:
        return 2

    def patch_y(self, handle: int) -> int:
        return 3

    def patch_z(self, handle: int) -> int:
        return 4

    def image_channels(self, handle: int) -> int:
        return self.channels

    def fill_batch(self, handle: int | None, start: int, batch_size: int, image: Any, label: Any) -> int:
        self.calls.append((start, batch_size))
        return self.fill_return if self.fill_return is not None else batch_size

    def free(self, handle: int) -> None:
        self.freed.append(handle)


def test_ffi_batch_iterable_opens_batches_and_reopens_after_pickle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = reload_dataset_module(monkeypatch)
    monkeypatch.setattr(ds, "_MedkitFfi", FakeFfi)
    FakeFfi.records = 3
    FakeFfi.channels = 2
    FakeFfi.fill_return = None
    FakeFfi.freed = []

    dataset = ds.MedkitFfiBatchIterableDataset(
        tmp_path, tmp_path / "patches.jsonl", length=3, batch_size=2, library_path=tmp_path / "lib"
    )

    batches = list(dataset)
    assert len(dataset) == 3
    assert batches[0]["image"].shape == (2, 2, 4, 3, 2)
    assert batches[1]["label"].shape == (1, 1, 4, 3, 2)
    assert dataset.__getstate__()["_handle"] is None
    assert dataset.__getstate__()["_ffi"] is None
    dataset.__del__()
    assert FakeFfi.freed == [77]

    with pytest.raises(ValueError, match="batch_size must be greater than zero"):
        ds.MedkitFfiBatchIterableDataset(tmp_path, tmp_path / "patches.jsonl", batch_size=0)

    FakeFfi.records = 0
    with pytest.raises(ValueError, match="patch plan contains no records"):
        ds.MedkitFfiBatchIterableDataset(
            tmp_path, tmp_path / "patches.jsonl", library_path=tmp_path / "lib"
        )


def test_ffi_batch_iterable_worker_shard_and_short_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = types.SimpleNamespace(id=1, num_workers=2)
    ds = reload_dataset_module(monkeypatch, worker)
    monkeypatch.setattr(ds, "_MedkitFfi", FakeFfi)
    FakeFfi.records = 6
    FakeFfi.channels = 1
    FakeFfi.fill_return = 1

    dataset = ds.MedkitFfiBatchIterableDataset(
        tmp_path, tmp_path / "patches.jsonl", length=5, batch_size=2, library_path=tmp_path / "lib"
    )
    batches = list(dataset)

    assert len(batches) == 1
    assert batches[0]["image"].shape == (1, 1, 4, 3, 2)
    assert dataset._ffi.calls == [(2, 2)]


def test_ffi_batch_iterable_reports_zero_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = reload_dataset_module(monkeypatch)
    monkeypatch.setattr(ds, "_MedkitFfi", FakeFfi)
    FakeFfi.records = 2
    FakeFfi.channels = 1
    FakeFfi.fill_return = 0

    dataset = ds.MedkitFfiBatchIterableDataset(
        tmp_path, tmp_path / "patches.jsonl", batch_size=2, library_path=tmp_path / "lib"
    )

    with pytest.raises(RuntimeError, match="returned no samples"):
        list(dataset)


class FakeNativeDatasetHandle:
    records = 3

    def __init__(self, cache_dir: Path, patches_path: Path, storage: str):
        self.cache_dir = cache_dir
        self.patches_path = patches_path
        self.storage = storage
        self.patch_x = 2
        self.patch_y = 3
        self.patch_z = 4

    def allocate_batch(self, batch_size: int, pin_memory: bool = False) -> object:
        return {"capacity": batch_size, "pin_memory": pin_memory}

    def fill_batch_buffer(self, buffer: object, start: int, current: int) -> dict[str, int]:
        return {"start": start, "count": current}


class FakeCxrCacheHandle:
    records = 4
    last_prefetch_args: tuple[Any, ...] | None = None
    last_prefetch_kwargs: dict[str, Any] | None = None
    last_read_mode: str | None = None

    def __init__(self, cache_dir: Path, split: str, read_mode: str = "mmap"):
        self.cache_dir = cache_dir
        self.split = split
        self.read_mode = read_mode
        self.__class__.last_read_mode = read_mode

    def targets(self) -> list[str]:
        return ["No Finding", "Pneumonia"]

    def image_shape(self) -> tuple[int, int, int, int]:
        return (self.records, 1, 8, 8)

    def allocate_cxr_batch(self, batch_size: int, pin_memory: bool = False) -> object:
        return {"capacity": batch_size, "pin_memory": pin_memory}

    def fill_cxr_batch_buffer(
        self,
        buffer: object,
        start: int,
        current: int,
        include_metadata: bool = False,
    ) -> dict[str, Any]:
        return {
            "mode": "range",
            "start": start,
            "count": current,
            "include_metadata": include_metadata,
        }

    def fill_cxr_indices_buffer(
        self,
        buffer: object,
        indices: list[int],
        include_metadata: bool = False,
    ) -> dict[str, Any]:
        return {
            "mode": "indices",
            "indices": indices,
            "include_metadata": include_metadata,
        }

    def create_cxr_prefetcher(self, *args: Any, **kwargs: Any) -> "FakePrefetcher":
        self.__class__.last_prefetch_args = args
        self.__class__.last_prefetch_kwargs = kwargs
        return FakePrefetcher()


class FakePrefetcher:
    last_instance: "FakePrefetcher | None" = None

    def __init__(self):
        self.ready = [(0, {"batch": 1}), (0, {"batch": 2})]
        self.released: list[int] = []
        self.closed = False
        FakePrefetcher.last_instance = self

    def next(self):
        if self.ready:
            return self.ready.pop(0)
        return None

    def release(self, slot: int) -> None:
        self.released.append(slot)

    def close(self) -> None:
        self.closed = True

    def stats(self) -> dict[str, int]:
        return {
            "batches": 2,
            "indexed_batches": 2,
            "indexed_runs": 3,
            "read_bytes": 1024,
            "scatter_bytes": 2048,
            "read_micros": 1200,
            "scatter_micros": 800,
            "read_workers": 2,
        }


def install_fake_native(monkeypatch: pytest.MonkeyPatch, ds: Any) -> None:
    native = types.SimpleNamespace(
        DatasetHandle=FakeNativeDatasetHandle,
        CxrCacheHandle=FakeCxrCacheHandle,
    )
    monkeypatch.setattr(ds, "_native_module", lambda: native)


def test_native_batch_iterable_uses_pyo3_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = reload_dataset_module(monkeypatch)
    install_fake_native(monkeypatch, ds)

    dataset = ds.MedkitNativeBatchIterableDataset(
        tmp_path, tmp_path / "patches.jsonl", length=3, batch_size=2, storage="chunked"
    )

    assert list(dataset) == [{"start": 0, "count": 2}, {"start": 2, "count": 1}]
    assert dataset.__getstate__()["_handle"] is None
    assert dataset._ensure_open() is dataset._handle
    with pytest.raises(ValueError, match="batch_size must be greater than zero"):
        ds.MedkitNativeBatchIterableDataset(tmp_path, tmp_path / "patches.jsonl", batch_size=0)
    with pytest.raises(ValueError, match="storage must be"):
        ds.MedkitNativeBatchIterableDataset(tmp_path, tmp_path / "patches.jsonl", storage="mmap")

    FakeNativeDatasetHandle.records = 0
    with pytest.raises(ValueError, match="patch plan contains no records"):
        ds.MedkitNativeBatchIterableDataset(tmp_path, tmp_path / "patches.jsonl")
    FakeNativeDatasetHandle.records = 3


def test_native_batch_iterable_worker_stride(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = types.SimpleNamespace(id=1, num_workers=2)
    ds = reload_dataset_module(monkeypatch, worker)
    install_fake_native(monkeypatch, ds)

    dataset = ds.MedkitNativeBatchIterableDataset(
        tmp_path, tmp_path / "patches.jsonl", length=5, batch_size=2
    )

    assert list(dataset) == [{"start": 2, "count": 2}]


def test_cxr_native_batch_iterable_covers_range_shuffle_and_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = reload_dataset_module(monkeypatch)
    install_fake_native(monkeypatch, ds)
    FakeCxrCacheHandle.records = 4

    dataset = ds.MedkitCxrNativeBatchIterableDataset(
        tmp_path,
        split="train",
        length=3,
        batch_size=2,
        pin_memory=True,
        read_mode="stream",
    )
    assert FakeCxrCacheHandle.last_read_mode == "stream"
    assert dataset.targets == ["No Finding", "Pneumonia"]
    assert dataset.image_shape == (4, 1, 8, 8)
    assert list(dataset) == [
        {"mode": "range", "start": 0, "count": 2, "include_metadata": False},
        {"mode": "range", "start": 2, "count": 1, "include_metadata": False},
    ]
    assert dataset.report_metadata()["drop_last"] is False
    assert dataset.__getstate__()["_handle"] is None
    assert dataset._ensure_open() is dataset._handle

    drop_last = ds.MedkitCxrNativeBatchIterableDataset(
        tmp_path, length=3, batch_size=2, drop_last=True
    )
    assert list(drop_last) == [
        {"mode": "range", "start": 0, "count": 2, "include_metadata": False}
    ]
    drop_last_report = drop_last.report_metadata()
    assert drop_last_report["drop_last"] is True
    assert drop_last_report["num_samples"] == 3
    assert drop_last_report["yielded_samples"] == 2
    assert drop_last_report["dropped_samples"] == 1
    assert drop_last_report["num_batches"] == 1

    shuffled = ds.MedkitCxrNativeBatchIterableDataset(
        tmp_path, length=4, batch_size=3, shuffle=True, seed=7
    )
    batches = list(shuffled)
    assert [batch["mode"] for batch in batches] == ["indices", "indices"]
    assert sorted(batches[0]["indices"] + batches[1]["indices"]) == [0, 1, 2, 3]
    assert all(batch["include_metadata"] is False for batch in batches)

    block_shuffled = ds.MedkitCxrNativeBatchIterableDataset(
        tmp_path, length=4, batch_size=2, shuffle=True, seed=7, shuffle_block_batches=1
    )
    block_batches = list(block_shuffled)
    assert block_batches[0]["indices"] in ([0, 1], [2, 3])
    assert block_batches[1]["indices"] in ([0, 1], [2, 3])
    assert block_batches[0]["indices"] != block_batches[1]["indices"]
    assert block_shuffled.report_metadata()["shuffle_block_batches"] == 1

    shuffled_drop_last = ds.MedkitCxrNativeBatchIterableDataset(
        tmp_path, length=3, batch_size=2, shuffle=True, seed=7, drop_last=True
    )
    shuffled_drop_batches = list(shuffled_drop_last)
    assert len(shuffled_drop_batches) == 1
    assert len(shuffled_drop_batches[0]["indices"]) == 2
    assert set(shuffled_drop_batches[0]["indices"]).issubset({0, 1, 2})

    with_metadata = ds.MedkitCxrNativeBatchIterableDataset(
        tmp_path, length=2, batch_size=2, include_metadata=True
    )
    assert list(with_metadata) == [
        {"mode": "range", "start": 0, "count": 2, "include_metadata": True}
    ]

    with pytest.raises(ValueError, match="batch_size must be greater than zero"):
        ds.MedkitCxrNativeBatchIterableDataset(tmp_path, batch_size=0)
    with pytest.raises(ValueError, match="length cannot exceed"):
        ds.MedkitCxrNativeBatchIterableDataset(tmp_path, length=5)
    with pytest.raises(ValueError, match="read_mode must be"):
        ds.MedkitCxrNativeBatchIterableDataset(tmp_path, read_mode="resident")
    with pytest.raises(ValueError, match="shuffle_block_batches must be non-negative"):
        ds.MedkitCxrNativeBatchIterableDataset(tmp_path, shuffle_block_batches=-1)

    worker_ds = reload_dataset_module(
        monkeypatch, types.SimpleNamespace(id=1, num_workers=2)
    )
    install_fake_native(monkeypatch, worker_ds)
    FakeCxrCacheHandle.records = 4
    worker_dataset = worker_ds.MedkitCxrNativeBatchIterableDataset(
        tmp_path, length=4, batch_size=2, shuffle=True, seed=3
    )
    worker_batches = list(worker_dataset)
    assert len(worker_batches) == 1
    assert worker_batches[0]["mode"] == "indices"

    FakeCxrCacheHandle.records = 0
    with pytest.raises(ValueError, match="contains no records"):
        worker_ds.MedkitCxrNativeBatchIterableDataset(tmp_path)


def test_cxr_prefetch_iterable_batches_releases_and_worker_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = reload_dataset_module(monkeypatch)
    install_fake_native(monkeypatch, ds)
    FakeCxrCacheHandle.records = 4

    dataset = ds.MedkitCxrNativePrefetchDataset(
        tmp_path,
        length=4,
        batch_size=2,
        shuffle=True,
        seed=2,
        prefetch_depth=2,
        read_workers=2,
        read_mode="stream",
        include_metadata=True,
    )
    assert FakeCxrCacheHandle.last_read_mode == "stream"

    assert len(dataset._batch_indices()) == 2
    assert len(
        ds.MedkitCxrNativePrefetchDataset(tmp_path, length=4, batch_size=2, shuffle=False)
        ._batch_indices()
    ) == 2
    assert ds.MedkitCxrNativePrefetchDataset(
        tmp_path, length=4, batch_size=2, shuffle=True, seed=7, shuffle_block_batches=1
    )._batch_indices() in ([[0, 1], [2, 3]], [[2, 3], [0, 1]])
    drop_last = ds.MedkitCxrNativePrefetchDataset(
        tmp_path, length=3, batch_size=2, drop_last=True
    )
    assert drop_last._batch_indices() == [[0, 1]]
    drop_last_report = drop_last.report_metadata()
    assert drop_last_report["drop_last"] is True
    assert drop_last_report["prefetch"] is True
    assert drop_last_report["yielded_samples"] == 2
    assert drop_last_report["dropped_samples"] == 1
    assert drop_last_report["num_batches"] == 1
    assert dataset.targets == ["No Finding", "Pneumonia"]
    assert dataset.image_shape == (4, 1, 8, 8)
    assert list(dataset) == [{"batch": 1}, {"batch": 2}]
    prefetcher = FakePrefetcher.last_instance
    assert prefetcher is not None
    assert prefetcher.released == [0, 0]
    assert prefetcher.closed is True
    assert dataset.report_metadata()["native_prefetch_stats"] == {
        "batches": 2,
        "indexed_batches": 2,
        "indexed_runs": 3,
        "read_bytes": 1024,
        "scatter_bytes": 2048,
        "read_micros": 1200,
        "scatter_micros": 800,
        "read_workers": 2,
    }
    assert dataset.__getstate__()["_handle"] is None
    assert FakeCxrCacheHandle.last_prefetch_kwargs is not None
    assert FakeCxrCacheHandle.last_prefetch_kwargs["include_metadata"] is True

    worker_ds = reload_dataset_module(
        monkeypatch, types.SimpleNamespace(id=0, num_workers=2)
    )
    install_fake_native(monkeypatch, worker_ds)
    with pytest.raises(RuntimeError, match="num_workers=0"):
        list(worker_ds.MedkitCxrNativePrefetchDataset(tmp_path))

    with pytest.raises(ValueError, match="batch_size must be greater than zero"):
        ds.MedkitCxrNativePrefetchDataset(tmp_path, batch_size=0)
    with pytest.raises(ValueError, match="prefetch_depth must be greater than zero"):
        ds.MedkitCxrNativePrefetchDataset(tmp_path, prefetch_depth=0)
    with pytest.raises(ValueError, match="read_workers must be greater than zero"):
        ds.MedkitCxrNativePrefetchDataset(tmp_path, read_workers=0)
    with pytest.raises(ValueError, match="length cannot exceed"):
        ds.MedkitCxrNativePrefetchDataset(tmp_path, length=5)
    with pytest.raises(ValueError, match="read_mode must be"):
        ds.MedkitCxrNativePrefetchDataset(tmp_path, read_mode="resident")
    with pytest.raises(ValueError, match="shuffle_block_batches must be non-negative"):
        ds.MedkitCxrNativePrefetchDataset(tmp_path, shuffle_block_batches=-1)
    FakeCxrCacheHandle.records = 0
    with pytest.raises(ValueError, match="contains no records"):
        ds.MedkitCxrNativePrefetchDataset(tmp_path)
    FakeCxrCacheHandle.records = 4

    ds = reload_dataset_module(monkeypatch)
    install_fake_native(monkeypatch, ds)
    early_close = ds.MedkitCxrNativePrefetchDataset(tmp_path, length=2, batch_size=1)
    generator = iter(early_close)
    assert next(generator) == {"batch": 1}
    generator.close()
    prefetcher = FakePrefetcher.last_instance
    assert prefetcher is not None
    assert prefetcher.released == [0]
    assert prefetcher.closed is True


def test_view_batch_iterable_uses_tensor_views_and_prefix_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = reload_dataset_module(monkeypatch)
    cache, patches = write_patch_fixture(tmp_path)

    dataset = ds.MedkitViewBatchIterableDataset(cache, patches, length=2, batch_size=2)
    batches = list(dataset)

    assert len(dataset) == 2
    assert len(batches) == 1
    assert len(batches[0]["image"]) == 2
    assert batches[0]["image"][0].shape == (1, 2, 2, 1)
    assert batches[0]["label_sum"] == 4
    assert dataset._resolve("prefix.raw") == cache / "prefix.raw"
    assert dataset._resolve(str(cache / "prefix.raw")) == cache / "prefix.raw"
    assert dataset._resolve("missing.raw") == Path("missing.raw")

    empty = tmp_path / "empty-view.jsonl"
    empty.write_text("\n")
    with pytest.raises(ValueError, match="patch plan contains no records"):
        ds.MedkitViewBatchIterableDataset(cache, empty)
    with pytest.raises(ValueError, match="batch_size must be greater than zero"):
        ds.MedkitViewBatchIterableDataset(cache, patches, batch_size=0)
    with pytest.raises(ValueError, match="length must be greater than zero"):
        ds.MedkitViewBatchIterableDataset(cache, patches, length=0)

    worker_ds = reload_dataset_module(
        monkeypatch, types.SimpleNamespace(id=1, num_workers=2)
    )
    worker_view = worker_ds.MedkitViewBatchIterableDataset(cache, patches, length=4, batch_size=1)
    assert len(list(worker_view)) == 2


def test_view_batch_iterable_reads_multichannel_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = reload_dataset_module(monkeypatch)
    cache, patches = write_patch_fixture(tmp_path)
    stacked = np.concatenate(
        [
            np.arange(8, dtype="<f4"),
            np.arange(8, dtype="<f4") + 100.0,
        ]
    )
    stacked.tofile(cache / "image.raw")
    manifest = json.loads((cache / "cache_manifest.json").read_text())
    manifest["cases"][0]["image_channel_count"] = 2
    (cache / "cache_manifest.json").write_text(json.dumps(manifest))

    batch = next(iter(ds.MedkitViewBatchIterableDataset(cache, patches, length=1)))

    image = batch["image"][0]
    assert image.shape == (2, 2, 2, 1)
    assert image.data[1, 0, 0, 0] == 100.0


def test_low_level_helpers_cover_pointer_prefix_and_library_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = reload_dataset_module(monkeypatch)
    fake_module_path = tmp_path / "pkg" / "medkit_rs" / "dataset.py"
    fake_module_path.parent.mkdir(parents=True)
    fake_module_path.touch()
    monkeypatch.setattr(ds, "__file__", str(fake_module_path))
    array = np.arange(4, dtype=np.float32)

    assert ds._writable_pointer(array).value == array.ctypes.data
    with pytest.raises(ValueError, match="must be contiguous"):
        ds._writable_pointer(FakeTensor(shape=(1,), contiguous=False))
    assert ds._writable_pointer(FakeTensor(shape=(1,))).value == 123456

    prefix = np.zeros((2, 2, 2), dtype=np.uint32)
    prefix[1, 1, 1] = 3
    assert ds._prefix_count(prefix, 0, 0, 0, 1, 1, 1) == 3

    release_dir = fake_module_path.resolve().parents[2] / "target" / "release"
    release_dir.mkdir(parents=True, exist_ok=True)
    platform_to_name = {
        "linux": "libmedkit_python_ffi.so",
        "darwin": "libmedkit_python_ffi.dylib",
        "win32": "medkit_python_ffi.dll",
    }
    for platform, filename in platform_to_name.items():
        path = release_dir / filename
        path.touch()
        monkeypatch.setattr(ds.sys, "platform", platform)
        assert ds._default_ffi_library() == path
        path.unlink()

    monkeypatch.setattr(ds.sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="missing medkit FFI library"):
        ds._default_ffi_library()

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="PyTorch is required"):
        ds._torch()


class FakeCFunc:
    def __init__(self, return_value: Any = 0):
        self.return_value = return_value
        self.calls: list[tuple[Any, ...]] = []
        self.argtypes = None
        self.restype = None

    def __call__(self, *args: Any) -> Any:
        self.calls.append(args)
        return self.return_value


class FakeCLib:
    def __init__(self, open_value: int = 88):
        self.medkit_dataset_open = FakeCFunc(open_value)
        self.medkit_dataset_free = FakeCFunc(None)
        self.medkit_dataset_len = FakeCFunc(5)
        self.medkit_dataset_patch_x = FakeCFunc(2)
        self.medkit_dataset_patch_y = FakeCFunc(3)
        self.medkit_dataset_patch_z = FakeCFunc(4)
        self.medkit_dataset_image_channels = FakeCFunc(2)
        self.medkit_dataset_fill_batch = FakeCFunc(0)
        self.medkit_dataset_fill_batch_f32_labels = FakeCFunc(2)


def test_ctypes_ffi_wrapper_configures_and_calls_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = reload_dataset_module(monkeypatch)
    fake_lib = FakeCLib()
    monkeypatch.setattr(ds.ctypes, "CDLL", lambda path: fake_lib)

    ffi = ds._MedkitFfi(tmp_path / "libffi.so")

    assert ffi.open(tmp_path, tmp_path / "patches.jsonl") == 88
    assert ffi.len(88) == 5
    assert ffi.patch_x(88) == 2
    assert ffi.patch_y(88) == 3
    assert ffi.patch_z(88) == 4
    assert ffi.image_channels(88) == 2
    array = np.zeros((2,), dtype=np.float32)
    assert ffi.fill_batch(88, 1, 2, array, array) == 2
    ffi.free(88)
    assert fake_lib.medkit_dataset_free.calls
    with pytest.raises(RuntimeError, match="dataset is not open"):
        ffi.fill_batch(None, 0, 1, array, array)

    fake_lib = FakeCLib(open_value=0)
    monkeypatch.setattr(ds.ctypes, "CDLL", lambda path: fake_lib)
    with pytest.raises(RuntimeError, match="failed to open"):
        ds._MedkitFfi(tmp_path / "libffi.so").open(tmp_path, tmp_path / "patches.jsonl")


def test_native_module_helper_reports_missing_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ds = reload_dataset_module(monkeypatch)
    sys.modules.pop("medkit_rs._native", None)

    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if level == 1 and "_native" in fromlist:
            raise ImportError("native extension unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="missing medkit_rs._native"):
        ds._native_module()

    fake_native = object()
    package = sys.modules["medkit_rs"]
    package._native = fake_native
    monkeypatch.setattr(builtins, "__import__", real_import)
    assert ds._native_module() is fake_native


def test_cxr_facade_routes_options_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = reload_dataset_module(monkeypatch)
    install_fake_native(monkeypatch, ds)
    cxr = importlib.import_module("medkit_rs.cxr")
    FakeCxrCacheHandle.records = 4
    assert cxr.presets()["speed"] == {
        "pin_memory": True,
        "prefetch": True,
        "prefetch_depth": 2,
        "read_workers": 4,
        "read_mode": "stream",
    }
    preset_copy = cxr.presets()
    preset_copy["speed"]["read_workers"] = 99
    assert cxr.presets()["speed"]["read_workers"] == 4
    (tmp_path / "cache-metadata.json").write_text(
        json.dumps(
            {
                "cache_schema_version": 1,
                "report_schema_version": 1,
                "label_policy": {"uncertain": "ignore", "missing": "ignore"},
                "transform_fingerprint": "abc",
                "source_manifest_checksum": "def",
                "splits": {"train": {"samples": 4}},
            }
        )
    )

    dataset = cxr.Dataset(
        tmp_path,
        split="train",
        length=4,
        batch_size=2,
        prefetch=False,
        read_workers=1,
        read_mode="stream",
    )
    assert dataset.num_samples == 4
    assert dataset.num_batches == 2
    assert dataset.targets == ["No Finding", "Pneumonia"]
    assert dataset.image_shape == (4, 1, 8, 8)
    report = dataset.report_metadata()
    assert report["cache_schema_version"] == 1
    assert report["transform_fingerprint"] == "abc"
    assert report["read_mode"] == "stream"
    assert report["include_metadata"] is False
    assert report["drop_last"] is False
    assert report["yielded_samples"] == 4
    assert report["dropped_samples"] == 0
    assert report["worker_mode"] == "single_process"
    assert len(dataset) == 2
    assert list(dataset) == [
        {"mode": "range", "start": 0, "count": 2, "include_metadata": False},
        {"mode": "range", "start": 2, "count": 2, "include_metadata": False},
    ]

    drop_last_dataset = cxr.Dataset(
        tmp_path,
        length=3,
        batch_size=2,
        prefetch=False,
        drop_last=True,
    )
    assert drop_last_dataset.num_batches == 1
    assert len(drop_last_dataset) == 1
    assert list(drop_last_dataset) == [
        {"mode": "range", "start": 0, "count": 2, "include_metadata": False}
    ]
    drop_last_report = drop_last_dataset.report_metadata()
    assert drop_last_report["drop_last"] is True
    assert drop_last_report["num_samples"] == 3
    assert drop_last_report["yielded_samples"] == 2
    assert drop_last_report["dropped_samples"] == 1

    speed_dataset = cxr.Dataset(tmp_path, length=3, batch_size=2, preset="speed")
    assert speed_dataset.preset == "speed"
    assert speed_dataset.pin_memory is True
    assert speed_dataset.prefetch is True
    assert speed_dataset.prefetch_depth == 2
    assert speed_dataset.read_workers == 4
    assert speed_dataset.read_mode == "stream"
    speed_report = speed_dataset.report_metadata()
    assert speed_report["preset"] == "speed"
    assert speed_report["prefetch_depth"] == 2
    assert speed_report["read_workers"] == 4
    assert speed_report["prefetch_read_workers"] == 4

    loader = cxr.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        seed=9,
        pin_memory=True,
        read_mode="mmap",
        include_metadata=True,
        drop_last=True,
        shuffle_block_batches=2,
    )
    assert isinstance(loader, FakeLoader)
    assert loader.dataset.batch_size == 1
    assert loader.dataset.shuffle is True
    assert loader.dataset.read_mode == "mmap"
    assert loader.dataset.include_metadata is True
    assert loader.dataset.drop_last is True
    assert loader.dataset.shuffle_block_batches == 2
    assert loader.report_metadata()["batch_size"] == 1
    assert loader.report_metadata()["include_metadata"] is True
    assert loader.report_metadata()["drop_last"] is True
    assert loader.report_metadata()["shuffle_block_batches"] == 2

    preset_loader = cxr.DataLoader(
        cxr.Dataset(tmp_path, length=3, batch_size=2, prefetch=False),
        preset="speed",
        drop_last=True,
    )
    assert preset_loader.dataset.preset == "speed"
    assert preset_loader.dataset.pin_memory is True
    assert preset_loader.dataset.prefetch is True
    assert preset_loader.dataset.prefetch_depth == 2
    assert preset_loader.dataset.read_workers == 4
    assert preset_loader.dataset.read_mode == "stream"
    assert preset_loader.dataset.drop_last is True

    split_map = cxr.datasets(
        tmp_path,
        batch_size=1,
        splits=("train", "val"),
        prefetch=False,
        preset="memory",
        include_metadata=True,
        shuffle_block_batches=3,
    )
    assert sorted(split_map) == ["train", "val"]
    assert split_map["train"].shuffle is True
    assert split_map["val"].shuffle is False
    assert split_map["train"].preset == "memory"
    assert split_map["train"].read_mode == "stream"
    assert split_map["train"].pin_memory is False
    assert split_map["train"].prefetch is False
    assert split_map["train"].include_metadata is True
    assert split_map["train"].shuffle_block_batches == 3

    bad_metadata = tmp_path / "bad"
    bad_metadata.mkdir()
    (bad_metadata / "cache-metadata.json").write_text("{")
    assert cxr.Dataset(bad_metadata, prefetch=False)._cache_metadata() == {}
    missing_metadata = tmp_path / "missing-metadata"
    missing_metadata.mkdir()
    assert cxr.Dataset(missing_metadata, prefetch=False)._cache_metadata() == {}

    with pytest.raises(ValueError, match="prefetch_depth must be greater than zero"):
        cxr.Dataset(tmp_path, prefetch_depth=0)
    with pytest.raises(ValueError, match="read_workers must be greater than zero"):
        cxr.Dataset(tmp_path, read_workers=0)
    with pytest.raises(ValueError, match="read_mode must be"):
        cxr.Dataset(tmp_path, read_mode="resident")
    with pytest.raises(ValueError, match="shuffle_block_batches must be non-negative"):
        cxr.Dataset(tmp_path, shuffle_block_batches=-1)

    prefetch_dataset = cxr.Dataset(tmp_path, length=2, batch_size=1, prefetch=True)
    assert prefetch_dataset.report_metadata()["prefetch_depth"] == 1
    assert prefetch_dataset.report_metadata()["prefetch_read_workers"] == 1
    assert list(prefetch_dataset) == [{"batch": 1}, {"batch": 2}]
    assert FakeCxrCacheHandle.last_prefetch_kwargs is not None
    assert FakeCxrCacheHandle.last_prefetch_kwargs["prefetch_depth"] == 1
    assert FakeCxrCacheHandle.last_prefetch_kwargs["read_workers"] == 1
    assert FakeCxrCacheHandle.last_prefetch_kwargs["include_metadata"] is False

    loader_without_overrides = cxr.DataLoader(prefetch_dataset)
    assert loader_without_overrides.dataset is prefetch_dataset
