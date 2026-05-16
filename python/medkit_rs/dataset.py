"""PyTorch Dataset adapters for medkit-rs sampled patch plans."""

from __future__ import annotations

import json
import ctypes
import sys
import random
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch as _torch_module  # type: ignore
except ImportError:
    _torch_module = None

_MapDatasetBase = (
    object if _torch_module is None else _torch_module.utils.data.Dataset
)
_IterableDatasetBase = (
    object if _torch_module is None else _torch_module.utils.data.IterableDataset
)


class MedkitPatchDataset(_MapDatasetBase):
    """Map-style PyTorch dataset backed by a medkit cache and JSONL patch plan."""

    def __init__(
        self,
        cache_dir: str | Path,
        patches_path: str | Path,
        length: int | None = None,
        include_metadata: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.patches_path = Path(patches_path)
        self.include_metadata = include_metadata
        self.manifest = json.loads((self.cache_dir / "cache_manifest.json").read_text())
        self.records = [
            json.loads(line)
            for line in self.patches_path.read_text().splitlines()
            if line.strip()
        ]
        if not self.records:
            raise ValueError(f"patch plan contains no records: {self.patches_path}")
        self.length = int(length) if length is not None else len(self.records)
        if self.length <= 0:
            raise ValueError("length must be greater than zero")
        self.cases = {case["case_id"]: case for case in self.manifest["cases"]}
        self._volumes: dict[str, tuple[np.memmap, np.memmap]] = {}

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch = _torch()
        record = self.records[index % len(self.records)]
        case = self.cases[record["case_id"]]
        image, label = self._volumes_for_case(case)
        x, y, z = record["patch_start"]
        sx, sy, sz = record["patch_size"]
        image_patch = image[z : z + sz, y : y + sy, x : x + sx]
        label_patch = label[z : z + sz, y : y + sy, x : x + sx]
        sample = {
            "image": torch.from_numpy(image_patch[None, ...]),
            "label": torch.from_numpy(label_patch[None, ...]),
        }
        if self.include_metadata:
            sample["case_id"] = record["case_id"]
            sample["patch_start"] = torch.tensor(record["patch_start"], dtype=torch.int64)
        return sample

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_volumes"] = {}
        return state

    def _volumes_for_case(self, case: dict[str, Any]) -> tuple[np.memmap, np.memmap]:
        case_id = case["case_id"]
        if case_id in self._volumes:
            return self._volumes[case_id]
        x, y, z = case["shape"]
        image = np.memmap(
            self._resolve(case["image_cache_path"]),
            dtype="<f4",
            mode="c",
            shape=(z, y, x),
        )
        label = np.memmap(
            self._resolve(case["label_cache_path"]),
            dtype="<u2",
            mode="c",
            shape=(z, y, x),
        )
        self._volumes[case_id] = (image, label)
        return image, label

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute() or path.exists():
            return path
        candidate = self.cache_dir / path
        if candidate.exists():
            return candidate
        return path


class MedkitPatchIterableDataset(_IterableDatasetBase):
    """Iterable PyTorch dataset that streams a medkit patch plan cyclically."""

    def __init__(self, cache_dir: str | Path, patches_path: str | Path, length: int | None = None):
        self.dataset = MedkitPatchDataset(cache_dir, patches_path, length)

    def __iter__(self):
        torch = _torch()
        worker = torch.utils.data.get_worker_info()
        if worker is None:
            start = 0
            step = 1
            limit = len(self.dataset)
        else:
            start = worker.id
            step = worker.num_workers
            limit = len(self.dataset)
        index = start
        while index < limit:
            yield self.dataset[index]
            index += step


class MedkitFfiBatchIterableDataset(_IterableDatasetBase):
    """Iterable dataset that asks the Rust FFI bridge to fill whole batches."""

    def __init__(
        self,
        cache_dir: str | Path,
        patches_path: str | Path,
        length: int | None = None,
        batch_size: int = 16,
        library_path: str | Path | None = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.cache_dir = Path(cache_dir)
        self.patches_path = Path(patches_path)
        self.requested_length = length
        self.batch_size = batch_size
        self.library_path = Path(library_path) if library_path else _default_ffi_library()
        self._ffi: _MedkitFfi | None = None
        self._handle: int | None = None
        self._records = 0
        self._patch = (0, 0, 0)
        self._ensure_open()

    def __iter__(self):
        torch = _torch()
        ffi = self._ensure_open()
        worker = torch.utils.data.get_worker_info()
        length = len(self)
        if worker is None:
            start = 0
            step = self.batch_size
        else:
            start = worker.id * self.batch_size
            step = worker.num_workers * self.batch_size
        index = start
        z, y, x = self._patch[2], self._patch[1], self._patch[0]
        image_buffer = torch.empty((self.batch_size, 1, z, y, x), dtype=torch.float32)
        label_buffer = torch.empty_like(image_buffer)
        while index < length:
            current = min(self.batch_size, length - index)
            image = image_buffer if current == self.batch_size else image_buffer[:current]
            label = label_buffer if current == self.batch_size else label_buffer[:current]
            written = ffi.fill_batch(self._handle, index % self._records, current, image, label)
            if written <= 0:
                raise RuntimeError("medkit FFI returned no samples")
            if written != current:
                image = image[:written]
                label = label[:written]
            yield {
                "image": image,
                "label": label,
            }
            index += step

    def __len__(self) -> int:
        self._ensure_open()
        return self.requested_length or self._records

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_ffi"] = None
        state["_handle"] = None
        return state

    def __del__(self):
        handle = getattr(self, "_handle", None)
        ffi = getattr(self, "_ffi", None)
        if handle and ffi:
            ffi.free(handle)
            self._handle = None

    def _ensure_open(self) -> "_MedkitFfi":
        if self._ffi is not None and self._handle:
            return self._ffi
        ffi = _MedkitFfi(self.library_path)
        handle = ffi.open(self.cache_dir, self.patches_path)
        records = ffi.len(handle)
        if records <= 0:
            ffi.free(handle)
            raise ValueError(f"patch plan contains no records: {self.patches_path}")
        self._ffi = ffi
        self._handle = handle
        self._records = records
        self._patch = (ffi.patch_x(handle), ffi.patch_y(handle), ffi.patch_z(handle))
        return ffi


class MedkitNativeBatchIterableDataset(_IterableDatasetBase):
    """Iterable dataset that fills whole batches through the PyO3 native module."""

    def __init__(
        self,
        cache_dir: str | Path,
        patches_path: str | Path,
        length: int | None = None,
        batch_size: int = 16,
        storage: str = "resident",
        pin_memory: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if storage not in {"resident", "chunked"}:
            raise ValueError("storage must be 'resident' or 'chunked'")
        self.cache_dir = Path(cache_dir)
        self.patches_path = Path(patches_path)
        self.requested_length = length
        self.batch_size = batch_size
        self.storage = storage
        self.pin_memory = pin_memory
        self._handle = None
        self._records = 0
        self._patch = (0, 0, 0)
        self._ensure_open()

    def __iter__(self):
        torch = _torch()
        handle = self._ensure_open()
        worker = torch.utils.data.get_worker_info()
        length = len(self)
        if worker is None:
            start = 0
            step = self.batch_size
        else:
            start = worker.id * self.batch_size
            step = worker.num_workers * self.batch_size
        index = start
        buffer = handle.allocate_batch(self.batch_size, pin_memory=self.pin_memory)
        while index < length:
            current = min(self.batch_size, length - index)
            yield handle.fill_batch_buffer(buffer, index % self._records, current)
            index += step

    def __len__(self) -> int:
        self._ensure_open()
        return self.requested_length or self._records

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    def _ensure_open(self):
        if self._handle is not None:
            return self._handle
        native = _native_module()
        handle = native.DatasetHandle(self.cache_dir, self.patches_path, self.storage)
        records = int(handle.records)
        if records <= 0:
            raise ValueError(f"patch plan contains no records: {self.patches_path}")
        self._handle = handle
        self._records = records
        self._patch = (int(handle.patch_x), int(handle.patch_y), int(handle.patch_z))
        return handle


class MedkitCxrNativeBatchIterableDataset(_IterableDatasetBase):
    """Iterable dataset that fills CXR batches through the PyO3 native module."""

    def __init__(
        self,
        cache_dir: str | Path,
        split: str = "train",
        length: int | None = None,
        batch_size: int = 64,
        pin_memory: bool = False,
        shuffle: bool = False,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.requested_length = length
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.seed = seed
        self._handle = None
        self._records = 0
        self._targets: list[str] = []
        self._image_shape = (0, 0, 0, 0)
        self._ensure_open()

    def __iter__(self):
        torch = _torch()
        handle = self._ensure_open()
        worker = torch.utils.data.get_worker_info()
        length = len(self)
        if worker is None:
            start = 0
            step = self.batch_size
        else:
            start = worker.id * self.batch_size
            step = worker.num_workers * self.batch_size
        index = start
        buffer = handle.allocate_cxr_batch(self.batch_size, pin_memory=self.pin_memory)
        if self.shuffle:
            order = list(range(length))
            random.Random(self.seed).shuffle(order)
            while index < length:
                indices = order[index : min(index + self.batch_size, length)]
                yield handle.fill_cxr_indices_buffer(buffer, indices)
                index += step
            return
        while index < length:
            start_index = index % self._records
            current = min(self.batch_size, length - index, self._records - start_index)
            yield handle.fill_cxr_batch_buffer(buffer, start_index, current)
            index += step

    def __len__(self) -> int:
        self._ensure_open()
        return self.requested_length or self._records

    @property
    def targets(self) -> list[str]:
        self._ensure_open()
        return list(self._targets)

    @property
    def image_shape(self) -> tuple[int, int, int, int]:
        self._ensure_open()
        return self._image_shape

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    def _ensure_open(self):
        if self._handle is not None:
            return self._handle
        native = _native_module()
        handle = native.CxrCacheHandle(self.cache_dir, self.split)
        records = int(handle.records)
        if records <= 0:
            raise ValueError(f"CXR cache split contains no records: {self.split}")
        if self.requested_length is not None and self.requested_length > records:
            raise ValueError(
                "length cannot exceed the CXR cache split size for the native "
                "batch adapter"
            )
        self._handle = handle
        self._records = records
        self._targets = list(handle.targets())
        self._image_shape = tuple(int(value) for value in handle.image_shape())
        return handle


class MedkitCxrNativePrefetchDataset(_IterableDatasetBase):
    """Iterable dataset backed by a Rust-owned CXR batch prefetch thread."""

    def __init__(
        self,
        cache_dir: str | Path,
        split: str = "train",
        length: int | None = None,
        batch_size: int = 64,
        pin_memory: bool = False,
        shuffle: bool = False,
        seed: int = 0,
        prefetch_depth: int = 3,
        read_workers: int = 1,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if prefetch_depth <= 0:
            raise ValueError("prefetch_depth must be greater than zero")
        if read_workers <= 0:
            raise ValueError("read_workers must be greater than zero")
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.requested_length = length
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.seed = seed
        self.prefetch_depth = prefetch_depth
        self.read_workers = read_workers
        self._handle = None
        self._records = 0
        self._targets: list[str] = []
        self._image_shape = (0, 0, 0, 0)
        self._ensure_open()

    def __iter__(self):
        torch = _torch()
        worker = torch.utils.data.get_worker_info()
        if worker is not None:
            raise RuntimeError(
                "MedkitCxrNativePrefetchDataset must be used with num_workers=0; "
                "it manages native prefetch threads internally"
            )
        handle = self._ensure_open()
        batches = self._batch_indices()
        prefetcher = handle.create_cxr_prefetcher(
            self.batch_size,
            batches,
            pin_memory=self.pin_memory,
            prefetch_depth=self.prefetch_depth,
            read_workers=self.read_workers,
        )
        leased_slot: int | None = None
        try:
            while True:
                if leased_slot is not None:
                    prefetcher.release(leased_slot)
                    leased_slot = None
                ready = prefetcher.next()
                if ready is None:
                    break
                leased_slot, batch = ready
                yield batch
        finally:
            if leased_slot is not None:
                prefetcher.release(leased_slot)
            prefetcher.close()

    def __len__(self) -> int:
        self._ensure_open()
        return self.requested_length or self._records

    @property
    def targets(self) -> list[str]:
        self._ensure_open()
        return list(self._targets)

    @property
    def image_shape(self) -> tuple[int, int, int, int]:
        self._ensure_open()
        return self._image_shape

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    def _batch_indices(self) -> list[list[int]]:
        length = len(self)
        order = list(range(length))
        if self.shuffle:
            random.Random(self.seed).shuffle(order)
        return [
            order[index : min(index + self.batch_size, length)]
            for index in range(0, length, self.batch_size)
        ]

    def _ensure_open(self):
        if self._handle is not None:
            return self._handle
        native = _native_module()
        handle = native.CxrCacheHandle(self.cache_dir, self.split)
        records = int(handle.records)
        if records <= 0:
            raise ValueError(f"CXR cache split contains no records: {self.split}")
        if self.requested_length is not None and self.requested_length > records:
            raise ValueError(
                "length cannot exceed the CXR cache split size for the native "
                "prefetch adapter"
            )
        self._handle = handle
        self._records = records
        self._targets = list(handle.targets())
        self._image_shape = tuple(int(value) for value in handle.image_shape())
        return handle


class MedkitViewBatchIterableDataset(_IterableDatasetBase):
    """Iterable dataset that yields no-copy batches of patch tensor views."""

    def __init__(
        self,
        cache_dir: str | Path,
        patches_path: str | Path,
        length: int | None = None,
        batch_size: int = 16,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.cache_dir = Path(cache_dir)
        self.patches_path = Path(patches_path)
        self.batch_size = batch_size
        self.manifest = json.loads((self.cache_dir / "cache_manifest.json").read_text())
        self._volumes = []
        case_indices = {}
        for case_index, case in enumerate(self.manifest["cases"]):
            case_indices[case["case_id"]] = case_index
            x, y, z = case["shape"]
            image = np.fromfile(self._resolve(case["image_cache_path"]), dtype="<f4").reshape(
                (z, y, x)
            )
            label = np.fromfile(self._resolve(case["label_cache_path"]), dtype="<u2").astype(
                np.float32
            ).reshape((z, y, x))
            prefix = np.fromfile(
                self._resolve(case["foreground_prefix_path"]),
                dtype="<u4",
            ).reshape((z + 1, y + 1, x + 1))
            self._volumes.append(
                (_torch().from_numpy(image), _torch().from_numpy(label), prefix)
            )
        records = []
        for line in self.patches_path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            x, y, z = record["patch_start"]
            sx, sy, sz = record["patch_size"]
            records.append((case_indices[record["case_id"]], x, y, z, sx, sy, sz))
        if not records:
            raise ValueError(f"patch plan contains no records: {self.patches_path}")
        self.records = records
        self.length = int(length) if length is not None else len(records)
        if self.length <= 0:
            raise ValueError("length must be greater than zero")

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        worker = _torch().utils.data.get_worker_info()
        if worker is None:
            start = 0
            step = self.batch_size
        else:
            start = worker.id * self.batch_size
            step = worker.num_workers * self.batch_size
        index = start
        while index < self.length:
            current = min(self.batch_size, self.length - index)
            images = []
            labels = []
            label_sum = 0
            for offset in range(current):
                case_index, x, y, z, sx, sy, sz = self.records[
                    (index + offset) % len(self.records)
                ]
                image, label, prefix = self._volumes[case_index]
                images.append(image[z : z + sz, y : y + sy, x : x + sx].unsqueeze(0))
                labels.append(label[z : z + sz, y : y + sy, x : x + sx].unsqueeze(0))
                label_sum += _prefix_count(prefix, x, y, z, sx, sy, sz)
            yield {"image": images, "label": labels, "label_sum": label_sum}
            index += step

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute() or path.exists():
            return path
        candidate = self.cache_dir / path
        if candidate.exists():
            return candidate
        return path


class _MedkitFfi:
    def __init__(self, library_path: Path):
        self.library_path = library_path
        self.lib = ctypes.CDLL(str(library_path))
        self.lib.medkit_dataset_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.lib.medkit_dataset_open.restype = ctypes.c_void_p
        self.lib.medkit_dataset_free.argtypes = [ctypes.c_void_p]
        self.lib.medkit_dataset_free.restype = None
        self.lib.medkit_dataset_len.argtypes = [ctypes.c_void_p]
        self.lib.medkit_dataset_len.restype = ctypes.c_size_t
        self.lib.medkit_dataset_patch_x.argtypes = [ctypes.c_void_p]
        self.lib.medkit_dataset_patch_x.restype = ctypes.c_size_t
        self.lib.medkit_dataset_patch_y.argtypes = [ctypes.c_void_p]
        self.lib.medkit_dataset_patch_y.restype = ctypes.c_size_t
        self.lib.medkit_dataset_patch_z.argtypes = [ctypes.c_void_p]
        self.lib.medkit_dataset_patch_z.restype = ctypes.c_size_t
        self.lib.medkit_dataset_fill_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.medkit_dataset_fill_batch.restype = ctypes.c_size_t
        self.lib.medkit_dataset_fill_batch_f32_labels.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.medkit_dataset_fill_batch_f32_labels.restype = ctypes.c_size_t

    def open(self, cache_dir: Path, patches_path: Path) -> int:
        handle = self.lib.medkit_dataset_open(
            str(cache_dir).encode(),
            str(patches_path).encode(),
        )
        if not handle:
            raise RuntimeError(
                f"failed to open medkit FFI dataset with {self.library_path}"
            )
        return int(handle)

    def free(self, handle: int) -> None:
        self.lib.medkit_dataset_free(ctypes.c_void_p(handle))

    def len(self, handle: int) -> int:
        return int(self.lib.medkit_dataset_len(ctypes.c_void_p(handle)))

    def patch_x(self, handle: int) -> int:
        return int(self.lib.medkit_dataset_patch_x(ctypes.c_void_p(handle)))

    def patch_y(self, handle: int) -> int:
        return int(self.lib.medkit_dataset_patch_y(ctypes.c_void_p(handle)))

    def patch_z(self, handle: int) -> int:
        return int(self.lib.medkit_dataset_patch_z(ctypes.c_void_p(handle)))

    def fill_batch(
        self,
        handle: int | None,
        start: int,
        batch_size: int,
        image: Any,
        label: Any,
    ) -> int:
        if handle is None:
            raise RuntimeError("medkit FFI dataset is not open")
        image_ptr = _writable_pointer(image)
        label_ptr = _writable_pointer(label)
        return int(
            self.lib.medkit_dataset_fill_batch_f32_labels(
                ctypes.c_void_p(handle),
                start,
                batch_size,
                image_ptr,
                label_ptr,
            )
        )


def _default_ffi_library() -> Path:
    root = Path(__file__).resolve().parents[2]
    stem = "medkit_python_ffi"
    if sys.platform == "darwin":
        filename = f"lib{stem}.dylib"
    elif sys.platform == "win32":
        filename = f"{stem}.dll"
    else:
        filename = f"lib{stem}.so"
    path = root / "target" / "release" / filename
    if not path.exists():
        raise RuntimeError(
            f"missing medkit FFI library at {path}; run "
            "`cargo build -p medkit-python-ffi --release`"
        )
    return path


def _prefix_count(
    prefix: np.ndarray,
    x: int,
    y: int,
    z: int,
    sx: int,
    sy: int,
    sz: int,
) -> int:
    x1 = x + sx
    y1 = y + sy
    z1 = z + sz
    value = (
        int(prefix[z1, y1, x1])
        - int(prefix[z1, y1, x])
        - int(prefix[z1, y, x1])
        - int(prefix[z, y1, x1])
        + int(prefix[z1, y, x])
        + int(prefix[z, y1, x])
        + int(prefix[z, y, x1])
        - int(prefix[z, y, x])
    )
    return value


def _writable_pointer(value: Any) -> ctypes.c_void_p:
    if hasattr(value, "data_ptr"):
        if not value.is_contiguous():
            raise ValueError("FFI batch tensors must be contiguous")
        return ctypes.c_void_p(int(value.data_ptr()))
    return value.ctypes.data_as(ctypes.c_void_p)


def _native_module():
    try:
        from . import _native  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "missing medkit_rs._native; build it with "
            "`uv run maturin develop --release` or copy the release extension "
            "into python/medkit_rs"
        ) from error
    return _native


def _torch():
    try:
        import torch  # type: ignore
    except ImportError as error:
        raise RuntimeError("PyTorch is required to use medkit_rs dataset adapters") from error
    return torch
