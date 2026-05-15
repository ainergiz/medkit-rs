"""PyTorch Dataset adapters for medkit-rs sampled patch plans."""

from __future__ import annotations

import json
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


def _torch():
    try:
        import torch  # type: ignore
    except ImportError as error:
        raise RuntimeError("PyTorch is required to use medkit_rs dataset adapters") from error
    return torch
