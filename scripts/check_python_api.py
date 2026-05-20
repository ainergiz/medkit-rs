"""Check the public Python API expected for the 0.1.1 package."""

from __future__ import annotations

import inspect

import medkit_rs as medkit


def main() -> None:
    assert medkit.__version__ == "0.1.1"
    assert hasattr(medkit, "cxr")
    assert hasattr(medkit.cxr, "Dataset")
    assert hasattr(medkit.cxr, "DataLoader")
    assert hasattr(medkit.cxr, "datasets")
    assert "MedkitFfiBatchIterableDataset" not in medkit.__all__

    dataset_sig = inspect.signature(medkit.cxr.Dataset)
    loader_sig = inspect.signature(medkit.cxr.DataLoader)
    for name in ("cache_dir", "split", "batch_size", "shuffle", "pin_memory", "prefetch"):
        assert name in dataset_sig.parameters, name
    for name in ("dataset", "batch_size", "shuffle", "num_workers", "pin_memory", "drop_last"):
        assert name in loader_sig.parameters, name

    print("medkit_rs public Python API contract ok")


if __name__ == "__main__":
    main()
