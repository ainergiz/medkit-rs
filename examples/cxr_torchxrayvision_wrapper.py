"""Use medkit CXR batches with TorchXRayVision-style model code.

The medkit cache remains responsible for preprocessing, split safety, target
order, and label masks. This wrapper only renames batch keys and optionally
aligns medkit targets to a TorchXRayVision model's pathology order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import medkit_rs as medkit
import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--preset", choices=("speed", "memory"), default="speed")
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--weights", default="densenet121-res224-all")
    parser.add_argument("--run-model", action="store_true")
    args = parser.parse_args()

    dataset = medkit.cxr.Dataset(cache_dir=args.cache_dir, split=args.split, preset=args.preset)
    loader = medkit.cxr.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.split == "train",
        pin_memory=torch.cuda.is_available() if args.preset == "speed" else None,
        drop_last=False,
    )

    pathologies = list(dataset.targets)
    model = None
    if args.run_model:
        try:
            import torchxrayvision as xrv
        except ImportError as error:
            raise SystemExit(
                "Install TorchXRayVision to run the model path: "
                "uv run --with torchxrayvision "
                "examples/cxr_torchxrayvision_wrapper.py --run-model ..."
            ) from error
        model = xrv.models.DenseNet(weights=args.weights)
        pathologies = list(getattr(model, "pathologies", pathologies))
        model.eval()

    adapter = TorchXRayVisionBatchAdapter(loader, dataset.targets, pathologies)
    for index, batch in enumerate(adapter):
        report = {
            "img_shape": list(batch["img"].shape),
            "lab_shape": list(batch["lab"].shape),
            "mask_shape": list(batch["mask"].shape),
            "pathologies": pathologies,
            "missing_pathologies": batch["missing_pathologies"],
            "loader_metadata": loader.report_metadata(),
        }
        if model is not None:
            with torch.no_grad():
                output = model(batch["img"].float())
            report["model_output_shape"] = list(output.shape)
        print(json.dumps(report, sort_keys=True))
        if index + 1 >= args.max_batches:
            break
    return 0


class TorchXRayVisionBatchAdapter:
    """Adapt medkit dict batches to common TorchXRayVision names."""

    def __init__(
        self,
        loader: Iterable[dict[str, Any]],
        medkit_targets: list[str],
        pathologies: list[str],
    ):
        self.loader = loader
        self.medkit_targets = list(medkit_targets)
        self.pathologies = list(pathologies)
        self.indices = [
            self.medkit_targets.index(pathology)
            if pathology in self.medkit_targets
            else None
            for pathology in self.pathologies
        ]
        self.missing_pathologies = [
            pathology
            for pathology, index in zip(self.pathologies, self.indices)
            if index is None
        ]

    def __iter__(self):
        for batch in self.loader:
            labels, masks = self._align(batch["labels"], batch["mask"])
            yield {
                "img": batch["image"],
                "lab": labels,
                "mask": masks,
                "medkit_targets": self.medkit_targets,
                "pathologies": self.pathologies,
                "missing_pathologies": self.missing_pathologies,
            }

    def _align(self, labels: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        aligned_labels = labels.new_zeros((labels.shape[0], len(self.pathologies)))
        aligned_masks = masks.new_zeros((masks.shape[0], len(self.pathologies)))
        for output_index, source_index in enumerate(self.indices):
            if source_index is None:
                continue
            aligned_labels[:, output_index] = labels[:, source_index]
            aligned_masks[:, output_index] = masks[:, source_index]
        return aligned_labels, aligned_masks


if __name__ == "__main__":
    raise SystemExit(main())
