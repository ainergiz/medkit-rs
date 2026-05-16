"""Train a timm CXR classifier through a Lightning DataModule.

This example keeps medkit responsible for manifest/cache/split semantics and
uses Lightning only for the training loop structure. It intentionally does not
advertise DDP: distributed CXR training needs native rank-aware sharding before
it can make exact sample-coverage claims.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import medkit_rs as medkit

try:
    import torch
    import timm
except ImportError:
    torch = None
    timm = None

try:
    import lightning.pytorch as pl
except ImportError:
    try:
        import pytorch_lightning as pl
    except ImportError:
        pl = None

LightningDataModuleBase = pl.LightningDataModule if pl is not None else object
LightningModuleBase = pl.LightningModule if pl is not None else object


def main() -> None:
    args = parse_args()
    require_training_stack()

    datamodule = MedkitCXRDataModule(
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        pin_memory=args.pin_memory,
        prefetch=not args.no_prefetch,
        read_workers=args.read_workers,
    )
    datamodule.setup("fit")

    if args.dry_run:
        print(datamodule.train_dataset.report_metadata())
        return

    model = TimmCXRClassifier(
        model_name=args.model,
        num_targets=len(datamodule.train_dataset.targets),
        learning_rate=args.learning_rate,
    )
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        log_every_n_steps=10,
    )
    trainer.fit(model, datamodule=datamodule)


class MedkitCXRDataModule(LightningDataModuleBase):
    """Lightning-compatible DataModule over a medkit CXR cache."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        batch_size: int,
        seed: int = 0,
        pin_memory: bool = False,
        prefetch: bool = True,
        read_workers: int = 1,
    ):
        require_training_stack()
        super().__init__()
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.seed = seed
        self.pin_memory = pin_memory
        self.prefetch = prefetch
        self.read_workers = read_workers

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = self._dataset("train", shuffle=True)
            self.val_dataset = self._dataset("val", shuffle=False)
        if stage in (None, "test"):
            self.test_dataset = self._dataset("test", shuffle=False)

    def train_dataloader(self) -> Any:
        return medkit.cxr.DataLoader(self.train_dataset)

    def val_dataloader(self) -> Any:
        return medkit.cxr.DataLoader(self.val_dataset)

    def test_dataloader(self) -> Any:
        return medkit.cxr.DataLoader(self.test_dataset)

    def _dataset(self, split: str, *, shuffle: bool) -> medkit.cxr.Dataset:
        return medkit.cxr.Dataset(
            self.cache_dir,
            split=split,
            batch_size=self.batch_size,
            shuffle=shuffle,
            seed=self.seed,
            pin_memory=self.pin_memory,
            prefetch=self.prefetch,
            read_workers=self.read_workers,
        )


class TimmCXRClassifier(LightningModuleBase):
    """Minimal LightningModule for masked multilabel CXR classification."""

    def __init__(
        self,
        *,
        model_name: str,
        num_targets: int,
        learning_rate: float,
    ):
        require_training_stack()
        super().__init__()
        self.learning_rate = learning_rate
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            in_chans=1,
            num_classes=num_targets,
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Any:
        return self._step(batch, "train_loss")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> Any:
        self._step(batch, "val_loss")

    def configure_optimizers(self) -> Any:
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def _step(self, batch: dict[str, Any], metric_name: str) -> Any:
        logits = self.backbone(batch["image"])
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            batch["labels"],
            reduction="none",
        )
        masked_loss = (losses * batch["mask"]).sum() / batch["mask"].sum().clamp_min(1.0)
        self.log(metric_name, masked_loss, prog_bar=True)
        return masked_loss


def require_training_stack() -> None:
    if torch is None or timm is None:
        raise SystemExit(
            "Install training dependencies to run this example: "
            "uv run --with torch --with timm --with lightning "
            "examples/cxr_lightning_timm_datamodule.py ..."
        )
    if pl is None:
        raise SystemExit(
            "Install Lightning to run this example: "
            "uv run --with lightning examples/cxr_lightning_timm_datamodule.py ..."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--read-workers", type=int, default=1)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-prefetch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
