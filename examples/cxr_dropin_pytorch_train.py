"""Plain PyTorch CXR training loop using the medkit drop-in data API."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import medkit_rs as medkit
import torch
import torchvision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--preset", choices=("speed", "memory"), default="speed")
    parser.add_argument("--drop-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle-block-batches", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--prefetch-depth", type=int)
    parser.add_argument("--read-workers", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("cxr-dropin-report.json"))
    args = parser.parse_args()

    device = torch.device(args.device)
    pin_memory = device.type == "cuda" if args.preset == "speed" else None
    train_loader = make_loader(args, split="train", shuffle=True, pin_memory=pin_memory)
    val_loader = make_loader(args, split="val", shuffle=False, pin_memory=pin_memory)
    targets = train_loader.dataset.targets

    model = torchvision.models.densenet121(weights=None)
    model.features.conv0 = torch.nn.Conv2d(
        1,
        model.features.conv0.out_channels,
        kernel_size=model.features.conv0.kernel_size,
        stride=model.features.conv0.stride,
        padding=model.features.conv0.padding,
        bias=False,
    )
    model.classifier = torch.nn.Linear(model.classifier.in_features, len(targets))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=1.0e-4)

    started = time.perf_counter()
    train_report = train(
        model=model,
        optimizer=optimizer,
        loader=train_loader,
        device=device,
        epochs=args.epochs,
        max_batches=args.max_train_batches,
    )
    val_report = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        max_batches=args.max_val_batches,
    )
    report = {
        "cache_dir": str(args.cache_dir),
        "targets": targets,
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "train": train_report,
        "validation": val_report,
        "loader_metadata": train_loader.report_metadata(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def make_loader(args: argparse.Namespace, *, split: str, shuffle: bool, pin_memory: bool | None):
    dataset = medkit.cxr.Dataset(
        cache_dir=args.cache_dir,
        split=split,
        preset=args.preset,
    )
    return medkit.cxr.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        prefetch_depth=args.prefetch_depth,
        read_workers=args.read_workers,
        drop_last=args.drop_last if split == "train" else False,
        shuffle_block_batches=args.shuffle_block_batches if split == "train" else 0,
    )


def train(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: Any,
    device: torch.device,
    epochs: int,
    max_batches: int,
) -> dict[str, Any]:
    model.train()
    losses: list[float] = []
    samples = 0
    batches = 0
    data_wait = 0.0
    step_time = 0.0
    started = time.perf_counter()
    for _epoch in range(epochs):
        iterator = iter(loader)
        while True:
            wait_start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            data_wait += time.perf_counter() - wait_start
            step_start = time.perf_counter()
            image = batch["image"].to(device, non_blocking=True).float()
            labels = batch["labels"].to(device, non_blocking=True).float()
            mask = batch["mask"].to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            raw_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                labels,
                reduction="none",
            )
            loss = (raw_loss * mask).sum() / mask.sum().clamp_min(1.0)
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_time += time.perf_counter() - step_start
            losses.append(float(loss.detach().cpu().item()))
            samples += int(image.shape[0])
            batches += 1
            if max_batches and batches >= max_batches:
                break
        if max_batches and batches >= max_batches:
            break
    elapsed = time.perf_counter() - started
    return {
        "samples": samples,
        "batches": batches,
        "loss_mean": sum(losses) / max(len(losses), 1),
        "loss_last": losses[-1] if losses else None,
        "elapsed_seconds": elapsed,
        "samples_per_second": samples / max(elapsed, 1.0e-12),
        "data_wait_percent": 100.0 * data_wait / max(elapsed, 1.0e-12),
        "model_step_seconds": step_time,
    }


@torch.no_grad()
def evaluate(
    *,
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    max_batches: int,
) -> dict[str, Any]:
    model.eval()
    samples = 0
    batches = 0
    loss_total = 0.0
    started = time.perf_counter()
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True).float()
        labels = batch["labels"].to(device, non_blocking=True).float()
        mask = batch["mask"].to(device, non_blocking=True).float()
        logits = model(image)
        raw_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
        )
        loss = (raw_loss * mask).sum() / mask.sum().clamp_min(1.0)
        loss_total += float(loss.detach().cpu().item())
        samples += int(image.shape[0])
        batches += 1
        if max_batches and batches >= max_batches:
            break
    elapsed = time.perf_counter() - started
    return {
        "samples": samples,
        "batches": batches,
        "loss_mean": loss_total / max(batches, 1),
        "elapsed_seconds": elapsed,
        "samples_per_second": samples / max(elapsed, 1.0e-12),
    }


if __name__ == "__main__":
    raise SystemExit(main())
