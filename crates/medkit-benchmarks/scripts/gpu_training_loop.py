"""Benchmark medkit-rs CPU extraction feeding a CUDA training loop.

This script intentionally bypasses PyTorch DataLoader workers. It measures the
runtime path we want for high-end training: Rust fills reusable host tensors,
PyTorch copies them to CUDA, and a tiny synthetic model step runs while the next
host batch is prepared.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark medkit native batches with pinned-memory CUDA prefetch."
    )
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--patches", required=True, type=Path)
    parser.add_argument("--samples", default=1024, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--storage", choices=["resident", "chunked"], default="resident")
    parser.add_argument("--prefetch-batches", default=3, type=int)
    parser.add_argument("--warmup-batches", default=4, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model-step",
        choices=["transfer-only", "forward", "train"],
        default="forward",
    )
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = run(args)
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)
    return 0


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.samples <= 0:
        raise ValueError("--samples must be greater than zero")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    if args.prefetch_batches <= 0:
        raise ValueError("--prefetch-batches must be greater than zero")
    if args.warmup_batches < 0:
        raise ValueError("--warmup-batches must be non-negative")

    torch = import_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gpu_training_loop.py")

    from medkit_rs.dataset import _native_module

    device_name = "cuda:0" if args.device == "cuda" else args.device
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("--device must identify a CUDA device")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True

    init_start = time.perf_counter()
    native = _native_module()
    handle = native.DatasetHandle(args.cache, args.patches, args.storage)
    records = int(handle.records)
    patch = (int(handle.patch_x), int(handle.patch_y), int(handle.patch_z))
    model, optimizer = make_model(torch, args.model_step, device)
    init_elapsed = time.perf_counter() - init_start

    if args.warmup_batches:
        warmup_samples = args.warmup_batches * args.batch_size
        run_pipeline(
            torch=torch,
            handle=handle,
            samples=warmup_samples,
            batch_size=args.batch_size,
            records=records,
            device=device,
            pin_memory=args.pin_memory,
            prefetch_batches=args.prefetch_batches,
            model_step=args.model_step,
            model=model,
            optimizer=optimizer,
        )
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    torch.cuda.synchronize(device)
    measured_start = time.perf_counter()
    loop_report = run_pipeline(
        torch=torch,
        handle=handle,
        samples=args.samples,
        batch_size=args.batch_size,
        records=records,
        device=device,
        pin_memory=args.pin_memory,
        prefetch_batches=args.prefetch_batches,
        model_step=args.model_step,
        model=model,
        optimizer=optimizer,
    )
    torch.cuda.synchronize(device)
    measured_elapsed = time.perf_counter() - measured_start

    bytes_per_sample = patch[0] * patch[1] * patch[2] * 2 * 4
    return {
        "backend": "medkit_rs native CUDA prefetch loop",
        "cache": str(args.cache),
        "patches": str(args.patches),
        "records": records,
        "patch": patch,
        "samples": args.samples,
        "batch_size": args.batch_size,
        "storage": args.storage,
        "pin_memory": args.pin_memory,
        "prefetch_batches": args.prefetch_batches,
        "warmup_batches": args.warmup_batches,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "model_step": args.model_step,
        "init_ms": init_elapsed * 1000.0,
        "loop_ms": measured_elapsed * 1000.0,
        "samples_per_second": args.samples / max(measured_elapsed, sys.float_info.epsilon),
        "host_to_device_gb_per_second": (args.samples * bytes_per_sample / (1024.0**3))
        / max(measured_elapsed, sys.float_info.epsilon),
        "host_fill_ms": loop_report["host_fill_seconds"] * 1000.0,
        "copy_enqueue_ms": loop_report["copy_enqueue_seconds"] * 1000.0,
        "batches": loop_report["batches"],
        "last_loss": loop_report["last_loss"],
        "last_label_sum": loop_report["last_label_sum"],
        "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0),
    }


def run_pipeline(
    *,
    torch: Any,
    handle: Any,
    samples: int,
    batch_size: int,
    records: int,
    device: Any,
    pin_memory: bool,
    prefetch_batches: int,
    model_step: str,
    model: Any,
    optimizer: Any,
) -> dict[str, Any]:
    total_batches = math.ceil(samples / batch_size)
    ring_size = min(prefetch_batches, total_batches)
    copy_stream = torch.cuda.Stream(device=device)
    current_stream = torch.cuda.current_stream(device)
    buffers = [
        handle.allocate_batch(batch_size, pin_memory=pin_memory) for _ in range(ring_size)
    ]
    slots: list[dict[str, Any]] = [
        {"event": None, "batch": None, "samples": 0} for _ in range(ring_size)
    ]
    host_fill_seconds = 0.0
    copy_enqueue_seconds = 0.0
    next_batch = 0
    last_loss_tensor = None
    last_label_tensor = None

    def fill_and_copy(slot_index: int, batch_number: int) -> None:
        nonlocal host_fill_seconds, copy_enqueue_seconds
        event = slots[slot_index]["event"]
        if event is not None:
            event.synchronize()
        start_index = batch_number * batch_size
        current = min(batch_size, samples - start_index)
        fill_start = time.perf_counter()
        cpu_batch = handle.fill_batch_buffer(
            buffers[slot_index],
            start_index % records,
            current,
        )
        host_fill_seconds += time.perf_counter() - fill_start

        copy_start = time.perf_counter()
        with torch.cuda.stream(copy_stream):
            gpu_batch = {
                "image": cpu_batch["image"].to(device, non_blocking=pin_memory),
                "label": cpu_batch["label"].to(device, non_blocking=pin_memory),
            }
            copy_event = torch.cuda.Event()
            copy_event.record(copy_stream)
        copy_enqueue_seconds += time.perf_counter() - copy_start
        slots[slot_index] = {
            "event": copy_event,
            "batch": gpu_batch,
            "samples": current,
        }

    while next_batch < ring_size:
        fill_and_copy(next_batch, next_batch)
        next_batch += 1

    for batch_number in range(total_batches):
        slot_index = batch_number % ring_size
        slot = slots[slot_index]
        current_stream.wait_event(slot["event"])
        gpu_batch = slot["batch"]
        last_label_tensor = gpu_batch["label"]
        last_loss_tensor = run_model_step(
            torch,
            gpu_batch,
            model_step,
            model,
            optimizer,
        )
        if next_batch < total_batches:
            fill_and_copy(slot_index, next_batch)
            next_batch += 1

    last_label_sum = 0
    if last_label_tensor is not None:
        last_label_sum = int(last_label_tensor.detach().sum().cpu().item())
    last_loss = None
    if last_loss_tensor is not None:
        last_loss = float(last_loss_tensor.detach().cpu().item())

    return {
        "host_fill_seconds": host_fill_seconds,
        "copy_enqueue_seconds": copy_enqueue_seconds,
        "batches": total_batches,
        "last_loss": last_loss,
        "last_label_sum": last_label_sum,
    }


def make_model(torch: Any, model_step: str, device: Any) -> tuple[Any, Any]:
    if model_step == "transfer-only":
        return None, None
    model = torch.nn.Sequential(
        torch.nn.Conv3d(1, 4, kernel_size=3, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv3d(4, 1, kernel_size=1),
    ).to(device)
    optimizer = None
    if model_step == "train":
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    return model, optimizer


def run_model_step(
    torch: Any,
    gpu_batch: dict[str, Any],
    model_step: str,
    model: Any,
    optimizer: Any,
) -> Any:
    if model_step == "transfer-only":
        return None
    image = gpu_batch["image"]
    label = gpu_batch["label"]
    if model_step == "forward":
        with torch.inference_mode():
            out = model(image)
            loss = torch.nn.functional.mse_loss(out, label)
        return loss
    optimizer.zero_grad(set_to_none=True)
    out = model(image)
    loss = torch.nn.functional.mse_loss(out, label)
    loss.backward()
    optimizer.step()
    return loss


def import_torch():
    try:
        import torch  # type: ignore
    except ImportError as error:
        raise RuntimeError("PyTorch is required for this benchmark") from error
    return torch


if __name__ == "__main__":
    raise SystemExit(main())
