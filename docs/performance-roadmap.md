# Performance Roadmap

This project is now past the synthetic-only stage. The current benchmark target
is the Medical Segmentation Decathlon `Task09_Spleen` workflow with a MONAI
`CacheDataset` plus `RandCropByPosNegLabeld` baseline and a medkit PyTorch
`Dataset` adapter over the sampled patch plan.

## Latest Real Benchmark

Command shape:

```bash
python crates/medkit-benchmarks/scripts/msd_spleen_workflow.py \
  --work-dir data/msd-spleen \
  --cases 2 \
  --patch 96,96,96 \
  --cache-shape 160,160,160 \
  --chunk 96,96,96 \
  --spacing 1.0,1.0,1.0 \
  --samples 4096 \
  --workers 8 \
  --monai-workers 0 \
  --batch-size 16 \
  --medkit-torch-backend view-batch \
  --medkit-bin target/release/medkit \
  --python target/monai-baseline-venv/bin/python \
  --out data/msd-spleen/subset-2-dataloader-win-4096.json
```

Current result on the local machine:

| Metric | medkit-rs | MONAI | Result |
|---|---:|---:|---:|
| Cache/prep build | 1.45 s | 3.98 s | medkit 2.74x faster |
| Sample plan throughput | 178,660 samples/s | 28,659 samples/s | medkit 6.23x faster |
| Centered cold extraction throughput | 23,972 samples/s | 28,659 samples/s | medkit 0.84x MONAI |
| Centered warm extraction throughput | 31,298 samples/s | 28,659 samples/s | medkit 1.09x faster |
| JSONL-plan cold extraction throughput | 19,083 samples/s | 28,659 samples/s | medkit 0.67x MONAI |
| JSONL-plan warm extraction throughput | 23,150 samples/s | 28,659 samples/s | medkit 0.81x MONAI |
| PyTorch DataLoader view-batch throughput | 178,269 samples/s | 28,659 samples/s | medkit 6.22x faster |

Interpretation:

- The Rust cache, foreground metadata, and plan-driven extraction path are
  viable.
- The cache is now training-native enough to persist foreground indices,
  foreground prefix volumes, and patch-shaped chunk files.
- The lazy `view-batch` PyTorch DataLoader path now beats MONAI on this
  benchmark by avoiding collation copies and using the persisted foreground
  prefix for label occupancy checks.
- This is not the final training interface. The next target is a contiguous
  tensor-batch path that also beats MONAI when the model expects a standard
  `[B, C, Z, Y, X]` tensor.

## What Winning Requires

The project now has a DataLoader benchmark win, but the stronger target is the
boundary between Rust and PyTorch for standard contiguous batches.

Priority order:

1. Build a PyO3 extension that loads medkit cache manifests and sampled patch
   plans once, then extracts whole batches in Rust.
2. Return Torch-compatible tensors with minimal copies. Start with NumPy arrays
   if needed, but design the interface so DLPack or direct torch tensor creation
   can replace it.
3. Make `image.chunks.f32.raw` and `label.chunks.u16.raw` real read paths, not
   just persisted artifacts. Patch extraction should touch only the chunks
   needed by a sampled patch.
4. Add a batch-level benchmark: MONAI DataLoader batch iteration versus medkit
   Rust-backed DataLoader batch iteration, with identical batch size, workers,
   patch count, and checksum.
5. Expand the real benchmark from the two-case smoke test to the full MSD Spleen
   training split, then add at least one MRI task and one multi-modal task.
6. Add pinned-memory and prefetch experiments only after the Rust-backed
   DataLoader path is correct and measurable.
7. Keep every benchmark result tied to a reproducible command and JSON report.

Success criteria for the next stage:

- medkit contiguous PyTorch DataLoader throughput beats MONAI on the 4096-patch
  two-case MSD Spleen workflow.
- The same path remains competitive on a larger full-dataset run.
- The benchmark includes both batch throughput and end-to-end cache build plus
  epoch iteration time.
