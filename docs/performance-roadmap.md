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
  --medkit-bin target/release/medkit \
  --python target/monai-baseline-venv/bin/python \
  --out data/msd-spleen/subset-2-training-native-4096.json
```

Current result on the local machine:

| Metric | medkit-rs | MONAI | Result |
|---|---:|---:|---:|
| Cache/prep build | 1.51 s | 4.34 s | medkit 2.88x faster |
| Sample plan throughput | 186,738 samples/s | 17,864 samples/s | medkit 10.45x faster |
| Centered cold extraction throughput | 21,588 samples/s | 17,864 samples/s | medkit 1.21x faster |
| Centered warm extraction throughput | 25,648 samples/s | 17,864 samples/s | medkit 1.44x faster |
| JSONL-plan cold extraction throughput | 16,396 samples/s | 17,864 samples/s | medkit 0.92x MONAI |
| JSONL-plan warm extraction throughput | 20,041 samples/s | 17,864 samples/s | medkit 1.12x faster |
| PyTorch DataLoader adapter throughput | 1,439 samples/s | 17,864 samples/s | medkit 0.08x MONAI |

Interpretation:

- The Rust cache, foreground metadata, and plan-driven extraction path are
  viable.
- The cache is now training-native enough to persist foreground indices,
  foreground prefix volumes, and patch-shaped chunk files.
- The Python DataLoader bridge is not competitive yet. It proves the integration
  shape, but it is still a Python memmap adapter instead of a Rust extension
  filling batches directly.

## What Winning Requires

The next performance target is not another CLI micro-optimization. The next
target is the boundary between Rust and PyTorch.

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

- medkit PyTorch DataLoader throughput beats MONAI on the 4096-patch two-case
  MSD Spleen workflow.
- The same path remains competitive on a larger full-dataset run.
- The benchmark includes both batch throughput and end-to-end cache build plus
  epoch iteration time.
