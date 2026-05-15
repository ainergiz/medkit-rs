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
  --medkit-torch-backend ffi-batch \
  --medkit-bin target/release/medkit \
  --python target/monai-baseline-venv/bin/python \
  --out data/msd-spleen/subset-2-contiguous-ffi-batch-4096.json
```

Current result on the local machine:

| Metric | medkit-rs | MONAI | Result |
|---|---:|---:|---:|
| Cache/prep build | 2.02 s | 6.17 s | medkit 3.06x faster |
| Sample plan throughput | 121,523 samples/s | 828 samples/s | medkit 146.82x faster |
| Centered cold extraction throughput | 19,002 samples/s | 828 samples/s | medkit 22.96x faster |
| Centered warm extraction throughput | 13,741 samples/s | 828 samples/s | medkit 16.60x faster |
| JSONL-plan cold extraction throughput | 10,010 samples/s | 828 samples/s | medkit 12.09x faster |
| JSONL-plan warm extraction throughput | 12,002 samples/s | 828 samples/s | medkit 14.50x faster |
| PyTorch DataLoader contiguous `ffi-batch` throughput | 5,009 samples/s | 828 samples/s | medkit 6.05x faster |

Interpretation:

- The Rust cache, foreground metadata, and plan-driven extraction path are
  viable.
- The cache is now training-native enough to persist foreground indices,
  foreground prefix volumes, and patch-shaped chunk files.
- The Rust-backed `ffi-batch` PyTorch DataLoader path now beats a corrected
  MONAI baseline while yielding standard contiguous `[B, C, Z, Y, X]` tensors.
- The MONAI baseline uses `monai.data.DataLoader`,
  `RandCropByPosNegLabeld(num_samples=batch_size)`, and explicit epoch
  iteration so repeated samples are re-cropped rather than replayed from
  `itertools.cycle`.

## What Winning Requires

The project now has a contiguous DataLoader benchmark win on the two-case MSD
Spleen smoke workflow. The stronger target is making that win survive larger
datasets, more modalities, and less favorable cache locality.

Priority order:

1. Replace the C-ABI experiment with a PyO3 extension that owns the cache
   manifest, sampled patch plan, thread pool, and reusable batch buffers.
2. Return Torch-compatible tensors with minimal copies through DLPack or direct
   torch tensor creation, then add pinned-memory and prefetch rings.
3. Make `image.chunks.f32.raw` and `label.chunks.u16.raw` real read paths, not
   just persisted artifacts. Patch extraction should touch only the chunks
   needed by a sampled patch.
4. Add benchmark guards for the MONAI baseline semantics: MONAI DataLoader,
   effective patch batch size, explicit epoch iteration, and no cached replay
   through `itertools.cycle`.
5. Expand the real benchmark from the two-case smoke test to the full MSD Spleen
   training split, then add at least one MRI task and one multi-modal task.
6. Keep every benchmark result tied to a reproducible command and JSON report.

Success criteria for the next stage:

- medkit contiguous PyTorch DataLoader throughput stays ahead of MONAI on the
  full MSD Spleen workflow.
- The same path remains competitive on an MRI task and a multimodal task.
- The benchmark includes both batch throughput and end-to-end cache build plus
  epoch iteration time.
