# Performance Roadmap

This project is now past the synthetic-only stage. The current benchmark target
is the Medical Segmentation Decathlon `Task09_Spleen` workflow with a MONAI
`CacheDataset` plus `RandCropByPosNegLabeld` baseline, medkit PyTorch adapters
over sampled patch plans, and a Modal L40S GPU-fed training loop.

## Latest Real Benchmark

Current strongest command:

```bash
modal run crates/medkit-benchmarks/scripts/modal_msd_gpu.py \
  --samples 10000 \
  --batch-size 16 \
  --cases 0 \
  --medkit-workers 16 \
  --monai-workers 0 \
  --prefetch-batches 3 \
  --warmup-batches 4 \
  --model-step forward \
  --include-chunked
```

Current result on Modal L40S, all 41 MSD Spleen training cases, 10,000 sampled
`96^3` patches:

| Metric | medkit-rs | MONAI | Result |
|---|---:|---:|---:|
| Cache/prep build | 65.06 s | 165.78 s | medkit 2.55x faster |
| Sample plan throughput | 11,356 samples/s | 132 samples/s | medkit 86.31x faster |
| JSONL-plan cold extraction throughput | 3,184 samples/s | 132 samples/s | medkit 24.20x faster |
| JSONL-plan warm extraction throughput | 14,222 samples/s | 132 samples/s | medkit 108.10x faster |
| PyTorch DataLoader `native-batch` resident | 3,409 samples/s | 132 samples/s | medkit 25.91x faster |
| PyTorch DataLoader `native-chunk-batch` mmap chunks | 2,260 samples/s | 132 samples/s | medkit 17.17x faster |
| Pinned CUDA prefetch, resident, forward model | 1,252 samples/s | n/a | 8.25 GB/s H2D effective |
| Pinned CUDA prefetch, chunked, forward model | 1,134 samples/s | n/a | 7.48 GB/s H2D effective |

Modal report prefix:
`/cache/results/msd-spleen-cases0-samples10000-batch16-20260515-210805-*`.

Secondary full-split shape:

| Shape | medkit resident DataLoader | medkit chunked DataLoader | MONAI | resident GPU forward | chunked GPU forward |
|---|---:|---:|---:|---:|---:|
| 4,096 samples, batch 8 | 2,331/s | 1,644/s | 286/s | 1,158/s | 984/s |

Secondary report prefix:
`/cache/results/msd-spleen-cases0-samples4096-batch8-20260515-211454-*`.

The local two-case smoke command remains useful for fast iteration:

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
  --medkit-torch-backend native-batch \
  --medkit-bin target/release/medkit \
  --python target/monai-baseline-venv/bin/python \
  --out data/msd-spleen/subset-2-native-batch-4096.json
```

Current two-case result on the local machine:

| Metric | medkit-rs | MONAI | Result |
|---|---:|---:|---:|
| Cache/prep build | 2.07 s | 5.96 s | medkit 2.88x faster |
| Sample plan throughput | 110,306 samples/s | 858 samples/s | medkit 128.59x faster |
| Centered cold extraction throughput | 17,919 samples/s | 858 samples/s | medkit 20.89x faster |
| Centered warm extraction throughput | 15,453 samples/s | 858 samples/s | medkit 18.01x faster |
| JSONL-plan cold extraction throughput | 12,325 samples/s | 858 samples/s | medkit 14.37x faster |
| JSONL-plan warm extraction throughput | 12,167 samples/s | 858 samples/s | medkit 14.18x faster |
| PyTorch DataLoader contiguous `native-batch` throughput | 5,354 samples/s | 858 samples/s | medkit 6.24x faster |

Interpretation:

- The Rust cache, foreground metadata, and plan-driven extraction path are
  viable.
- The cache is now training-native enough to persist foreground indices,
  foreground prefix volumes, and patch-shaped chunk files.
- The PyO3-backed `native-batch` PyTorch DataLoader path now beats a corrected
  MONAI baseline while yielding standard contiguous `[B, C, Z, Y, X]` tensors,
  and this holds on the full 41-case MSD Spleen split.
- The mmap chunk-backed path is no longer just a persisted artifact. It is a
  real runtime storage mode and reaches 66% of resident DataLoader throughput
  on the 10,000-sample full-split run, then narrows to 91% of resident in the
  pinned CUDA forward loop where transfer/model work dominate.
- The first Modal GPU loop proves pinned host batches and CUDA stream copies
  work, but it also shows the next bottleneck clearly: on the 10,000-sample
  resident run, host fill was 2.24 s of a 7.99 s loop and copy enqueue accounted
  for 5.45 s. The next GPU work should focus on true background production,
  fewer Python dispatches, and lower-overhead tensor handoff.
- The MONAI baseline uses `monai.data.DataLoader`,
  `RandCropByPosNegLabeld(num_samples=batch_size)`, and explicit epoch
  iteration so repeated samples are re-cropped rather than replayed from
  `itertools.cycle`.

## Next Big Iteration: Memory-Efficient CXR Training

The installable-package CXR run on Modal L4 used the published
`medkit-rs==0.1.0` Python wheel against the public NIH ChestX-ray14 320px
Hugging Face parquet export, not MIMIC-CXR. The fixed-split comparison used
6,000 records, 3,998 train samples, 224px images, batch size 64, a DenseNet121
training loop, and the same requested split file for both paths. Patient safety
is based on NIH filename-derived patient IDs. The table below is a post-cache
training-time comparison: the float32 medkit cache build was about 24.5 s and
is excluded from the throughput numbers. The current Modal CXR benchmark cache
is still materialized by the Python benchmark harness, although the file layout
is compatible with the Rust CXR reader. The historical artifacts behind this
table reported peak RSS only; newer runs should use the richer
RSS/PSS/USS/file-backed/private/pinned telemetry in the benchmark harness.

| Metric | Raw PyTorch | medkit CXR loader | Result |
|---|---:|---:|---:|
| Loader throughput | 1,046 samples/s | 3,262 samples/s | medkit 3.12x faster |
| Train throughput | 146.1 samples/s | 159.4 samples/s | medkit 1.09x faster |
| Data wait | 2.24% | 0.33% | medkit waits less |
| Time to first batch | 572 ms | 1,080 ms | raw starts faster |
| Host peak RSS | 4,155 MB | 4,996 MB | medkit uses 1.20x RSS |
| CUDA peak memory | 4,236 MB | 4,236 MB | equal |

Follow-up ablations showed that the higher RSS is not mainly caused by pinned
memory or a deep prefetch queue:

| medkit path | Loader throughput | Train throughput | Data wait | Peak RSS |
|---|---:|---:|---:|---:|
| Pinned prefetch, depth 4 | 3,262 samples/s | 159.4 samples/s | 0.33% | 4,996 MB |
| Unpinned prefetch, depth 1 | 3,285 samples/s | 157.3 samples/s | 1.10% | 4,943 MB |
| Native no-prefetch batch path | 26,111 samples/s | 163.6 samples/s | 0.89% | 4,937 MB |

Interpretation:

- The measured CXR cache path is speed-biased. In this run it moves PNG/JPEG
  decode, grayscale conversion, resize, scaling, normalization, and label-mask
  assembly into a deterministic `float32` cache. DICOM should be measured
  separately.
- The measured training gain is modest on this small 224px L4 run: 159.4 vs
  146.1 samples/s, because DenseNet121 compute dominates once data wait is
  reduced.
- Medkit speed-mode RSS is about 841 MB higher than raw PyTorch. This is
  consistent with the 802.4 MB train image cache and native cached access path,
  but peak RSS alone cannot separate file-backed pages, private heap, pinned
  host memory, or allocator effects.
- Reducing `prefetch_depth` from 4 to 1 and disabling `pin_memory` saves about
  52 MB RSS in this benchmark; the native no-prefetch path saves about 58 MB
  versus pinned depth 4. Queue tuning therefore is unlikely to be the primary
  memory fix.

The next big CXR performance iteration should make medkit memory-conscious
without giving up the deterministic cache contract, and it should also prove
that the same cache design scales on larger GPUs where data loading can become
visible again once model throughput increases. The L4 runs below are the
memory-pressure regression target, not the ceiling for optimization.

2026-05-19 local-source Modal validation has now implemented and measured the
first lever, `read_mode = "mmap" | "stream"`, on the same L4/224px/batch64
shape with one baseline per Modal process:

| medkit path | Read mode | Loader PSS | GPU-loop PSS | Loader throughput | Train throughput |
|---|---:|---:|---:|---:|---:|
| Native prefetch | `mmap` | 4,952 MB | 7,436 MB | 23,420 samples/s | 166.2 samples/s |
| Native prefetch | `stream` | 4,193 MB | 5,719 MB | 5,431 samples/s | 134.0 samples/s |
| Native prefetch pinned | `mmap` | 4,952 MB | 7,439 MB | 11,040 samples/s | 136.0 samples/s |
| Native prefetch pinned | `stream` | 4,199 MB | 5,707 MB | 5,925 samples/s | 134.0 samples/s |

The streaming path is useful for memory-constrained pinned CXR training: it cut
loader PSS by about 753 MB and GPU-loop PSS by about 1.73 GB while leaving
pinned train throughput essentially flat. It should not become the universal
default yet, because unpinned native prefetch lost substantial throughput.
`smaps_pss_file_mb` reported 0.0 in all four rows, so those historical rows
should rely on total PSS/USS and train throughput. The next rerun should use the
new smaps parser that classifies mappings by address headers and known cache
paths, then verify that mmap rows report nonzero cache-file PSS after the image
cache has been touched.

Next implementation priorities:

1. Keep CXR reader memory policy explicit: `read_mode = "mmap" | "stream"`.
   `mmap` remains the speed default. `stream` is a memory-pressure lever and
   should use persistent file handles plus positioned reads rather than mapping
   the whole split image cache.
2. Keep CXR cache dtype policy explicit:
   `cache_dtype = "float32" | "float16" | "uint8"`. Apply dtype compaction to
   the image cache first; labels and masks stay float32 until there is evidence
   that compact label storage matters.
3. Keep metadata opt-in for training batches: `include_metadata = false`.
   Training loops normally need only `image`, `labels`, and `mask`; metadata
   strings remain available for audit/debug sidecars.
4. Keep user-facing defaults on a balanced policy:
   `prefetch_depth = 1`, `read_workers = 1`, and `pin_memory = false` unless a
   benchmark shows the GPU is data-starved. Keep pinned/deeper prefetch as
   explicit speed knobs.
5. Improve memory telemetry in every report. Summaries should include RSS, PSS,
   USS/private dirty memory, cache-file PSS buckets, other file-backed mappings,
   cache file size, and estimated pinned batch footprint.
6. Run two benchmark tracks, not one:
   - Memory-pressure track: L4/224px/batch64 remains the comparable regression
     target for host-memory and throughput tradeoffs.
   - Scale-up track: run 512px and higher-throughput configurations on a larger
     GPU target such as L40S, A100, or H100 depending on Modal availability. The
     project should prove that medkit remains useful when training moves to
     larger GPUs, not only when optimizing around constrained hosts.

Success criteria:

- On the same L4 CXR benchmark, the memory-conscious mode should keep medkit
  train throughput ahead of raw PyTorch while reducing host RSS materially.
- On the larger-GPU CXR benchmark, speed mode and memory-conscious mode should
  both be measured against a same-run raw PyTorch baseline so we can see whether
  cache dtype, read mode, metadata suppression, and prefetch settings still
  matter when GPU throughput rises.
- The speed mode should keep the current low data-wait profile and document the
  extra RSS as an explicit cache/prefetch tradeoff.
- Benchmark reports should make it clear whether memory is private heap,
  file-backed cache pages, pinned host memory, or GPU allocation.

## What Winning Requires

The project now has a contiguous DataLoader benchmark win on the full MSD Spleen
workflow, plus a chunk-backed and GPU-fed proof point. The stronger target is
making that win survive more modalities, less favorable cache locality, and
real training code rather than a synthetic forward benchmark.

Priority order:

1. Convert the GPU prefetch benchmark into a reusable Python iterator/training
   bridge while keeping the benchmark harness as a regression test.
2. Reduce CUDA-loop overhead: background CPU producer, reusable GPU buffers
   where safe, fewer Python calls per batch, and DLPack/direct tensor handoff if
   it beats the current Torch allocation route.
3. Optimize chunk-backed extraction: remove unnecessary zero-fills, speed up
   `u16` to `f32` label conversion, cache chunk overlap metadata, and benchmark
   less patch-aligned chunk shapes.
4. Add benchmark guards for the MONAI baseline semantics: MONAI DataLoader,
   effective patch batch size, explicit epoch iteration, and no cached replay
   through `itertools.cycle`.
5. Add at least one MRI task and one multi-modal task, then rerun the same
   resident/chunked/GPU tables.
6. Keep every benchmark result tied to a reproducible command and JSON report.

Success criteria for the next stage:

- medkit contiguous PyTorch DataLoader throughput stays ahead of MONAI on the
  full MSD Spleen workflow.
- The same path remains competitive on an MRI task and a multimodal task.
- The benchmark includes both batch throughput and end-to-end cache build plus
  epoch iteration time.
