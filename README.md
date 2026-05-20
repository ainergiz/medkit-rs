# medkit-rs

Rust-first medical imaging data tooling for dataset validation, deterministic
preprocessing caches, and training-time batch access.

## Workspace

- `medkit-core`: spatial image contracts, geometry, dtype, axes, metadata, and provenance.
- `medkit-io`: metadata readers for imaging formats such as NIfTI.
- `medkit-dataset`: dataset scanning, case pairing, validation, manifests, and reports.
- `medkit-transform`: deterministic preprocessing plans and kernels.
- `medkit-cache`: content-addressed preprocessing cache.
- `medkit-sampler`: foreground-balanced patch planning and extraction.
- `medkit-cxr`: CXR manifests, validation, patient-safe splits, and 2D cache creation.
- `medkit-python`: PyO3 extension for Rust-owned batch extraction.
- `medkit-python-ffi`: C-ABI bridge retained as a baseline.
- `medkit-cli`: command-line workflows exposed through the `medkit` binary.

## Core Workflows

Validate an nnU-Net-style NIfTI segmentation dataset:

```bash
cargo run -p medkit-cli -- dataset validate ./data \
  --images imagesTr \
  --labels labelsTr \
  --layout nnunet \
  --out manifest.json \
  --report report.txt
```

Prepare a deterministic cache:

```bash
cargo run -p medkit-cli -- prepare ./data \
  --manifest manifest.json \
  --plan ct-segmentation.toml \
  --cache .medkit/cache \
  --chunk 96,96,96
```

Sample and benchmark training patches:

```bash
cargo run -p medkit-cli -- sample .medkit/cache \
  --patch 96,96,96 \
  --strategy foreground-balanced \
  --count 10000 \
  --seed 123 \
  --epoch 0 \
  --worker 0 \
  --out patches.jsonl

cargo run -p medkit-cli -- bench-plan .medkit/cache \
  --patches patches.jsonl \
  --workers 8 \
  --samples 10000
```

Prepare a CXR cache for the Python drop-in loader:

```bash
cargo run -p medkit-cli -- cxr manifest --images data/cxr/files --metadata metadata.csv --labels labels.csv --out data/cxr/manifest.jsonl
cargo run -p medkit-cli -- cxr validate data/cxr/manifest.jsonl --require-frontal --check-patient-leakage --check-duplicates --report data/cxr/validation.md
cargo run -p medkit-cli -- cxr split data/cxr/manifest.jsonl --by patient_id --train 0.8 --val 0.1 --test 0.1 --seed 0 --out data/cxr/splits.json
cargo run -p medkit-cli -- cxr cache data/cxr/manifest.jsonl --splits data/cxr/splits.json --plan recipes/cxr-512.toml --cache data/cxr/.medkit/cache
cargo run -p medkit-cli -- cxr validate-cache data/cxr/.medkit/cache --split train --plan recipes/cxr-512.toml --report data/cxr/cache-validation.md
uv run --with torch examples/cxr_dropin_pytorch_train.py --cache-dir data/cxr/.medkit/cache --batch-size 32
```

### CXR Benchmark Gates

For current-source Modal benchmark runs, use the local package build until the
published PyPI package includes the latest CXR prefetch arguments. The matrix
launcher has repeatable raw+medkit gate presets that put all rows under one
batch id, force a single Modal GPU selector, and run three profiled repeats per
logical row by default. Use `--repeats 1` only for ad hoc debugging.

Dry-run a gate first when Modal is not available or before spending GPU time:

```bash
MEDKIT_MODAL_USE_PYPI=0 \
MEDKIT_MODAL_CLI="uvx --python 3.11 modal" \
python crates/medkit-benchmarks/scripts/modal_cxr_parallel_matrix.py \
  --gate h100-512-b32 \
  --batch-id cxr-gate-h100-512-b32-$(date -u +%Y%m%d-%H%M) \
  --dry-run
```

Run the H100 512/b32 gate:

```bash
MEDKIT_MODAL_USE_PYPI=0 \
MEDKIT_MODAL_CLI="uvx --python 3.11 modal" \
python crates/medkit-benchmarks/scripts/modal_cxr_parallel_matrix.py \
  --gate h100-512-b32 \
  --batch-id cxr-gate-h100-512-b32-$(date -u +%Y%m%d-%H%M)
```

Run the L4 224/b64 gate:

```bash
MEDKIT_MODAL_USE_PYPI=0 \
MEDKIT_MODAL_CLI="uvx --python 3.11 modal" \
python crates/medkit-benchmarks/scripts/modal_cxr_parallel_matrix.py \
  --gate l4-224-b64 \
  --batch-id cxr-gate-l4-224-b64-$(date -u +%Y%m%d-%H%M)
```

Re-audit an existing batch without launching Modal:

```bash
python crates/medkit-benchmarks/scripts/modal_cxr_parallel_matrix.py \
  --audit-batch target/reports/cxr-current-tools/<batch-id>
```

`MEDKIT_MODAL_CLI` is optional, but the Python 3.11 Modal client avoided local
heartbeat hangs observed with the globally installed Python 3.13 client during
these gate runs.

The current fastest confirmed CXR path is native prefetch with pinned batches,
stream reads, `prefetch_depth=2`, and `prefetch_read_workers=4`. Audited
three-repeat May 20, 2026 release gates on the public NIH ChestX-ray14 cache
produced:

| Gate | Raw PyTorch | medkit pinned stream | Result |
|---|---:|---:|---:|
| H100 512/b32, float32 | 265.1/s | 376.0/s | medkit 1.42x faster |
| H100 512/b32, uint8 | 265.1/s | 378.7/s | medkit 1.43x faster |
| L4 224/b64, float32 | 310.8/s | 362.4/s | medkit 1.17x faster |

All rows used 6,000 records, `--drop-last-train`, full profile windows, and
passed batch audit. The H100 release batch is
`cxr-release-h100-512-b32-20260520-codex-r0`; the L4 release batch is
`cxr-release-l4-224-b64-20260520-codex-r0`. Stream rows and raw rows reported
near-zero cache-image PSS. Mean GPU-loop PSS was about 5.54 GB for H100 raw,
5.71-5.73 GB for H100 medkit stream, 5.66 GB for L4 raw, and 5.75 GB for L4
medkit stream. Gate rows must retain `step-profile.json`,
`summary-consistency.json`, `run-summary.json`, `environment.json`, and the
row/batch summaries under `target/reports/cxr-current-tools/<batch-id>/`.
Each gate also writes `repeat-summary.json`, aggregating train throughput,
profile end-to-end throughput, loader throughput, data wait, GPU PSS, and
cache-image PSS across repeats, plus mean-speedup comparisons against the raw
PyTorch control row. The launcher treats missing profiler,
provenance, summary-consistency, or smaps/PSS telemetry as a row failure; gate
presets fail fast after the first invalid row.
Use `--shuffle-block-batches N` as an opt-in locality experiment for medkit
rows: it shuffles contiguous blocks of `N` native batches instead of individual
sample indices, preserving longer stream reads while still changing epoch order.
Use `--gpu-prefetch-batches N` as an opt-in CUDA handoff experiment; it keeps
CPU pinned batches copied ahead on a dedicated CUDA stream while the model step
runs. The gate presets leave this at `0` until repeat evidence justifies
promoting it.

Run the L4 quality gate when the question is training behavior rather than
loader speed:

```bash
MEDKIT_MODAL_USE_PYPI=0 \
MEDKIT_MODAL_CLI="uvx --python 3.11 modal" \
python crates/medkit-benchmarks/scripts/modal_cxr_parallel_matrix.py \
  --gate l4-quality-224-b64 \
  --batch-id cxr-quality-l4-224-b64-$(date -u +%Y%m%d-%H%M)
```

The quality gate uses full validation evaluation, balanced positive-class BCE,
patient/study/hash leakage checks, reproducible split checksums, and writes
`model-quality.json`, `threshold-report.json`, `quality-gate.json`,
`label-balance.json`, and `split-audit.json`.

The first audited quality batch,
`cxr-quality-l4-224-b64-20260520-codex-r0`, passed for raw PyTorch and medkit.
Both rows used two epochs, 7/7 measurable targets, and zero patient/study/hash
overlap. Raw reached 306.2 train samples/s, macro AUROC 0.636, and macro AUPRC
0.175. Medkit reached 363.8 train samples/s, macro AUROC 0.690, and macro AUPRC
0.180.

The first next-layer CUDA-prefetch optimization screens did not justify changing
the speed preset:

| Experiment | Shape | Mean train throughput | Decision |
|---|---:|---:|---|
| baseline medkit stream | L4 224/b64 float32 | 362.4/s | keep |
| `--gpu-prefetch-batches 1` | L4 224/b64 float32 | 359.5/s | do not promote |
| baseline medkit stream | H100 512/b32 float32 | 376.0/s | keep |
| `--gpu-prefetch-batches 1` | H100 512/b32 float32 | 376.5/s | neutral, do not promote |
| baseline medkit stream | H100 512/b32 uint8 | 378.7/s | keep |
| `--gpu-prefetch-batches 1` | H100 512/b32 uint8 | 374.1/s | do not promote |

CUDA-prefetch evidence batches:
`cxr-opt-l4-224-b64-gpuprefetch1-20260520-codex-r0`,
and `cxr-opt-h100-512-b32-gpuprefetch1-20260520-codex-r0`.

Initial block-shuffle batches from 2026-05-20 are not used as promotion
evidence because the native prefetch benchmark reported `shuffle_block_batches`
without passing it into `MedkitCxrNativePrefetchDataset`. Re-run block-shuffle
screens after that wiring fix before changing the speed preset.

## DICOM Decoder Policy

The default DICOM pixel backend is `medkit-native`. It keeps normal builds
small and covers the initial CXR-focused support matrix: uncompressed little and
big endian, RLE Lossless, and JPEG Baseline.

An opt-in DICOM-rs backend is available for broader pure-Rust codec coverage:

```bash
cargo test -p medkit-dicom --features dicom-rs-codecs
cargo run -p medkit-cli --features dicom-rs-codecs -- dicom pixels --explain image.dcm --decoder-backend dicom-rs
cargo run -p medkit-cli --features dicom-rs-codecs -- cxr cache manifest.jsonl --splits splits.json --plan recipes/cxr-512.toml --cache .medkit/cxr-cache --dicom-decoder-backend auto
```

Native codec stacks for JPEG-LS or JPEG 2000 are intentionally not enabled in
the default package until real fixtures and packaging tradeoffs are verified.

## Python Surface

The CXR drop-in API exposes PyTorch-style dataset and loader helpers:

```python
import medkit_rs as medkit

train_ds = medkit.cxr.Dataset("data/cxr/.medkit/cache", split="train", preset="speed")
train_loader = medkit.cxr.DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    drop_last=True,
)

print(train_loader.report_metadata())
```

Batches use stable keys: `image`, `labels`, `mask`, and metadata sidecars such
as `sample_id`, `patient_id`, `study_id`, and `image_id`.
`preset="speed"` selects the current benchmark path: pinned native prefetch,
stream reads, `prefetch_depth=2`, and `read_workers=4`. `preset="memory"` keeps
stream reads with a shallow unpinned queue. For shuffled stream training,
`shuffle_block_batches=N` can preserve local contiguous reads by shuffling
native-batch blocks instead of individual samples.

The examples under `examples/` use the same product surface:
`cxr_dropin_pytorch_train.py` for plain PyTorch,
`cxr_lightning_timm_datamodule.py` for Lightning/timm,
`cxr_torchxrayvision_wrapper.py` for TorchXRayVision-style batches, and
`cxr_monai_datalist_adapter.py` for MONAI datalist compatibility. The medkit
loader uses `num_workers=0` intentionally; Rust-native prefetch threads own the
background work, so passing PyTorch worker processes is rejected with a clear
error.

Free-threaded CPython builds such as `3.13t` and `3.14t` are not currently
supported. The published wheels target normal CPython, and the optimized CXR
path already moves hot data loading work into Rust-native prefetch threads
rather than Python bytecode threads. Supporting free-threaded Python is future
work: it requires a PyO3/maturin upgrade, dedicated `cp313t`/`cp314t` wheels,
and a thread-safety audit of the PyO3/Torch boundary, especially raw tensor
pointer writes, shared batch buffers, and native prefetch slot ownership.

## Development

Create the development environment and build the native Python extension:

```bash
uv sync --dev
uv run maturin develop --release
```

Run the full test suite:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked --exclude medkit-python
uv run python scripts/run_medkit_python_rust_tests.py
uv run python scripts/check_python_api.py
uv run python -m compileall python tests scripts examples crates/medkit-benchmarks/scripts
uv run pytest tests/python -q
```

Internal planning, benchmark notes, and generated reports are intentionally
ignored by git.
