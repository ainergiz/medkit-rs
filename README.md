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

### CXR H100 Benchmark Recipe

For current-source Modal benchmark runs, use the local package build until the
published PyPI package includes the latest CXR prefetch arguments:

```bash
MEDKIT_MODAL_USE_PYPI=0 python crates/medkit-benchmarks/scripts/modal_cxr_parallel_matrix.py \
  --batch-id cxr-confirm-h100-pinned-d2-rw4-local \
  --baselines pytorch_raw,medkit_native_prefetch_pinned \
  --cache-dtypes float32,uint8 \
  --read-modes stream \
  --image-size 512 \
  --batch-size 32 \
  --workers 8 \
  --max-samples 6000 \
  --max-train 4096 \
  --max-val 1024 \
  --max-test 1024 \
  --epochs 1 \
  --loader-batches 64 \
  --warmup-batches 4 \
  --profile-batches 128 \
  --drop-last-train \
  --prefetch-depth 2 \
  --prefetch-read-workers 4 \
  --no-include-metadata \
  --max-eval-batches 1 \
  --modal-gpu H100
```

The current fastest confirmed CXR path is native prefetch with pinned batches,
stream reads, `prefetch_depth=2`, and `prefetch_read_workers=4`. In the May 20,
2026 H100 confirmation run on the public NIH ChestX-ray14 cache, raw PyTorch
reached 194.7 train samples/s, while medkit pinned stream reached 379.8
samples/s with float32 cache data and 377.4 samples/s with uint8 cache data.
Both medkit rows used about 64 MB of estimated pinned batch memory and reported
near-zero cache-image PSS.

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

train_ds = medkit.cxr.Dataset("data/cxr/.medkit/cache", split="train")
train_loader = medkit.cxr.DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    prefetch=True,
)
```

Batches use stable keys: `image`, `labels`, `mask`, and metadata sidecars such
as `sample_id`, `patient_id`, `study_id`, and `image_id`.

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
