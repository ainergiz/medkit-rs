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

## Python Surface

The CXR drop-in API exposes PyTorch-style dataset and loader helpers:

```python
import medkit_rs as medkit

train_ds = medkit.cxr.Dataset("data/cxr/.medkit/cache", split="train")
train_loader = medkit.cxr.DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    pin_memory=True,
    prefetch=True,
)
```

Batches use stable keys: `image`, `labels`, `mask`, and metadata sidecars such
as `sample_id`, `patient_id`, `study_id`, and `image_id`.

## Development

Create the development environment and build the native Python extension:

```bash
uv sync --dev
uv run maturin develop --release
```

Run the full test suite:

```bash
cargo test --workspace --all-targets
uv run pytest
```

Internal planning, benchmark notes, and generated reports are intentionally
ignored by git.
