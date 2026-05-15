# Benchmarks

`medkit-benchmarks` is the dedicated performance harness for the project. It is
separate from the production `medkit bench` command so benchmark fixtures,
microbenchmarks, macrobenchmarks, and external baselines can evolve without
polluting the runtime crates.

## Synthetic And Cached Fixtures

The harness can generate nnU-Net-shaped synthetic CT segmentation data:

- `imagesTr/case_0000_0000.nii`
- `labelsTr/case_0000.nii`
- `ct-segmentation.toml`

The fixture code also has a cached mode that runs dataset validation and cache
preparation through the Rust libraries. Criterion benches use this path for
repeatable cache-read and patch-extraction workloads.

## Criterion Microbenchmarks

Run all benchmark targets:

```bash
cargo bench -p medkit-benchmarks
```

Run one target:

```bash
cargo bench -p medkit-benchmarks --bench transform_micro
cargo bench -p medkit-benchmarks --bench sampler_micro
cargo bench -p medkit-benchmarks --bench cache_micro
```

Current microbenchmarks cover:

- transform plan execution with geometry-aware resampling;
- aligned image/label patch extraction;
- Python-free batch planning;
- cached volume reads;
- the existing cold/warm `medkit-bench` cache extraction path.

## CLI Macrobenchmarks

Build the two binaries first:

```bash
cargo build -p medkit-cli -p medkit-benchmarks --release
```

Run the end-to-end CLI macrobenchmark:

```bash
cargo run -p medkit-benchmarks --release -- run \
  --cases 4 \
  --shape 128,128,128 \
  --cache-shape 128,128,128 \
  --patch 96,96,96 \
  --samples 10000 \
  --workers 8 \
  --medkit-bin target/release/medkit \
  --out target/medkit-macrobench.json
```

The runner creates a synthetic dataset and times these actual CLI stages:

```bash
medkit dataset validate
medkit prepare
medkit sample
medkit bench
```

The JSON report captures the command line, elapsed wall time, stdout, stderr,
and exit code for each stage.

For apples-to-apples comparisons, use the same `--samples`, `--patch`,
`--workers`, fixture root, and spacing in both `medkit-benchmarks run` and
`scripts/monai_baseline.py`. The macrobenchmark runner passes `--samples`
through to `medkit bench` so the cold/warm extraction metrics use the same
sample count as the MONAI baseline.

## Python MONAI Baseline

The baseline script compares against a MONAI `CacheDataset` pipeline with
`LoadImaged`, `EnsureChannelFirstd`, `Spacingd`, `ScaleIntensityRanged`,
`CropForegroundd`, and `RandCropByPosNegLabeld`.
It uses `monai.data.DataLoader`, configures
`RandCropByPosNegLabeld(num_samples=batch_size)`, and iterates explicit epochs
instead of `itertools.cycle(loader)` so repeated samples are actually re-cropped
rather than replayed from `cycle`'s internal cache.

Install optional Python dependencies in your Python environment:

```bash
python -m pip install monai nibabel torch
```

Run it against a generated synthetic fixture root:

```bash
python crates/medkit-benchmarks/scripts/monai_baseline.py \
  --data-root /path/to/synthetic/root \
  --patch 96,96,96 \
  --samples 10000 \
  --workers 8 \
  --out target/monai-baseline.json
```

The MONAI docs describe dictionary transforms operating on file paths via
`LoadImaged`, channel-first tensors for medical transforms, and `CacheDataset`
caching deterministic transforms before randomized transforms such as
`RandCropByPosNegLabeld`:

- https://docs.monai.io/en/latest/transforms.html
- https://monai.readthedocs.io/en/1.3.0/_modules/monai/data/dataset.html

## Real Dataset Workflow: MSD Task09 Spleen

Synthetic fixtures are useful for regression tests, but real performance work
needs real volumes, real labels, and the workflow a researcher would actually
run. The first supported real-data workflow uses the Medical Segmentation
Decathlon `Task09_Spleen` dataset:

- CT modality;
- spleen segmentation target;
- NIfTI `.nii.gz` images and labels;
- nnU-Net-like `imagesTr` / `labelsTr` folder structure;
- commonly used by MONAI tutorials and examples.

The full archive is about 1.5 GB, so the workflow script supports extracting a
subset of training cases for faster iteration before running the full dataset.

Build release binaries first:

```bash
cargo build -p medkit-cli -p medkit-benchmarks --release
```

Install the MONAI baseline environment if it does not already exist:

```bash
python3.11 -m venv target/monai-baseline-venv
target/monai-baseline-venv/bin/python -m pip install --upgrade pip
target/monai-baseline-venv/bin/python -m pip install monai nibabel torch
```

Run a two-case real-data subset:

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

This is the current canonical real-data smoke benchmark. It uses a small
two-case subset for iteration speed, but the work itself matches a normal
research training input pipeline: deterministic preprocessing cache, balanced
3D patch sampling, repeated patch extraction for an epoch-like stream, and a
PyTorch `Dataset` adapter over the sampled patch plan.
Very small sample counts, such as 512 patches, are useful for startup testing
but over-weight one-time load and process overhead.

Run the full extracted training set by setting `--cases 0`:

```bash
python crates/medkit-benchmarks/scripts/msd_spleen_workflow.py \
  --work-dir data/msd-spleen \
  --cases 0 \
  --patch 96,96,96 \
  --cache-shape 160,160,160 \
  --samples 10000 \
  --medkit-bin target/release/medkit \
  --python target/monai-baseline-venv/bin/python \
  --out data/msd-spleen/full-comparison.json
```

The script downloads `Task09_Spleen.tar` if needed, extracts either the selected
subset or all training cases, writes a medkit CT spleen transform plan, runs:

```bash
medkit dataset validate
medkit prepare
medkit sample
medkit bench
medkit bench-plan
medkit_torch_dataset_baseline.py
monai_baseline.py
```

and emits one JSON comparison with medkit stage timings, MONAI timing, and
relative throughput ratios.

Current two-case MSD Spleen result, 4096 patches, patch size `96,96,96`,
cache shape `160,160,160`, chunk shape `96,96,96`, medkit workers `8`,
MONAI workers `0`, effective DataLoader patch batch size `16`:

| Metric | medkit-rs | MONAI | Result |
|---|---:|---:|---:|
| Cache/prep build | 2.02 s | 6.17 s | medkit 3.06x faster |
| Sample plan throughput | 121,523 samples/s | 828 samples/s | medkit 146.82x faster |
| Centered cold extraction throughput | 19,002 samples/s | 828 samples/s | medkit 22.96x faster |
| Centered warm extraction throughput | 13,741 samples/s | 828 samples/s | medkit 16.60x faster |
| JSONL-plan cold extraction throughput | 10,010 samples/s | 828 samples/s | medkit 12.09x faster |
| JSONL-plan warm extraction throughput | 12,002 samples/s | 828 samples/s | medkit 14.50x faster |
| PyTorch DataLoader contiguous `ffi-batch` throughput | 5,009 samples/s | 828 samples/s | medkit 6.05x faster |

Report path:
`data/msd-spleen/subset-2-contiguous-ffi-batch-4096.json`.

The PyTorch DataLoader row now uses the `ffi-batch` backend. It asks the Rust
FFI bridge to fill standard contiguous `[B, C, Z, Y, X]` CPU tensors directly
from the medkit cache and sampled patch plan. The older `view-batch` result is
still useful as a zero-copy ceiling, but the contiguous path is the relevant
training-loop target.

An additional same-worker-count run is also useful when evaluating how the
pipeline behaves under process/thread orchestration overhead:
`data/msd-spleen/subset-2-comparison-4096-final-workers8v8.json`. On the
current machine, MONAI was faster with `--monai-workers 0` for this small
cached subset, so the table above keeps that stronger MONAI baseline.

Primary dataset references:

- https://medicaldecathlon.com/
- https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar
- https://huggingface.co/datasets/Angelou0516/msd-spleen
