# medkit-rs

`medkit-rs` is an ambitious Rust-first medical imaging data engine for high-performance image manipulation, dataset preparation, and training-time data access.

The project starts from a deliberately systems-heavy premise: medical AI data loading should be treated as a full stack performance problem, not as a collection of incidental preprocessing scripts.

## Initial Aim

Build an end-to-end toolchain for medical image data that can:

- ingest clinical and research imaging formats;
- preserve geometry, metadata, provenance, and clinical meaning;
- validate datasets before training;
- transform and cache data efficiently;
- sample volumes, patches, tiles, labels, and multimodal inputs;
- expose fast Rust APIs;
- provide Python bindings for PyTorch, MONAI, nnU-Net, JAX, and related ecosystems;
- eventually compete with or replace parts of existing tools where performance, correctness, or maintainability justify it.

Compatibility matters, but the project is not constrained to being a thin adapter around existing Python workflows. If a full replacement for parts of MONAI, TorchIO, or one-off preprocessing stacks becomes the right outcome, the architecture should allow that.

## Philosophy

This is a passion project, so the engineering bar is intentionally high.

We will optimize aggressively, but in a compounding way:

- start with narrow, benchmarkable building blocks;
- make correctness and geometry explicit;
- profile before claiming performance wins;
- design APIs that can survive broader format and training support;
- keep Python interoperability practical;
- avoid prematurely depending on Python for core execution;
- build toward reusable primitives rather than a single-purpose app.

## Core Thesis

Medical images are not just tensors. They are spatial, clinical, and provenance-bearing objects that can be materialized into tensors for training.

That means the core representation must track:

- shape;
- dtype;
- axis order;
- voxel spacing;
- origin;
- orientation;
- affine transforms;
- modality;
- acquisition/source metadata;
- labels and annotation geometry;
- transformation history.

The tensor is the final training view, not the source of truth.

## Target Stack

The long-term stack is expected to include:

- Rust core libraries for IO, geometry, transforms, sampling, caching, and batching;
- command-line tools for inspection, validation, conversion, and benchmarking;
- Python bindings via PyO3 or a similar bridge;
- zero-copy or low-copy handoff to NumPy, PyTorch, and JAX where possible;
- compatibility exports for MONAI, nnU-Net, TorchIO, and common dataset manifests;
- optional accelerated backends for SIMD, multithreading, GPU-oriented IO, and chunked storage.

## Early Direction

The first serious target should be radiology volumes, especially NIfTI and DICOM-derived CT/MRI/PET datasets.

Initial focus:

- NIfTI metadata and array loading;
- DICOM metadata indexing and series grouping;
- geometry validation;
- image/mask consistency checks;
- train/validation/test leakage checks;
- fast crop and patch sampling;
- transform planning;
- reproducible caching;
- benchmark comparisons against common Python pipelines.

## Current Building Blocks

The workspace currently contains:

- `medkit-core`: non-pixel spatial image contracts, geometry, dtype, axes, metadata, provenance, and compatibility validation.
- `medkit-io`: metadata-only image readers that convert format headers into `medkit-core` specs.
- `medkit-dataset`: dataset scanning, case pairing, image/label geometry validation, JSON manifests, and text reports.
- `medkit-transform`: lazy transform plans plus deterministic preprocessing kernels for CT windowing, normalization, foreground cropping, pad/crop, and geometry-aware 3D resampling.
- `medkit-cache`: content-addressed deterministic preprocessing cache with raw f32 image/u16 label volumes, persisted foreground indices/prefixes, optional patch-friendly chunk files, and source/output geometry provenance.
- `medkit-sampler`: deterministic foreground-balanced patch planning and aligned image/label patch extraction.
- `medkit-bench`: cold/warm cache loading and patch extraction throughput metrics.
- `medkit-benchmarks`: synthetic/cached fixtures, Criterion microbenchmarks, CLI macrobenchmarks, and a Python MONAI baseline script.
- `medkit-python-ffi`: a C-ABI Rust bridge used by Python experiments for batch extraction.
- `medkit-cli`: command-line workflows exposed through the `medkit` binary.

The first IO adapter is a NIfTI-1 metadata reader for `.nii`, `.nii.gz`, and `.hdr` files. It reads only the header, maps shape/dtype/spacing/origin/direction into `ImageSpec`, and handles sform/qform geometry without loading pixel arrays.

Example Rust usage:

```rust
use medkit_io::{ImageMetadataReader, NiftiMetadataReader};

let reader = NiftiMetadataReader::new();
let spec = reader.read_spec("volume.nii.gz".as_ref())?;
println!("{:?}", spec.geometry().shape());
```

The first end-to-end workflow validates an nnU-Net-shaped NIfTI segmentation dataset:

```bash
cargo run -p medkit-cli -- dataset validate ./data \
  --images imagesTr \
  --labels labelsTr \
  --out manifest.json \
  --report report.txt
```

The command:

- scans image and label folders recursively;
- pairs cases by filename, including image channel suffixes such as `_0000`;
- reads NIfTI metadata without loading pixels;
- validates image/label geometry;
- reports missing images, missing labels, duplicate mappings, read errors, and geometry mismatches;
- writes a machine-readable JSON manifest;
- writes a human-readable validation report.

The first training-runtime workflow turns a validated dataset into cached training data:

```bash
cargo run -p medkit-cli -- prepare ./data \
  --manifest manifest.json \
  --plan ct-segmentation.toml \
  --cache .medkit/cache \
  --chunk 96,96,96
```

The transform plan is TOML:

```toml
name = "ct-segmentation"
image_interpolation = "linear"
label_interpolation = "nearest"

[[operations]]
op = "ct_window"
min = -1000.0
max = 1000.0

[[operations]]
op = "normalize"
mean = 0.0
std = 1.0

[[operations]]
op = "crop_foreground"
margin = 2

[[operations]]
op = "pad_crop"
size = [96, 96, 96]

[[operations]]
op = "resample"
spacing = [1.0, 1.0, 1.0]
```

Sample deterministic patch records:

```bash
cargo run -p medkit-cli -- sample .medkit/cache \
  --patch 96,96,96 \
  --strategy foreground-balanced \
  --count 10000 \
  --out patches.jsonl
```

Benchmark cold and warm cache extraction:

```bash
cargo run -p medkit-cli -- bench .medkit/cache \
  --patch 96,96,96 \
  --workers 8 \
  --samples 10000
```

Benchmark extraction from an actual sampled patch plan:

```bash
cargo run -p medkit-cli -- bench-plan .medkit/cache \
  --patches patches.jsonl \
  --workers 8 \
  --samples 10000
```

For deeper performance work, use the dedicated benchmark harness:

```bash
cargo bench -p medkit-benchmarks
cargo run -p medkit-benchmarks -- run --cases 4 --shape 64,64,64 --patch 32,32,32
python crates/medkit-benchmarks/scripts/monai_baseline.py --data-root /path/to/fixture --patch 32,32,32
```

The real-data benchmark target is the Medical Segmentation Decathlon
`Task09_Spleen` workflow. It downloads or reuses the public dataset, extracts
NIfTI image/label cases, runs the medkit validation, cache, sampling, and
extraction commands, runs a medkit PyTorch `Dataset` adapter over the sampled
patch plan, then runs a comparable MONAI `CacheDataset` plus
`RandCropByPosNegLabeld` baseline.

The latest local DataLoader comparison uses medkit's lazy `view-batch` adapter
and beats MONAI on the two-case MSD Spleen smoke benchmark. The next target is
to make the same win hold for standard contiguous tensor batches.

See [docs/benchmarks.md](docs/benchmarks.md) for fixture, microbenchmark,
macrobenchmark, real MSD Spleen workflow, and MONAI baseline details. See
[docs/performance-roadmap.md](docs/performance-roadmap.md) for the latest
training-native benchmark result and the next steps needed to beat MONAI inside
a real PyTorch `DataLoader`.

Whole-slide pathology, DICOMweb, DICOM-SEG, RTSTRUCT, multimodal reports, and distributed training support are important future targets, but they should be added through reusable abstractions rather than bolted on.

## Non-Goals For The First Phase

- Building a model training framework first.
- Reimplementing every DICOM edge case before useful dataset tooling exists.
- Replacing MONAI or nnU-Net by declaration.
- Optimizing without benchmarks.
- Collapsing medical images into anonymous tensors too early.

## Success Shape

The project is successful if it becomes a tool that researchers and engineers can use to answer:

- Is my dataset geometrically valid?
- Are my labels aligned with my images?
- Did I accidentally leak patients across splits?
- Can I reproduce this preprocessing exactly?
- Can I cache deterministic preprocessing once?
- Can I sample training patches faster?
- Can I feed the same data into MONAI, nnU-Net, PyTorch, or JAX?
- Can I inspect what happened to every image from source to tensor?

Performance is the entry point. Trustworthy medical data handling is the reason it matters.
