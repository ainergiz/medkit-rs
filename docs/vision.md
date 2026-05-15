# Vision

`medkit-rs` aims to become a high-performance, end-to-end medical imaging data engine.

The project should be useful to people training medical imaging models, but it is also explicitly a learning vehicle for systems programming, medical image geometry, data pipeline architecture, and performance engineering.

## Ambition

The ambition is broad:

- manipulate medical images correctly and quickly;
- handle training-oriented workflows from raw data to sampled tensors;
- provide reusable building blocks across the stack;
- interoperate with existing Python tools;
- replace existing components when a Rust implementation can be meaningfully better;
- become a serious foundation for medical imaging AI data infrastructure.

This should not be approached like a fast-shipping application. The project should favor durable abstractions, careful benchmarks, and deep understanding of the stack.

## Guiding Principles

### Preserve Meaning Before Tensorization

Medical images carry physical and clinical meaning.

The engine should preserve spacing, orientation, affine transforms, modality, metadata, provenance, labels, and annotation geometry until the final training representation is required.

### Performance Is A Design Constraint

Performance should be designed into the architecture:

- memory-mapped IO where appropriate;
- crop-first and slice-first loading;
- lazy transform graphs;
- transform fusion;
- SIMD-aware kernels;
- parallel metadata scanning;
- cache-aware chunking;
- deterministic random sampling;
- low-copy Python interop;
- benchmarkable hot paths.

### Compatibility Without Subservience

The project should export to existing ecosystems:

- MONAI datalists;
- nnU-Net folder layouts;
- PyTorch datasets;
- NumPy arrays;
- JAX-compatible buffers;
- common metadata manifests.

But the Rust core should not be reduced to glue code. It should own its internal model of images, geometry, transforms, and execution.

### Build Compounding Blocks

Each layer should make later layers easier:

1. core geometry and array types;
2. metadata and provenance;
3. IO adapters;
4. validation;
5. lazy transforms;
6. physical execution planning;
7. caching;
8. sampling;
9. batching;
10. Python bindings;
11. distributed and accelerator-aware workflows.

## System Model

The desired long-term flow is:

```text
source data
  -> indexed clinical/research objects
  -> validated spatial images and annotations
  -> lazy transform graph
  -> optimized execution plan
  -> cacheable materialized chunks
  -> sampled training views
  -> framework tensors
```

The core engine should be able to explain and reproduce every step.

## Candidate Crates

The project may eventually split into crates such as:

- `medkit-core`: geometry, dtype, axis, metadata, provenance;
- `medkit-io`: NIfTI, DICOM metadata, DICOM series, WSI adapters;
- `medkit-array`: array views, chunked arrays, mmap buffers;
- `medkit-transform`: transforms, lazy graphs, interpolation;
- `medkit-cache`: deterministic cache keys and chunk stores;
- `medkit-sampler`: volume, patch, label-aware, and tile sampling;
- `medkit-python`: Python bindings;
- `medkit-cli`: validation, conversion, inspection, benchmarking.

This split should happen only when the code earns it.

## First Research Track

The first track should focus on 3D radiology datasets:

- NIfTI volumes;
- DICOM-derived CT/MRI/PET series;
- segmentation masks;
- image/mask geometry checks;
- nnU-Net-style datasets;
- MONAI-style datalists;
- patch-based training.

The first benchmarkable claim should be narrow:

> Given a volume and label mask, `medkit-rs` can validate geometry, apply deterministic preprocessing, cache the deterministic prefix, and sample training patches faster and more reproducibly than a naive Python pipeline.

## Open Questions

- Should the first internal array representation build directly on `ndarray`, or use a smaller custom view type?
- Should cache storage start with Zarr, a custom binary format, or both?
- How much DICOM pixel decoding should be owned by Rust versus delegated initially?
- How should transform graphs represent physical-space operations versus index-space operations?
- What is the minimum Python API that makes the Rust core genuinely useful?
- How should benchmark datasets be structured so comparisons are fair and reproducible?

## Tone Of The Project

The project should be ambitious, technical, and honest.

It should avoid vague performance claims. Every optimization should eventually be tied to a benchmark, trace, or clear architectural reason.

The long-term goal is not just to load medical images. It is to make medical imaging data pipelines faster, more correct, more inspectable, and easier to trust.

