#![deny(missing_docs)]

//! Deterministic foreground-balanced patch sampling from cached 3D volumes.

mod error;
mod rng;
mod sampler;

pub use error::{Result, SamplerError};
pub use sampler::{
    extract_patch_pair, extract_patch_pair_chunked_into, extract_patch_pair_into,
    extract_patch_pair_mmap_into, foreground_voxels_in_patch, load_cached_cases,
    load_chunked_cached_cases, load_mmap_cached_cases, plan_batches, sample_cache, BatchPlan,
    CachedPatch, ChunkedCachedCase, ForegroundPrefix, LoadedCachedCase, MmapCachedCase,
    PatchRecord, SampleConfig, SampleSummary, SamplingStrategy,
};
