#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Deterministic foreground-balanced patch sampling from cached 3D volumes.

mod error;
mod rng;
mod sampler;

pub use error::{Result, SamplerError};
pub use sampler::{
    extract_patch_pair, extract_patch_pair_into, foreground_voxels_in_patch, load_cached_cases,
    plan_batches, sample_cache, BatchPlan, CachedPatch, ForegroundPrefix, LoadedCachedCase,
    PatchRecord, SampleConfig, SampleSummary, SamplingStrategy,
};
