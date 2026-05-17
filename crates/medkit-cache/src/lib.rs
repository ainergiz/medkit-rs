#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Content-addressed preprocessing cache for validated image/label datasets.

mod error;
mod inspect;
mod nifti_pixels;
mod prepare;

pub use error::{CacheError, Result};
pub use inspect::{
    inspect_cache, validate_cache, CacheCaseInspection, CacheInspection, CacheStorageKind,
};
pub use prepare::{
    prepare_cache, read_cache_manifest, CacheManifest, CacheSummary, CachedCase, PrepareConfig,
};
