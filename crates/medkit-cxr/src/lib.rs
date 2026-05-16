#![forbid(unsafe_code)]

mod cache;
mod error;
mod manifest;
mod split;
mod types;
mod util;

pub use cache::{cache_cxr, read_cache_summary, validate_cache_cxr};
pub use error::CxrError;
pub use manifest::{index_cxr, read_manifest, validate_cxr};
pub use split::split_cxr;
pub use types::*;

#[cfg(test)]
mod tests;
