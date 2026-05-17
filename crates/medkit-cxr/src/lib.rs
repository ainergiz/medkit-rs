#![forbid(unsafe_code)]

mod cache;
mod error;
mod ingest;
mod manifest;
mod recipe;
mod split;
mod types;
mod util;

pub use cache::{
    cache_cxr, cache_cxr_with_options, read_cache_summary, validate_cache_cxr, CxrCacheOptions,
};
pub use error::CxrError;
pub use ingest::ingest_cxr_dicom;
pub use manifest::{index_cxr, read_manifest, validate_cxr};
pub use recipe::{
    read_cxr_dicom_recipe, recipe_fingerprint, validate_cxr_dicom_recipe, CxrDicomRecipe,
    RecipeDicomSection, RecipeFingerprint, RecipeImageSection, RecipeLabelsSection,
    RecipePresentationSection, RecipeSplitSection,
};
pub use split::split_cxr;
pub use types::*;

#[cfg(test)]
mod tests;
