use std::path::Path;

use medkit_core::ImageSpec;

use crate::Result;

/// Reads image metadata into an [`ImageSpec`] without loading pixel data.
pub trait ImageMetadataReader {
    /// Reads image metadata from `path`.
    fn read_spec(&self, path: &Path) -> Result<ImageSpec>;
}
