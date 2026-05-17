#![forbid(unsafe_code)]

mod error;
mod parser;
mod pixel;
mod scan;
mod types;
mod view;

pub use error::DicomError;
pub use parser::{inspect_dicom_file, DicomDataSet, DicomElement};
pub use pixel::{explain_pixels, present_dicom_pixels, PixelExplanation, PresentedImage};
pub use scan::{scan_dicom, write_scan_outputs};
pub use types::*;
pub use view::{render_unicode, RenderOptions};

pub type Result<T> = std::result::Result<T, DicomError>;

#[cfg(test)]
mod tests;
