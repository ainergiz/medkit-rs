#![forbid(unsafe_code)]

mod error;
mod graph;
mod parser;
mod pixel;
mod scan;
mod types;
mod view;

pub use error::DicomError;
pub use graph::{browse_dicom, construct_dicom_graph, write_graph_outputs};
pub use parser::{inspect_dicom_file, DicomDataSet, DicomElement};
pub use pixel::{
    explain_pixels, present_dicom_pixels, present_dicom_pixels_with_backend,
    present_dicom_pixels_with_options, DecodedPixelData, DecoderBackend, DicomPresentationOptions,
    DicomVoiStrategy, NativeDecoderBackend, PixelExplanation, PresentedImage,
};
pub use scan::{scan_dicom, scan_dicom_with_workers, write_scan_outputs};
pub use types::*;
pub use view::{render_unicode, RenderOptions};

pub type Result<T> = std::result::Result<T, DicomError>;

#[cfg(test)]
mod tests;
