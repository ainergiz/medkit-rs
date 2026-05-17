#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Dataset scanning, image/label pairing, validation, and manifest generation.
//!
//! The first workflow is intentionally NIfTI- and segmentation-oriented:
//! scan image and label directories, pair files into cases, read metadata
//! through `medkit-io`, validate geometry through `medkit-core`, then emit a
//! machine-readable manifest and a human-readable report.

mod error;
mod manifest;
mod pairing;
mod report;
mod validate;

pub use error::{DatasetError, Result};
pub use manifest::{
    CaseImage, CaseManifest, CaseStatus, DatasetManifest, ImageRecord, Problem, ProblemCode,
    ValidationSummary,
};
pub use pairing::{case_id_from_image_path, case_id_from_label_path, DatasetLayout};
pub use report::render_report;
pub use validate::{validate_dataset, write_manifest_json, write_report, ValidationConfig};
