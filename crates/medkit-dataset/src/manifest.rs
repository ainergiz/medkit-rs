use std::path::Path;

use medkit_core::{GeometryMismatch, ImageSpec};
use serde::{Deserialize, Serialize};

use crate::pairing::DatasetLayout;

/// Machine-readable validation output for a dataset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetManifest {
    /// Dataset root path.
    pub dataset_root: String,
    /// Image directory path.
    pub images_dir: String,
    /// Label directory path.
    pub labels_dir: String,
    /// Image naming layout used while pairing images and labels.
    pub layout: DatasetLayout,
    /// Aggregate validation counts.
    pub summary: ValidationSummary,
    /// Per-case validation records.
    pub cases: Vec<CaseManifest>,
}

/// Aggregate validation counts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationSummary {
    /// Total case count.
    pub total_cases: usize,
    /// Number of valid cases.
    pub valid_cases: usize,
    /// Number of invalid cases.
    pub invalid_cases: usize,
    /// Number of cases missing an image.
    pub missing_images: usize,
    /// Number of cases missing a label.
    pub missing_labels: usize,
    /// Number of cases with geometry mismatches.
    pub geometry_mismatches: usize,
    /// Number of cases with metadata read errors.
    pub read_errors: usize,
}

/// Validation status for a case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseStatus {
    /// Case is valid.
    Valid,
    /// Case is invalid.
    Invalid,
}

/// Per-case manifest entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseManifest {
    /// Stable case identifier.
    pub case_id: String,
    /// Case validation status.
    pub status: CaseStatus,
    /// Image file path, if present.
    pub image_path: Option<String>,
    /// Label file path, if present.
    pub label_path: Option<String>,
    /// Image metadata, if successfully read.
    pub image: Option<ImageRecord>,
    /// Structured image channels for layouts such as nnU-Net multi-modal cases.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<CaseImage>,
    /// Label metadata, if successfully read.
    pub label: Option<ImageRecord>,
    /// Problems found for this case.
    pub problems: Vec<Problem>,
}

impl CaseManifest {
    /// Builds a case manifest and infers status from its problem list.
    pub fn new(
        case_id: impl Into<String>,
        image_path: Option<String>,
        label_path: Option<String>,
        image: Option<ImageRecord>,
        label: Option<ImageRecord>,
        problems: Vec<Problem>,
    ) -> Self {
        let status = if problems.is_empty() {
            CaseStatus::Valid
        } else {
            CaseStatus::Invalid
        };
        Self {
            case_id: case_id.into(),
            status,
            image_path,
            label_path,
            image,
            images: Vec::new(),
            label,
            problems,
        }
    }
}

/// One image channel belonging to a manifest case.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseImage {
    /// Image path.
    pub path: String,
    /// Optional nnU-Net-style channel index parsed from `_dddd` suffixes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel_index: Option<u16>,
    /// Optional modality label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modality: Option<String>,
    /// Image metadata, if successfully read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<ImageRecord>,
}

/// Serializable metadata extracted from an image spec.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageRecord {
    /// Scalar dtype.
    pub dtype: String,
    /// Modality.
    pub modality: String,
    /// Shape dimensions.
    pub shape: Vec<usize>,
    /// Voxel spacing.
    pub spacing: Vec<f64>,
    /// Physical origin.
    pub origin: Vec<f64>,
    /// Row-major direction matrix.
    pub direction: Vec<f64>,
    /// Coordinate system.
    pub coordinate_system: String,
    /// Source kind.
    pub source_kind: String,
    /// Source URI/path.
    pub source_uri: String,
}

impl ImageRecord {
    /// Creates a serializable image record from a core image spec.
    pub fn from_spec(spec: &ImageSpec) -> Self {
        Self {
            dtype: format!("{:?}", spec.dtype()),
            modality: format!("{:?}", spec.modality()),
            shape: spec.geometry().shape().as_slice().to_vec(),
            spacing: spec.geometry().spacing().to_vec(),
            origin: spec.geometry().origin().to_vec(),
            direction: spec.geometry().direction().to_vec(),
            coordinate_system: format!("{:?}", spec.geometry().coordinate_system()),
            source_kind: format!("{:?}", spec.provenance().source().kind()),
            source_uri: spec.provenance().source().uri().to_string(),
        }
    }
}

/// Problem code used in manifests and reports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProblemCode {
    /// No image file was found for the case.
    MissingImage,
    /// No label file was found for the case.
    MissingLabel,
    /// Multiple image files mapped to the same case id.
    DuplicateImage,
    /// Multiple label files mapped to the same case id.
    DuplicateLabel,
    /// Image metadata could not be read.
    ImageReadError,
    /// Label metadata could not be read.
    LabelReadError,
    /// Image and label geometry are incompatible.
    GeometryMismatch,
}

/// A validation problem for a case.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Problem {
    /// Problem code.
    pub code: ProblemCode,
    /// Human-readable problem message.
    pub message: String,
}

impl Problem {
    /// Creates a validation problem.
    pub fn new(code: ProblemCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    /// Creates a geometry problem from core mismatch details.
    pub fn geometry(mismatch: &GeometryMismatch) -> Self {
        Self::new(
            ProblemCode::GeometryMismatch,
            format!(
                "image/label geometry mismatch: {}",
                describe_geometry_mismatch(mismatch)
            ),
        )
    }
}

/// Computes aggregate validation summary from cases.
pub fn summarize(cases: &[CaseManifest]) -> ValidationSummary {
    let mut summary = ValidationSummary {
        total_cases: cases.len(),
        ..ValidationSummary::default()
    };
    for case in cases {
        match case.status {
            CaseStatus::Valid => summary.valid_cases += 1,
            CaseStatus::Invalid => summary.invalid_cases += 1,
        }
        for problem in &case.problems {
            match problem.code {
                ProblemCode::MissingImage => summary.missing_images += 1,
                ProblemCode::MissingLabel => summary.missing_labels += 1,
                ProblemCode::GeometryMismatch => summary.geometry_mismatches += 1,
                ProblemCode::ImageReadError | ProblemCode::LabelReadError => {
                    summary.read_errors += 1
                }
                ProblemCode::DuplicateImage | ProblemCode::DuplicateLabel => {}
            }
        }
    }
    summary
}

/// Converts a path to a stable string for manifests.
pub fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn describe_geometry_mismatch(mismatch: &GeometryMismatch) -> String {
    match mismatch {
        GeometryMismatch::Shape { left, right } => {
            format!("shape differs: image={left:?}, label={right:?}")
        }
        GeometryMismatch::CoordinateSystem { left, right } => {
            format!("coordinate system differs: image={left}, label={right}")
        }
        GeometryMismatch::Spacing { index, left, right } => {
            format!("spacing[{index}] differs: image={left}, label={right}")
        }
        GeometryMismatch::Origin { index, left, right } => {
            format!("origin[{index}] differs: image={left}, label={right}")
        }
        GeometryMismatch::Direction { index, left, right } => {
            format!("direction[{index}] differs: image={left}, label={right}")
        }
    }
}
