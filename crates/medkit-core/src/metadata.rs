use crate::{MedkitCoreError, Result};

/// Medical image modality.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ImageModality {
    /// Computed tomography.
    CT,
    /// Magnetic resonance imaging.
    MR,
    /// Positron emission tomography.
    PET,
    /// Ultrasound.
    US,
    /// X-ray or projection radiography.
    XR,
    /// Pathology or microscopy image.
    Pathology,
    /// Segmentation, mask, or label map.
    Segmentation,
    /// Other modality.
    Other(String),
}

/// Kind of source backing an image.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SourceKind {
    /// DICOM file or series.
    Dicom,
    /// NIfTI file.
    Nifti,
    /// Whole-slide image.
    WholeSlide,
    /// Zarr hierarchy or array.
    Zarr,
    /// In-memory synthetic or derived data.
    Memory,
    /// Other source kind.
    Other(String),
}

/// Reference to the source of an image or derived object.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SourceRef {
    kind: SourceKind,
    uri: String,
}

impl SourceRef {
    /// Creates a source reference.
    pub fn new(kind: SourceKind, uri: impl Into<String>) -> Result<Self> {
        let uri = uri.into();
        if uri.is_empty() {
            return Err(MedkitCoreError::EmptySourceUri);
        }
        Ok(Self { kind, uri })
    }

    /// Returns the source kind.
    pub fn kind(&self) -> &SourceKind {
        &self.kind
    }

    /// Returns the source URI.
    pub fn uri(&self) -> &str {
        &self.uri
    }
}

/// Provenance trail for a source or derived image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provenance {
    source: SourceRef,
    operations: Vec<String>,
}

impl Provenance {
    /// Creates a provenance trail from a source.
    pub fn new(source: SourceRef) -> Self {
        Self {
            source,
            operations: Vec::new(),
        }
    }

    /// Adds an operation to the provenance trail.
    pub fn add_operation(&mut self, operation: impl Into<String>) -> Result<()> {
        let operation = operation.into();
        if operation.is_empty() {
            return Err(MedkitCoreError::EmptyProvenanceOperation);
        }
        self.operations.push(operation);
        Ok(())
    }

    /// Returns the source reference.
    pub fn source(&self) -> &SourceRef {
        &self.source
    }

    /// Returns the ordered operations applied after the source was acquired.
    pub fn operations(&self) -> &[String] {
        &self.operations
    }
}
