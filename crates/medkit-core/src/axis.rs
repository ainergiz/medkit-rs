use crate::{MedkitCoreError, Result};

/// Semantic role of an image axis.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AxisKind {
    /// Left/right patient-space axis.
    LeftRight,
    /// Posterior/anterior patient-space axis.
    PosteriorAnterior,
    /// Inferior/superior patient-space axis.
    InferiorSuperior,
    /// Time axis.
    Time,
    /// Channel or modality axis.
    Channel,
    /// Unknown or domain-specific axis.
    Other(String),
}

/// A named axis with a semantic role.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Axis {
    kind: AxisKind,
    label: String,
}

impl Axis {
    /// Creates an axis.
    pub fn new(kind: AxisKind, label: impl Into<String>) -> Result<Self> {
        let label = label.into();
        if label.is_empty() {
            return Err(MedkitCoreError::EmptyAxisLabel);
        }
        Ok(Self { kind, label })
    }

    /// Creates a left/right axis labeled `x`.
    pub fn x() -> Self {
        Self {
            kind: AxisKind::LeftRight,
            label: "x".to_string(),
        }
    }

    /// Creates a posterior/anterior axis labeled `y`.
    pub fn y() -> Self {
        Self {
            kind: AxisKind::PosteriorAnterior,
            label: "y".to_string(),
        }
    }

    /// Creates an inferior/superior axis labeled `z`.
    pub fn z() -> Self {
        Self {
            kind: AxisKind::InferiorSuperior,
            label: "z".to_string(),
        }
    }

    /// Creates a channel axis labeled `c`.
    pub fn channel() -> Self {
        Self {
            kind: AxisKind::Channel,
            label: "c".to_string(),
        }
    }

    /// Returns the axis kind.
    pub fn kind(&self) -> &AxisKind {
        &self.kind
    }

    /// Returns the axis label.
    pub fn label(&self) -> &str {
        &self.label
    }
}
