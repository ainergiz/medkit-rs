use std::fmt;

/// Result type used by `medkit-transform`.
pub type Result<T> = std::result::Result<T, TransformError>;

/// Errors produced by transform planning and execution.
#[derive(Debug, Clone, PartialEq)]
pub enum TransformError {
    /// TOML plan parsing failed.
    PlanParse {
        /// Parser message.
        message: String,
    },
    /// JSON plan serialization failed.
    PlanSerialize {
        /// Serialization message.
        message: String,
    },
    /// Volume dimensions and data length disagree.
    InvalidVolume {
        /// Shape that was provided.
        shape: [usize; 3],
        /// Number of elements that were provided.
        len: usize,
    },
    /// Patch or crop size is invalid.
    InvalidSize {
        /// Invalid size.
        size: [usize; 3],
    },
    /// Image and label volumes do not share a shape.
    ShapeMismatch {
        /// Image shape.
        image: [usize; 3],
        /// Label shape.
        label: [usize; 3],
    },
    /// Volume and geometry shapes do not agree.
    GeometryShapeMismatch {
        /// Volume shape.
        volume: [usize; 3],
        /// Geometry shape.
        geometry: [usize; 3],
    },
    /// Spacing is invalid.
    InvalidSpacing {
        /// Invalid spacing.
        spacing: [f64; 3],
    },
    /// Physical origin is invalid.
    InvalidOrigin {
        /// Invalid origin.
        origin: [f64; 3],
    },
    /// Direction matrix is invalid or singular.
    InvalidDirection {
        /// Direction determinant.
        determinant: f64,
    },
    /// Label resampling policy can corrupt discrete segmentation labels.
    InvalidLabelInterpolation {
        /// Human-readable reason.
        reason: String,
    },
}

impl fmt::Display for TransformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PlanParse { message } => write!(f, "failed to parse transform plan: {message}"),
            Self::PlanSerialize { message } => {
                write!(f, "failed to serialize transform plan: {message}")
            }
            Self::InvalidVolume { shape, len } => {
                write!(f, "invalid volume shape {shape:?} for {len} values")
            }
            Self::InvalidSize { size } => write!(f, "invalid 3D size {size:?}"),
            Self::ShapeMismatch { image, label } => {
                write!(
                    f,
                    "image shape {image:?} does not match label shape {label:?}"
                )
            }
            Self::GeometryShapeMismatch { volume, geometry } => {
                write!(
                    f,
                    "volume shape {volume:?} does not match geometry shape {geometry:?}"
                )
            }
            Self::InvalidSpacing { spacing } => write!(f, "invalid spacing {spacing:?}"),
            Self::InvalidOrigin { origin } => write!(f, "invalid origin {origin:?}"),
            Self::InvalidDirection { determinant } => {
                write!(f, "invalid direction matrix with determinant {determinant}")
            }
            Self::InvalidLabelInterpolation { reason } => {
                write!(f, "invalid label interpolation: {reason}")
            }
        }
    }
}

impl std::error::Error for TransformError {}
