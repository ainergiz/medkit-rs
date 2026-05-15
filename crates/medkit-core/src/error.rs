use std::fmt;

/// Result type used by `medkit-core`.
pub type Result<T> = std::result::Result<T, MedkitCoreError>;

/// Errors produced by core type construction and validation.
#[derive(Debug, Clone, PartialEq)]
pub enum MedkitCoreError {
    /// A shape was empty.
    EmptyShape,
    /// A dimension was zero.
    ZeroDimension {
        /// Dimension index.
        index: usize,
    },
    /// An axis label was empty.
    EmptyAxisLabel,
    /// Axis count did not match image rank.
    AxisRankMismatch {
        /// Number of provided axes.
        axes: usize,
        /// Expected image rank.
        rank: usize,
    },
    /// Spacing count did not match image rank.
    SpacingRankMismatch {
        /// Number of spacing values.
        spacing: usize,
        /// Expected image rank.
        rank: usize,
    },
    /// Origin count did not match image rank.
    OriginRankMismatch {
        /// Number of origin values.
        origin: usize,
        /// Expected image rank.
        rank: usize,
    },
    /// Direction matrix had an invalid number of values.
    DirectionSizeMismatch {
        /// Number of direction values.
        values: usize,
        /// Expected number of direction values.
        expected: usize,
    },
    /// A spacing value was not positive and finite.
    InvalidSpacing {
        /// Spacing index.
        index: usize,
        /// Invalid spacing value.
        value: f64,
    },
    /// A geometry tolerance was negative or not finite.
    InvalidTolerance {
        /// Tolerance field name.
        field: &'static str,
        /// Invalid tolerance value.
        value: f64,
    },
    /// A required image id was empty.
    EmptyImageId,
    /// A required source URI was empty.
    EmptySourceUri,
    /// A required provenance operation was empty.
    EmptyProvenanceOperation,
}

impl fmt::Display for MedkitCoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyShape => write!(f, "shape must contain at least one dimension"),
            Self::ZeroDimension { index } => write!(f, "shape dimension {index} must be non-zero"),
            Self::EmptyAxisLabel => write!(f, "axis label must not be empty"),
            Self::AxisRankMismatch { axes, rank } => {
                write!(f, "axis count {axes} does not match rank {rank}")
            }
            Self::SpacingRankMismatch { spacing, rank } => {
                write!(f, "spacing count {spacing} does not match rank {rank}")
            }
            Self::OriginRankMismatch { origin, rank } => {
                write!(f, "origin count {origin} does not match rank {rank}")
            }
            Self::DirectionSizeMismatch { values, expected } => {
                write!(
                    f,
                    "direction matrix has {values} values, expected {expected}"
                )
            }
            Self::InvalidSpacing { index, value } => {
                write!(
                    f,
                    "spacing value {value} at index {index} must be finite and positive"
                )
            }
            Self::InvalidTolerance { field, value } => {
                write!(
                    f,
                    "tolerance {field}={value} must be finite and non-negative"
                )
            }
            Self::EmptyImageId => write!(f, "image id must not be empty"),
            Self::EmptySourceUri => write!(f, "source URI must not be empty"),
            Self::EmptyProvenanceOperation => {
                write!(f, "provenance operation must not be empty")
            }
        }
    }
}

impl std::error::Error for MedkitCoreError {}
