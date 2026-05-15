use std::{fmt, io, path::PathBuf};

/// Result type used by `medkit-cache`.
pub type Result<T> = std::result::Result<T, CacheError>;

/// Errors produced by cache preparation and cache reads.
#[derive(Debug)]
pub enum CacheError {
    /// Filesystem operation failed.
    Io {
        /// Related path.
        path: PathBuf,
        /// Source IO error.
        source: io::Error,
    },
    /// JSON serialization or parsing failed.
    Json {
        /// Related path.
        path: PathBuf,
        /// Source JSON error.
        source: serde_json::Error,
    },
    /// Transform plan parsing or execution failed.
    Transform(medkit_transform::TransformError),
    /// Cache input is invalid.
    InvalidInput {
        /// Human-readable reason.
        reason: String,
    },
    /// NIfTI pixel loading failed.
    Nifti {
        /// Related path.
        path: PathBuf,
        /// Human-readable reason.
        reason: String,
    },
}

impl CacheError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    pub(crate) fn json(path: impl Into<PathBuf>, source: serde_json::Error) -> Self {
        Self::Json {
            path: path.into(),
            source,
        }
    }

    pub(crate) fn invalid_input(reason: impl Into<String>) -> Self {
        Self::InvalidInput {
            reason: reason.into(),
        }
    }

    pub(crate) fn nifti(path: impl Into<PathBuf>, reason: impl Into<String>) -> Self {
        Self::Nifti {
            path: path.into(),
            reason: reason.into(),
        }
    }
}

impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(f, "filesystem error at {}: {source}", path.display())
            }
            Self::Json { path, source } => {
                write!(f, "JSON error at {}: {source}", path.display())
            }
            Self::Transform(error) => write!(f, "{error}"),
            Self::InvalidInput { reason } => write!(f, "invalid cache input: {reason}"),
            Self::Nifti { path, reason } => {
                write!(
                    f,
                    "failed to load NIfTI pixels from {}: {reason}",
                    path.display()
                )
            }
        }
    }
}

impl std::error::Error for CacheError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json { source, .. } => Some(source),
            Self::Transform(error) => Some(error),
            Self::InvalidInput { .. } | Self::Nifti { .. } => None,
        }
    }
}

impl From<medkit_transform::TransformError> for CacheError {
    fn from(value: medkit_transform::TransformError) -> Self {
        Self::Transform(value)
    }
}
