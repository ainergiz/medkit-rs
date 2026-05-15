use std::{fmt, io, path::PathBuf};

/// Result type used by `medkit-dataset`.
pub type Result<T> = std::result::Result<T, DatasetError>;

/// Errors produced while scanning, validating, or writing dataset artifacts.
#[derive(Debug)]
pub enum DatasetError {
    /// A filesystem operation failed.
    Io {
        /// Path associated with the failed operation.
        path: PathBuf,
        /// Source IO error.
        source: io::Error,
    },
    /// Manifest JSON serialization failed.
    Json {
        /// Path associated with the JSON operation.
        path: PathBuf,
        /// Source JSON error.
        source: serde_json::Error,
    },
    /// Dataset root or configured input directories are invalid.
    InvalidInput {
        /// Human-readable reason.
        reason: String,
    },
}

impl DatasetError {
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
}

impl fmt::Display for DatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(f, "filesystem error at {}: {source}", path.display())
            }
            Self::Json { path, source } => {
                write!(
                    f,
                    "failed to write JSON manifest {}: {source}",
                    path.display()
                )
            }
            Self::InvalidInput { reason } => write!(f, "invalid dataset input: {reason}"),
        }
    }
}

impl std::error::Error for DatasetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json { source, .. } => Some(source),
            Self::InvalidInput { .. } => None,
        }
    }
}
