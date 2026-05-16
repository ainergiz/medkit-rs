use std::{fmt, io, path::PathBuf};

/// Result type used by `medkit-sampler`.
pub type Result<T> = std::result::Result<T, SamplerError>;

/// Errors produced while sampling cached training data.
#[derive(Debug)]
pub enum SamplerError {
    /// Filesystem operation failed.
    Io {
        /// Related path.
        path: PathBuf,
        /// Source IO error.
        source: io::Error,
    },
    /// Cache manifest read failed.
    Cache(medkit_cache::CacheError),
    /// JSON serialization failed.
    Json(serde_json::Error),
    /// Sampling input is invalid.
    InvalidInput {
        /// Human-readable reason.
        reason: String,
    },
}

impl SamplerError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: io::Error) -> Self {
        Self::Io {
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

impl fmt::Display for SamplerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(f, "filesystem error at {}: {source}", path.display())
            }
            Self::Cache(error) => write!(f, "{error}"),
            Self::Json(error) => write!(f, "failed to write sample JSONL: {error}"),
            Self::InvalidInput { reason } => write!(f, "invalid sampler input: {reason}"),
        }
    }
}

impl std::error::Error for SamplerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Cache(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::InvalidInput { .. } => None,
        }
    }
}

impl From<medkit_cache::CacheError> for SamplerError {
    fn from(value: medkit_cache::CacheError) -> Self {
        Self::Cache(value)
    }
}

impl From<serde_json::Error> for SamplerError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_impls_are_covered() {
        let _ = SamplerError::from(medkit_cache::CacheError::InvalidInput {
            reason: "bad cache".to_string(),
        });
        let source = serde_json::from_str::<serde_json::Value>("{").unwrap_err();
        let _ = SamplerError::from(source);
    }
}
