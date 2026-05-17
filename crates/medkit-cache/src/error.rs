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

#[cfg(test)]
mod tests {
    use std::{error::Error as _, io, path::PathBuf};

    use medkit_transform::TransformError;

    use super::*;

    #[test]
    fn display_messages_include_error_context() {
        let io_error = CacheError::io("cache/out.raw", io::Error::other("denied"));
        assert_eq!(
            io_error.to_string(),
            "filesystem error at cache/out.raw: denied"
        );

        let json_source = serde_json::from_str::<serde_json::Value>("{").unwrap_err();
        let json_error = CacheError::json("cache/cache_manifest.json", json_source);
        assert!(json_error
            .to_string()
            .contains("JSON error at cache/cache_manifest.json"));

        let transform_error = CacheError::from(TransformError::InvalidSize { size: [0, 8, 8] });
        assert_eq!(transform_error.to_string(), "invalid 3D size [0, 8, 8]");

        let invalid_input = CacheError::invalid_input("case a has no label path");
        assert_eq!(
            invalid_input.to_string(),
            "invalid cache input: case a has no label path"
        );

        let nifti_error = CacheError::nifti(PathBuf::from("image.nii.gz"), "bad header");
        assert_eq!(
            nifti_error.to_string(),
            "failed to load NIfTI pixels from image.nii.gz: bad header"
        );
    }

    #[test]
    fn source_returns_wrapped_errors_when_available() {
        let io_error = CacheError::io("cache", io::Error::new(io::ErrorKind::NotFound, "missing"));
        assert_eq!(io_error.source().unwrap().to_string(), "missing");

        let json_source = serde_json::from_str::<serde_json::Value>("{").unwrap_err();
        let json_error = CacheError::json("manifest.json", json_source);
        assert!(json_error.source().is_some());

        let transform_error = CacheError::from(TransformError::ShapeMismatch {
            image: [1, 2, 3],
            label: [1, 2, 4],
        });
        assert_eq!(
            transform_error.source().unwrap().to_string(),
            "image shape [1, 2, 3] does not match label shape [1, 2, 4]"
        );

        assert!(CacheError::invalid_input("bad input").source().is_none());
        assert!(CacheError::nifti("image.nii", "bad pixels")
            .source()
            .is_none());
    }
}
