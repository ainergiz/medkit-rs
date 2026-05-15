use std::{fmt, io, path::PathBuf};

/// Result type used by `medkit-io`.
pub type Result<T> = std::result::Result<T, MedkitIoError>;

/// Errors produced by medical image metadata readers.
#[derive(Debug)]
pub enum MedkitIoError {
    /// The requested format is not supported by the selected reader.
    UnsupportedFormat {
        /// Path that could not be interpreted.
        path: PathBuf,
    },
    /// An IO operation failed.
    Io {
        /// Path being read.
        path: PathBuf,
        /// IO source error.
        source: io::Error,
    },
    /// Header bytes are structurally invalid.
    InvalidHeader {
        /// Human-readable reason.
        reason: String,
    },
    /// The NIfTI datatype is valid but not represented by `medkit-core` yet.
    UnsupportedDatatype {
        /// Raw NIfTI datatype code.
        code: i16,
    },
    /// A lower-level core type rejected converted metadata.
    Core(medkit_core::MedkitCoreError),
}

impl MedkitIoError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    pub(crate) fn invalid_header(reason: impl Into<String>) -> Self {
        Self::InvalidHeader {
            reason: reason.into(),
        }
    }
}

impl fmt::Display for MedkitIoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedFormat { path } => {
                write!(f, "unsupported image metadata format: {}", path.display())
            }
            Self::Io { path, source } => write!(f, "failed to read {}: {source}", path.display()),
            Self::InvalidHeader { reason } => write!(f, "invalid image header: {reason}"),
            Self::UnsupportedDatatype { code } => {
                write!(f, "unsupported NIfTI datatype code {code}")
            }
            Self::Core(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for MedkitIoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Core(error) => Some(error),
            Self::UnsupportedFormat { .. }
            | Self::InvalidHeader { .. }
            | Self::UnsupportedDatatype { .. } => None,
        }
    }
}

impl From<medkit_core::MedkitCoreError> for MedkitIoError {
    fn from(value: medkit_core::MedkitCoreError) -> Self {
        Self::Core(value)
    }
}
