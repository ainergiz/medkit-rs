use std::path::PathBuf;

#[derive(Debug)]
pub enum DicomError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    InvalidInput(String),
    Parse {
        path: PathBuf,
        message: String,
    },
    Unsupported {
        path: PathBuf,
        message: String,
    },
}

impl DicomError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    pub(crate) fn parse(path: impl Into<PathBuf>, message: impl Into<String>) -> Self {
        Self::Parse {
            path: path.into(),
            message: message.into(),
        }
    }

    pub(crate) fn unsupported(path: impl Into<PathBuf>, message: impl Into<String>) -> Self {
        Self::Unsupported {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for DicomError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Self::InvalidInput(message) => write!(f, "{message}"),
            Self::Parse { path, message } => write!(f, "{}: {message}", path.display()),
            Self::Unsupported { path, message } => write!(f, "{}: {message}", path.display()),
        }
    }
}

impl std::error::Error for DicomError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::InvalidInput(_) | Self::Parse { .. } | Self::Unsupported { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;

    #[test]
    fn display_and_sources_cover_all_error_variants() {
        let io = DicomError::io(
            "input.dcm",
            std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied"),
        );
        assert_eq!(io.to_string(), "input.dcm: denied");
        assert!(io.source().is_some());

        let invalid = DicomError::InvalidInput("bad width".to_string());
        assert_eq!(invalid.to_string(), "bad width");
        assert!(invalid.source().is_none());

        let parse = DicomError::parse("broken.dcm", "bad tag");
        assert_eq!(parse.to_string(), "broken.dcm: bad tag");
        assert!(parse.source().is_none());

        let unsupported = DicomError::unsupported("compressed.dcm", "jpeg");
        assert_eq!(unsupported.to_string(), "compressed.dcm: jpeg");
        assert!(unsupported.source().is_none());
    }
}
