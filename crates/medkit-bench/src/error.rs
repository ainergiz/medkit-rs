use std::fmt;

/// Result type used by `medkit-bench`.
pub type Result<T> = std::result::Result<T, BenchError>;

/// Errors produced by benchmark execution.
#[derive(Debug)]
pub enum BenchError {
    /// Sampler/cache read error.
    Sampler(medkit_sampler::SamplerError),
    /// Invalid benchmark input.
    InvalidInput {
        /// Human-readable reason.
        reason: String,
    },
}

impl BenchError {
    pub(crate) fn invalid_input(reason: impl Into<String>) -> Self {
        Self::InvalidInput {
            reason: reason.into(),
        }
    }
}

impl fmt::Display for BenchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sampler(error) => write!(f, "{error}"),
            Self::InvalidInput { reason } => write!(f, "invalid benchmark input: {reason}"),
        }
    }
}

impl std::error::Error for BenchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Sampler(error) => Some(error),
            Self::InvalidInput { .. } => None,
        }
    }
}

impl From<medkit_sampler::SamplerError> for BenchError {
    fn from(value: medkit_sampler::SamplerError) -> Self {
        Self::Sampler(value)
    }
}
