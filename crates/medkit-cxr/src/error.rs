#[derive(Debug)]
pub enum CxrError {
    Io(std::io::Error),
    Csv(csv::Error),
    Json(serde_json::Error),
    Image(image::ImageError),
    Toml(toml::de::Error),
    Message(String),
}

impl std::fmt::Display for CxrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::Csv(error) => write!(f, "{error}"),
            Self::Json(error) => write!(f, "{error}"),
            Self::Image(error) => write!(f, "{error}"),
            Self::Toml(error) => write!(f, "{error}"),
            Self::Message(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for CxrError {}

impl From<std::io::Error> for CxrError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<csv::Error> for CxrError {
    fn from(value: csv::Error) -> Self {
        Self::Csv(value)
    }
}

impl From<serde_json::Error> for CxrError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl From<image::ImageError> for CxrError {
    fn from(value: image::ImageError) -> Self {
        Self::Image(value)
    }
}

impl From<toml::de::Error> for CxrError {
    fn from(value: toml::de::Error) -> Self {
        Self::Toml(value)
    }
}
