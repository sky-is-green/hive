use std::io;

/// One error type for the CLI core. The CLI prints these on stderr and picks
/// the exit code; library crates never exit the process.
#[derive(Debug, thiserror::Error)]
pub enum HiveError {
    #[error("{0}")]
    Io(#[from] io::Error),
    #[error("{0}")]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    NotFound(String),
}

pub type Result<T> = std::result::Result<T, HiveError>;

impl HiveError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::NotFound(message.into())
    }
}
