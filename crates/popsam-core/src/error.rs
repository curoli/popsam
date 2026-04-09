use thiserror::Error;

/// Convenience result type used throughout the crate.
pub type PopsamResult<T> = Result<T, PopsamError>;

/// Errors returned by the public `popsam-core` API.
#[derive(Debug, Error)]
pub enum PopsamError {
    /// The caller supplied no records.
    #[error("input data is empty")]
    EmptyInput,
    /// `report_last_k` must be at least `1`.
    #[error("report_last_k must be at least 1")]
    InvalidReportK,
    /// `elimination_fraction` must be within `(0, 1]`.
    #[error("elimination_fraction must be in the interval (0, 1]")]
    InvalidEliminationFraction,
    /// An embedding vector was empty.
    #[error("embedding vector for id '{id}' is empty")]
    EmptyEmbedding {
        /// The record identifier.
        id: String,
    },
    /// Embedding vectors in one election must all have the same dimension.
    #[error("embedding vector lengths differ: expected {expected}, got {actual} for id '{id}'")]
    DimensionMismatch {
        /// The record identifier.
        id: String,
        /// The expected embedding length.
        expected: usize,
        /// The embedding length that was actually provided.
        actual: usize,
    },
    /// An embedding vector had zero norm and cannot be normalized.
    #[error("embedding vector for id '{id}' has zero norm")]
    ZeroNorm {
        /// The record identifier.
        id: String,
    },
    /// The configured embedding provider returned an error.
    #[error("embedding provider error: {0}")]
    Provider(String),
    /// Loading a local model or tokenizer failed.
    #[error("model load error: {0}")]
    ModelLoad(String),
    /// File I/O failed.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON serialization or deserialization failed.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    /// CSV parsing failed.
    #[error("csv error: {0}")]
    Csv(#[from] csv::Error),
}
