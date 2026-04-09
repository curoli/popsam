//! Core library for selecting semantically representative texts from a larger collection.
//!
//! The public API is organized around three concerns:
//! - input and result data models in [`model`]
//! - embedding providers in [`embedding`]
//! - the elimination algorithm in [`election`]

#![deny(missing_docs)]

/// Embedding provider traits and implementations.
pub mod embedding;
/// Election configuration and execution.
pub mod election;
/// Error types returned by the library.
pub mod error;
/// Serializable data models used by the public API.
pub mod model;

pub use embedding::{
    CandleEmbeddingModelFiles, CandleEmbeddingModelSpec, CandleEmbeddingProvider, EmbeddingProvider,
    OpenAiCompatibleEmbeddingProvider,
};
pub use election::{run_election, ElectionConfig};
pub use error::{PopsamError, PopsamResult};
pub use model::{
    CandidateRoundVotes, EmbeddedText, EmbeddedTextInput, ElectionResult, InputRecord, RoundSummary,
};
