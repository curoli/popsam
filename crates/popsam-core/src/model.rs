use serde::{Deserialize, Serialize};

/// Raw input text identified by a caller-provided ID.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputRecord {
    /// Stable caller-provided identifier for the record.
    pub id: String,
    /// Optional original text content.
    #[serde(default)]
    pub text: Option<String>,
}

/// Input record with a caller-provided embedding vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedTextInput {
    /// Stable caller-provided identifier for the record.
    pub id: String,
    /// Optional original text content.
    #[serde(default)]
    pub text: Option<String>,
    /// Embedding vector for the record.
    pub embedding: Vec<f32>,
}

/// Record returned by an embedding provider or included in an election result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedText {
    /// Stable caller-provided identifier for the record.
    pub id: String,
    /// Optional original text content.
    #[serde(default)]
    pub text: Option<String>,
    /// Normalized embedding vector for the record.
    pub embedding: Vec<f32>,
}

/// Vote totals for a candidate in a single round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateRoundVotes {
    /// Candidate identifier.
    pub id: String,
    /// Number of first-preference votes.
    pub first_votes: u32,
    /// Number of second-preference votes.
    pub second_votes: u32,
    /// Number of third-preference votes.
    pub third_votes: u32,
}

/// Summary of a reported round among the final `k` candidates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundSummary {
    /// One-based round index within the reported suffix of the election.
    pub round_index: usize,
    /// Number of candidates still active at the start of the round.
    pub active_candidates: usize,
    /// Candidate IDs eliminated at the end of the round.
    pub eliminated_candidate_ids: Vec<String>,
    /// Vote totals for all active candidates in that round.
    pub votes: Vec<CandidateRoundVotes>,
}

/// Result of running the elimination algorithm on a collection of embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectionResult {
    /// ID of the final surviving candidate.
    pub winner_id: String,
    /// IDs of the last `k` candidates, captured when reporting begins.
    pub representative_ids: Vec<String>,
    /// Full ranking from winner to first eliminated candidate.
    pub all_ranked_ids: Vec<String>,
    /// Reported round summaries for the final `k` rounds.
    pub rounds: Vec<RoundSummary>,
    /// Normalized embeddings associated with the processed records.
    pub embeddings: Vec<EmbeddedText>,
}
