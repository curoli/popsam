use crate::error::{PopsamError, PopsamResult};
use crate::model::{CandidateRoundVotes, EmbeddedText, EmbeddedTextInput, ElectionResult, RoundSummary};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::cmp::Ordering;

/// Configuration for the representative-text election.
#[derive(Debug, Clone)]
pub struct ElectionConfig {
    /// Number of final candidates to keep and report.
    pub report_last_k: usize,
    /// Fraction of active candidates to eliminate in early rounds.
    pub elimination_fraction: f32,
    /// Seed used for all random tie-breaks.
    pub random_seed: u64,
}

impl Default for ElectionConfig {
    fn default() -> Self {
        Self {
            report_last_k: 10,
            elimination_fraction: 0.05,
            random_seed: 42,
        }
    }
}

/// Runs the elimination process on a collection of embedded texts.
///
/// The function normalizes all embeddings, repeatedly assigns first/second/third
/// preference votes by cosine similarity, and eliminates the weakest candidates
/// until a single winner remains.
///
/// The returned [`ElectionResult`] includes:
/// - the final winner
/// - the last `k` surviving candidates
/// - round-by-round vote totals for the reported suffix
/// - normalized embeddings for all processed inputs
pub fn run_election(
    inputs: Vec<EmbeddedTextInput>,
    config: ElectionConfig,
) -> PopsamResult<ElectionResult> {
    if inputs.is_empty() {
        return Err(PopsamError::EmptyInput);
    }
    if config.report_last_k == 0 {
        return Err(PopsamError::InvalidReportK);
    }
    if !(0.0 < config.elimination_fraction && config.elimination_fraction <= 1.0) {
        return Err(PopsamError::InvalidEliminationFraction);
    }

    let dimension = inputs[0].embedding.len();
    let mut embeddings = Vec::with_capacity(inputs.len());
    for input in inputs {
        if input.embedding.is_empty() {
            return Err(PopsamError::EmptyEmbedding { id: input.id });
        }
        if input.embedding.len() != dimension {
            return Err(PopsamError::DimensionMismatch {
                id: input.id,
                expected: dimension,
                actual: input.embedding.len(),
            });
        }
        let normalized = normalize(&input.id, input.embedding)?;
        embeddings.push(EmbeddedText {
            id: input.id,
            text: input.text,
            embedding: normalized,
        });
    }

    let report_k = config.report_last_k.min(embeddings.len());
    let mut rng = ChaCha8Rng::seed_from_u64(config.random_seed);
    let mut active: Vec<usize> = (0..embeddings.len()).collect();
    let mut eliminated_ranked = Vec::with_capacity(embeddings.len().saturating_sub(1));
    let mut rounds = Vec::new();
    let mut representative_ids = Vec::new();

    while active.len() > 1 {
        let vote_rows = tally_votes(&embeddings, &active, &mut rng);
        let eliminate_count = elimination_count(active.len(), report_k, config.elimination_fraction);
        let weakest = weakest_candidates(&vote_rows, eliminate_count, &mut rng);

        if active.len() <= report_k {
            if representative_ids.is_empty() {
                representative_ids = active
                    .iter()
                    .map(|&idx| embeddings[idx].id.clone())
                    .collect();
            }
            rounds.push(build_round_summary(&embeddings, &active, &vote_rows, &weakest, rounds.len() + 1));
        }

        let mut removed_flags = vec![false; embeddings.len()];
        for &idx in &weakest {
            removed_flags[idx] = true;
            eliminated_ranked.push(embeddings[idx].id.clone());
        }
        active.retain(|idx| !removed_flags[*idx]);
    }

    let winner_idx = active[0];
    let winner_id = embeddings[winner_idx].id.clone();
    if representative_ids.is_empty() {
        representative_ids.push(winner_id.clone());
    }
    if report_k > 0 {
        let final_votes = tally_votes(&embeddings, &active, &mut rng);
        rounds.push(build_round_summary(
            &embeddings,
            &active,
            &final_votes,
            &[],
            rounds.len() + 1,
        ));
    }
    let mut all_ranked_ids = vec![winner_id.clone()];
    eliminated_ranked.reverse();
    all_ranked_ids.extend(eliminated_ranked);

    Ok(ElectionResult {
        winner_id,
        representative_ids,
        all_ranked_ids,
        rounds,
        embeddings,
    })
}

fn normalize(id: &str, embedding: Vec<f32>) -> PopsamResult<Vec<f32>> {
    let norm = embedding
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if norm == 0.0 {
        return Err(PopsamError::ZeroNorm {
            id: id.to_string(),
        });
    }
    Ok(embedding.into_iter().map(|value| value / norm).collect())
}

fn elimination_count(active: usize, report_k: usize, elimination_fraction: f32) -> usize {
    if active <= report_k {
        return 1;
    }
    let max_allowed = active - report_k;
    let batch = ((active as f32) * elimination_fraction).ceil() as usize;
    batch.max(1).min(max_allowed)
}

fn tally_votes(
    embeddings: &[EmbeddedText],
    active: &[usize],
    rng: &mut ChaCha8Rng,
) -> Vec<(usize, CandidateRoundVotes)> {
    let mut candidate_positions = vec![None; embeddings.len()];
    for (position, &candidate_idx) in active.iter().enumerate() {
        candidate_positions[candidate_idx] = Some(position);
    }

    let mut tallies: Vec<CandidateRoundVotes> = active
        .iter()
        .map(|&idx| CandidateRoundVotes {
            id: embeddings[idx].id.clone(),
            first_votes: 0,
            second_votes: 0,
            third_votes: 0,
        })
        .collect();

    for voter in embeddings {
        let mut ranked = active
            .iter()
            .map(|&candidate_idx| {
                (
                    candidate_idx,
                    cosine_similarity(&voter.embedding, &embeddings[candidate_idx].embedding),
                    rng.random::<u64>(),
                )
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| compare_similarity(right, left));

        for (rank, (candidate_idx, _, _)) in ranked.into_iter().take(3).enumerate() {
            let tally_index = candidate_positions[candidate_idx].expect("candidate must exist");
            match rank {
                0 => tallies[tally_index].first_votes += 1,
                1 => tallies[tally_index].second_votes += 1,
                2 => tallies[tally_index].third_votes += 1,
                _ => {}
            }
        }
    }

    active.iter().copied().zip(tallies).collect()
}

fn weakest_candidates(
    vote_rows: &[(usize, CandidateRoundVotes)],
    eliminate_count: usize,
    rng: &mut ChaCha8Rng,
) -> Vec<usize> {
    let mut ranked = vote_rows
        .iter()
        .map(|(idx, votes)| (*idx, votes.clone(), rng.random::<u64>()))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        left.1
            .first_votes
            .cmp(&right.1.first_votes)
            .then(left.1.second_votes.cmp(&right.1.second_votes))
            .then(left.1.third_votes.cmp(&right.1.third_votes))
            .then(left.2.cmp(&right.2))
    });
    ranked
        .into_iter()
        .take(eliminate_count)
        .map(|entry| entry.0)
        .collect()
}

fn build_round_summary(
    embeddings: &[EmbeddedText],
    active: &[usize],
    vote_rows: &[(usize, CandidateRoundVotes)],
    eliminated: &[usize],
    round_index: usize,
) -> RoundSummary {
    let mut votes = vote_rows.iter().map(|(_, votes)| votes.clone()).collect::<Vec<_>>();
    votes.sort_by(|left, right| {
        right
            .first_votes
            .cmp(&left.first_votes)
            .then(right.second_votes.cmp(&left.second_votes))
            .then(right.third_votes.cmp(&left.third_votes))
            .then(left.id.cmp(&right.id))
    });

    RoundSummary {
        round_index,
        active_candidates: active.len(),
        eliminated_candidate_ids: eliminated
            .iter()
            .map(|&idx| embeddings[idx].id.clone())
            .collect(),
        votes,
    }
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
}

fn compare_similarity(
    left: &(usize, f32, u64),
    right: &(usize, f32, u64),
) -> Ordering {
    left.1
        .total_cmp(&right.1)
        .then_with(|| left.2.cmp(&right.2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_last_k_rounds_and_winner() {
        let inputs = vec![
            EmbeddedTextInput { id: "a".into(), text: None, embedding: vec![1.0, 0.0] },
            EmbeddedTextInput { id: "b".into(), text: None, embedding: vec![0.9, 0.1] },
            EmbeddedTextInput { id: "c".into(), text: None, embedding: vec![0.8, 0.2] },
            EmbeddedTextInput { id: "d".into(), text: None, embedding: vec![0.0, 1.0] },
        ];

        let result = run_election(
            inputs,
            ElectionConfig {
                report_last_k: 3,
                elimination_fraction: 0.25,
                random_seed: 7,
            },
        )
        .expect("election should succeed");

        assert_eq!(result.representative_ids.len(), 3);
        assert_eq!(result.rounds.len(), 3);
        assert_eq!(result.rounds[0].active_candidates, 3);
        assert_eq!(result.rounds[2].active_candidates, 1);
        assert_eq!(result.winner_id, result.all_ranked_ids[0]);
    }

    #[test]
    fn rejects_dimension_mismatch() {
        let inputs = vec![
            EmbeddedTextInput { id: "a".into(), text: None, embedding: vec![1.0, 0.0] },
            EmbeddedTextInput { id: "b".into(), text: None, embedding: vec![1.0] },
        ];

        let error = run_election(inputs, ElectionConfig::default()).expect_err("must fail");
        assert!(matches!(error, PopsamError::DimensionMismatch { .. }));
    }
}
