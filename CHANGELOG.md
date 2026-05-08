# Changelog

## 0.1.1

- Added `candidate_best_results` to `ElectionResult` for Rust, CLI JSON output, and Python bindings.
- Added `full_round_index` to candidate best results to distinguish full-election rounds from reported suffix rounds.
- Ordered `candidate_best_results` like `all_ranked_ids`, from winner to first eliminated candidate.
- Added candidate best-result sections to CLI table, CSV, and Markdown output.

## 0.1.0

- Initial release with core election logic, CLI, local/OpenAI-compatible embeddings, and Python bindings.
