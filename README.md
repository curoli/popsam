# popsam

`popsam` uses AI to select a small set of texts from a much larger collection so that the selected texts cover the semantic space of the full collection as well as possible.

Here, "sample" is not meant in the classical statistical sense, where one item stands in for many identical items. Instead, a selected text represents many other texts with similar meaning. To do this, `popsam` computes AI-based embeddings and compares texts not only by wording, but by semantic proximity. This is useful when you want to understand a large volume of feedback, comments, or support tickets without having to read everything. Instead of going through thousands of similar phrasings one by one, you get a small number of concrete texts that expose the main patterns in the data.

A simple example: suppose 20,000 customer comments are mainly about delivery delays, strong product quality, complicated returns, and helpful support. In that case, `popsam` should not just pick any 20 comments at random. It should find a small set in which those thematic patterns are actually visible. Each selected text therefore stands not for identical wording, but for many semantically similar responses.

The workspace contains three components:

- `popsam-core`: Rust library with data models, the embedding provider API, and the election algorithm
- `popsam-cli`: command-line application for file inputs and JSON output
- `popsam-py`: Python bindings via `pyo3` and `maturin`

## Election Algorithm

Each text is both a voter and a candidate. In each round:

1. Each voter gives a first, second, and third vote to the most similar remaining candidates.
2. Similarity is computed as cosine similarity on normalized embeddings.
3. The weakest candidate is eliminated based on first, second, and third votes.
4. Exact ties are resolved with a reproducible random seed.

To reduce very large candidate sets faster, `popsam` can eliminate multiple candidates at once in early rounds. Once only `k` candidates remain, the algorithm switches to single-candidate elimination so that the final `k` rounds can be reported in full.

## Status

The first version implements:

- the core logic for selection and round-by-round evaluation
- a local Candle embedding provider with a multilingual MiniLM default
- CLI import for precomputed embeddings in JSONL and CSV
- a CLI workflow for raw text using either local or OpenAI-compatible embedding generation
- an OpenAI-compatible embedding provider for API-based usage
- Python bindings for the core logic

## CLI

Select from precomputed embeddings:

```bash
cargo run -p popsam-cli -- elect --input feedback.jsonl --format jsonl --pretty
```

Embed raw texts locally and run the selection:

```bash
cargo run -p popsam-cli -- embed-elect --input feedback.txt --format plain --pretty
```

Compact table output:

```bash
cargo run -p popsam-cli -- embed-elect --input feedback.txt --format plain --output table
```

With the final representative texts shown first:

```bash
cargo run -p popsam-cli -- embed-elect --input feedback.txt --format plain --output table --include-text
```

CSV output:

```bash
cargo run -p popsam-cli -- embed-elect --input feedback.txt --format plain --output csv
```

Markdown output:

```bash
cargo run -p popsam-cli -- embed-elect --input feedback.txt --format plain --output markdown
```

Embed raw texts via an API and run the selection:

```bash
OPENAI_API_KEY=... cargo run -p popsam-cli -- embed-elect --backend openai --input feedback.csv --format csv
```

## Python

Precomputed embeddings:

```python
import popsam

records = [
    popsam.PyEmbeddedTextInput("a", [1.0, 0.0], "good"),
    popsam.PyEmbeddedTextInput("b", [0.9, 0.1], "also good"),
]
result = popsam.elect(records, report_last_k=2)
print(result["winner_id"])
```

Raw texts with local Candle embeddings:

```python
import popsam

records = [
    popsam.PyInputRecord("1", "The product is fast and robust."),
    popsam.PyInputRecord("2", "Delivery was slow."),
]
result = popsam.elect_texts_local(records, report_last_k=2)
print(result["representative_ids"])
```
