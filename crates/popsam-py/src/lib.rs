use candle_core::Device;
use popsam_core::{
    run_election, CandleEmbeddingModelFiles, CandleEmbeddingModelSpec, CandleEmbeddingProvider,
    CandidateRoundVotes, ElectionConfig, ElectionResult, EmbeddedText, EmbeddedTextInput,
    EmbeddingProvider, InputRecord, OpenAiCompatibleEmbeddingProvider, RoundSummary,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[pyfunction]
#[pyo3(signature = (records, report_last_k=10, elimination_fraction=0.05, seed=42))]
fn elect(
    py: Python<'_>,
    records: Vec<PyEmbeddedTextInput>,
    report_last_k: usize,
    elimination_fraction: f32,
    seed: u64,
) -> PyResult<Py<PyDict>> {
    let inputs = records
        .into_iter()
        .map(|record| EmbeddedTextInput {
            id: record.id,
            text: record.text,
            embedding: record.embedding,
        })
        .collect();

    let result = run_election(
        inputs,
        ElectionConfig {
            report_last_k,
            elimination_fraction,
            random_seed: seed,
        },
    )
    .map_err(|err| PyValueError::new_err(err.to_string()))?;

    election_result_to_py(py, result)
}

#[pyfunction]
#[pyo3(signature = (
    records,
    report_last_k=10,
    elimination_fraction=0.05,
    seed=42,
    model_id=None,
    revision="main".to_string(),
    max_length=512,
    config_file=None,
    tokenizer_file=None,
    weights_file=None
))]
fn elect_texts_local(
    py: Python<'_>,
    records: Vec<PyInputRecord>,
    report_last_k: usize,
    elimination_fraction: f32,
    seed: u64,
    model_id: Option<String>,
    revision: String,
    max_length: usize,
    config_file: Option<String>,
    tokenizer_file: Option<String>,
    weights_file: Option<String>,
) -> PyResult<Py<PyDict>> {
    let inputs = records
        .into_iter()
        .map(|record| InputRecord {
            id: record.id,
            text: record.text,
        })
        .collect::<Vec<_>>();

    let provider = if let (Some(config), Some(tokenizer), Some(weights)) =
        (config_file, tokenizer_file, weights_file)
    {
        CandleEmbeddingProvider::from_local_files(
            &CandleEmbeddingModelFiles {
                config: config.into(),
                tokenizer: tokenizer.into(),
                weights: weights.into(),
            },
            Device::Cpu,
            max_length,
        )
    } else {
        let mut spec = CandleEmbeddingModelSpec::multilingual_default();
        if let Some(model_id) = model_id {
            spec.model_id = model_id;
        }
        spec.revision = revision;
        CandleEmbeddingProvider::from_hf_hub(&spec, Device::Cpu, max_length)
    }
    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

    elect_from_provider(py, inputs, &provider, report_last_k, elimination_fraction, seed)
}

#[pyfunction]
#[pyo3(signature = (
    records,
    api_key,
    report_last_k=10,
    elimination_fraction=0.05,
    seed=42,
    model="text-embedding-3-small".to_string(),
    base_url="https://api.openai.com/v1".to_string()
))]
fn elect_texts_openai(
    py: Python<'_>,
    records: Vec<PyInputRecord>,
    api_key: String,
    report_last_k: usize,
    elimination_fraction: f32,
    seed: u64,
    model: String,
    base_url: String,
) -> PyResult<Py<PyDict>> {
    let inputs = records
        .into_iter()
        .map(|record| InputRecord {
            id: record.id,
            text: record.text,
        })
        .collect::<Vec<_>>();

    let provider = OpenAiCompatibleEmbeddingProvider::new(base_url, api_key, model);
    elect_from_provider(py, inputs, &provider, report_last_k, elimination_fraction, seed)
}

fn elect_from_provider<P: EmbeddingProvider>(
    py: Python<'_>,
    inputs: Vec<InputRecord>,
    provider: &P,
    report_last_k: usize,
    elimination_fraction: f32,
    seed: u64,
) -> PyResult<Py<PyDict>> {
    let embedded = provider
        .embed(&inputs)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

    let result = run_election(
        embedded
            .into_iter()
            .map(|item| EmbeddedTextInput {
                id: item.id,
                text: item.text,
                embedding: item.embedding,
            })
            .collect(),
        ElectionConfig {
            report_last_k,
            elimination_fraction,
            random_seed: seed,
        },
    )
    .map_err(|err| PyValueError::new_err(err.to_string()))?;

    election_result_to_py(py, result)
}

fn election_result_to_py(py: Python<'_>, result: ElectionResult) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("winner_id", result.winner_id)?;
    dict.set_item("representative_ids", result.representative_ids)?;
    dict.set_item("all_ranked_ids", result.all_ranked_ids)?;

    let rounds = PyList::empty(py);
    for round in result.rounds {
        rounds.append(round_summary_to_py(py, round)?)?;
    }
    dict.set_item("rounds", rounds)?;

    let embeddings = PyList::empty(py);
    for embedding in result.embeddings {
        embeddings.append(embedded_text_to_py(py, embedding)?)?;
    }
    dict.set_item("embeddings", embeddings)?;

    Ok(dict.unbind())
}

fn round_summary_to_py(py: Python<'_>, round: RoundSummary) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("round_index", round.round_index)?;
    dict.set_item("active_candidates", round.active_candidates)?;
    dict.set_item("eliminated_candidate_ids", round.eliminated_candidate_ids)?;

    let votes = PyList::empty(py);
    for vote in round.votes {
        votes.append(candidate_round_votes_to_py(py, vote)?)?;
    }
    dict.set_item("votes", votes)?;

    Ok(dict.unbind())
}

fn candidate_round_votes_to_py(
    py: Python<'_>,
    vote: CandidateRoundVotes,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", vote.id)?;
    dict.set_item("first_votes", vote.first_votes)?;
    dict.set_item("second_votes", vote.second_votes)?;
    dict.set_item("third_votes", vote.third_votes)?;
    Ok(dict.unbind())
}

fn embedded_text_to_py(py: Python<'_>, item: EmbeddedText) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", item.id)?;
    dict.set_item("text", item.text)?;
    dict.set_item("embedding", item.embedding)?;
    Ok(dict.unbind())
}

#[pyclass]
#[derive(Clone)]
struct PyEmbeddedTextInput {
    #[pyo3(get, set)]
    id: String,
    #[pyo3(get, set)]
    text: Option<String>,
    #[pyo3(get, set)]
    embedding: Vec<f32>,
}

#[pyclass]
#[derive(Clone)]
struct PyInputRecord {
    #[pyo3(get, set)]
    id: String,
    #[pyo3(get, set)]
    text: Option<String>,
}

#[pymethods]
impl PyEmbeddedTextInput {
    #[new]
    #[pyo3(signature = (id, embedding, text=None))]
    fn new(id: String, embedding: Vec<f32>, text: Option<String>) -> Self {
        Self { id, text, embedding }
    }
}

#[pymethods]
impl PyInputRecord {
    #[new]
    #[pyo3(signature = (id, text=None))]
    fn new(id: String, text: Option<String>) -> Self {
        Self { id, text }
    }
}

#[pymodule]
fn popsam(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEmbeddedTextInput>()?;
    m.add_class::<PyInputRecord>()?;
    m.add_function(wrap_pyfunction!(elect, m)?)?;
    m.add_function(wrap_pyfunction!(elect_texts_local, m)?)?;
    m.add_function(wrap_pyfunction!(elect_texts_openai, m)?)?;
    Ok(())
}
