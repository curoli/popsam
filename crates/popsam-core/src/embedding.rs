use crate::error::{PopsamError, PopsamResult};
use crate::model::{EmbeddedText, InputRecord};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE as BERT_DTYPE};
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

/// Trait implemented by embedding backends that can turn raw texts into vectors.
pub trait EmbeddingProvider {
    /// Embeds a batch of records and returns the corresponding vectors.
    fn embed(&self, records: &[InputRecord]) -> PopsamResult<Vec<EmbeddedText>>;
}

/// Embedding provider for OpenAI-compatible `/embeddings` HTTP APIs.
#[derive(Debug, Clone)]
pub struct OpenAiCompatibleEmbeddingProvider {
    client: Client,
    base_url: String,
    api_key: String,
    model: String,
}

impl OpenAiCompatibleEmbeddingProvider {
    /// Creates a new OpenAI-compatible embedding provider.
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
            api_key: api_key.into(),
            model: model.into(),
        }
    }
}

impl EmbeddingProvider for OpenAiCompatibleEmbeddingProvider {
    fn embed(&self, records: &[InputRecord]) -> PopsamResult<Vec<EmbeddedText>> {
        let inputs: Vec<String> = records
            .iter()
            .map(|record| record.text.clone().unwrap_or_default())
            .collect();

        let response: EmbeddingResponse = self
            .client
            .post(format!("{}/embeddings", self.base_url.trim_end_matches('/')))
            .bearer_auth(&self.api_key)
            .json(&EmbeddingRequest {
                model: self.model.clone(),
                input: inputs,
            })
            .send()
            .map_err(|err| PopsamError::Provider(err.to_string()))?
            .error_for_status()
            .map_err(|err| PopsamError::Provider(err.to_string()))?
            .json()
            .map_err(|err| PopsamError::Provider(err.to_string()))?;

        let mut by_index = response.data;
        by_index.sort_by_key(|item| item.index);

        if by_index.len() != records.len() {
            return Err(PopsamError::Provider(format!(
                "embedding API returned {} vectors for {} inputs",
                by_index.len(),
                records.len()
            )));
        }

        Ok(records
            .iter()
            .zip(by_index)
            .map(|(record, item)| EmbeddedText {
                id: record.id.clone(),
                text: record.text.clone(),
                embedding: item.embedding,
            })
            .collect())
    }
}

/// Local embedding provider backed by Candle and a BERT-style sentence model.
pub struct CandleEmbeddingProvider {
    tokenizer: Tokenizer,
    model: BertModel,
    device: Device,
    max_length: usize,
}

/// Local file paths needed to load a Candle embedding model.
#[derive(Debug, Clone)]
pub struct CandleEmbeddingModelFiles {
    /// Path to the Hugging Face `config.json`.
    pub config: PathBuf,
    /// Path to the tokenizer JSON file.
    pub tokenizer: PathBuf,
    /// Path to the model weights in `safetensors` format.
    pub weights: PathBuf,
}

/// Model specification used to download a sentence embedding model from Hugging Face.
#[derive(Debug, Clone)]
pub struct CandleEmbeddingModelSpec {
    /// Hugging Face model ID.
    pub model_id: String,
    /// Revision, branch, or tag to resolve.
    pub revision: String,
    /// Config filename inside the repository.
    pub config_filename: String,
    /// Tokenizer filename inside the repository.
    pub tokenizer_filename: String,
    /// Weights filename inside the repository.
    pub weights_filename: String,
}

impl CandleEmbeddingModelSpec {
    /// Returns the default multilingual sentence-transformer model specification.
    pub fn multilingual_default() -> Self {
        Self {
            model_id: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2".to_string(),
            revision: "main".to_string(),
            config_filename: "config.json".to_string(),
            tokenizer_filename: "tokenizer.json".to_string(),
            weights_filename: "model.safetensors".to_string(),
        }
    }
}

impl CandleEmbeddingProvider {
    /// Loads a local Candle embedding provider from explicit model files.
    pub fn from_local_files(
        files: &CandleEmbeddingModelFiles,
        device: Device,
        max_length: usize,
    ) -> PopsamResult<Self> {
        let mut tokenizer = Tokenizer::from_file(&files.tokenizer)
            .map_err(|err| PopsamError::ModelLoad(err.to_string()))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length,
                ..Default::default()
            }))
            .map_err(|err| PopsamError::ModelLoad(err.to_string()))?;

        let config_text = std::fs::read_to_string(&files.config)?;
        let config: BertConfig =
            serde_json::from_str(&config_text).map_err(|err| PopsamError::ModelLoad(err.to_string()))?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[files.weights.clone()], BERT_DTYPE, &device)
                .map_err(|err| PopsamError::ModelLoad(err.to_string()))?
        };
        let model = BertModel::load(vb, &config).map_err(|err| PopsamError::ModelLoad(err.to_string()))?;

        Ok(Self {
            tokenizer,
            model,
            device,
            max_length,
        })
    }

    /// Downloads the configured model files from Hugging Face and loads them locally.
    pub fn from_hf_hub(spec: &CandleEmbeddingModelSpec, device: Device, max_length: usize) -> PopsamResult<Self> {
        let api = Api::new().map_err(|err| PopsamError::ModelLoad(err.to_string()))?;
        let repo = api.repo(Repo::with_revision(
            spec.model_id.clone(),
            RepoType::Model,
            spec.revision.clone(),
        ));
        let files = CandleEmbeddingModelFiles {
            config: repo
                .get(&spec.config_filename)
                .map_err(|err| PopsamError::ModelLoad(err.to_string()))?,
            tokenizer: repo
                .get(&spec.tokenizer_filename)
                .map_err(|err| PopsamError::ModelLoad(err.to_string()))?,
            weights: repo
                .get(&spec.weights_filename)
                .map_err(|err| PopsamError::ModelLoad(err.to_string()))?,
        };
        Self::from_local_files(&files, device, max_length)
    }

    /// Convenience constructor for the default CPU-backed local model.
    ///
    /// The current implementation always resolves to the multilingual default model.
    pub fn cpu(multilingual_default: bool) -> PopsamResult<Self> {
        let spec = if multilingual_default {
            CandleEmbeddingModelSpec::multilingual_default()
        } else {
            CandleEmbeddingModelSpec::multilingual_default()
        };
        Self::from_hf_hub(&spec, Device::Cpu, 512)
    }
}

impl EmbeddingProvider for CandleEmbeddingProvider {
    fn embed(&self, records: &[InputRecord]) -> PopsamResult<Vec<EmbeddedText>> {
        if records.is_empty() {
            return Ok(Vec::new());
        }

        let texts = records
            .iter()
            .map(|record| record.text.clone().unwrap_or_default())
            .collect::<Vec<_>>();
        let encodings = self
            .tokenizer
            .encode_batch(texts, true)
            .map_err(|err| PopsamError::Provider(err.to_string()))?;

        let max_seq_len = encodings
            .iter()
            .map(|encoding| encoding.len())
            .max()
            .unwrap_or(0)
            .min(self.max_length);

        let mut input_ids = Vec::with_capacity(records.len() * max_seq_len);
        let mut attention_mask = Vec::with_capacity(records.len() * max_seq_len);
        let token_type_ids = vec![0_u32; records.len() * max_seq_len];

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let pad_len = max_seq_len.saturating_sub(ids.len());

            input_ids.extend_from_slice(ids);
            input_ids.extend(std::iter::repeat_n(0_u32, pad_len));

            attention_mask.extend_from_slice(mask);
            attention_mask.extend(std::iter::repeat_n(0_u32, pad_len));
        }

        let input_ids = Tensor::new(input_ids.as_slice(), &self.device)
            .map_err(|err| PopsamError::Provider(err.to_string()))?
            .reshape((records.len(), max_seq_len))
            .map_err(|err| PopsamError::Provider(err.to_string()))?;
        let attention_mask = Tensor::new(attention_mask.as_slice(), &self.device)
            .map_err(|err| PopsamError::Provider(err.to_string()))?
            .reshape((records.len(), max_seq_len))
            .map_err(|err| PopsamError::Provider(err.to_string()))?;
        let token_type_ids = Tensor::new(token_type_ids.as_slice(), &self.device)
            .map_err(|err| PopsamError::Provider(err.to_string()))?
            .reshape((records.len(), max_seq_len))
            .map_err(|err| PopsamError::Provider(err.to_string()))?;

        let hidden = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(|err| PopsamError::Provider(err.to_string()))?;
        let pooled = mean_pool(&hidden, &attention_mask)?;
        let embeddings = pooled
            .to_dtype(DType::F32)
            .map_err(|err| PopsamError::Provider(err.to_string()))?
            .to_vec2::<f32>()
            .map_err(|err| PopsamError::Provider(err.to_string()))?;

        Ok(records
            .iter()
            .zip(embeddings)
            .map(|(record, embedding)| EmbeddedText {
                id: record.id.clone(),
                text: record.text.clone(),
                embedding,
            })
            .collect())
    }
}

fn mean_pool(hidden: &Tensor, attention_mask: &Tensor) -> PopsamResult<Tensor> {
    let mask = attention_mask
        .to_dtype(DType::F32)
        .map_err(|err| PopsamError::Provider(err.to_string()))?
        .unsqueeze(2)
        .map_err(|err| PopsamError::Provider(err.to_string()))?;
    let masked_hidden = hidden
        .broadcast_mul(&mask)
        .map_err(|err| PopsamError::Provider(err.to_string()))?;
    let sum_hidden = masked_hidden
        .sum(1)
        .map_err(|err| PopsamError::Provider(err.to_string()))?;
    let sum_mask = mask
        .sum(1)
        .map_err(|err| PopsamError::Provider(err.to_string()))?
        .broadcast_maximum(&Tensor::new(&[1e-9_f32], hidden.device()).map_err(|err| PopsamError::Provider(err.to_string()))?)
        .map_err(|err| PopsamError::Provider(err.to_string()))?;
    sum_hidden
        .broadcast_div(&sum_mask)
        .map_err(|err| PopsamError::Provider(err.to_string()))
}

#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    index: usize,
    embedding: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multilingual_default_points_to_sentence_transformers_model() {
        let spec = CandleEmbeddingModelSpec::multilingual_default();
        assert_eq!(
            spec.model_id,
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        );
        assert_eq!(spec.weights_filename, "model.safetensors");
    }
}
