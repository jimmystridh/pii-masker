use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::debertav2::{
    Config as DebertaV2Config, DTYPE, DebertaV2NERModel, Id2Label,
};
use hf_hub::{Repo, RepoType, api::sync::Api};
use serde::Serialize;
use thiserror::Error;
use tokenizers::Tokenizer;

const MODEL_REPO_ID: &str = "hydroxai/pii_model_weight";
const MODEL_WEIGHTS_FILE: &str = "model.safetensors";
const CONFIG_JSON: &str = include_str!("../assets/deberta3base_1024/config.json");
const TOKENIZER_JSON: &[u8] = include_bytes!("../assets/deberta3base_1024/tokenizer.json");
const WEIGHTS_ENV_VAR: &str = "PII_MASKER_MODEL_WEIGHTS";
const MODEL_DIR_WEIGHTS_CANDIDATE: &str = "model/model.safetensors";

pub type Result<T> = std::result::Result<T, MaskerError>;

#[derive(Debug, Error)]
pub enum MaskerError {
    #[error("failed to parse model config: {0}")]
    Config(#[from] serde_json::Error),
    #[error("failed to read model weights: {0}")]
    Io(#[from] std::io::Error),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("model error: {0}")]
    Model(String),
    #[error("missing id2label in model config")]
    MissingId2Label,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PiiEntity {
    pub label: String,
    pub start: usize,
    pub end: usize,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct MaskResult {
    pub masked_text: String,
    pub pii: BTreeMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Default)]
pub struct PiiMaskerBuilder {
    weights_path: Option<PathBuf>,
}

impl PiiMaskerBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn weights_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.weights_path = Some(path.into());
        self
    }

    pub fn build(self) -> Result<PiiMasker> {
        let weights_path = match self.weights_path {
            Some(path) => path,
            None => default_weights_path()?,
        };

        PiiMasker::from_weights_path(weights_path)
    }
}

pub struct PiiMasker {
    tokenizer: Tokenizer,
    model: DebertaV2NERModel,
    id2label: Id2Label,
    device: Device,
    weights_path: PathBuf,
}

impl PiiMasker {
    pub fn builder() -> PiiMaskerBuilder {
        PiiMaskerBuilder::new()
    }

    pub fn new() -> Result<Self> {
        Self::builder().build()
    }

    pub fn from_weights_path(path: impl Into<PathBuf>) -> Result<Self> {
        let weights_path = path.into();
        let config: DebertaV2Config = serde_json::from_str(CONFIG_JSON)?;
        let id2label = config
            .id2label
            .clone()
            .ok_or(MaskerError::MissingId2Label)?;
        let tokenizer = Tokenizer::from_bytes(TOKENIZER_JSON)
            .map_err(|err| MaskerError::Tokenizer(err.to_string()))?;
        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&weights_path], DTYPE, &device)
                .map_err(|err| MaskerError::Model(err.to_string()))?
        };
        let vb = vb.set_prefix("deberta");
        let model = DebertaV2NERModel::load(vb, &config, Some(id2label.clone()))
            .map_err(|err| MaskerError::Model(err.to_string()))?;

        Ok(Self {
            tokenizer,
            model,
            id2label,
            device,
            weights_path,
        })
    }

    pub fn weights_path(&self) -> &Path {
        &self.weights_path
    }

    pub fn detect_pii(&self, input: &str) -> Result<Vec<PiiEntity>> {
        let encoding = self
            .tokenizer
            .encode(input, true)
            .map_err(|err| MaskerError::Tokenizer(err.to_string()))?;

        let input_ids = Tensor::stack(
            &[Tensor::new(encoding.get_ids(), &self.device)
                .map_err(|err| MaskerError::Model(err.to_string()))?],
            0,
        )
        .map_err(|err| MaskerError::Model(err.to_string()))?;
        let attention_mask = Tensor::stack(
            &[Tensor::new(encoding.get_attention_mask(), &self.device)
                .map_err(|err| MaskerError::Model(err.to_string()))?],
            0,
        )
        .map_err(|err| MaskerError::Model(err.to_string()))?;
        let token_type_ids = Tensor::stack(
            &[Tensor::new(encoding.get_type_ids(), &self.device)
                .map_err(|err| MaskerError::Model(err.to_string()))?],
            0,
        )
        .map_err(|err| MaskerError::Model(err.to_string()))?;

        let logits = self
            .model
            .forward(&input_ids, Some(token_type_ids), Some(attention_mask))
            .map_err(|err| MaskerError::Model(err.to_string()))?;
        let predictions = logits
            .argmax(2)
            .map_err(|err| MaskerError::Model(err.to_string()))?
            .to_vec2::<u32>()
            .map_err(|err| MaskerError::Model(err.to_string()))?;

        let labels = &predictions[0];
        let special_mask = encoding.get_special_tokens_mask();
        let offsets = encoding.get_offsets();

        let mut entities = Vec::new();
        let mut current: Option<(String, usize, usize)> = None;

        for (index, label_id) in labels.iter().enumerate() {
            if special_mask.get(index).copied().unwrap_or_default() == 1 {
                continue;
            }

            let Some(&(start, end)) = offsets.get(index) else {
                continue;
            };
            if start == end {
                continue;
            }

            let raw_label = self
                .id2label
                .get(label_id)
                .map(String::as_str)
                .unwrap_or("O");
            if raw_label == "O" {
                flush_entity(&mut entities, &mut current, input);
                continue;
            }

            let normalized_label = normalize_label(raw_label);
            let can_extend = current.as_ref().is_some_and(|(label, _, current_end)| {
                label == &normalized_label && start <= *current_end + 1
            });

            if can_extend {
                if let Some((_, _, current_end)) = current.as_mut() {
                    *current_end = end.max(*current_end);
                }
                continue;
            }

            flush_entity(&mut entities, &mut current, input);
            current = Some((normalized_label, start, end));
        }

        flush_entity(&mut entities, &mut current, input);
        Ok(entities)
    }

    pub fn mask(&self, input: &str) -> Result<MaskResult> {
        let (masked_text, pii) = self.mask_pii(input)?;
        Ok(MaskResult { masked_text, pii })
    }

    pub fn mask_pii(&self, input: &str) -> Result<(String, BTreeMap<String, Vec<String>>)> {
        let entities = self.detect_pii(input)?;
        let mut masked_text = String::with_capacity(input.len());
        let mut pii = BTreeMap::<String, Vec<String>>::new();
        let mut cursor = 0;

        for entity in &entities {
            masked_text.push_str(&input[cursor..entity.start]);
            masked_text.push('[');
            masked_text.push_str(&entity.label);
            masked_text.push(']');
            cursor = entity.end;

            let values = pii.entry(entity.label.clone()).or_default();
            if !values.iter().any(|value| value == &entity.text) {
                values.push(entity.text.clone());
            }
        }

        masked_text.push_str(&input[cursor..]);
        Ok((masked_text, pii))
    }
}

fn default_weights_path() -> Result<PathBuf> {
    if let Ok(path) = std::env::var(WEIGHTS_ENV_VAR) {
        return Ok(PathBuf::from(path));
    }

    let local_candidate =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(MODEL_DIR_WEIGHTS_CANDIDATE);
    if local_candidate.exists() {
        return Ok(local_candidate);
    }

    download_weights_from_hub()
}

fn download_weights_from_hub() -> Result<PathBuf> {
    let api = Api::new().map_err(|err| MaskerError::Model(err.to_string()))?;
    let repo = Repo::new(MODEL_REPO_ID.to_owned(), RepoType::Model);
    let api = api.repo(repo);
    api.get(MODEL_WEIGHTS_FILE)
        .map_err(|err| MaskerError::Model(err.to_string()))
}

fn normalize_label(label: &str) -> String {
    let cleaned = label
        .strip_prefix("B-")
        .or_else(|| label.strip_prefix("I-"))
        .unwrap_or(label);

    match cleaned {
        "ID_NUM" => "ID".to_string(),
        "NAME_STUDENT" => "NAME".to_string(),
        "PHONE_NUM" => "PHONE".to_string(),
        "STREET_ADDRESS" => "ADDRESS".to_string(),
        "URL_PERSONAL" => "URL".to_string(),
        other => other.to_string(),
    }
}

fn flush_entity(
    entities: &mut Vec<PiiEntity>,
    current: &mut Option<(String, usize, usize)>,
    input: &str,
) {
    let Some((label, start, end)) = current.take() else {
        return;
    };

    let (start, end) = trim_span(input, start, end);
    if start >= end {
        return;
    }

    entities.push(PiiEntity {
        label,
        start,
        end,
        text: input[start..end].to_string(),
    });
}

fn trim_span(input: &str, start: usize, end: usize) -> (usize, usize) {
    let segment = &input[start..end];
    let leading = segment.len() - segment.trim_start_matches(char::is_whitespace).len();
    let trailing = segment.len() - segment.trim_end_matches(char::is_whitespace).len();
    (start + leading, end - trailing)
}

#[cfg(test)]
mod tests {
    use super::{MODEL_DIR_WEIGHTS_CANDIDATE, PiiMaskerBuilder, normalize_label, trim_span};
    use std::path::PathBuf;

    const TEST_WEIGHTS_ENV_VAR: &str = "PII_MASKER_TEST_MODEL_WEIGHTS";

    #[test]
    fn normalizes_model_labels() {
        assert_eq!(normalize_label("B-NAME_STUDENT"), "NAME");
        assert_eq!(normalize_label("I-STREET_ADDRESS"), "ADDRESS");
        assert_eq!(normalize_label("B-EMAIL"), "EMAIL");
    }

    #[test]
    fn trims_surrounding_whitespace() {
        let input = " hello ";
        assert_eq!(trim_span(input, 0, input.len()), (1, 6));
    }

    #[test]
    fn masks_with_local_model_weights() {
        let Some(weights) = optional_test_weights() else {
            eprintln!("Skipping model-backed test because no test weights were configured.");
            return;
        };

        let masker = PiiMaskerBuilder::new()
            .weights_path(weights)
            .build()
            .expect("load local model");

        let result = masker
            .mask("John Doe lives at 1234 Elm St.")
            .expect("mask text");
        assert_eq!(result.masked_text, "John Doe lives at [ADDRESS].");
        assert_eq!(
            result.pii.get("ADDRESS").expect("address label"),
            &vec!["1234 Elm St".to_string()]
        );
    }

    fn optional_test_weights() -> Option<PathBuf> {
        if let Ok(path) = std::env::var(TEST_WEIGHTS_ENV_VAR) {
            let path = PathBuf::from(path);
            if path.exists() {
                return Some(path);
            }
        }

        let repo_local =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(MODEL_DIR_WEIGHTS_CANDIDATE);
        if repo_local.exists() {
            return Some(repo_local);
        }

        None
    }
}
