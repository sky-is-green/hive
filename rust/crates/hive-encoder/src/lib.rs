//! Native encoder experiments: load the cached paraphrase-MiniLM-L3-v2 ONNX
//! model and produce sentence embeddings without Python or torch.

use std::path::{Path, PathBuf};

use tokenizers::Encoding;
use tokenizers::tokenizer::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

const DEFAULT_MODEL: &str = "sentence-transformers/paraphrase-MiniLM-L3-v2";
const MAX_SEQ: usize = 128;

pub struct Encoder {
    session: ort::session::Session,
    tokenizer: Tokenizer,
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub dimension: usize,
}

impl Encoder {
    pub fn load() -> Result<Self, String> {
        let (model_path, tokenizer_path) = find_model()?;
        Self::load_from(&model_path, &tokenizer_path)
    }

    pub fn load_from(model_path: &Path, tokenizer_path: &Path) -> Result<Self, String> {
        let mut tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|error| format!("cannot load {}: {error}", tokenizer_path.display()))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_SEQ,
                ..Default::default()
            }))
            .map_err(|error| format!("cannot configure truncation: {error}"))?;
        let env = ort::init()
            .with_name("hive-encoder")
            .build()
            .map_err(|error| format!("cannot create ONNX environment: {error}"))?;
        let threads = std::env::var("HIVE_ENCODER_THREADS")
            .ok()
            .and_then(|text| text.parse::<usize>().ok())
            .filter(|count| *count > 0)
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|count| count.get())
                    .unwrap_or(1)
            });
        let session = ort::session::Session::builder(&env)
            .map_err(|error| format!("cannot create ONNX session: {error}"))?
            .with_intra_threads(threads)
            .map_err(|error| format!("cannot configure ONNX threads: {error}"))?
            .commit_from_file(model_path)
            .map_err(|error| format!("cannot load {}: {error}", model_path.display()))?;
        let dimension = session
            .outputs()
            .iter()
            .find(|output| output.name() == "sentence_embedding")
            .and_then(|output| output.dtype().tensor_shape())
            .and_then(|shape| shape.last().copied())
            .map(|dim| dim as usize)
            .filter(|dim| *dim > 0)
            .unwrap_or(384);
        Ok(Self {
            session,
            tokenizer,
            model_path: model_path.to_path_buf(),
            tokenizer_path: tokenizer_path.to_path_buf(),
            dimension,
        })
    }

    pub fn encode(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|error| format!("tokenization failed: {error}"))?;
        let batch = encodings.len();
        let sequence = encodings
            .iter()
            .map(|encoding| encoding.get_ids().len())
            .max()
            .unwrap_or(1)
            .max(1);
        let mut ids = Vec::with_capacity(batch * sequence);
        let mut mask = Vec::with_capacity(batch * sequence);
        let mut types = Vec::with_capacity(batch * sequence);
        for encoding in &encodings {
            append_encoding(encoding, sequence, &mut ids, &mut mask, &mut types);
        }
        let input_ids = ort::value::Tensor::from_array(([batch as i64, sequence as i64], ids))
            .map_err(|error| format!("input_ids tensor: {error}"))?;
        let attention_mask =
            ort::value::Tensor::from_array(([batch as i64, sequence as i64], mask.clone()))
                .map_err(|error| format!("attention_mask tensor: {error}"))?;
        let token_type_ids =
            ort::value::Tensor::from_array(([batch as i64, sequence as i64], types))
                .map_err(|error| format!("token_type_ids tensor: {error}"))?;
        let outputs = self
            .session
            .run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
                "token_type_ids" => token_type_ids,
            ])
            .map_err(|error| format!("inference failed: {error}"))?;
        if let Some(value) = outputs.get("sentence_embedding") {
            let (shape, data) = value
                .try_extract_tensor::<f32>()
                .map_err(|error| format!("sentence_embedding: {error}"))?;
            let dim = *shape.last().unwrap_or(&0) as usize;
            if dim == 0 {
                return Err("sentence_embedding has no feature dimension".into());
            }
            return Ok(data.chunks_exact(dim).map(<[f32]>::to_vec).collect());
        }
        let value = outputs
            .get("last_hidden_state")
            .or_else(|| outputs.get("token_embeddings"))
            .ok_or_else(|| "model exposes no sentence_embedding or last_hidden_state".to_owned())?;
        let (shape, data) = value
            .try_extract_tensor::<f32>()
            .map_err(|error| format!("last_hidden_state: {error}"))?;
        let dims: Vec<usize> = shape.iter().map(|dim| *dim as usize).collect();
        if dims.len() != 3 {
            return Err(format!("expected [batch, seq, dim], got {dims:?}"));
        }
        let (batch_dim, sequence_dim, feature_dim) = (dims[0], dims[1], dims[2]);
        let mut out = Vec::with_capacity(batch_dim);
        for row in 0..batch_dim {
            let mut vector = vec![0.0f32; feature_dim];
            let mut count = 0.0f32;
            for token in 0..sequence_dim {
                let offset = (row * sequence_dim + token) * feature_dim;
                if mask[row * sequence_dim + token] == 0 {
                    continue;
                }
                count += 1.0;
                for feature in 0..feature_dim {
                    vector[feature] += data[offset + feature];
                }
            }
            if count > 0.0 {
                for value in &mut vector {
                    *value /= count;
                }
            }
            out.push(vector);
        }
        Ok(out)
    }
}

fn append_encoding(
    encoding: &Encoding,
    sequence: usize,
    ids: &mut Vec<i64>,
    mask: &mut Vec<i64>,
    types: &mut Vec<i64>,
) {
    let length = encoding.get_ids().len().min(sequence);
    ids.extend(encoding.get_ids()[..length].iter().map(|id| *id as i64));
    mask.extend(
        encoding.get_attention_mask()[..length]
            .iter()
            .map(|value| *value as i64),
    );
    types.extend(
        encoding.get_type_ids()[..length]
            .iter()
            .map(|id| *id as i64),
    );
    let padding = sequence.saturating_sub(length);
    ids.extend(std::iter::repeat_n(0i64, padding));
    mask.extend(std::iter::repeat_n(0i64, padding));
    types.extend(std::iter::repeat_n(0i64, padding));
}

pub fn default_model_name() -> &'static str {
    DEFAULT_MODEL
}

fn find_model() -> Result<(PathBuf, PathBuf), String> {
    if let Some(raw) = std::env::var_os("HIVE_ENCODER_DIR") {
        let path = hive_core::paths::expand(Path::new(&raw));
        if let Some(found) = model_in(&path) {
            return Ok(found);
        }
        return Err(format!(
            "HIVE_ENCODER_DIR points at {}, which has no ONNX model/tokenizer",
            path.display()
        ));
    }
    let paths = hive_core::paths::Paths::discover(None, None);
    if let Some(found) = model_in(&paths.repo.join("models/encoder")) {
        return Ok(found);
    }
    let cache = std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| hive_core::paths::expand(Path::new("~/.cache/huggingface")));
    let hub = cache.join("hub");
    let model_dir = hub.join("models--sentence-transformers--paraphrase-MiniLM-L3-v2");
    let snapshots = model_dir.join("snapshots");
    if let Ok(entries) = std::fs::read_dir(&snapshots) {
        for entry in entries.flatten() {
            if let Some(found) = model_in(&entry.path()) {
                return Ok(found);
            }
        }
    }
    Err(format!(
        "no cached {DEFAULT_MODEL}; set HIVE_ENCODER_DIR to a folder with onnx/model.onnx and tokenizer.json"
    ))
}

fn model_in(root: &Path) -> Option<(PathBuf, PathBuf)> {
    let model = [root.join("onnx/model.onnx"), root.join("model.onnx")]
        .into_iter()
        .find(|path| path.is_file())?;
    let tokenizer = root.join("tokenizer.json");
    tokenizer.is_file().then_some((model, tokenizer))
}
