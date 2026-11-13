use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use hive_encoder::Encoder;
use hive_memory::{ContextStore, apply_decay, cosine_scores};
use serde::Deserialize;

#[derive(Debug, Parser)]
#[command(
    name = "hive-memory",
    version,
    about = "Native retention-store experiments."
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Repair legacy-codec mojibake
    Codec {
        #[arg(long, allow_hyphen_values = true)]
        text: String,
        #[arg(long)]
        json: bool,
    },
    /// Run the write-side hygiene pipeline
    Prepare {
        #[arg(long, allow_hyphen_values = true)]
        text: String,
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        sanitize: bool,
        #[arg(long)]
        json: bool,
    },
    /// Replay add/turn operations and print the resulting store
    Replay {
        #[arg(long)]
        file: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Deduplicate chunks with supplied embeddings
    Dedup {
        #[arg(long)]
        chunks: PathBuf,
        #[arg(long)]
        embeddings: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Assemble context for one query
    Assemble {
        #[arg(long)]
        chunks: PathBuf,
        #[arg(long)]
        query: String,
        #[arg(long, default_value_t = 0)]
        current_turn: u32,
        #[arg(long, default_value = "ultra_small")]
        route: String,
        #[arg(long, default_value_t = 8192)]
        max_context: u32,
        #[arg(long)]
        json: bool,
    },
    /// Run the remembrance pass over chunks
    Remembrance {
        #[arg(long)]
        chunks: PathBuf,
        #[arg(long)]
        topic: String,
        #[arg(long)]
        json: bool,
    },
    /// Measure topic drift between recent and historical chunks
    Drift {
        #[arg(long)]
        chunks: PathBuf,
        #[arg(long, default_value_t = 3)]
        recent: usize,
        #[arg(long, default_value_t = 0.6)]
        threshold: f64,
        #[arg(long)]
        json: bool,
    },
    /// Score and decay chunks for one query
    Retrieve {
        #[arg(long)]
        chunks: PathBuf,
        #[arg(long)]
        query: String,
        #[arg(long, default_value_t = 10)]
        top_k: usize,
        #[arg(long, default_value_t = 0)]
        current_turn: u32,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Deserialize)]
struct Operation {
    op: String,
    #[serde(default)]
    turn: u32,
    #[serde(default)]
    content: String,
    #[serde(default)]
    chunk_id: Option<String>,
    #[serde(default)]
    n: usize,
}

fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("hive-memory: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Args) -> Result<(), String> {
    match args.command {
        Command::Codec {
            text,
            json: as_json,
        } => {
            let repaired = hive_memory::codec::normalize_codec(&text);
            if as_json {
                print!(
                    "{}",
                    hive_core::json::pretty(&serde_json::json!({"text": repaired}))
                );
            } else {
                println!("{repaired}");
            }
            Ok(())
        }
        Command::Prepare {
            text,
            sanitize,
            json: as_json,
        } => {
            let prefixes: Vec<String> = hive_memory::hygiene::default_ingest_block_prefixes()
                .iter()
                .map(|prefix| (*prefix).to_owned())
                .collect();
            let prepared = hive_memory::hygiene::prepare_for_storage(
                &text,
                hive_memory::hygiene::DEFAULT_MAX_CHUNK_CHARS,
                &prefixes,
                sanitize,
            );
            if as_json {
                print!(
                    "{}",
                    hive_core::json::pretty(&serde_json::json!({"text": prepared}))
                );
            } else {
                println!("{}", prepared.unwrap_or_default());
            }
            Ok(())
        }
        Command::Replay {
            file,
            json: as_json,
        } => {
            let raw = std::fs::read_to_string(&file)
                .map_err(|error| format!("cannot read {}: {error}", file.display()))?;
            let operations: Vec<Operation> =
                serde_json::from_str(&raw).map_err(|error| format!("invalid ops: {error}"))?;
            let mut store = ContextStore::new(None);
            for operation in operations {
                match operation.op.as_str() {
                    "add" => {
                        store.add_chunk(
                            operation.turn,
                            &operation.content,
                            operation.chunk_id.as_deref(),
                        );
                    }
                    "set_max" => store.max_chunks = Some(operation.n),
                    "sanitize" => store.sanitize = operation.n != 0,
                    "block" => store.ingest_block_prefixes.push(operation.content),
                    other => return Err(format!("unknown op {other:?}")),
                }
            }
            let payload = store.to_json();
            if as_json {
                print!("{}", hive_core::json::pretty(&payload));
            } else {
                println!("{} chunks", store.count());
            }
            Ok(())
        }
        Command::Dedup {
            chunks,
            embeddings,
            json: as_json,
        } => {
            let rows = read_chunks(&chunks)?;
            let embedding_rows: Vec<serde_json::Value> = serde_json::from_str(
                &std::fs::read_to_string(&embeddings)
                    .map_err(|error| format!("cannot read {}: {error}", embeddings.display()))?,
            )
            .map_err(|error| format!("invalid embeddings: {error}"))?;
            let mut by_id = BTreeMap::new();
            for row in embedding_rows {
                if let (Some(id), Some(vector)) = (
                    row.get("id").and_then(|value| value.as_str()),
                    row.get("vector").and_then(|value| value.as_array()),
                ) {
                    by_id.insert(
                        id.to_owned(),
                        vector
                            .iter()
                            .filter_map(|value| value.as_f64().map(|v| v as f32))
                            .collect::<Vec<_>>(),
                    );
                }
            }
            let refs: Vec<&hive_memory::Chunk> = rows.iter().collect();
            let embeddings: Vec<Vec<f32>> = refs
                .iter()
                .map(|chunk| by_id.get(&chunk.id).cloned().unwrap_or_default())
                .collect();
            let vocab = vocab()?;
            let dedup = hive_memory::dedup::ContextDeduplicator::new(&vocab, 0.92);
            let (survivors, refresh) = dedup.deduplicate(&refs, &embeddings);
            let payload = serde_json::json!({
                "survivors": survivors.iter().map(|index| refs[*index].id.clone()).collect::<Vec<_>>(),
                "refresh": refresh,
            });
            if as_json {
                print!("{}", hive_core::json::pretty(&payload));
            } else {
                println!("{} survivors", survivors.len());
            }
            Ok(())
        }
        Command::Assemble {
            chunks,
            query,
            current_turn,
            route,
            max_context,
            json: as_json,
        } => {
            let rows = read_chunks(&chunks)?;
            let store = hive_memory::ContextStore::from_chunks(rows);
            let mut encoder = Encoder::load()?;
            let vocab = vocab()?;
            let options = hive_memory::assemble::AssembleOptions {
                route,
                current_turn,
                max_context,
                ..Default::default()
            };
            let assembled =
                hive_memory::assemble::assemble(&store, &query, &mut encoder, &vocab, &options)?;
            let payload = serde_json::json!({
                "content": assembled.content,
                "token_count": assembled.token_count,
                "budget": assembled.budget,
                "selected_ids": assembled.selected_ids,
                "drift_detected": assembled.drift_detected,
                "drift_score": assembled.drift_score,
                "raw_scores": assembled.raw_scores,
                "effective_scores": assembled.effective_scores,
            });
            if as_json {
                print!("{}", hive_core::json::pretty(&payload));
            } else {
                println!(
                    "{} tokens · {} chunks",
                    assembled.token_count,
                    assembled.selected_ids.len()
                );
            }
            Ok(())
        }
        Command::Remembrance {
            chunks,
            topic,
            json: as_json,
        } => {
            let mut rows = read_chunks(&chunks)?;
            let mut encoder = Encoder::load()?;
            let texts: Vec<String> = rows.iter().map(|chunk| chunk.content.clone()).collect();
            let mut all = vec![topic.clone()];
            all.extend(texts);
            let vectors = encoder.encode(&all)?;
            let scores = hive_memory::cosine_scores(&vectors[0], &vectors[1..]);
            let by_content: BTreeMap<String, f64> = rows
                .iter()
                .zip(scores)
                .map(|(chunk, score)| (chunk.content.clone(), score))
                .collect();
            let vocab = vocab()?;
            let pass = hive_memory::remembrance::RemembrancePass::new(&vocab);
            let results = pass.process(&mut rows, &topic, |text| {
                by_content.get(text).copied().unwrap_or(0.0)
            });
            let payload = serde_json::json!({
                "results": results.iter().map(|result| serde_json::json!({
                    "chunk_id": result.chunk_id,
                    "saved": result.saved,
                    "relevance_score": result.relevance_score,
                    "compressed_content": result.compressed_content,
                    "new_decay": result.new_decay,
                })).collect::<Vec<_>>(),
            });
            if as_json {
                print!("{}", hive_core::json::pretty(&payload));
            } else {
                println!("{} reminders", results.iter().filter(|r| r.saved).count());
            }
            Ok(())
        }
        Command::Drift {
            chunks,
            recent,
            threshold,
            json: as_json,
        } => {
            let rows = read_chunks(&chunks)?;
            let refs: Vec<&hive_memory::Chunk> = rows.iter().collect();
            let recent_refs: Vec<&hive_memory::Chunk> = refs
                .iter()
                .rev()
                .take(recent)
                .copied()
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            let mut encoder = Encoder::load()?;
            let texts: Vec<String> = rows.iter().map(|chunk| chunk.content.clone()).collect();
            let mut recent_text = recent_refs
                .iter()
                .map(|chunk| chunk.content.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            let historical_text = rows[..rows.len() / 2]
                .iter()
                .map(|chunk| chunk.content.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            let mut all_texts = vec![recent_text.clone(), historical_text.clone()];
            all_texts.extend(texts);
            let vectors = encoder.encode(&all_texts)?;
            recent_text.clear();
            let detector = hive_memory::drift::TopicDriftDetector { threshold };
            let fixed = [vectors[0].clone()];
            let historical = &vectors[1];
            let _ = fixed;
            let similarity = hive_memory::cosine(&vectors[0], historical);
            let score = 1.0 - similarity;
            let (score, reset) = (score, score > detector.threshold);
            let payload = serde_json::json!({"drift_score": score, "should_reset": reset});
            if as_json {
                print!("{}", hive_core::json::pretty(&payload));
            } else {
                println!("drift {score:.4} reset={reset}");
            }
            Ok(())
        }
        Command::Retrieve {
            chunks,
            query,
            top_k,
            current_turn,
            json: as_json,
        } => {
            let raw = std::fs::read_to_string(&chunks)
                .map_err(|error| format!("cannot read {}: {error}", chunks.display()))?;
            let rows: Vec<serde_json::Value> =
                serde_json::from_str(&raw).map_err(|error| format!("invalid chunks: {error}"))?;
            let mut encoder = Encoder::load()?;
            let contents: Vec<String> = rows
                .iter()
                .filter_map(|row| row.get("content").and_then(|value| value.as_str()))
                .map(str::to_owned)
                .collect();
            let mut texts = vec![query.clone()];
            texts.extend(contents);
            let vectors = encoder.encode(&texts)?;
            let scores = cosine_scores(&vectors[0], &vectors[1..]);
            let chunks: Vec<hive_memory::Chunk> = rows
                .iter()
                .zip(scores)
                .map(|(row, raw)| {
                    let mut chunk: hive_memory::Chunk = serde_json::from_value(row.clone())
                        .unwrap_or(hive_memory::Chunk {
                            id: row
                                .get("id")
                                .and_then(|value| value.as_str())
                                .unwrap_or("")
                                .to_owned(),
                            content: row
                                .get("content")
                                .and_then(|value| value.as_str())
                                .unwrap_or("")
                                .to_owned(),
                            turn: row
                                .get("turn")
                                .and_then(|value| value.as_u64())
                                .unwrap_or(0) as u32,
                            fingerprint: String::new(),
                            timestamp: String::new(),
                            decay_multiplier: 1.0,
                            times_saved: 0,
                            last_referenced_turn: None,
                            relevance_history: Vec::new(),
                        });
                    chunk.relevance_history.push((current_turn, raw));
                    chunk
                })
                .collect();
            let raw_scores: BTreeMap<String, f64> = chunks
                .iter()
                .map(|chunk| {
                    (
                        chunk.id.clone(),
                        chunk
                            .relevance_history
                            .last()
                            .map(|(_, score)| *score)
                            .unwrap_or(0.0),
                    )
                })
                .collect();
            let refs: Vec<&hive_memory::Chunk> = chunks.iter().collect();
            let decayed = apply_decay(
                &refs,
                current_turn,
                &raw_scores,
                &BTreeMap::new(),
                &[],
                None,
                None,
            );
            let mut rows: Vec<(String, f64, f64)> = chunks
                .iter()
                .map(|chunk| {
                    (
                        chunk.id.clone(),
                        raw_scores.get(&chunk.id).copied().unwrap_or(0.0),
                        decayed.get(&chunk.id).copied().unwrap_or(0.0),
                    )
                })
                .collect();
            rows.sort_by(|left, right| right.2.total_cmp(&left.2));
            rows.truncate(top_k.max(1));
            if as_json {
                let payload = serde_json::json!({
                    "query": query,
                    "dimension": encoder.dimension,
                    "results": rows.iter().map(|(id, raw, effective)| {
                        serde_json::json!({"id": id, "raw": raw, "effective": effective})
                    }).collect::<Vec<_>>(),
                });
                print!("{}", hive_core::json::pretty(&payload));
            } else {
                for (id, raw, effective) in rows {
                    println!("{id}  raw={raw:.4} effective={effective:.4}");
                }
            }
            Ok(())
        }
    }
}

fn read_chunks(path: &std::path::Path) -> Result<Vec<hive_memory::Chunk>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    serde_json::from_str(&text).map_err(|error| format!("invalid chunks: {error}"))
}

fn vocab() -> Result<hive_memory::vocab::Vocabulary, String> {
    let paths = hive_core::paths::Paths::discover(None, None);
    let dir = paths.repo.join("splinter/splinter/vocab");
    hive_memory::vocab::Vocabulary::load(&dir, &["code", "general"])
}
