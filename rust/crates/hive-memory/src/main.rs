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
    /// Replay add/turn operations and print the resulting store
    Replay {
        #[arg(long)]
        file: PathBuf,
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
