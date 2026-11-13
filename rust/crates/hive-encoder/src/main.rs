use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, Subcommand};
use hive_encoder::Encoder;
use serde_json::json;

#[derive(Debug, Parser)]
#[command(
    name = "hive-encoder",
    version,
    about = "Native ONNX encoder for the Hive memory core."
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Print the resolved model, tokenizer, and vector dimension
    Info {
        #[arg(long)]
        json: bool,
    },
    /// Encode one text or every line of a file
    Encode {
        #[arg(long)]
        text: Option<String>,
        #[arg(long)]
        file: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Time repeated batches over a text file
    Bench {
        #[arg(long)]
        file: PathBuf,
        #[arg(long, default_value_t = 10)]
        repeat: usize,
        #[arg(long, default_value_t = 32)]
        batch: usize,
        #[arg(long)]
        json: bool,
    },
}

fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("hive-encoder: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Args) -> Result<(), String> {
    match args.command {
        Command::Info { json: as_json } => {
            let encoder = Encoder::load()?;
            let payload = json!({
                "model": encoder.model_path.to_string_lossy(),
                "tokenizer": encoder.tokenizer_path.to_string_lossy(),
                "dimension": encoder.dimension,
                "implementation": "onnxruntime",
            });
            if as_json {
                print!("{}", hive_core::json::pretty(&payload));
            } else {
                println!("model      {}", encoder.model_path.display());
                println!("tokenizer  {}", encoder.tokenizer_path.display());
                println!("dimension  {}", encoder.dimension);
            }
            Ok(())
        }
        Command::Encode {
            text,
            file,
            json: as_json,
        } => {
            let texts = read_texts(text, file)?;
            let mut encoder = Encoder::load()?;
            let vectors = encoder.encode(&texts)?;
            if as_json {
                let rows: Vec<_> = texts
                    .iter()
                    .zip(&vectors)
                    .map(|(text, vector)| json!({"text": text, "vector": vector}))
                    .collect();
                print!(
                    "{}",
                    hive_core::json::pretty(&json!({"dimension": encoder.dimension, "rows": rows}))
                );
            } else {
                for (text, vector) in texts.iter().zip(&vectors) {
                    println!("{text}");
                    println!("  {}", preview(vector));
                }
            }
            Ok(())
        }
        Command::Bench {
            file,
            repeat,
            batch,
            json: as_json,
        } => {
            let texts = read_texts(None, Some(file))?;
            if texts.is_empty() {
                return Err("benchmark file has no texts".into());
            }
            let mut encoder = Encoder::load()?;
            encoder.encode(&texts[..1])?;
            let started = Instant::now();
            let mut vectors = 0usize;
            for _ in 0..repeat.max(1) {
                for chunk in texts.chunks(batch.max(1)) {
                    vectors += encoder.encode(chunk)?.len();
                }
            }
            let elapsed = started.elapsed().as_secs_f64();
            let payload = json!({
                "texts": texts.len(),
                "repeat": repeat.max(1),
                "batch": batch.max(1),
                "vectors": vectors,
                "seconds": elapsed,
                "texts_per_second": vectors as f64 / elapsed.max(1e-9),
                "max_rss_mb": max_rss_mb(),
            });
            if as_json {
                print!("{}", hive_core::json::pretty(&payload));
            } else {
                println!(
                    "{} vectors in {:.3}s · {:.1} texts/s",
                    vectors,
                    elapsed,
                    vectors as f64 / elapsed.max(1e-9)
                );
            }
            Ok(())
        }
    }
}

fn read_texts(text: Option<String>, file: Option<PathBuf>) -> Result<Vec<String>, String> {
    if let Some(text) = text {
        return Ok(vec![text]);
    }
    let Some(file) = file else {
        return Err("--text or --file is required".into());
    };
    let text = std::fs::read_to_string(&file)
        .map_err(|error| format!("cannot read {}: {error}", file.display()))?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect())
}

fn max_rss_mb() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(value) = line.strip_prefix("VmHWM:") {
            let kib = value.split_whitespace().next()?.parse::<f64>().ok()?;
            return Some((kib / 1024.0 * 10.0).round() / 10.0);
        }
    }
    None
}

fn preview(vector: &[f32]) -> String {
    vector
        .iter()
        .take(6)
        .map(|value| format!("{value:.4}"))
        .collect::<Vec<_>>()
        .join(", ")
}
