use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::{Arc, Mutex};
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
    /// Run a loopback JSON encoder service for the Python sidecar
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8172)]
        port: u16,
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
        Command::Serve { host, port } => serve(&host, port),
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

fn serve(host: &str, port: u16) -> Result<(), String> {
    let encoder = Arc::new(Mutex::new(Encoder::load()?));
    let listener = TcpListener::bind((host, port))
        .map_err(|error| format!("cannot bind {host}:{port}: {error}"))?;
    let address = listener.local_addr().map_err(|error| error.to_string())?;
    println!("hive-encoder listening on http://{address}");
    let _ = std::io::stdout().flush();
    for stream in listener.incoming() {
        let Ok(mut stream) = stream else {
            continue;
        };
        let encoder = Arc::clone(&encoder);
        std::thread::spawn(move || {
            let _ = handle_connection(&mut stream, &encoder);
        });
    }
    Ok(())
}

fn handle_connection(
    stream: &mut std::net::TcpStream,
    encoder: &Arc<Mutex<Encoder>>,
) -> Result<(), String> {
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(30)))
        .ok();
    let mut raw = Vec::new();
    let mut buffer = [0u8; 8192];
    let header_end = loop {
        let read = stream
            .read(&mut buffer)
            .map_err(|error| error.to_string())?;
        if read == 0 {
            return Ok(());
        }
        raw.extend_from_slice(&buffer[..read]);
        if let Some(position) = raw.windows(4).position(|window| window == b"\r\n\r\n") {
            break position + 4;
        }
        if raw.len() > 1 << 20 {
            return Ok(());
        }
    };
    let headers = String::from_utf8_lossy(&raw[..header_end]).to_string();
    let mut lines = headers.lines();
    let request_line = lines.next().unwrap_or_default();
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("/");
    let content_length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);
    while raw.len() < header_end + content_length {
        let read = stream
            .read(&mut buffer)
            .map_err(|error| error.to_string())?;
        if read == 0 {
            break;
        }
        raw.extend_from_slice(&buffer[..read]);
    }
    let body = &raw[header_end..header_end + content_length.min(raw.len() - header_end)];
    if method == "GET" && path == "/health" {
        return respond(stream, 200, &json!({"ok": true, "ready": true}));
    }
    if method != "POST" || path != "/encode" {
        return respond(stream, 404, &json!({"detail": "not found"}));
    }
    let request: serde_json::Value = match serde_json::from_slice(body) {
        Ok(value) => value,
        Err(error) => {
            return respond(
                stream,
                400,
                &json!({"detail": format!("invalid JSON: {error}")}),
            );
        }
    };
    let texts: Vec<String> = request
        .get("texts")
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    if texts.is_empty() {
        return respond(stream, 422, &json!({"detail": "texts is required"}));
    }
    let result = {
        let mut encoder = encoder.lock().map_err(|error| error.to_string())?;
        encoder.encode(&texts)
    };
    match result {
        Ok(vectors) => respond(
            stream,
            200,
            &json!({"dimension": vectors.first().map(Vec::len).unwrap_or(0), "vectors": vectors}),
        ),
        Err(error) => respond(stream, 500, &json!({"detail": error})),
    }
}

fn respond(
    stream: &mut std::net::TcpStream,
    status: u16,
    payload: &serde_json::Value,
) -> Result<(), String> {
    let body = serde_json::to_vec(payload).map_err(|error| error.to_string())?;
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        422 => "Unprocessable Entity",
        _ => "Internal Server Error",
    };
    let header = format!(
        "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        body.len()
    );
    stream
        .write_all(header.as_bytes())
        .and_then(|()| stream.write_all(&body))
        .map_err(|error| error.to_string())
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
