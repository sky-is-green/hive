use std::process::ExitCode;

use clap::Parser;
use hive_daemon::{DaemonConfig, run};

/// Hive Phase 2 daemon: native routes plus a Python sidecar proxy.
#[derive(Debug, Parser)]
#[command(name = "hive-daemon", version)]
struct Args {
    /// bind host (default 127.0.0.1)
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    /// public API port (default 8765)
    #[arg(long, default_value_t = 8765)]
    port: u16,
    /// Python sidecar base URL (default HIVE_SIDECAR_URL or :8766)
    #[arg(
        long,
        env = "HIVE_SIDECAR_URL",
        default_value = "http://127.0.0.1:8766"
    )]
    sidecar_url: String,
}

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "hive_daemon=info,tower_http=warn".into()),
        )
        .init();
    let args = Args::parse();
    let config = DaemonConfig::new(args.host, args.port, Some(args.sidecar_url));
    match run(config).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("hive-daemon: {error}");
            ExitCode::FAILURE
        }
    }
}
