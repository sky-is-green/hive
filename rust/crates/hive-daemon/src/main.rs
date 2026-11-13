use std::process::ExitCode;

use clap::Parser;
use hive_daemon::{DaemonConfig, run};

/// Hive Phase 2 daemon: native routes plus a Python sidecar proxy.
///
/// With no `--sidecar-url`, the daemon starts and supervises its own
/// `python -m harness` sidecar using `HIVE_PYTHON`.
#[derive(Debug, Parser)]
#[command(name = "hive-daemon", version)]
struct Args {
    /// bind host (default 127.0.0.1)
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    /// public API port (default 8765)
    #[arg(long, default_value_t = 8765)]
    port: u16,
    /// use an already-running sidecar instead of spawning one
    /// (default HIVE_SIDECAR_URL)
    #[arg(long, env = "HIVE_SIDECAR_URL", conflicts_with = "no_sidecar")]
    sidecar_url: Option<String>,
    /// private loopback port for the managed sidecar; 0 picks a free port
    #[arg(long, default_value_t = 8766)]
    sidecar_port: u16,
    /// do not spawn or proxy a sidecar (native routes only)
    #[arg(long)]
    no_sidecar: bool,
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
    let mut config = DaemonConfig::new(args.host, args.port, args.sidecar_url);
    config.spawn_sidecar = config.sidecar_url.is_none() && !args.no_sidecar;
    config.sidecar_port = args.sidecar_port;
    match run(config).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("hive-daemon: {error}");
            ExitCode::FAILURE
        }
    }
}
