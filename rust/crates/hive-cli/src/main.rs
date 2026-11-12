mod commands;
mod help;
mod probe;
mod serve;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

/// Hive — serve and manage local GGUF stacks, engine-blind.
#[derive(Debug, Parser)]
#[command(
    name = "hive",
    version,
    about = "Hive — serve and manage local GGUF stacks, engine-blind.",
    long_about = "Hive — serve and manage local GGUF stacks, engine-blind.",
    disable_help_subcommand = true
)]
pub struct Cli {
    /// disable colour even on a TTY
    #[arg(long)]
    pub no_color: bool,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Debug, Subcommand)]
#[allow(clippy::large_enum_variant)]
pub enum Command {
    /// apply a stack and serve it (foreground or detached)
    #[command(long_about = "Apply a stack (one process per tier, any engine) and keep it serving.")]
    Serve {
        /// stack name under stacks/, or a path to a .json stack
        stack: Option<String>,
        /// list the saved stacks and exit
        #[arg(long)]
        list: bool,
        /// list the declared server builds and exit
        #[arg(long)]
        backends: bool,
        /// HuggingFace repo (or repo:file), or a local .gguf path
        #[arg(long)]
        model: Option<String>,
        /// GGUF file inside the repo (with --model repo)
        #[arg(long)]
        file: Option<String>,
        /// how many detected cards to use (default: all)
        #[arg(long)]
        cards: Option<u32>,
        /// explicit HIP indices, e.g. 0,1 (overrides --cards)
        #[arg(long)]
        devices: Option<String>,
        /// context size for a generated stack (default 8192)
        #[arg(long, default_value_t = 8192)]
        ctx: u32,
        /// server backend (default: the first installed of rocm, vulkan)
        #[arg(long)]
        backend: Option<String>,
        /// fetch a missing engine backend from the manifest before applying
        #[arg(long)]
        fetch: bool,
        /// role to give a generated tier (default: face)
        #[arg(long, default_value = "face")]
        role: String,
        /// write a generated stack to stacks/<name>.json
        #[arg(long)]
        save: Option<String>,
        /// name this serve (default: the stack name)
        #[arg(long)]
        name: Option<String>,
        /// run in the background; manage with status/stop/logs
        #[arg(long)]
        detach: bool,
        /// stop a running serve with the same name first
        #[arg(long)]
        replace: bool,
        /// directory of stack files (default: <repo>/stacks)
        #[arg(long)]
        stacks_dir: Option<PathBuf>,
        /// GGUF model directory
        #[arg(long)]
        models_dir: Option<PathBuf>,
        /// server binary
        #[arg(long)]
        llama_server: Option<PathBuf>,
        /// engine log directory
        #[arg(long)]
        log_dir: Option<PathBuf>,
        /// host the tiers bind/publish on (default 127.0.0.1)
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// per-tier startup timeout in seconds (default 300)
        #[arg(long, default_value_t = 300.0)]
        timeout: f64,
        /// apply, print the endpoints, unload, exit (smoke test)
        #[arg(long)]
        once: bool,
        /// print status as JSON instead of a table
        #[arg(long)]
        json: bool,
        /// internal: the detached child
        #[arg(long = "_child", hide = true)]
        child: bool,
    },

    /// show running serves and their endpoints
    #[command(
        visible_alias = "ps",
        long_about = "List recorded serves, their tier endpoints, and live health. \
                      Records are advisory and checked against their process."
    )]
    Status {
        /// machine-readable output
        #[arg(long)]
        json: bool,
        /// do not call the endpoints
        #[arg(long)]
        no_probe: bool,
        /// remove records whose process is gone
        #[arg(long)]
        prune: bool,
    },

    /// stop a serve (name; or --all)
    #[command(
        long_about = "Stop one recorded serve, the only running serve, or every \
                      recorded serve with --all. Stale records are cleaned without \
                      signalling a reused PID."
    )]
    Stop {
        /// serve name (the only one if omitted)
        name: Option<String>,
        /// stop every recorded serve
        #[arg(long)]
        all: bool,
        /// SIGKILL after the timeout
        #[arg(long)]
        force: bool,
        /// seconds to wait for a clean stop (default 15)
        #[arg(long, default_value_t = 15.0)]
        timeout: f64,
        /// machine-readable JSON on stdout
        #[arg(long)]
        json: bool,
    },

    /// show or follow a serve's log
    #[command(
        long_about = "Show a serve's supervisor log, or one tier's engine log with \
                      --tier. Use --follow to stream new lines."
    )]
    Logs {
        /// serve name (the only one if omitted)
        name: Option<String>,
        /// a tier role's engine log
        #[arg(long)]
        tier: Option<String>,
        /// lines to show; 0 means all (default 80)
        #[arg(short = 'n', long, default_value_t = 80)]
        lines: i64,
        /// follow the file
        #[arg(short = 'f', long)]
        follow: bool,
        /// machine-readable JSON on stdout
        #[arg(long)]
        json: bool,
    },

    /// local GGUF library
    #[command(
        long_about = "Browse, search, download, import, or remove GGUF models in \
                      the configured local library."
    )]
    Models {
        #[command(subcommand)]
        command: ModelsCommand,
    },

    /// saved stacks
    #[command(
        long_about = "List, inspect, or validate saved stack files without starting \
                      an engine."
    )]
    Stacks {
        #[command(subcommand)]
        command: StacksCommand,
    },

    /// check the box before serving
    #[command(
        long_about = "Check Python, paths, engines, cards, host memory, ports, and \
                      the Hugging Face token before a serve. --strict treats warnings \
                      as failures."
    )]
    Doctor {
        /// GGUF model directory (default: the configured library)
        #[arg(long)]
        models_dir: Option<PathBuf>,
        /// directory of stack files (default: <repo>/stacks)
        #[arg(long)]
        stacks_dir: Option<PathBuf>,
        /// treat warnings as failures
        #[arg(long)]
        strict: bool,
        /// machine-readable JSON on stdout
        #[arg(long)]
        json: bool,
    },

    /// resolved paths and versions
    #[command(
        long_about = "Print the resolved model, stack, state, log, and backend \
                      paths plus versions. Read-only."
    )]
    Config {
        /// GGUF model directory (default: the configured library)
        #[arg(long)]
        models_dir: Option<PathBuf>,
        /// directory of stack files (default: <repo>/stacks)
        #[arg(long)]
        stacks_dir: Option<PathBuf>,
        /// machine-readable JSON on stdout
        #[arg(long)]
        json: bool,
    },

    /// print the Hive version
    #[command(long_about = "Print Hive, Python, and platform versions.")]
    Version {
        /// machine-readable JSON on stdout
        #[arg(long)]
        json: bool,
    },

    /// show help or a machine-readable command catalog
    #[command(long_about = "Print human help for a command, or emit the complete \
                      command/option catalog as JSON for agents and scripts.")]
    Help {
        /// command or subcommand, e.g. 'serve' or 'models list'
        topic: Vec<String>,
        /// emit the command/option catalog as JSON
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Subcommand)]
pub enum ModelsCommand {
    /// list local GGUFs
    #[command(long_about = "List GGUF files in the configured local model library.")]
    List {
        /// GGUF model directory (default: the configured library)
        #[arg(long)]
        models_dir: Option<PathBuf>,
        /// machine-readable JSON on stdout
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Subcommand)]
pub enum StacksCommand {
    /// list saved stacks
    #[command(long_about = "List stack files in the stacks directory.")]
    List {
        /// directory of stack files (default: <repo>/stacks)
        #[arg(long)]
        stacks_dir: Option<PathBuf>,
        /// machine-readable JSON on stdout
        #[arg(long)]
        json: bool,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    commands::run(cli)
}
