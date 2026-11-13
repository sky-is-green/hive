mod backends_cmd;
mod commands;
mod help;
mod models_cmd;
mod probe;
mod serve;
mod stacks_cmd;
mod ui;

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

    /// interactive terminal dashboard
    #[command(long_about = "Open the interactive dashboard (Status, Models, Stacks, Runs).")]
    Tui,

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

    /// server builds declared, installed or fetched
    #[command(long_about = "Inspect the declared engine server builds, or fetch a \
                      portable llama.cpp release.")]
    Backends {
        /// machine-readable output
        #[arg(long)]
        json: bool,
        #[command(subcommand)]
        command: Option<BackendsCommand>,
    },

    /// onboard this box: engine, model/stacks/state paths
    #[command(
        long_about = "Create missing paths, fetch a portable engine when needed, \
                      and print the next commands. It does not download a model."
    )]
    Setup {
        /// GGUF model directory (default: the configured library)
        #[arg(long)]
        models_dir: Option<PathBuf>,
        /// directory of stack files (default: <repo>/stacks)
        #[arg(long)]
        stacks_dir: Option<PathBuf>,
        /// do not fetch a portable engine if none is installed
        #[arg(long)]
        no_fetch: bool,
        /// machine-readable JSON on stdout
        #[arg(long)]
        json: bool,
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

    /// switch between terminal and web interfaces
    #[command(
        long_about = "Use the terminal dashboard or the web Studio. With no mode, \
                      report the web endpoint and the switch commands."
    )]
    Ui {
        /// target interface (default: status)
        #[arg(
            value_parser = ["web", "terminal", "status"],
            default_value = "status"
        )]
        mode: String,
        /// web bind host (default 127.0.0.1)
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// API port (default 8765)
        #[arg(long, default_value_t = 8765)]
        port: u16,
        /// web console port (default 8170)
        #[arg(long, default_value_t = 8170)]
        console_port: u16,
        /// start the web Studio in the background
        #[arg(long)]
        detach: bool,
        /// do not open a browser
        #[arg(long)]
        no_open: bool,
        /// start the Studio in model-free mock mode
        #[arg(long)]
        mock: bool,
        /// studio state directory (passed through to the web UI)
        #[arg(long)]
        state_dir: Option<PathBuf>,
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

    /// search Hugging Face for GGUF repos
    #[command(
        long_about = "Search the Hugging Face Hub for GGUF repositories. Omit \
                      the query to list popular repos."
    )]
    Search {
        /// words to search for (omit for popular repos)
        #[arg(default_value = "")]
        query: String,
        /// number of repos (default 12)
        #[arg(long, default_value_t = 12)]
        limit: i64,
        /// GGUF model directory (default: the configured library)
        #[arg(long)]
        models_dir: Option<PathBuf>,
        /// machine-readable JSON on stdout
        #[arg(long)]
        json: bool,
    },

    /// list the GGUFs in a Hugging Face repo
    #[command(long_about = "List the GGUF files and sizes in one Hugging Face repo.")]
    Files {
        /// org/repo
        repo: String,
        /// GGUF model directory (default: the configured library)
        #[arg(long)]
        models_dir: Option<PathBuf>,
        /// machine-readable JSON on stdout
        #[arg(long)]
        json: bool,
    },

    /// show or set the models directory
    #[command(
        long_about = "Print the resolved GGUF library path, or set a new one with \
                      PATH/--set. The setting persists."
    )]
    Dir {
        /// a directory to use from now on
        path: Option<String>,
        /// set the models directory (same as the positional)
        #[arg(long)]
        set: Option<String>,
        /// GGUF model directory (default: the configured library)
        #[arg(long)]
        models_dir: Option<PathBuf>,
        /// machine-readable JSON on stdout
        #[arg(long)]
        json: bool,
    },

    /// download one GGUF from Hugging Face
    #[command(
        long_about = "Download one GGUF from a Hugging Face repo into the local \
                      library. Use --file or repo:file to select the file."
    )]
    Download {
        /// org/repo, or org/repo:file.gguf
        repo: String,
        /// GGUF file inside the repo
        #[arg(long)]
        file: Option<String>,
        /// HF revision (default main)
        #[arg(long, default_value = "main")]
        revision: String,
        /// GGUF model directory (default: the configured library)
        #[arg(long)]
        models_dir: Option<PathBuf>,
        /// machine-readable JSON on stdout
        #[arg(long)]
        json: bool,
    },

    /// link a local GGUF or folder into the library
    #[command(
        long_about = "Link one local .gguf file, or every .gguf in a folder, into \
                      the library. --copy copies instead of linking."
    )]
    Import {
        /// a .gguf file or a folder containing GGUFs
        path: String,
        /// copy the file instead of hardlink/symlink
        #[arg(long)]
        copy: bool,
        /// GGUF model directory (default: the configured library)
        #[arg(long)]
        models_dir: Option<PathBuf>,
        /// machine-readable JSON on stdout
        #[arg(long)]
        json: bool,
    },

    /// delete one GGUF from the library
    #[command(
        long_about = "Delete one GGUF from the library. --yes is required; the \
                      command never deletes outside the configured library."
    )]
    Remove {
        /// file name or relative path inside the library
        file: String,
        /// confirm the deletion
        #[arg(long)]
        yes: bool,
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

    /// print one stack as JSON
    #[command(
        long_about = "Print one stack's resolved JSON. --json is accepted for \
                      symmetry; this command always prints JSON."
    )]
    Show {
        /// stack name or path
        stack: String,
        /// directory of stack files (default: <repo>/stacks)
        #[arg(long)]
        stacks_dir: Option<PathBuf>,
        /// accepted for symmetry; show always prints JSON
        #[arg(long)]
        json: bool,
    },

    /// check a stack without starting it
    #[command(
        long_about = "Validate a stack file without launching an engine. --strict \
                      treats warnings as failures."
    )]
    Validate {
        /// stack name or path
        stack: String,
        /// directory of stack files (default: <repo>/stacks)
        #[arg(long)]
        stacks_dir: Option<PathBuf>,
        /// GGUF model directory (default: the configured library)
        #[arg(long)]
        models_dir: Option<PathBuf>,
        /// treat warnings as failures
        #[arg(long)]
        strict: bool,
        /// machine-readable JSON on stdout
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Subcommand)]
pub enum BackendsCommand {
    /// list declared backends
    #[command(long_about = "List declared backends and whether each is installed.")]
    List {
        /// machine-readable JSON on stdout
        #[arg(long)]
        json: bool,
    },

    /// download a portable llama.cpp release
    #[command(
        long_about = "Download and install a portable llama.cpp backend release. \
                      --force replaces an existing directory."
    )]
    Fetch {
        /// backend to install (default: vulkan)
        name: Option<String>,
        /// GitHub release tag (default: newest build)
        #[arg(long)]
        tag: Option<String>,
        /// replace an existing backend directory
        #[arg(long)]
        force: bool,
        /// machine-readable JSON on stdout
        #[arg(long)]
        json: bool,
    },
}

/// Verbs that still live in the Python reference. They are proxied rather
/// than reimplemented until their phase lands, so `hive <verb>` keeps working.
const PYTHON_ONLY_VERBS: &[&str] = &[
    "cascade",
    "profile",
    "controller",
    "encoder",
    "tools",
    "activity",
    "sandbox",
    "web",
    "browser",
];

fn main() -> ExitCode {
    if let Some(code) = proxy_python_only() {
        return code;
    }
    let cli = Cli::parse();
    commands::run(cli)
}

fn proxy_python_only() -> Option<ExitCode> {
    let mut argv: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    while argv.first().and_then(|arg| arg.to_str()) == Some("--no-color") {
        argv.remove(0);
    }
    let verb = argv.first()?.to_string_lossy().into_owned();
    let proxied = if verb == "help" {
        argv.get(1)
            .map(|topic| topic.to_string_lossy().into_owned())
            .is_some_and(|topic| PYTHON_ONLY_VERBS.contains(&topic.as_str()))
    } else {
        PYTHON_ONLY_VERBS.contains(&verb.as_str())
    };
    if !proxied {
        return None;
    }
    let paths = hive_core::paths::Paths::discover(None, None);
    let python = match hive_core::python::resolve(&paths) {
        Ok(python) => python,
        Err(error) => {
            eprintln!("hive {verb}: {error}");
            return Some(ExitCode::FAILURE);
        }
    };
    let status = std::process::Command::new(python)
        .arg("-m")
        .arg("harness.cli")
        .args(&argv)
        .current_dir(&paths.repo)
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status();
    match status {
        Ok(status) if status.success() => Some(ExitCode::SUCCESS),
        Ok(_) => Some(ExitCode::FAILURE),
        Err(error) => {
            eprintln!("hive {verb}: cannot start the Python reference: {error}");
            Some(ExitCode::FAILURE)
        }
    }
}
