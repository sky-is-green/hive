use std::ffi::OsString;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Command, ExitCode};

use clap::Parser;
use hive_sandbox::{EXIT_UNSUPPORTED, Limits};

#[derive(Debug, Parser)]
#[command(
    name = "hive-sandbox",
    version,
    about = "Run one command under Landlock and seccomp."
)]
struct Args {
    /// Read-write workspace directory.
    #[arg(long)]
    workspace: Option<PathBuf>,
    /// Read-write scratch directory (usually a per-call temp dir).
    #[arg(long)]
    temp: Option<PathBuf>,
    /// Additional read-only path.
    #[arg(long = "read", value_name = "PATH")]
    read: Vec<PathBuf>,
    #[arg(long, default_value_t = 512)]
    memory_mb: u64,
    #[arg(long, default_value_t = 60)]
    cpu_seconds: u64,
    #[arg(long, default_value_t = 256)]
    pids: u64,
    #[arg(long, default_value_t = 256)]
    nofile: u64,
    /// Permit socket syscalls (filesystem confinement still applies).
    #[arg(long)]
    allow_network: bool,
    /// Command and arguments after `--`.
    #[arg(last = true, required = true, num_args = 1.., allow_hyphen_values = true)]
    command: Vec<OsString>,
}

fn main() -> ExitCode {
    let args = Args::parse();
    if args.workspace.is_none() && args.temp.is_none() {
        eprintln!("hive-sandbox: --workspace or --temp is required");
        return ExitCode::from(2);
    }
    let limits = Limits {
        memory_mb: Some(args.memory_mb),
        cpu_seconds: Some(args.cpu_seconds),
        pids: args.pids,
        nofile: args.nofile,
    };
    if let Err(error) = hive_sandbox::apply(
        args.workspace.as_deref(),
        args.temp.as_deref(),
        &args.read,
        &limits,
        args.allow_network,
    ) {
        eprintln!("hive-sandbox: {error}");
        return ExitCode::from(EXIT_UNSUPPORTED as u8);
    }

    let Some(program) = args.command.first() else {
        eprintln!("hive-sandbox: no command");
        return ExitCode::from(2);
    };
    let root = args
        .workspace
        .as_deref()
        .or(args.temp.as_deref())
        .unwrap_or_else(|| std::path::Path::new("/"));
    let scratch = args.temp.as_deref().or(args.workspace.as_deref());

    let mut command = Command::new(program);
    command.args(&args.command[1..]);
    command.current_dir(root);
    command.env_clear();
    command.env("PATH", "/usr/local/bin:/usr/bin:/bin");
    command.env("HOME", root);
    command.env("LANG", "C.UTF-8");
    command.env("LC_ALL", "C.UTF-8");
    command.env("TERM", "dumb");
    if let Some(scratch) = scratch {
        command.env("TMPDIR", scratch);
    }
    let error = command.exec();
    eprintln!(
        "hive-sandbox: cannot execute {}: {error}",
        program.to_string_lossy()
    );
    ExitCode::from(127)
}
