//! Native Linux execution isolation.
//!
//! The daemon never applies Landlock/seccomp from a threaded `pre_exec` hook.
//! Instead it spawns this tiny single-threaded helper, which:
//!
//! 1. sets resource limits and `PR_SET_NO_NEW_PRIVS`,
//! 2. installs a Landlock ruleset (read-only system paths, read-write
//!    workspace/temp),
//! 3. installs a seccomp deny-list (network and privilege-escalation syscalls),
//! 4. execs the requested command with a scrubbed environment.
//!
//! Failure to enforce Landlock is fatal (exit 78), never a silent host exec.

use std::path::{Path, PathBuf};

use landlock::{
    ABI, Access, AccessFs, CompatLevel, Compatible, Ruleset, RulesetAttr, RulesetCreatedAttr,
    RulesetStatus, path_beneath_rules,
};

pub const EXIT_UNSUPPORTED: i32 = 78;

#[derive(Debug, Clone)]
pub struct Limits {
    pub memory_mb: Option<u64>,
    pub cpu_seconds: Option<u64>,
    pub pids: u64,
    pub nofile: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            memory_mb: Some(512),
            cpu_seconds: Some(60),
            pids: 256,
            nofile: 256,
        }
    }
}

/// Apply the confinement to the current (single-threaded) process.
pub fn apply(
    workspace: Option<&Path>,
    temp: Option<&Path>,
    extra_read: &[PathBuf],
    limits: &Limits,
    allow_network: bool,
) -> Result<(), String> {
    let workspace = workspace.map(canonical_dir).transpose()?;
    let temp = temp.map(canonical_dir).transpose()?;
    if workspace.is_none() && temp.is_none() {
        return Err("a workspace or temp directory is required".into());
    }
    apply_limits(limits)?;
    apply_landlock(workspace.as_deref(), temp.as_deref(), extra_read)?;
    apply_seccomp(allow_network)?;
    Ok(())
}

fn canonical_dir(path: &Path) -> Result<PathBuf, String> {
    let path = std::fs::canonicalize(path)
        .map_err(|error| format!("cannot resolve {}: {error}", path.display()))?;
    if !path.is_dir() {
        return Err(format!("{} is not a directory", path.display()));
    }
    Ok(path)
}

fn apply_limits(limits: &Limits) -> Result<(), String> {
    if let Some(memory_mb) = limits.memory_mb {
        let bytes = memory_mb.saturating_mul(1024 * 1024) as libc::rlim_t;
        set_limit(libc::RLIMIT_AS, bytes, bytes, "RLIMIT_AS")?;
    }
    if let Some(cpu_seconds) = limits.cpu_seconds {
        let seconds = cpu_seconds as libc::rlim_t;
        set_limit(libc::RLIMIT_CPU, seconds, seconds + 1, "RLIMIT_CPU")?;
    }
    set_limit(
        libc::RLIMIT_NOFILE,
        limits.nofile as libc::rlim_t,
        limits.nofile as libc::rlim_t,
        "RLIMIT_NOFILE",
    )?;
    set_limit(
        libc::RLIMIT_NPROC,
        limits.pids as libc::rlim_t,
        limits.pids as libc::rlim_t,
        "RLIMIT_NPROC",
    )?;
    set_limit(libc::RLIMIT_CORE, 0, 0, "RLIMIT_CORE")?;
    Ok(())
}

fn set_limit(
    resource: libc::__rlimit_resource_t,
    soft: libc::rlim_t,
    hard: libc::rlim_t,
    name: &str,
) -> Result<(), String> {
    let limit = libc::rlimit {
        rlim_cur: soft,
        rlim_max: hard,
    };
    // SAFETY: `limit` is a valid rlimit for the duration of the call.
    let result = unsafe { libc::setrlimit(resource, &limit) };
    if result == 0 {
        Ok(())
    } else {
        Err(format!(
            "cannot set {name}: {}",
            std::io::Error::last_os_error()
        ))
    }
}

fn apply_landlock(
    workspace: Option<&Path>,
    temp: Option<&Path>,
    extra_read: &[PathBuf],
) -> Result<(), String> {
    let abi = ABI::V4;
    let mut read_paths: Vec<PathBuf> = [
        "/usr",
        "/bin",
        "/sbin",
        "/lib",
        "/lib64",
        "/etc/ld.so.cache",
        "/etc/ld.so.conf",
        "/etc/ld.so.conf.d",
        "/etc/localtime",
        "/etc/passwd",
        "/etc/group",
        "/etc/nsswitch.conf",
        "/etc/resolv.conf",
        "/etc/ssl/certs",
        "/dev/null",
        "/dev/zero",
        "/dev/random",
        "/dev/urandom",
        "/proc/self",
        "/proc/thread-self",
        "/proc/cpuinfo",
        "/proc/meminfo",
        "/proc/stat",
    ]
    .into_iter()
    .map(PathBuf::from)
    .collect();
    read_paths.extend(extra_read.iter().cloned());

    let mut write_paths: Vec<PathBuf> = Vec::new();
    if let Some(workspace) = workspace {
        write_paths.push(workspace.to_path_buf());
    }
    if let Some(temp) = temp {
        write_paths.push(temp.to_path_buf());
    }

    let status = Ruleset::default()
        .handle_access(AccessFs::from_all(abi))
        .map_err(|error| format!("cannot declare Landlock access: {error}"))?
        .create()
        .map_err(|error| format!("cannot create Landlock ruleset: {error}"))?
        .add_rules(path_beneath_rules(
            read_paths.iter(),
            AccessFs::from_read(abi),
        ))
        .map_err(|error| format!("cannot add read rules: {error}"))?
        .add_rules(path_beneath_rules(
            write_paths.iter(),
            AccessFs::from_all(abi),
        ))
        .map_err(|error| format!("cannot add write rules: {error}"))?
        .set_compatibility(CompatLevel::BestEffort)
        .restrict_self()
        .map_err(|error| format!("cannot enforce Landlock ruleset: {error}"))?;

    match status.ruleset {
        RulesetStatus::FullyEnforced => {}
        RulesetStatus::PartiallyEnforced => {
            return Err("Landlock was only partially enforced; refusing host exec".into());
        }
        RulesetStatus::NotEnforced => {
            return Err("Landlock is not enforced by this kernel; refusing host exec".into());
        }
    }
    if !status.no_new_privs {
        return Err("PR_SET_NO_NEW_PRIVS was not applied".into());
    }
    Ok(())
}

fn apply_seccomp(allow_network: bool) -> Result<(), String> {
    let arch = seccompiler::TargetArch::try_from(std::env::consts::ARCH)
        .map_err(|error| format!("unsupported seccomp architecture: {error}"))?;

    let mut denied: Vec<i64> = vec![
        libc::SYS_ptrace,
        libc::SYS_mount,
        libc::SYS_umount2,
        libc::SYS_pivot_root,
        libc::SYS_chroot,
        libc::SYS_setns,
        libc::SYS_unshare,
        libc::SYS_bpf,
        libc::SYS_kexec_load,
        libc::SYS_init_module,
        libc::SYS_finit_module,
        libc::SYS_delete_module,
        libc::SYS_open_by_handle_at,
        libc::SYS_name_to_handle_at,
        libc::SYS_userfaultfd,
        libc::SYS_perf_event_open,
        libc::SYS_io_uring_setup,
        libc::SYS_io_uring_enter,
        libc::SYS_io_uring_register,
        libc::SYS_add_key,
        libc::SYS_request_key,
        libc::SYS_keyctl,
        libc::SYS_process_vm_readv,
        libc::SYS_process_vm_writev,
    ];
    if !allow_network {
        denied.extend([
            libc::SYS_socket,
            libc::SYS_socketpair,
            libc::SYS_connect,
            libc::SYS_bind,
            libc::SYS_listen,
            libc::SYS_accept,
            libc::SYS_accept4,
            libc::SYS_sendto,
            libc::SYS_recvfrom,
            libc::SYS_sendmsg,
            libc::SYS_recvmsg,
            libc::SYS_sendmmsg,
            libc::SYS_recvmmsg,
            libc::SYS_setsockopt,
            libc::SYS_getsockopt,
            libc::SYS_shutdown,
        ]);
    }
    let rules: std::collections::BTreeMap<i64, Vec<seccompiler::SeccompRule>> = denied
        .into_iter()
        .map(|syscall| (syscall, Vec::new()))
        .collect();
    let filter = seccompiler::SeccompFilter::new(
        rules,
        seccompiler::SeccompAction::Allow,
        seccompiler::SeccompAction::Errno(libc::EPERM as u32),
        arch,
    )
    .map_err(|error| format!("cannot build seccomp filter: {error}"))?;
    let program: seccompiler::BpfProgram = filter
        .try_into()
        .map_err(|error| format!("cannot compile seccomp filter: {error}"))?;
    seccompiler::apply_filter(&program)
        .map_err(|error| format!("cannot install seccomp filter: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_are_positive() {
        let limits = Limits::default();
        assert!(limits.memory_mb.unwrap_or(0) > 0);
        assert!(limits.cpu_seconds.unwrap_or(0) > 0);
    }
}
