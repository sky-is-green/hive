use std::path::Path;
use std::process::{Command, Output};

fn run(workspace: &Path, script: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_hive-sandbox"))
        .arg("--workspace")
        .arg(workspace)
        .arg("--")
        .arg("/bin/sh")
        .arg("-c")
        .arg(script)
        .env("HIVE_TEST_SECRET", "must-not-leak")
        .output()
        .expect("spawn sandbox")
}

fn unsupported(output: &Output) -> bool {
    output.status.code() == Some(hive_sandbox::EXIT_UNSUPPORTED)
}

#[test]
fn writes_inside_the_workspace_and_denies_outside_writes() {
    let root = tempfile::tempdir().unwrap();
    let workspace = root.path().join("ws");
    std::fs::create_dir(&workspace).unwrap();
    let outside = root.path().join("escape.txt");

    let output = run(&workspace, "echo ok > inside.txt && cat inside.txt");
    if unsupported(&output) {
        eprintln!("skipping: Landlock is unavailable");
        return;
    }
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "ok");
    assert!(workspace.join("inside.txt").is_file());

    let output = run(&workspace, &format!("echo bad > '{}'", outside.display()));
    assert!(!output.status.success());
    assert!(!outside.exists(), "sandbox escaped the workspace");
}

#[test]
fn denies_network_and_clears_the_environment() {
    let root = tempfile::tempdir().unwrap();
    let workspace = root.path().join("ws");
    std::fs::create_dir(&workspace).unwrap();

    let output = run(&workspace, "/usr/bin/env");
    if unsupported(&output) {
        eprintln!("skipping: Landlock is unavailable");
        return;
    }
    assert!(output.status.success());
    let env = String::from_utf8_lossy(&output.stdout);
    assert!(
        !env.contains("HIVE_TEST_SECRET"),
        "environment leaked: {env}"
    );
    assert!(env.contains("PATH="));

    if Path::new("/usr/bin/python3").is_file() {
        let output = run(
            &workspace,
            "/usr/bin/python3 -c 'import socket; socket.socket()'",
        );
        assert!(!output.status.success(), "socket syscall was allowed");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("Operation not permitted") || stderr.contains("PermissionError"),
            "unexpected network error: {stderr}"
        );
    }
}

#[test]
fn denies_sensitive_host_files() {
    if unsafe { libc::geteuid() } == 0 {
        eprintln!("skipping: root can read /etc/shadow regardless of Landlock");
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let workspace = root.path().join("ws");
    std::fs::create_dir(&workspace).unwrap();
    let output = run(&workspace, "head -1 /etc/shadow");
    if unsupported(&output) {
        eprintln!("skipping: Landlock is unavailable");
        return;
    }
    assert!(!output.status.success());
    assert!(!output.stderr.is_empty() || !output.stdout.is_empty());
}
