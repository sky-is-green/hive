//! Types, paths, config, and host probes shared by the Hive CLI crates.

pub mod backends;
pub mod error;
pub mod hardware;
pub mod http;
pub mod json;
pub mod paths;
pub mod python;
pub mod stacks;

pub use error::{HiveError, Result};
pub use paths::Paths;

/// Python's `repr(str)`: single quotes, with backslashes and quotes escaped.
pub fn py_repr(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('\'');
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out.push('\'');
    out
}

/// The package version the CLI reports (`hive version`).
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// `platform.system()`-style values, e.g. `("Linux", "x86_64")`.
pub fn platform() -> (&'static str, &'static str) {
    let system = match std::env::consts::OS {
        "linux" => "Linux",
        "macos" => "Darwin",
        "windows" => "Windows",
        other => other,
    };
    (system, std::env::consts::ARCH)
}

/// `"Linux x86_64"`, matching the Python payload.
pub fn platform_display() -> String {
    let (system, machine) = platform();
    format!("{system} {machine}")
}
