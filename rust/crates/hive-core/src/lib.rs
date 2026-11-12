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
