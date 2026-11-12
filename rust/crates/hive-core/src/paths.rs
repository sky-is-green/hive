//! The mutable Hive root and the paths derived from it.
//!
//! A source checkout keeps `stacks/`, `models/gguf/`, `logs/` and
//! `harness_state/` beside `pyproject.toml`. An installed binary uses
//! `HIVE_HOME` (default `~/.hive`) instead. `HIVE_STATE_DIR` overrides only
//! the runtime state root. This mirrors `harness/paths.py`, so the Rust CLI
//! and the Python Studio read the same files.

use std::env;
use std::path::{Path, PathBuf};

/// All resolved roots for one CLI invocation.
#[derive(Debug, Clone)]
pub struct Paths {
    /// The checkout root when the binary lives in one.
    pub source_root: Option<PathBuf>,
    /// The path reported by `hive config` as `repo`.
    pub repo: PathBuf,
    /// The writable Hive root (`HIVE_HOME`, checkout, or `~/.hive`).
    pub home: PathBuf,
    pub state_dir: PathBuf,
    pub serve_dir: PathBuf,
    pub stacks_dir: PathBuf,
    pub models_dir: PathBuf,
    pub log_dir: PathBuf,
    pub manifest: Option<PathBuf>,
}

impl Paths {
    /// Resolve every path. `models_dir`/`stacks_dir` are the CLI overrides.
    pub fn discover(models_dir: Option<&Path>, stacks_dir: Option<&Path>) -> Self {
        let source_root = discover_source_root();
        let home = env::var_os("HIVE_HOME")
            .map(PathBuf::from)
            .map(|path| expand(&path))
            .or_else(|| source_root.clone())
            .unwrap_or_else(|| expand(Path::new("~/.hive")));
        let repo = source_root.clone().unwrap_or_else(|| home.clone());
        let state_dir = env::var_os("HIVE_STATE_DIR")
            .map(PathBuf::from)
            .map(|path| expand(&path))
            .unwrap_or_else(|| home.join("harness_state"));

        let models_dir = match models_dir {
            Some(path) => expand(path),
            None => persisted_models_dir(&state_dir).unwrap_or_else(|| default_models_dir(&home)),
        };
        let stacks_dir = match stacks_dir {
            Some(path) => expand(path),
            None => home.join("stacks"),
        };
        let log_dir = home.join("logs");
        let manifest = {
            let checkout = repo.join("tools/backends/manifest.json");
            let packaged = repo.join("harness/data/backends-manifest.json");
            if checkout.is_file() {
                Some(checkout)
            } else if packaged.is_file() {
                Some(packaged)
            } else {
                None
            }
        };

        Self {
            source_root,
            repo,
            home,
            serve_dir: state_dir.join("serve"),
            state_dir,
            stacks_dir,
            models_dir,
            log_dir,
            manifest,
        }
    }

    /// `hive config` and `doctor` create the library on first use.
    pub fn ensure_models_dir(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.models_dir)
    }

    pub fn ensure_serve_dir(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.serve_dir)
    }
}

/// Walk up from the executable (then the cwd) looking for a checkout.
fn discover_source_root() -> Option<PathBuf> {
    if let Some(raw) = env::var_os("HIVE_SOURCE_ROOT") {
        let path = expand(PathBuf::from(raw).as_path());
        if is_source_root(&path) {
            return Some(path);
        }
    }
    if let Ok(exe) = env::current_exe()
        && let Some(root) = ancestors_find(&exe)
    {
        return Some(root);
    }
    if let Ok(cwd) = env::current_dir()
        && let Some(root) = ancestors_find(&cwd)
    {
        return Some(root);
    }
    None
}

fn ancestors_find(start: &Path) -> Option<PathBuf> {
    let start = start.canonicalize().unwrap_or_else(|_| start.to_path_buf());
    for ancestor in start.ancestors() {
        if is_source_root(ancestor) {
            return Some(ancestor.to_path_buf());
        }
    }
    None
}

fn is_source_root(path: &Path) -> bool {
    path.join("pyproject.toml").is_file() && path.join("harness/paths.py").is_file()
}

/// The library override persisted by `hive models dir --set`.
pub fn persisted_models_dir(state_dir: &Path) -> Option<PathBuf> {
    let text = std::fs::read_to_string(state_dir.join("models_dir.txt")).ok()?;
    let path = expand(Path::new(text.trim()));
    path.is_dir().then_some(path)
}

/// LM Studio's library when the operator already keeps one, then Hive's own.
pub fn default_models_dir(home: &Path) -> PathBuf {
    for candidate in lmstudio_candidates() {
        if candidate.is_dir() {
            return candidate;
        }
    }
    home.join("models/gguf")
}

fn lmstudio_candidates() -> Vec<PathBuf> {
    let home = expand(Path::new("~"));
    let mut out = vec![
        home.join(".lmstudio/models"),
        home.join(".cache/lm-studio/models"),
    ];
    if let Some(local) = env::var_os("LOCALAPPDATA") {
        out.push(PathBuf::from(local).join("LM Studio/models"));
    }
    if let Some(roaming) = env::var_os("APPDATA") {
        out.push(PathBuf::from(roaming).join("LM Studio/models"));
    }
    out
}

/// `~` expansion for flags, matching `Path.expanduser`.
pub fn expand(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    if text == "~" {
        return home_dir();
    }
    if let Some(rest) = text.strip_prefix("~/") {
        return home_dir().join(rest);
    }
    path.to_path_buf()
}

fn home_dir() -> PathBuf {
    env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_a_leading_tilde_only() {
        let home = home_dir();
        assert_eq!(expand(Path::new("~/models")), home.join("models"));
        assert_eq!(
            expand(Path::new("/tmp/~/models")),
            PathBuf::from("/tmp/~/models")
        );
    }

    #[test]
    fn a_checkout_is_detected_from_an_ancestor() {
        let cwd = env::current_dir().expect("cwd");
        assert!(
            ancestors_find(&cwd).is_some(),
            "tests run inside the checkout"
        );
    }
}
