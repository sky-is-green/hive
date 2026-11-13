//! Small library operations shared by the CLI and the supervisor.

use std::path::{Path, PathBuf};

/// A model arg is a real file, a library-relative path, or a unique stem match.
pub fn resolve_model(name: &str, models_dir: &Path) -> Option<PathBuf> {
    if name.trim().is_empty() {
        return None;
    }
    let candidate = Path::new(name);
    if candidate.is_file() {
        return Some(
            candidate
                .canonicalize()
                .unwrap_or_else(|_| candidate.to_path_buf()),
        );
    }
    let direct = models_dir.join(name);
    if direct.is_file() {
        return Some(direct.canonicalize().unwrap_or(direct));
    }
    let mut matches: Vec<PathBuf> = walk_gguf(models_dir)
        .into_iter()
        .filter(|path| {
            path.file_name().map(|file| file == name).unwrap_or(false)
                || path
                    .file_stem()
                    .map(|stem| stem.to_string_lossy().contains(name))
                    .unwrap_or(false)
        })
        .collect();
    matches.sort();
    matches
        .into_iter()
        .next()
        .map(|path| path.canonicalize().unwrap_or(path))
}

/// Every `.gguf` under `root`, recursively (symlinked directories are not
/// followed, matching Python's `Path.rglob`).
pub fn walk_gguf(root: &Path) -> Vec<PathBuf> {
    let mut visited = std::collections::HashSet::new();
    let mut out = Vec::new();
    walk_gguf_into(root, &mut visited, &mut out);
    out
}

fn walk_gguf_into(
    root: &Path,
    visited: &mut std::collections::HashSet<PathBuf>,
    out: &mut Vec<PathBuf>,
) {
    let Ok(canonical) = root.canonicalize() else {
        return;
    };
    if !visited.insert(canonical) {
        return;
    }
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(metadata) = std::fs::metadata(&path) else {
            continue;
        };
        if metadata.is_dir() {
            walk_gguf_into(&path, visited, out);
        } else if metadata.is_file() && path.extension().is_some_and(|ext| ext == "gguf") {
            out.push(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_an_exact_file_and_a_library_name() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("Tiny-1.7B-Q4_K_M.gguf");
        std::fs::write(&model, b"GGUF").unwrap();
        assert_eq!(
            resolve_model("Tiny-1.7B-Q4_K_M.gguf", dir.path()),
            Some(model.canonicalize().unwrap())
        );
        assert_eq!(
            resolve_model("1.7B", dir.path()),
            Some(model.canonicalize().unwrap())
        );
    }

    #[test]
    fn an_unknown_name_does_not_resolve() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(resolve_model("nope", dir.path()), None);
        assert_eq!(resolve_model("", dir.path()), None);
    }
}
