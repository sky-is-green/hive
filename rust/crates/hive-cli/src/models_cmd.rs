//! `hive models dir/import/remove`.

use std::fs;
use std::path::{Component, Path, PathBuf};

use hive_core::json;
use hive_core::paths::{Paths, expand};
use hive_models::Library;
use serde_json::{Map, Value};

type Cmd = Result<u8, String>;

pub fn dir(paths: &Paths, path: Option<&str>, set: Option<&str>, as_json: bool) -> Cmd {
    let target = set.or(path);
    if let Some(target) = target {
        let mut folder = expand(Path::new(target));
        if !folder.exists() {
            fs::create_dir_all(&folder).map_err(|error| {
                format!(
                    "hive models dir: cannot create folder {}: {error}",
                    folder.display()
                )
            })?;
        }
        if !folder.is_dir() {
            return Err(format!(
                "hive models dir: not a directory: {}",
                folder.display()
            ));
        }
        if !readable(&folder) {
            return Err(format!(
                "hive models dir: cannot read folder: {}",
                folder.display()
            ));
        }
        folder = folder
            .canonicalize()
            .map_err(|error| format!("hive models dir: {}: {error}", folder.display()))?;
        let config = paths.state_dir.join("models_dir.txt");
        if let Some(parent) = config.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::write(&config, folder.to_string_lossy().as_bytes());
        let payload = {
            let mut object = Map::new();
            object.insert(
                "models_dir".into(),
                Value::String(folder.to_string_lossy().into_owned()),
            );
            object.insert("models".into(), Value::Array(Library::new(&folder).rows()));
            Value::Object(object)
        };
        if as_json {
            print!("{}", json::pretty(&payload));
        } else {
            println!("{}", folder.display());
        }
        return Ok(0);
    }

    paths
        .ensure_models_dir()
        .map_err(|error| format!("hive models dir: {error}"))?;
    let library = Library::new(&paths.models_dir);
    let payload = {
        let mut object = Map::new();
        object.insert(
            "models_dir".into(),
            Value::String(paths.models_dir.to_string_lossy().into_owned()),
        );
        object.insert("exists".into(), Value::Bool(paths.models_dir.is_dir()));
        object.insert("gguf_count".into(), Value::from(library.count() as u64));
        Value::Object(object)
    };
    if as_json {
        print!("{}", json::pretty(&payload));
    } else {
        println!("{}", paths.models_dir.display());
    }
    Ok(0)
}

pub fn import(paths: &Paths, source: &str, copy: bool, as_json: bool) -> Cmd {
    let source = expand(Path::new(source));
    if source.is_dir() {
        return import_dir(paths, &source, as_json);
    }
    if !source.is_file() {
        eprintln!(
            "hive models import: no such file or folder: {}",
            source.display()
        );
        return Ok(1);
    }
    if source
        .extension()
        .map(|ext| !ext.eq_ignore_ascii_case("gguf"))
        .unwrap_or(true)
    {
        eprintln!(
            "hive models import: not a .gguf file: {}",
            source.file_name().unwrap_or_default().to_string_lossy()
        );
        return Ok(2);
    }
    paths
        .ensure_models_dir()
        .map_err(|error| format!("hive models import: {error}"))?;
    let source = source.canonicalize().unwrap_or(source);
    let models_dir = paths
        .models_dir
        .canonicalize()
        .unwrap_or_else(|_| paths.models_dir.clone());
    let name = source.file_name().unwrap_or_default().to_os_string();

    if source.parent() == Some(models_dir.as_path()) {
        let payload = {
            let mut object = Map::new();
            object.insert(
                "imported".into(),
                Value::String(name.to_string_lossy().into_owned()),
            );
            object.insert(
                "path".into(),
                Value::String(source.to_string_lossy().into_owned()),
            );
            object.insert("linked".into(), Value::Bool(false));
            Value::Object(object)
        };
        if as_json {
            print!("{}", json::pretty(&payload));
        } else {
            println!("{}", source.display());
        }
        return Ok(0);
    }

    let dest = models_dir.join(&name);
    if dest.exists() {
        eprintln!(
            "hive models import: {} is already in {}",
            dest.file_name().unwrap_or_default().to_string_lossy(),
            models_dir.display()
        );
        return Ok(1);
    }
    let how = if copy {
        fs::copy(&source, &dest)
            .map_err(|error| format!("hive models import: copy failed: {error}"))?;
        "copy"
    } else if fs::hard_link(&source, &dest).is_ok() {
        "hardlink"
    } else if symlink(&source, &dest).is_ok() {
        "symlink"
    } else {
        fs::copy(&source, &dest)
            .map_err(|error| format!("hive models import: copy failed: {error}"))?;
        "copy"
    };
    let payload = {
        let mut object = Map::new();
        object.insert(
            "imported".into(),
            Value::String(name.to_string_lossy().into_owned()),
        );
        object.insert(
            "path".into(),
            Value::String(dest.to_string_lossy().into_owned()),
        );
        object.insert("linked".into(), Value::Bool(how != "copy"));
        object.insert("how".into(), Value::String(how.into()));
        Value::Object(object)
    };
    if as_json {
        print!("{}", json::pretty(&payload));
    } else {
        println!("imported {} · {how}", name.to_string_lossy());
        println!("  {}", dest.display());
    }
    Ok(0)
}

fn import_dir(paths: &Paths, source: &Path, as_json: bool) -> Cmd {
    let source = source
        .canonicalize()
        .unwrap_or_else(|_| source.to_path_buf());
    let ggufs = hive_models::ops::walk_gguf(&source);
    if ggufs.is_empty() {
        eprintln!(
            "hive models import: no .gguf files under {}",
            source.display()
        );
        return Ok(1);
    }
    paths
        .ensure_models_dir()
        .map_err(|error| format!("hive models import: {error}"))?;
    let models_dir = paths
        .models_dir
        .canonicalize()
        .unwrap_or_else(|_| paths.models_dir.clone());

    if source.starts_with(&models_dir) {
        let imported: Vec<Value> = ggufs
            .iter()
            .take(50)
            .map(|path| {
                Value::String(
                    path.strip_prefix(&models_dir)
                        .unwrap_or(path)
                        .to_string_lossy()
                        .into_owned(),
                )
            })
            .collect();
        let payload = {
            let mut object = Map::new();
            object.insert("imported".into(), Value::Array(imported));
            object.insert("linked".into(), Value::Bool(false));
            object.insert(
                "models_dir".into(),
                Value::String(models_dir.to_string_lossy().into_owned()),
            );
            Value::Object(object)
        };
        print_import(&payload, as_json, &source);
        return Ok(0);
    }

    let link_name = source
        .file_name()
        .filter(|name| !name.is_empty())
        .map(|name| name.to_os_string())
        .unwrap_or_else(|| "imported".into());
    let mut dest = models_dir.join(&link_name);
    if dest.exists()
        && (dest.is_symlink() || dest.is_dir())
        && dest.canonicalize().ok().as_deref() == Some(source.as_path())
    {
        let imported: Vec<Value> = Library::new(&models_dir)
            .rows()
            .into_iter()
            .filter(|row| {
                row.get("file")
                    .and_then(Value::as_str)
                    .map(|file| file.contains(&*link_name.to_string_lossy()))
                    .unwrap_or(false)
            })
            .filter_map(|row| row.get("file").cloned())
            .collect();
        let payload = {
            let mut object = Map::new();
            object.insert("imported".into(), Value::Array(imported));
            object.insert("linked".into(), Value::Bool(false));
            object.insert(
                "models_dir".into(),
                Value::String(models_dir.to_string_lossy().into_owned()),
            );
            Value::Object(object)
        };
        print_import(&payload, as_json, &source);
        return Ok(0);
    }
    if dest.exists() {
        let base = link_name.to_string_lossy().into_owned();
        let mut n = 2;
        loop {
            dest = models_dir.join(format!("{base}-{n}"));
            if !dest.exists() {
                break;
            }
            n += 1;
        }
    }
    symlink(&source, &dest).map_err(|error| {
        format!(
            "hive models import: failed to link {} -> {}: {error}",
            source.display(),
            dest.display()
        )
    })?;
    let mut imported: Vec<PathBuf> = hive_models::ops::walk_gguf(&dest);
    imported.sort();
    let imported: Vec<Value> = imported
        .iter()
        .map(|path| {
            Value::String(
                path.strip_prefix(&models_dir)
                    .unwrap_or(path)
                    .to_string_lossy()
                    .into_owned(),
            )
        })
        .collect();
    let payload = {
        let mut object = Map::new();
        object.insert("imported".into(), Value::Array(imported));
        object.insert(
            "linked".into(),
            Value::String(dest.to_string_lossy().into_owned()),
        );
        object.insert(
            "models_dir".into(),
            Value::String(models_dir.to_string_lossy().into_owned()),
        );
        Value::Object(object)
    };
    print_import(&payload, as_json, &source);
    Ok(0)
}

fn print_import(payload: &Value, as_json: bool, source: &Path) {
    if as_json {
        print!("{}", json::pretty(payload));
        return;
    }
    println!(
        "linked {}",
        source.file_name().unwrap_or_default().to_string_lossy()
    );
    for item in payload
        .get("imported")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .take(20)
    {
        if let Some(item) = item.as_str() {
            println!("  {item}");
        }
    }
}

pub fn remove(paths: &Paths, file: &str, yes: bool, as_json: bool) -> Cmd {
    if !yes {
        eprintln!("hive models remove: refusing to delete without --yes");
        return Ok(2);
    }
    paths
        .ensure_models_dir()
        .map_err(|error| format!("hive models remove: {error}"))?;
    let root = paths
        .models_dir
        .canonicalize()
        .unwrap_or_else(|_| paths.models_dir.clone());
    let candidate = root.join(file);
    let resolved = resolve_path(&candidate);
    if resolved != root && !resolved.starts_with(&root) {
        eprintln!("hive models remove: file must stay inside the models directory");
        return Ok(1);
    }
    if !resolved.is_file() {
        eprintln!("hive models remove: no such model file: {file}");
        return Ok(1);
    }
    fs::remove_file(&resolved)
        .map_err(|error| format!("hive models remove: cannot delete: {error}"))?;
    let payload = {
        let mut object = Map::new();
        object.insert("ok".into(), Value::Bool(true));
        object.insert(
            "deleted".into(),
            Value::String(resolved.to_string_lossy().into_owned()),
        );
        Value::Object(object)
    };
    if as_json {
        print!("{}", json::pretty(&payload));
    } else {
        println!("removed {}", resolved.display());
    }
    Ok(0)
}

/// `Path.resolve()`-style canonicalisation: resolve the existing parent and
/// append the final component when the target itself does not exist.
fn resolve_path(path: &Path) -> PathBuf {
    if let Ok(resolved) = path.canonicalize() {
        return resolved;
    }
    let Some(parent) = path.parent() else {
        return lexical(path);
    };
    match parent.canonicalize() {
        Ok(parent) => parent.join(path.file_name().unwrap_or_default()),
        Err(_) => lexical(path),
    }
}

fn lexical(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn readable(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let Ok(c_path) = std::ffi::CString::new(path.as_os_str().as_bytes()) else {
            return false;
        };
        unsafe { libc::access(c_path.as_ptr(), libc::R_OK) == 0 }
    }
    #[cfg(not(unix))]
    {
        path.is_dir()
    }
}

fn symlink(source: &Path, dest: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(source, dest)
    }
    #[cfg(not(unix))]
    {
        let _ = (source, dest);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "symlinks are not implemented on this platform",
        ))
    }
}
