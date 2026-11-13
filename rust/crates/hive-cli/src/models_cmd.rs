//! `hive models dir/import/remove`.

use std::fs;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use hive_core::json;
use hive_core::paths::{Paths, expand};
use hive_models::Library;
use serde_json::{Map, Value};

type Cmd = Result<u8, String>;

const HF_API: &str = "https://huggingface.co/api";

pub fn search(query: &str, limit: i64, as_json: bool) -> Cmd {
    let mut request = ureq::get(&format!("{HF_API}/models"))
        .query("filter", "gguf")
        .query("sort", "downloads")
        .query("limit", &limit.max(1).to_string())
        .timeout(Duration::from_secs(20));
    if !query.trim().is_empty() {
        request = request.query("search", query);
    }
    let response = request
        .call()
        .map_err(|error| format!("hive models search: {error}"))?;
    let text = response
        .into_string()
        .map_err(|error| format!("hive models search: {error}"))?;
    let data: Value =
        serde_json::from_str(&text).map_err(|error| format!("hive models search: {error}"))?;
    let items = data.as_array().cloned().unwrap_or_default();
    let rows: Vec<Value> = items
        .into_iter()
        .map(|entry| {
            let tags: Vec<String> = entry
                .get("tags")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect();
            let base_model = tags
                .iter()
                .find_map(|tag| tag.strip_prefix("base_model:"))
                .unwrap_or("")
                .to_owned();
            let license = tags
                .iter()
                .find_map(|tag| tag.strip_prefix("license:"))
                .unwrap_or("")
                .to_owned();
            let short_tags: Vec<Value> = tags
                .iter()
                .filter(|tag| {
                    !tag.starts_with("base_model:")
                        && !tag.starts_with("license:")
                        && !tag.starts_with("gguf:")
                })
                .take(8)
                .map(|tag| Value::String(tag.clone()))
                .collect();
            let mut row = Map::new();
            row.insert(
                "repo".into(),
                Value::String(
                    entry
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_owned(),
                ),
            );
            row.insert(
                "downloads".into(),
                Value::from(entry.get("downloads").and_then(Value::as_i64).unwrap_or(0)),
            );
            row.insert(
                "likes".into(),
                Value::from(entry.get("likes").and_then(Value::as_i64).unwrap_or(0)),
            );
            row.insert(
                "last_modified".into(),
                Value::String(short_date(entry.get("lastModified"))),
            );
            row.insert(
                "created_at".into(),
                Value::String(short_date(entry.get("createdAt"))),
            );
            row.insert(
                "pipeline_tag".into(),
                Value::String(
                    entry
                        .get("pipeline_tag")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_owned(),
                ),
            );
            row.insert("base_model".into(), Value::String(base_model));
            row.insert("license".into(), Value::String(license));
            row.insert("tags".into(), Value::Array(short_tags));
            Value::Object(row)
        })
        .collect();
    if as_json {
        print!("{}", json::pretty(&Value::Array(rows)));
        return Ok(0);
    }
    println!(
        "models · search  {}",
        if query.trim().is_empty() {
            "popular"
        } else {
            query
        }
    );
    if rows.is_empty() {
        println!("  no GGUF repos matched");
        return Ok(0);
    }
    for row in &rows {
        println!(
            "  {:<48} {:>10} {:>7} {:<12} {}",
            row.get("repo").and_then(Value::as_str).unwrap_or(""),
            row.get("downloads").and_then(Value::as_i64).unwrap_or(0),
            row.get("likes").and_then(Value::as_i64).unwrap_or(0),
            row.get("last_modified")
                .and_then(Value::as_str)
                .unwrap_or(""),
            row.get("license").and_then(Value::as_str).unwrap_or("-"),
        );
    }
    Ok(0)
}

pub fn files(repo: &str, as_json: bool) -> Cmd {
    let response = ureq::get(&format!("{HF_API}/models/{repo}/tree/main"))
        .timeout(Duration::from_secs(20))
        .call()
        .map_err(|error| format!("hive models files: {error}"))?;
    let text = response
        .into_string()
        .map_err(|error| format!("hive models files: {error}"))?;
    let data: Value =
        serde_json::from_str(&text).map_err(|error| format!("hive models files: {error}"))?;
    let rows: Vec<Value> = data
        .as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter(|entry| {
            entry
                .get("path")
                .and_then(Value::as_str)
                .map(|path| path.ends_with(".gguf"))
                .unwrap_or(false)
        })
        .map(|entry| {
            let size = entry.get("size").and_then(Value::as_f64).unwrap_or(0.0)
                / (1024.0 * 1024.0 * 1024.0);
            let mut row = Map::new();
            row.insert(
                "file".into(),
                Value::String(
                    entry
                        .get("path")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_owned(),
                ),
            );
            row.insert(
                "size_gb".into(),
                serde_json::Number::from_f64((size * 100.0).round() / 100.0)
                    .map(Value::Number)
                    .unwrap_or(Value::Null),
            );
            Value::Object(row)
        })
        .collect();
    if as_json {
        print!("{}", json::pretty(&Value::Array(rows)));
        return Ok(0);
    }
    println!("models · files  {repo}");
    if rows.is_empty() {
        println!("  {repo} has no .gguf files");
        return Ok(0);
    }
    for row in &rows {
        println!(
            "  {:<60} {:>8}",
            row.get("file").and_then(Value::as_str).unwrap_or(""),
            format!(
                "{:.2} GiB",
                row.get("size_gb").and_then(Value::as_f64).unwrap_or(0.0)
            )
        );
    }
    Ok(0)
}

fn short_date(value: Option<&Value>) -> String {
    value
        .and_then(Value::as_str)
        .unwrap_or("")
        .chars()
        .take(10)
        .collect()
}

pub fn download(
    paths: &Paths,
    repo_arg: &str,
    file_arg: Option<&str>,
    revision: &str,
    as_json: bool,
) -> Cmd {
    let (repo, mut file) = match file_arg {
        Some(file) => (repo_arg.to_owned(), Some(file.to_owned())),
        None => match repo_arg.split_once(':') {
            Some((repo, file)) if !file.is_empty() => (repo.to_owned(), Some(file.to_owned())),
            _ => (repo_arg.to_owned(), None),
        },
    };
    if file.is_none() {
        match hub_files(&repo) {
            Ok(rows) => {
                if rows.is_empty() {
                    eprintln!("hive models download: {repo} has no .gguf files");
                    return Ok(1);
                }
                if rows.len() > 1 {
                    eprintln!(
                        "hive models download: {repo} has several GGUFs; choose one with --file:"
                    );
                    for row in rows {
                        eprintln!(
                            "  {}  ({} GiB)",
                            row.get("file").and_then(Value::as_str).unwrap_or(""),
                            row.get("size_gb").and_then(Value::as_f64).unwrap_or(0.0)
                        );
                    }
                    return Ok(2);
                }
                file = rows[0]
                    .get("file")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
            }
            Err(error) => {
                eprintln!("hive models download: cannot list {repo}: {error}");
                return Ok(1);
            }
        }
    }
    let Some(file) = file else {
        eprintln!("hive models download: {repo} has no .gguf files");
        return Ok(1);
    };
    if file.split('/').any(|part| part == "..") || Path::new(&file).is_absolute() {
        eprintln!("hive models download: file must stay inside the models directory");
        return Ok(1);
    }
    paths
        .ensure_models_dir()
        .map_err(|error| format!("hive models download: {error}"))?;
    let dest = paths.models_dir.join(&file);
    if dest.is_file() && dest.metadata().map(|meta| meta.len() > 0).unwrap_or(false) {
        return finish_download(&repo, &file, &dest, as_json);
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|error| format!("hive models download: {error}"))?;
    }
    let part = PathBuf::from(format!("{}.part", dest.to_string_lossy()));
    let mut resume = fs::metadata(&part).map(|meta| meta.len()).unwrap_or(0);
    let url = format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        encode_path(&repo),
        encode_component(revision),
        encode_path(&file)
    );
    let token = std::env::var("HF_TOKEN")
        .ok()
        .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok());
    let mut request = ureq::get(&url).timeout(Duration::from_secs(3600));
    if let Some(token) = &token {
        request = request.set("Authorization", &format!("Bearer {token}"));
    }
    if resume > 0 {
        request = request.set("Range", &format!("bytes={resume}-"));
    }
    let response = match request.call() {
        Ok(response) => response,
        Err(error) => {
            eprintln!("hive models download: {repo}:{file}: {error}");
            return Ok(1);
        }
    };
    let status = response.status();
    if status != 200 && status != 206 {
        eprintln!("hive models download: {repo}:{file}: HTTP {status}");
        return Ok(1);
    }
    if status == 200 {
        resume = 0;
    }
    let total = response
        .header("content-length")
        .and_then(|value| value.parse::<u64>().ok())
        .map(|length| length + resume)
        .unwrap_or(0);
    let mut handle = match fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(resume > 0)
        .truncate(resume == 0)
        .open(&part)
    {
        Ok(handle) => handle,
        Err(error) => {
            eprintln!("hive models download: {repo}:{file}: {error}");
            return Ok(1);
        }
    };
    let mut reader = response.into_reader();
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut done = resume;
    let mut last_pct = -1i64;
    loop {
        let read = match reader.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => read,
            Err(error) => {
                eprintln!("hive models download: {repo}:{file}: {error}");
                return Ok(1);
            }
        };
        if let Err(error) = handle.write_all(&buffer[..read]) {
            eprintln!("hive models download: {repo}:{file}: {error}");
            return Ok(1);
        }
        done += read as u64;
        if !as_json && total > 0 {
            let pct = (done.saturating_mul(100) / total) as i64;
            if pct != last_pct {
                last_pct = pct;
                eprint!(
                    "\r  {pct:3}%  {:.1} / {:.1} MiB",
                    done as f64 / 1_048_576.0,
                    total as f64 / 1_048_576.0
                );
                let _ = std::io::stderr().flush();
            }
        }
    }
    if !as_json && total > 0 {
        eprintln!();
    }
    if let Err(error) = handle.sync_all() {
        eprintln!("hive models download: {repo}:{file}: {error}");
        return Ok(1);
    }
    drop(handle);
    if let Err(error) = fs::rename(&part, &dest) {
        eprintln!("hive models download: {repo}:{file}: {error}");
        return Ok(1);
    }
    finish_download(&repo, &file, &dest, as_json)
}

fn finish_download(repo: &str, file: &str, dest: &Path, as_json: bool) -> Cmd {
    let size =
        dest.metadata().map(|meta| meta.len()).unwrap_or(0) as f64 / (1024.0 * 1024.0 * 1024.0);
    let size_gb = (size * 100.0).round() / 100.0;
    let payload = {
        let mut object = Map::new();
        object.insert("repo".into(), Value::String(repo.into()));
        object.insert("file".into(), Value::String(file.into()));
        object.insert(
            "path".into(),
            Value::String(dest.to_string_lossy().into_owned()),
        );
        object.insert(
            "size_gb".into(),
            serde_json::Number::from_f64(size_gb)
                .map(Value::Number)
                .unwrap_or(Value::Null),
        );
        Value::Object(object)
    };
    if as_json {
        print!("{}", json::pretty(&payload));
    } else {
        println!("downloaded {repo}:{file}");
        println!("  path  {}", dest.display());
        println!("  size  {size_gb:.2} GiB");
    }
    Ok(0)
}

fn hub_files(repo: &str) -> Result<Vec<Value>, String> {
    let response = ureq::get(&format!("{HF_API}/models/{}/tree/main", encode_path(repo)))
        .timeout(Duration::from_secs(20))
        .call()
        .map_err(|error| error.to_string())?;
    let text = response.into_string().map_err(|error| error.to_string())?;
    let data: Value = serde_json::from_str(&text).map_err(|error| error.to_string())?;
    Ok(data
        .as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter(|entry| {
            entry
                .get("path")
                .and_then(Value::as_str)
                .map(|path| path.ends_with(".gguf"))
                .unwrap_or(false)
        })
        .map(|entry| {
            let size = entry.get("size").and_then(Value::as_f64).unwrap_or(0.0)
                / (1024.0 * 1024.0 * 1024.0);
            let mut row = Map::new();
            row.insert(
                "file".into(),
                Value::String(
                    entry
                        .get("path")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_owned(),
                ),
            );
            row.insert(
                "size_gb".into(),
                serde_json::Number::from_f64((size * 100.0).round() / 100.0)
                    .map(Value::Number)
                    .unwrap_or(Value::Null),
            );
            Value::Object(row)
        })
        .collect())
}

fn encode_path(path: &str) -> String {
    path.split('/')
        .map(encode_component)
        .collect::<Vec<_>>()
        .join("/")
}

fn encode_component(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

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
