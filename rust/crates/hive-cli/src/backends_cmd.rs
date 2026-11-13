//! `hive backends`, `hive backends list`, and `hive backends fetch`.

use std::fs;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use flate2::read::GzDecoder;
use hive_core::backends::Manifest;
use hive_core::json;
use hive_core::paths::Paths;
use serde_json::{Map, Value};

type Cmd = Result<u8, String>;

const GITHUB_API: &str = "https://api.github.com/repos";
const USER_AGENT: &str = "hivebench-backend-fetch";

pub fn list(paths: &Paths, as_json: bool) -> Cmd {
    let manifest = Manifest::load(paths.manifest.as_deref(), &paths.home)
        .map_err(|error| format!("hive backends: {error}"))?;
    let rows = manifest.rows();
    if as_json {
        print!("{}", json::pretty(&Value::Array(rows)));
        return Ok(0);
    }
    println!("backends");
    for row in rows {
        println!(
            "  {:<10} {:<10} {:<24} {:<48} {}",
            row.get("name").and_then(|v| v.as_str()).unwrap_or(""),
            if row
                .get("installed")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                "installed"
            } else {
                "-"
            },
            row.get("accelerators")
                .and_then(|v| v.as_array())
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|item| item.as_str())
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .unwrap_or_default(),
            row.get("binary").and_then(|v| v.as_str()).unwrap_or(""),
            row.get("label").and_then(|v| v.as_str()).unwrap_or("")
        );
    }
    Ok(0)
}

pub fn fetch(
    paths: &Paths,
    name: Option<&str>,
    tag: Option<&str>,
    force: bool,
    as_json: bool,
) -> Cmd {
    let mut captured = None;
    fetch_capture(paths, name, tag, force, as_json, true, &mut captured)
}

pub fn fetch_capture(
    paths: &Paths,
    name: Option<&str>,
    tag: Option<&str>,
    force: bool,
    as_json: bool,
    print_output: bool,
    captured: &mut Option<Value>,
) -> Cmd {
    let name = name.unwrap_or("vulkan").trim().to_lowercase();
    let manifest = Manifest::load(paths.manifest.as_deref(), &paths.home)
        .map_err(|error| format!("hive backends fetch: {error}"))?;
    let backend = manifest
        .resolve(&name)
        .map_err(|error| format!("hive backends fetch: {error}"))?;
    let target = backend
        .binary
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "hive backends fetch: backend has no directory".to_owned())?;
    if backend.installed && !force {
        let payload = already_payload(&name, &target, &backend.binary);
        *captured = Some(payload.clone());
        if print_output {
            print_fetch(&payload, as_json, &name);
        }
        return Ok(0);
    }
    let release = backend
        .raw
        .get("release")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            format!(
                "hive backends fetch: backend {name:?} has no downloadable release in the manifest"
            )
        })?;
    let repo = release
        .get("repo")
        .and_then(Value::as_str)
        .filter(|repo| !repo.is_empty())
        .ok_or_else(|| format!("hive backends fetch: backend {name:?} release has no repo"))?;
    let wanted_tag = tag.map(str::to_owned).or_else(|| {
        release
            .get("tag")
            .and_then(Value::as_str)
            .map(str::to_owned)
    });
    let (release_tag, assets) = github_release(repo, wanted_tag.as_deref())
        .map_err(|error| format!("hive backends fetch: {error}"))?;
    let key = platform_key();
    let pattern = release
        .get("assets")
        .and_then(Value::as_object)
        .and_then(|assets| assets.get(&key))
        .and_then(Value::as_str)
        .ok_or_else(|| {
            let known = release
                .get("assets")
                .and_then(Value::as_object)
                .map(|assets| assets.keys().cloned().collect::<Vec<_>>().join(", "))
                .unwrap_or_default();
            format!(
                "hive backends fetch: backend {name:?} has no release asset for {key} (known: {known})"
            )
        })?;
    let asset = select_asset(&assets, pattern).ok_or_else(|| {
        format!("hive backends fetch: release has no asset matching {pattern:?} for {key}")
    })?;
    if !as_json {
        eprintln!("resolving {name} for this host");
    }
    let temp = tempfile::Builder::new()
        .prefix("hive-backend-")
        .tempdir()
        .map_err(|error| format!("hive backends fetch: {error}"))?;
    let archive = temp.path().join(&asset.name);
    download(&asset.url, &archive, as_json)
        .map_err(|error| format!("hive backends fetch: {error}"))?;
    let extract = temp.path().join("extract");
    fs::create_dir_all(&extract).map_err(|error| format!("hive backends fetch: {error}"))?;
    extract_archive(&archive, &extract).map_err(|error| format!("hive backends fetch: {error}"))?;
    let binary = find_binary(&extract)
        .ok_or_else(|| "hive backends fetch: release archive has no llama-server".to_owned())?;
    let source = binary
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "hive backends fetch: release binary has no directory".to_owned())?;
    let staging = target.with_file_name(format!(
        ".{}.fetch",
        target.file_name().unwrap_or_default().to_string_lossy()
    ));
    if staging.exists() {
        let _ = fs::remove_dir_all(&staging);
    }
    copy_tree(&source, &staging).map_err(|error| format!("hive backends fetch: {error}"))?;
    let staged_binary = staging.join(binary.file_name().unwrap_or_default());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&staged_binary, fs::Permissions::from_mode(0o755));
    }
    verify(&staged_binary).map_err(|error| format!("hive backends fetch: {error}"))?;
    if target.exists() || target.is_symlink() {
        if !force {
            let _ = fs::remove_dir_all(&staging);
            return Err(format!(
                "hive backends fetch: backend directory {} already exists; use --force to replace it",
                target.display()
            ));
        }
        if target.is_symlink() || target.is_file() {
            let _ = fs::remove_file(&target);
        } else {
            fs::remove_dir_all(&target).map_err(|error| format!("hive backends fetch: {error}"))?;
        }
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent).map_err(|error| format!("hive backends fetch: {error}"))?;
    }
    fs::rename(&staging, &target).map_err(|error| format!("hive backends fetch: {error}"))?;
    let binary = target.join(
        binary
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .as_ref(),
    );
    let payload = {
        let mut object = Map::new();
        object.insert("name".into(), Value::String(name.clone()));
        object.insert(
            "path".into(),
            Value::String(target.to_string_lossy().into_owned()),
        );
        object.insert(
            "binary".into(),
            Value::String(binary.to_string_lossy().into_owned()),
        );
        object.insert("installed".into(), Value::Bool(true));
        object.insert("already".into(), Value::Bool(false));
        object.insert("tag".into(), Value::String(release_tag));
        object.insert("asset".into(), Value::String(asset.name));
        Value::Object(object)
    };
    *captured = Some(payload.clone());
    if print_output {
        print_fetch(&payload, as_json, &name);
    }
    let _ = paths;
    Ok(0)
}

fn already_payload(name: &str, target: &Path, binary: &Path) -> Value {
    let mut object = Map::new();
    object.insert("name".into(), Value::String(name.into()));
    object.insert(
        "path".into(),
        Value::String(target.to_string_lossy().into_owned()),
    );
    object.insert(
        "binary".into(),
        Value::String(binary.to_string_lossy().into_owned()),
    );
    object.insert("installed".into(), Value::Bool(true));
    object.insert("already".into(), Value::Bool(true));
    object.insert("tag".into(), Value::Null);
    object.insert("asset".into(), Value::Null);
    Value::Object(object)
}

fn print_fetch(payload: &Value, as_json: bool, name: &str) {
    if as_json {
        print!("{}", json::pretty(payload));
        return;
    }
    if payload.get("already").and_then(Value::as_bool) == Some(true) {
        println!(
            "{} is already installed at {} (use --force to replace it)",
            name,
            payload.get("path").and_then(Value::as_str).unwrap_or("")
        );
        return;
    }
    println!(
        "installed {} · {}",
        name,
        payload
            .get("tag")
            .and_then(Value::as_str)
            .unwrap_or("release")
    );
    println!(
        "  binary  {}",
        payload.get("binary").and_then(Value::as_str).unwrap_or("")
    );
    println!(
        "  asset   {}",
        payload.get("asset").and_then(Value::as_str).unwrap_or("-")
    );
}

#[derive(Debug)]
struct Asset {
    name: String,
    url: String,
}

fn github_release(repo: &str, tag: Option<&str>) -> Result<(String, Vec<Asset>), String> {
    let url = match tag {
        Some(tag) => format!("{GITHUB_API}/{repo}/releases/tags/{tag}"),
        None => format!("{GITHUB_API}/{repo}/releases?per_page=1"),
    };
    let response = ureq::get(&url)
        .set("User-Agent", USER_AGENT)
        .timeout(Duration::from_secs(30))
        .call()
        .map_err(|error| error.to_string())?;
    let text = response.into_string().map_err(|error| error.to_string())?;
    let data: Value = serde_json::from_str(&text).map_err(|error| error.to_string())?;
    let release = if tag.is_some() {
        data
    } else {
        data.as_array()
            .and_then(|items| items.first())
            .cloned()
            .ok_or_else(|| format!("{repo} has no releases"))?
    };
    if !release.is_object() {
        return Err(format!("unexpected release response for {repo}"));
    }
    let tag = release
        .get("tag_name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    let assets = release
        .get("assets")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|asset| {
            let name = asset.get("name").and_then(Value::as_str)?;
            let url = asset.get("browser_download_url").and_then(Value::as_str)?;
            Some(Asset {
                name: name.to_owned(),
                url: url.to_owned(),
            })
        })
        .collect();
    Ok((tag, assets))
}

fn select_asset(assets: &[Asset], pattern: &str) -> Option<Asset> {
    let mut matches: Vec<&Asset> = assets
        .iter()
        .filter(|asset| glob_match(pattern, &asset.name))
        .collect();
    matches.sort_by(|a, b| a.name.cmp(&b.name));
    matches.last().map(|asset| Asset {
        name: asset.name.clone(),
        url: asset.url.clone(),
    })
}

fn glob_match(pattern: &str, text: &str) -> bool {
    fn go(pattern: &[u8], text: &[u8]) -> bool {
        if pattern.is_empty() {
            return text.is_empty();
        }
        match pattern[0] {
            b'*' => go(&pattern[1..], text) || (!text.is_empty() && go(pattern, &text[1..])),
            b'?' => !text.is_empty() && go(&pattern[1..], &text[1..]),
            byte => !text.is_empty() && byte == text[0] && go(&pattern[1..], &text[1..]),
        }
    }
    go(pattern.as_bytes(), text.as_bytes())
}

fn platform_key() -> String {
    let arch = match std::env::consts::ARCH {
        "x86_64" | "amd64" => "x64",
        "aarch64" | "arm64" => "arm64",
        other => other,
    };
    match std::env::consts::OS {
        "linux" => format!("linux-{arch}"),
        "windows" => format!("win-{arch}"),
        other => format!("{other}-{arch}"),
    }
}

fn download(url: &str, destination: &Path, as_json: bool) -> Result<(), String> {
    let response = ureq::get(url)
        .set("User-Agent", USER_AGENT)
        .timeout(Duration::from_secs(600))
        .call()
        .map_err(|error| error.to_string())?;
    let total = response
        .header("content-length")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0);
    let mut handle = fs::File::create(destination).map_err(|error| error.to_string())?;
    let mut reader = response.into_reader();
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut done = 0u64;
    let mut last_pct = -1i64;
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|error| error.to_string())?;
        if read == 0 {
            break;
        }
        handle
            .write_all(&buffer[..read])
            .map_err(|error| error.to_string())?;
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
    Ok(())
}

fn extract_archive(archive: &Path, destination: &Path) -> Result<(), String> {
    let file = fs::File::open(archive).map_err(|error| error.to_string())?;
    let decoder = GzDecoder::new(file);
    let mut bundle = tar::Archive::new(decoder);
    for entry in bundle.entries().map_err(|error| error.to_string())? {
        let mut entry = entry.map_err(|error| error.to_string())?;
        let path = entry
            .path()
            .map_err(|error| error.to_string())?
            .into_owned();
        if path.is_absolute()
            || path
                .components()
                .any(|component| matches!(component, Component::ParentDir))
        {
            return Err(format!("archive member escapes: {}", path.display()));
        }
        entry
            .unpack_in(destination)
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn find_binary(root: &Path) -> Option<PathBuf> {
    let mut found = Vec::new();
    walk(root, &mut found);
    found.into_iter().find(|path| {
        path.file_name()
            .map(|name| name == "llama-server")
            .unwrap_or(false)
    })
}

fn walk(root: &Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            walk(&path, found);
        } else {
            found.push(path);
        }
    }
    found.sort();
}

fn copy_tree(source: &Path, destination: &Path) -> Result<(), String> {
    fs::create_dir_all(destination).map_err(|error| error.to_string())?;
    for entry in fs::read_dir(source).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let target = destination.join(entry.file_name());
        let file_type = entry.file_type().map_err(|error| error.to_string())?;
        if file_type.is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else if file_type.is_symlink() {
            #[cfg(unix)]
            {
                let link = fs::read_link(entry.path()).map_err(|error| error.to_string())?;
                std::os::unix::fs::symlink(link, &target).map_err(|error| error.to_string())?;
            }
        } else {
            fs::copy(entry.path(), &target).map_err(|error| error.to_string())?;
        }
    }
    Ok(())
}

fn verify(binary: &Path) -> Result<(), String> {
    let output = std::process::Command::new(binary)
        .arg("--version")
        .current_dir(binary.parent().unwrap_or(Path::new(".")))
        .env(
            "LD_LIBRARY_PATH",
            binary
                .parent()
                .map(|parent| parent.to_string_lossy().into_owned())
                .unwrap_or_default(),
        )
        .output()
        .map_err(|error| format!("{} did not run: {error}", binary.display()))?;
    if !output.status.success() {
        let tail = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "{} failed --version (exit {}): {}",
            binary.display(),
            output.status,
            tail.trim()
                .chars()
                .rev()
                .take(400)
                .collect::<String>()
                .chars()
                .rev()
                .collect::<String>()
        ));
    }
    Ok(())
}

pub fn setup(paths: &Paths, no_fetch: bool, as_json: bool) -> Cmd {
    let mut manifest = Manifest::load(paths.manifest.as_deref(), &paths.home)
        .map_err(|error| format!("hive setup: {error}"))?;
    let mut installed: Vec<String> = manifest
        .installed()
        .into_iter()
        .map(str::to_owned)
        .collect();
    let mut fetched = None;
    if installed.is_empty() {
        if no_fetch {
            eprintln!("hive setup: no engine backend installed; run `hive backends fetch vulkan`");
            return Ok(1);
        }
        if !as_json {
            eprintln!("no engine installed; fetching vulkan");
        }
        let mut payload = None;
        match fetch_capture(
            paths,
            Some("vulkan"),
            None,
            false,
            as_json,
            false,
            &mut payload,
        ) {
            Ok(0) => fetched = payload,
            Err(error) => {
                let detail = error
                    .strip_prefix("hive backends fetch: ")
                    .unwrap_or(&error);
                eprintln!("hive setup: cannot fetch vulkan: {detail}");
                return Ok(1);
            }
            Ok(code) => return Ok(code),
        }
        manifest = Manifest::load(paths.manifest.as_deref(), &paths.home)
            .map_err(|error| format!("hive setup: {error}"))?;
        installed = manifest
            .installed()
            .into_iter()
            .map(str::to_owned)
            .collect();
    }
    paths
        .ensure_models_dir()
        .map_err(|error| format!("hive setup: {error}"))?;
    std::fs::create_dir_all(&paths.stacks_dir).map_err(|error| format!("hive setup: {error}"))?;
    paths
        .ensure_serve_dir()
        .map_err(|error| format!("hive setup: {error}"))?;
    let payload = {
        let mut object = Map::new();
        object.insert("ok".into(), Value::Bool(true));
        object.insert("fetched".into(), fetched.unwrap_or(Value::Null));
        object.insert(
            "backends".into(),
            Value::Array(installed.iter().cloned().map(Value::String).collect()),
        );
        object.insert(
            "models_dir".into(),
            Value::String(paths.models_dir.to_string_lossy().into_owned()),
        );
        object.insert(
            "stacks_dir".into(),
            Value::String(paths.stacks_dir.to_string_lossy().into_owned()),
        );
        object.insert(
            "state_dir".into(),
            Value::String(paths.serve_dir.to_string_lossy().into_owned()),
        );
        Value::Object(object)
    };
    if as_json {
        print!("{}", json::pretty(&payload));
        return Ok(0);
    }
    println!("setup");
    println!(
        "  engine: {}",
        if installed.is_empty() {
            "none".to_owned()
        } else {
            installed.join(", ")
        }
    );
    println!("  models  {}", paths.models_dir.display());
    println!("  stacks  {}", paths.stacks_dir.display());
    println!("  state   {}", paths.serve_dir.display());
    println!();
    println!("  next");
    println!("    › hive models download <repo>[:<file>]");
    println!("    › hive serve --model <local.gguf> --cards 1");
    println!("    › hive doctor");
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_matches_llama_release_names() {
        assert!(glob_match(
            "llama-*-bin-ubuntu-x64.tar.gz",
            "llama-b12329-bin-ubuntu-x64.tar.gz"
        ));
        assert!(!glob_match(
            "llama-*-bin-ubuntu-x64.tar.gz",
            "llama-b12329-bin-win-x64.zip"
        ));
    }

    #[test]
    fn selects_the_lexicographically_newest_asset() {
        let assets = vec![
            Asset {
                name: "llama-b100-bin-ubuntu-x64.tar.gz".into(),
                url: "a".into(),
            },
            Asset {
                name: "llama-b200-bin-ubuntu-x64.tar.gz".into(),
                url: "b".into(),
            },
        ];
        let asset = select_asset(&assets, "llama-*-bin-ubuntu-x64.tar.gz").unwrap();
        assert_eq!(asset.name, "llama-b200-bin-ubuntu-x64.tar.gz");
    }
}
