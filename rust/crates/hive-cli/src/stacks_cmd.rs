//! `hive stacks show/validate`.

use std::path::{Path, PathBuf};

use hive_core::backends::Manifest;
use hive_core::json;
use hive_core::paths::{Paths, expand};
use hive_core::py_repr;
use hive_core::stacks::{check_stack_name, parse_stack, validate_shape};
use hive_models::resolve_model;
use serde::Serialize;
use serde_json::{Map, Value};

type Cmd = Result<u8, String>;

#[derive(Serialize)]
struct StackOut {
    name: String,
    version: i64,
    tiers: Vec<TierOut>,
    routing: Value,
}

#[derive(Serialize)]
struct TierOut {
    role: String,
    repo: String,
    file: String,
    ctx: i64,
    ngl: i64,
    backend: String,
    cache_k: String,
    cache_v: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    spec: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mmproj: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pin: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ts: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    engine_args: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    endpoint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    caps: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    launch: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expect_model: Option<String>,
}

pub fn show(paths: &Paths, target: &str) -> Cmd {
    let stack = load_stack(target, &paths.stacks_dir)?;
    let out = StackOut {
        name: text(stack.get("name")),
        version: stack.get("version").and_then(Value::as_i64).unwrap_or(1),
        tiers: stack
            .get("tiers")
            .and_then(Value::as_array)
            .map(|tiers| tiers.iter().map(tier_out).collect())
            .unwrap_or_default(),
        routing: stack.get("routing").cloned().unwrap_or_else(|| {
            let mut routing = Map::new();
            routing.insert("workers_as".into(), Value::String("subagent".into()));
            Value::Object(routing)
        }),
    };
    print!("{}", json::to_string_ordered(&out));
    println!();
    Ok(0)
}

fn tier_out(tier: &Value) -> TierOut {
    TierOut {
        role: text(tier.get("role")),
        repo: text(tier.get("repo")),
        file: text(tier.get("file")),
        ctx: tier.get("ctx").and_then(Value::as_i64).unwrap_or(8192),
        ngl: tier.get("ngl").and_then(Value::as_i64).unwrap_or(99),
        backend: optional_text(tier.get("backend")).unwrap_or_else(|| "vulkan".into()),
        cache_k: optional_text(tier.get("cache_k")).unwrap_or_else(|| "q8_0".into()),
        cache_v: optional_text(tier.get("cache_v")).unwrap_or_else(|| "q8_0".into()),
        spec: tier.get("spec").filter(|value| !value.is_null()).cloned(),
        mmproj: optional_text(tier.get("mmproj")),
        pin: optional_text(tier.get("pin")),
        ts: optional_text(tier.get("ts")),
        engine_args: tier
            .get("engine_args")
            .filter(|value| !value.is_null())
            .cloned(),
        endpoint: optional_text(tier.get("endpoint")),
        caps: tier.get("caps").filter(|value| !value.is_null()).cloned(),
        launch: tier.get("launch").filter(|value| !value.is_null()).cloned(),
        expect_model: optional_text(tier.get("expect_model")),
    }
}

fn text(value: Option<&Value>) -> String {
    value
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default()
        .to_owned()
}

fn optional_text(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_owned)
}

pub fn validate(paths: &Paths, target: &str, strict: bool, as_json: bool) -> Cmd {
    let stack = load_stack(target, &paths.stacks_dir)?;
    let stack_name = text(stack.get("name"));
    let problems = validate_shape(&stack);
    let (errors, warnings, tiers) = if problems.is_empty() {
        tier_validation(paths, &stack)?
    } else {
        (problems.clone(), Vec::new(), Vec::new())
    };
    let ok = errors.is_empty() && !(strict && !warnings.is_empty());
    if as_json {
        let mut payload = Map::new();
        payload.insert("ok".into(), Value::Bool(ok));
        payload.insert("stack".into(), Value::String(stack_name.clone()));
        payload.insert(
            "errors".into(),
            Value::Array(errors.iter().cloned().map(Value::String).collect()),
        );
        payload.insert(
            "warnings".into(),
            Value::Array(warnings.iter().cloned().map(Value::String).collect()),
        );
        payload.insert("tiers".into(), Value::Array(tiers.clone()));
        print!("{}", json::pretty(&Value::Object(payload)));
        return Ok(if ok { 0 } else { 1 });
    }
    println!("stacks · validate  {stack_name}");
    println!(
        "  {} tier(s)",
        stack
            .get("tiers")
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or(0)
    );
    for problem in &problems {
        println!("  ✘ {problem}");
    }
    for row in &tiers {
        let role = row.get("role").and_then(Value::as_str).unwrap_or("");
        let detail = row.get("detail").and_then(Value::as_str).unwrap_or("");
        let mark = match row.get("status").and_then(Value::as_str).unwrap_or("") {
            "ok" => "✔",
            "warn" => "!",
            _ => "✘",
        };
        println!("  {mark} {role:<8} {detail}");
    }
    for error in &errors {
        if !problems.contains(error) {
            println!("  ✘ {error}");
        }
    }
    for warning in &warnings {
        println!("  ! {warning}");
    }
    println!("  {}", if ok { "ok" } else { "validation failed" });
    Ok(if ok { 0 } else { 1 })
}

type TierValidation = (Vec<String>, Vec<String>, Vec<Value>);

fn tier_validation(paths: &Paths, stack: &Value) -> Result<TierValidation, String> {
    let manifest = Manifest::load(paths.manifest.as_deref(), &paths.home)
        .map_err(|error| format!("backend manifest: {error}"))?;
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    let mut details = Vec::new();
    let tiers = stack
        .get("tiers")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for tier in tiers {
        let role = text(tier.get("role"));
        let mut status = "ok";
        let mut detail = String::new();
        let mut launch_ok = true;
        if let Some(launch) = tier.get("launch").filter(|value| value.is_object()) {
            let raw_repo = launch
                .get("repo")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let repo_text = expand_vars(raw_repo);
            let repo = expand(Path::new(&repo_text));
            if repo_text.contains('$') {
                warnings.push(format!(
                    "tier {}: launch repo {raw_repo} is unresolved until the environment is set",
                    py_repr(&role)
                ));
            } else if !repo_text.is_empty() && !repo.exists() {
                launch_ok = false;
                errors.push(format!(
                    "tier {}: launch repo {} does not exist",
                    py_repr(&role),
                    repo.display()
                ));
                status = "fail";
                detail = "launch repo missing".into();
            }
        }
        if let Some(endpoint) = optional_text(tier.get("endpoint")) {
            if launch_ok {
                detail = format!("endpoint {endpoint}");
            }
            details.push(row(&role, status, &detail));
            continue;
        }

        let backend_name = optional_text(tier.get("backend")).unwrap_or_else(|| "vulkan".into());
        let mut backend_ok = true;
        match manifest.resolve(&backend_name) {
            Ok(backend) if backend.installed => {}
            Ok(backend) => {
                backend_ok = false;
                errors.push(format!(
                    "tier {}: backend {} is not installed ({})",
                    py_repr(&role),
                    py_repr(&backend_name),
                    backend.binary.display()
                ));
                status = "fail";
                detail = "backend missing".into();
            }
            Err(error) => {
                backend_ok = false;
                errors.push(format!("tier {}: {error}", py_repr(&role)));
                status = "fail";
                detail = error.to_string();
            }
        }
        if backend_ok {
            let file = text(tier.get("file"));
            let model = expand(Path::new(&file));
            let found = model.is_file() || resolve_model(&file, &paths.models_dir).is_some();
            let repo = text(tier.get("repo"));
            if found {
                detail = format!("{backend_name} · model ok");
            } else if !repo.is_empty() && repo != "local" {
                detail = format!("{backend_name} · remote (downloaded on serve)");
                warnings.push(format!(
                    "tier {}: model is not local; it will be fetched from {repo} on first serve",
                    py_repr(&role)
                ));
                status = "warn";
            } else {
                errors.push(format!(
                    "tier {}: model {} is not in {}",
                    py_repr(&role),
                    py_repr(&file),
                    paths.models_dir.display()
                ));
                status = "fail";
                detail = "model missing".into();
            }
        }
        details.push(row(&role, status, &detail));
    }
    Ok((errors, warnings, details))
}

fn row(role: &str, status: &str, detail: &str) -> Value {
    let mut object = Map::new();
    object.insert("role".into(), Value::String(role.into()));
    object.insert("status".into(), Value::String(status.into()));
    object.insert("detail".into(), Value::String(detail.into()));
    Value::Object(object)
}

/// Python's `os.path.expandvars` for the `$VAR`/`${VAR}` forms used in stacks.
fn expand_vars(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut chars = value.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '$' {
            out.push(ch);
            continue;
        }
        if chars.peek() == Some(&'{') {
            chars.next();
            let mut name = String::new();
            for next in chars.by_ref() {
                if next == '}' {
                    break;
                }
                name.push(next);
            }
            if let Some(value) = std::env::var_os(&name) {
                out.push_str(&value.to_string_lossy());
            } else {
                out.push_str("${");
                out.push_str(&name);
                out.push('}');
            }
            continue;
        }
        let mut name = String::new();
        while let Some(next) = chars.peek().copied() {
            if next.is_ascii_alphanumeric() || next == '_' {
                name.push(next);
                chars.next();
            } else {
                break;
            }
        }
        if name.is_empty() {
            out.push('$');
        } else if let Some(value) = std::env::var_os(&name) {
            out.push_str(&value.to_string_lossy());
        } else {
            out.push('$');
            out.push_str(&name);
        }
    }
    out
}

fn load_stack(target: &str, stacks_dir: &Path) -> Result<Value, String> {
    let path = stack_path(target, stacks_dir)?;
    let text = std::fs::read_to_string(&path)
        .map_err(|error| format!("cannot read stack at {}: {error}", path.display()))?;
    let data: Value = serde_json::from_str(&text)
        .map_err(|error| format!("stack file {} is not valid JSON: {error}", path.display()))?;
    let stack = parse_stack(&data).map_err(|error| error.to_string())?;
    let problems = validate_shape(&stack);
    if !problems.is_empty() {
        return Err(format!(
            "stack {} has an invalid shape: {}",
            py_repr(&text_value(stack.get("name"))),
            problems.join("; ")
        ));
    }
    Ok(stack)
}

fn stack_path(target: &str, stacks_dir: &Path) -> Result<PathBuf, String> {
    let candidate = PathBuf::from(target);
    if target.ends_with(".json") || target.contains('/') || candidate.is_file() {
        if !candidate.is_file() {
            return Err(format!("no stack file at {}", candidate.display()));
        }
        return Ok(candidate);
    }
    let name = check_stack_name(target)?;
    let path = stacks_dir.join(format!("{name}.json"));
    if !path.is_file() {
        return Err(format!("no saved stack at {}", path.display()));
    }
    Ok(path)
}

fn text_value(value: Option<&Value>) -> String {
    value.and_then(Value::as_str).unwrap_or_default().to_owned()
}
