//! `hive backends` and `hive backends list`.

use hive_core::backends::Manifest;
use hive_core::json;
use hive_core::paths::Paths;

type Cmd = Result<u8, String>;

pub fn list(paths: &Paths, as_json: bool) -> Cmd {
    let manifest = Manifest::load(paths.manifest.as_deref(), &paths.home)
        .map_err(|error| format!("hive backends: {error}"))?;
    let rows = manifest.rows();
    if as_json {
        print!("{}", json::pretty(&serde_json::Value::Array(rows)));
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
