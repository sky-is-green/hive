//! Native daemon OpenAPI document.
//!
//! The Rust routes are merged with the Python sidecar's `/openapi.json` at
//! request time, so the published spec still lists every proxied route. Native
//! operations win when both sides define the same method.
//!
//! The schemas are intentionally loose (`additionalProperties: true`) where
//! the wire shape is the frozen JSON contract; tightening them is tracked in
//! the rewrite caveat register.

use serde_json::{Map, Value, json};

use crate::OWNER_HEADER;

fn operation(summary: &str, tag: &str) -> Value {
    json!({
        "summary": summary,
        "tags": [tag],
        "responses": {
            "200": {"description": "successful response"},
            "4xx": {"description": "request error", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
            "5xx": {"description": "daemon or sidecar error", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
        },
        "x-hive-owner": "rust",
        "x-hive-response-headers": {
            OWNER_HEADER: {"schema": {"type": "string", "enum": ["rust", "python"]}},
            "x-hive-request-id": {"schema": {"type": "string"}},
        },
    })
}

fn path_item(operations: &[(&str, Value)]) -> Value {
    let mut object = Map::new();
    for (method, operation) in operations {
        object.insert((*method).into(), operation.clone());
    }
    Value::Object(object)
}

pub fn document() -> Value {
    let mut paths = Map::new();
    let mut add = |path: &str, operations: &[(&str, Value)]| {
        paths.insert(path.into(), path_item(operations));
    };

    add(
        "/health",
        &[("get", operation("Daemon and sidecar liveness", "health"))],
    );
    add(
        "/v1/setup/status",
        &[(
            "get",
            operation("Windows setup status (410 on Linux)", "health"),
        )],
    );
    add(
        "/v1/version",
        &[(
            "get",
            operation("Hive, Python, and platform versions", "meta"),
        )],
    );
    add(
        "/v1/config",
        &[(
            "get",
            operation("Resolved paths and installed backends", "meta"),
        )],
    );

    add(
        "/v1/stacks",
        &[("get", operation("List saved stacks", "stacks"))],
    );
    add(
        "/v1/stacks/status",
        &[("get", operation("Applied stack and tier status", "stacks"))],
    );
    add(
        "/v1/stacks/{name}",
        &[
            ("get", operation("Read one saved stack", "stacks")),
            (
                "put",
                operation("Create or replace one saved stack", "stacks"),
            ),
            ("delete", operation("Delete one saved stack", "stacks")),
        ],
    );
    add(
        "/v1/stacks/{name}/validate",
        &[("post", operation("Validate a saved stack", "stacks"))],
    );
    add(
        "/v1/stacks/{name}/apply",
        &[("post", operation("Apply a saved stack", "stacks"))],
    );
    add(
        "/v1/stacks/{name}/unload",
        &[("post", operation("Unload a saved stack", "stacks"))],
    );

    for (path, summary) in [
        ("/v1/server/status", "Loaded instances and hardware"),
        ("/v1/server/log", "Recent engine logs"),
        ("/v1/server/memory", "Daemon memory and thread counts"),
        (
            "/v1/server/metrics",
            "Prometheus gauges from the first engine",
        ),
        ("/v1/server/fit", "Where a model and its KV cache fit"),
    ] {
        add(path, &[("get", operation(summary, "server"))]);
    }
    for (path, summary) in [
        ("/v1/server/start", "Start a one-tier model server"),
        ("/v1/server/stop", "Stop one instance or the whole fleet"),
        ("/v1/server/unload", "Unload one instance"),
        ("/v1/server/unload-loaded", "Unload every loaded model"),
    ] {
        add(path, &[("post", operation(summary, "server"))]);
    }

    add(
        "/v1/models/local",
        &[
            ("get", operation("List the local GGUF library", "models")),
            ("delete", operation("Delete one local GGUF", "models")),
        ],
    );
    add(
        "/v1/models/local/path",
        &[(
            "post",
            operation("Switch and persist the library folder", "models"),
        )],
    );
    add(
        "/v1/models/local/import-path",
        &[("post", operation("Link an external model folder", "models"))],
    );
    add(
        "/v1/models/hub",
        &[("get", operation("Search Hugging Face GGUF repos", "models"))],
    );
    add(
        "/v1/models/hub/files/{repo}",
        &[("get", operation("List GGUF files in a hub repo", "models"))],
    );
    add(
        "/v1/models/hub/download",
        &[("post", operation("Queue a hub download", "models"))],
    );
    add(
        "/v1/models/hub/downloads",
        &[("get", operation("List download jobs", "models"))],
    );

    add(
        "/v1/runs",
        &[("get", operation("List run bundles", "runs"))],
    );
    add(
        "/v1/report/{run_dir}",
        &[("get", operation("Read a run report bundle", "runs"))],
    );
    add(
        "/openapi.json",
        &[(
            "get",
            operation("This document, merged with the sidecar", "meta"),
        )],
    );

    json!({
        "openapi": "3.1.0",
        "info": {
            "title": "Hive daemon API",
            "version": hive_core::version(),
            "description": "Native Rust routes plus the Python sidecar routes merged at request time.",
        },
        "paths": Value::Object(paths),
        "components": {
            "schemas": {
                "Error": {
                    "type": "object",
                    "required": ["detail"],
                    "properties": {"detail": {"type": "string"}},
                },
                "Health": {
                    "type": "object",
                    "required": ["ok"],
                    "properties": {
                        "ok": {"type": "boolean"},
                        "conversations": {"type": "integer"},
                        "error": {"type": "string"},
                        "sidecar": {
                            "type": "object",
                            "properties": {
                                "managed": {"type": "boolean"},
                                "url": {"type": "string"},
                                "pid": {"type": ["integer", "null"]},
                            },
                        },
                    },
                },
                "Stack": {
                    "type": "object",
                    "additionalProperties": true,
                    "required": ["name", "version", "tiers"],
                    "properties": {
                        "name": {"type": "string"},
                        "version": {"type": "integer"},
                        "tiers": {"type": "array", "items": {"type": "object", "additionalProperties": true}},
                        "routing": {"type": "object", "additionalProperties": true},
                    },
                },
                "Instance": {
                    "type": "object",
                    "additionalProperties": true,
                    "properties": {
                        "key": {"type": "string"},
                        "running": {"type": "boolean"},
                        "healthy": {"type": "boolean"},
                        "port": {"type": "integer"},
                        "base_url": {"type": "string"},
                        "model": {"type": "string"},
                        "pid": {"type": ["integer", "null"]},
                    },
                },
                "DownloadJob": {
                    "type": "object",
                    "additionalProperties": true,
                    "properties": {
                        "key": {"type": "string"},
                        "repo": {"type": "string"},
                        "filename": {"type": "string"},
                        "state": {"type": "string", "enum": ["queued", "downloading", "done", "error"]},
                        "error": {"type": "string"},
                        "path": {"type": ["string", "null"]},
                        "elapsed_s": {"type": "number"},
                    },
                },
            },
        },
    })
}

/// Merge a sidecar document into the native one. Native operations win.
pub fn merge(native: &mut Value, sidecar: Value) -> bool {
    let Some(remote_paths) = sidecar.get("paths").and_then(Value::as_object) else {
        return false;
    };
    if native.get("paths").and_then(Value::as_object).is_none() {
        return false;
    }
    let paths = native
        .get_mut("paths")
        .and_then(Value::as_object_mut)
        .expect("paths object");
    for (path, remote_item) in remote_paths {
        let entry = paths
            .entry(path.clone())
            .or_insert_with(|| Value::Object(Map::new()));
        let Some(entry) = entry.as_object_mut() else {
            continue;
        };
        if let Some(remote_item) = remote_item.as_object() {
            for (method, operation) in remote_item {
                entry
                    .entry(method.clone())
                    .or_insert_with(|| operation.clone());
            }
        }
    }
    if let (Some(remote), Some(local)) = (
        sidecar.get("components").and_then(Value::as_object),
        native.get_mut("components").and_then(Value::as_object_mut),
    ) && let Some(remote_schemas) = remote.get("schemas").and_then(Value::as_object)
        && let Some(local_schemas) = local.get_mut("schemas").and_then(Value::as_object_mut)
    {
        for (name, schema) in remote_schemas {
            local_schemas
                .entry(name.clone())
                .or_insert_with(|| schema.clone());
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_document_lists_the_owned_routes() {
        let document = document();
        let paths = document["paths"].as_object().unwrap();
        for path in [
            "/health",
            "/v1/stacks",
            "/v1/stacks/{name}/apply",
            "/v1/server/start",
            "/v1/models/hub/download",
            "/v1/runs",
            "/openapi.json",
        ] {
            assert!(paths.contains_key(path), "{path}");
        }
        assert_eq!(document["openapi"], "3.1.0");
    }

    #[test]
    fn merge_keeps_native_operations_and_adds_proxied_ones() {
        let mut native = document();
        let sidecar = json!({
            "paths": {
                "/v1/version": {"get": {"summary": "python version"}},
                "/v1/echo": {"post": {"summary": "echo"}},
            },
            "components": {"schemas": {"Extra": {"type": "object"}}},
        });
        assert!(merge(&mut native, sidecar));
        assert_eq!(
            native["paths"]["/v1/version"]["get"]["summary"],
            "Hive, Python, and platform versions"
        );
        assert_eq!(native["paths"]["/v1/echo"]["post"]["summary"], "echo");
        assert!(native["components"]["schemas"].get("Extra").is_some());
    }
}
