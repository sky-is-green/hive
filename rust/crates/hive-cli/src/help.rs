//! `hive help` and the machine-readable command catalog.
//!
//! The catalog is generated from the live clap command tree, not a checked-in
//! snapshot, so it cannot drift from `--help`. Field names follow the Python
//! CLI's `hive help --json` contract (`kind`, `name`, `flags`, `help`,
//! `nargs`, `required`, `default`, `choices`, `type`).

use clap::CommandFactory;
use clap::builder::ArgAction;
use serde_json::{Map, Value};

use crate::Cli;

pub fn run(topic: Vec<String>, as_json: bool) -> Result<u8, String> {
    let mut command = Cli::command();
    for token in &topic {
        let Some(next) = command.find_subcommand(token).cloned() else {
            eprintln!("hive help: unknown command {:?}", topic.join(" "));
            return Ok(2);
        };
        command = next;
    }
    if as_json {
        let path: Vec<String> = std::iter::once("hive".to_owned())
            .chain(topic.iter().cloned())
            .collect();
        let schema = schema(&command, &path);
        print!("{}", hive_core::json::pretty(&schema));
    } else {
        let _ = command.print_help();
        println!();
    }
    Ok(0)
}

fn schema(command: &clap::Command, path: &[String]) -> Value {
    let name = path.last().cloned().unwrap_or_else(|| "hive".into());
    let prog = path.join(" ");
    let description = command
        .get_long_about()
        .or_else(|| command.get_about())
        .map(ToString::to_string)
        .unwrap_or_default();

    let mut options = Vec::new();
    let mut arguments = Vec::new();
    let mut help_arg = None;
    for arg in command.get_arguments() {
        if arg.is_hide_set() && arg.get_id().as_str() != "help" {
            continue;
        }
        let is_option = arg.get_long().is_some() || arg.get_short().is_some();
        let row = argument_schema(arg);
        if is_option {
            if arg.get_id().as_str() == "help" {
                help_arg = Some(row);
            } else {
                options.push(row);
            }
        } else {
            arguments.push(row);
        }
    }
    // Paths are walked in the tree's natural order; the help flag is always
    // first in the Python catalog. clap's auto-generated help arg is not part
    // of `get_arguments()`, so synthesise the same row argparse reports.
    if help_arg.is_none() {
        help_arg = Some(argument_row(
            "help",
            "option",
            vec!["-h".to_owned(), "--help".to_owned()],
            "show this help message and exit",
            Some(Value::from(0)),
        ));
    }
    if let Some(help) = help_arg {
        options.insert(0, help);
    }
    if path.len() == 1
        && !options
            .iter()
            .any(|row| row.get("name") == Some(&Value::from("version")))
    {
        options.push(argument_row(
            "version",
            "option",
            vec!["--version".to_owned()],
            "show program's version number and exit",
            Some(Value::from(0)),
        ));
    }
    // The root catalog's order follows argparse: help, version, no-color.
    if path.len() == 1 {
        options.sort_by_key(|row| {
            let name = row.get("name").and_then(Value::as_str).unwrap_or("");
            match name {
                "help" => 0,
                "version" => 1,
                "no_color" => 2,
                _ => 3,
            }
        });
    }

    let mut commands = Vec::new();
    for child in command.get_subcommands() {
        if child.is_hide_set() {
            continue;
        }
        let mut child_path = path.to_vec();
        child_path.push(child.get_name().to_owned());
        let mut row = schema(child, &child_path);
        let aliases: Vec<Value> = child
            .get_visible_aliases()
            .map(|alias| Value::String(alias.to_owned()))
            .collect();
        if !aliases.is_empty() {
            row.as_object_mut()
                .expect("schema is an object")
                .insert("aliases".into(), Value::Array(aliases));
        }
        commands.push(row);
    }

    let mut out = Map::new();
    out.insert("name".into(), Value::String(name));
    out.insert("path".into(), Value::String(prog.clone()));
    out.insert("prog".into(), Value::String(prog));
    out.insert("description".into(), Value::String(description));
    out.insert("options".into(), Value::Array(options));
    out.insert("arguments".into(), Value::Array(arguments));
    out.insert("commands".into(), Value::Array(commands));
    Value::Object(out)
}

fn argument_schema(arg: &clap::Arg) -> Value {
    let is_option = arg.get_long().is_some() || arg.get_short().is_some();
    let id = arg.get_id().as_str();
    let name = if id == "json" { "as_json" } else { id };

    let mut flags = Vec::new();
    if let Some(short) = arg.get_short() {
        flags.push(format!("-{short}"));
    }
    if let Some(long) = arg.get_long() {
        flags.push(format!("--{long}"));
    }
    let help = match name {
        "help" => "show this help message and exit".to_owned(),
        "version" => "show program's version number and exit".to_owned(),
        _ => arg.get_help().map(ToString::to_string).unwrap_or_else(|| {
            if name == "as_json" {
                "machine-readable JSON on stdout".to_owned()
            } else {
                String::new()
            }
        }),
    };

    let flag_action = matches!(
        arg.get_action(),
        ArgAction::SetTrue | ArgAction::SetFalse | ArgAction::Help | ArgAction::Version
    );
    let nargs: Option<Value> = if flag_action {
        Some(Value::from(0))
    } else if !is_option {
        match arg.get_action() {
            ArgAction::Set => {
                if arg
                    .get_num_args()
                    .map(|range| range.min_values())
                    .unwrap_or(1)
                    == 0
                {
                    Some(Value::String("*".into()))
                } else if !arg.is_required_set() {
                    Some(Value::String("?".into()))
                } else {
                    None
                }
            }
            ArgAction::Append => Some(Value::String("*".into())),
            _ => None,
        }
    } else {
        None
    };

    let mut out = argument_row(
        name,
        if is_option { "option" } else { "argument" },
        flags,
        &help,
        nargs,
    );
    let object = out.as_object_mut().expect("argument row is an object");
    if arg.is_required_set() {
        object.insert("required".into(), Value::Bool(true));
    }
    if name != "help"
        && name != "version"
        && let Some(default) = default_value(arg, name)
    {
        object.insert("default".into(), default);
    }
    let choices: Vec<Value> = if flag_action {
        Vec::new()
    } else {
        arg.get_possible_values()
            .iter()
            .filter(|value| !value.is_hide_set())
            .map(|value| Value::String(value.get_name().to_owned()))
            .collect()
    };
    if !choices.is_empty() {
        object.insert("choices".into(), Value::Array(choices));
    }
    if let Some(kind) = python_type(name) {
        object.insert("type".into(), Value::String(kind.into()));
    }
    out
}

fn argument_row(
    name: &str,
    kind: &str,
    flags: Vec<String>,
    help: &str,
    nargs: Option<Value>,
) -> Value {
    let mut out = Map::new();
    out.insert("kind".into(), Value::String(kind.into()));
    out.insert("name".into(), Value::String(name.into()));
    out.insert(
        "flags".into(),
        Value::Array(flags.into_iter().map(Value::String).collect()),
    );
    out.insert("help".into(), Value::String(help.into()));
    if let Some(nargs) = nargs {
        out.insert("nargs".into(), nargs);
    }
    Value::Object(out)
}

fn default_value(arg: &clap::Arg, name: &str) -> Option<Value> {
    match arg.get_action() {
        ArgAction::SetTrue => return Some(Value::Bool(false)),
        ArgAction::SetFalse => return Some(Value::Bool(true)),
        _ => {}
    }
    let raw = arg
        .get_default_values()
        .first()?
        .to_string_lossy()
        .into_owned();
    match python_type(name) {
        Some("int") => raw.parse::<i64>().ok().map(Value::from),
        Some("float") => raw
            .parse::<f64>()
            .ok()
            .and_then(|value| serde_json::Number::from_f64(value).map(Value::Number)),
        _ => Some(Value::String(raw)),
    }
}

fn python_type(name: &str) -> Option<&'static str> {
    match name {
        "timeout" => Some("float"),
        "lines" | "cards" | "ctx" => Some("int"),
        _ => None,
    }
}
