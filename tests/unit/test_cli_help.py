"""The CLI help surface is both human-readable and catalog-readable."""

from __future__ import annotations

import json

from harness import cli


def _walk(node: dict):
    yield node
    for child in node.get("commands", []):
        yield from _walk(child)


def test_command_catalog_has_descriptions_and_option_help():
    schema = cli._command_schema(cli.build_parser(), ["hive"])
    nodes = list(_walk(schema))
    assert [node["name"] for node in schema["commands"]] == [
        "serve", "tui", "status", "stop", "logs", "models", "stacks",
        "cascade", "backends", "setup", "doctor", "config", "profile",
        "controller", "encoder", "ui", "tools", "sandbox", "version", "help",
    ]
    for node in nodes:
        assert node["description"], f"{node['path']} has no description"
        for option in node["options"]:
            assert option["help"], f"{node['path']} {option['flags']} has no help"
        for argument in node["arguments"]:
            assert argument["help"], f"{node['path']} {argument['name']} has no help"


def test_help_json_is_stable_and_includes_aliases(capsys):
    assert cli.main(["help", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "hive"
    status = next(row for row in payload["commands"] if row["name"] == "status")
    assert status["aliases"] == ["ps"]
    models = next(row for row in payload["commands"] if row["name"] == "models")
    assert any(row["name"] == "list" for row in models["commands"])
    json_flag = next(
        option for option in models["commands"][0]["options"]
        if "--json" in option["flags"]
    )
    assert json_flag["help"] == "machine-readable JSON on stdout"


def test_help_topic_json_returns_that_subtree(capsys):
    assert cli.main(["help", "models", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "models"
    assert payload["path"] == "hive models"
    assert any(row["name"] == "list" for row in payload["commands"])


def test_help_topic_prints_the_existing_command_help(capsys):
    assert cli.main(["help", "models", "list"]) == 0
    out = capsys.readouterr().out
    assert "List GGUF files in the configured local model library." in out
    assert "--json" in out
    assert "machine-readable JSON on stdout" in out


def test_help_unknown_topic_is_a_usage_error(capsys):
    assert cli.main(["help", "does-not-exist"]) == 2
    assert "unknown command" in capsys.readouterr().err
