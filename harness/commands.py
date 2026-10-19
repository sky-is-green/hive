"""Console slash commands — dsh command conventions over the Studio seam.

Mirrors ``@deepseek-ai/dsh-commands``: lowercase names without the slash,
immutable descriptors (``name``/``description``/``input.hint``) served to the
UI for discovery, ``/name raw-input`` parsing, and handlers that settle as
``{kind: 'success'|'error', text}``.

Scope note: dsh's own interactive commands (/plan, /compact, /goal) live in
its Web host; the SDK transport intentionally does not expose them. These are
the Studio's console commands — same shape, wired to the seams we own
(server, providers, engines, splinter conversations, benchmark, transcripts).
"""

from __future__ import annotations

import html
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from cortex.config import SplinterConfig


@dataclass
class CommandDescriptor:
    name: str
    description: str
    input_hint: Optional[str] = None

    def to_dict(self) -> dict:
        out = {"name": self.name, "description": self.description}
        if self.input_hint:
            out["input"] = {"hint": self.input_hint}
        return out


@dataclass
class CommandResult:
    kind: str  # success | error
    text: str = ""
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "text": self.text, **self.extras}


def parse_command(line: str) -> Optional[tuple[str, str]]:
    """/name raw-input -> (name, raw_input); None when not a command line.

    dsh convention: raw_input is the VERBATIM remainder including separator
    whitespace ('/goal\\ncreate' -> ('goal', '\\ncreate'))."""
    stripped = line.strip()
    if not stripped.startswith("/"):
        return None
    body = stripped[1:]
    if not body:
        return ("", "")
    match = re.match(r"^(\S+)(.*)$", body, flags=re.DOTALL)
    return (match.group(1).lower(), match.group(2)) if match else (body.lower(), "")


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "-", text).strip("-") or "conversation"


def local_shell_command(command: str) -> list[str]:
    """The argv that runs ``command`` in the local shell.

    The sandbox's fallback is a local subprocess, and the local shell is not
    the same everywhere: POSIX runs ``sh -c``, Windows runs ``cmd /c``.
    Hard-coding ``/bin/sh`` made every local sandbox call fail on Windows
    with a WinError 2 ("the system cannot find the file specified").
    """
    import os
    import shutil

    if os.name == "nt":
        shell = shutil.which("cmd") or shutil.which("cmd.exe")
        if shell is None:
            raise OSError("no local shell found (cmd.exe)")
        return [shell, "/c", command]
    return [shutil.which("sh") or "/bin/sh", "-c", command]


# ---------------------------------------------------------------------------
# transcript export (/save)
# ---------------------------------------------------------------------------
def _hive_markdown(st, conversation_id: str) -> list[str]:
    """Splinter-side turns from the persisted conversation store: per turn the
    query chunk was stored before the reply chunk."""
    path = st._conv_path(conversation_id)
    lines: list[str] = []
    if path is None or not path.exists():
        return lines
    data = json.loads(path.read_text(encoding="utf-8"))
    store = data.get("store") or {}
    chunks = store.get("chunks") or []
    by_turn: dict[int, dict[str, str]] = {}
    for chunk in chunks:
        turn = int(chunk.get("turn", 0))
        slot = by_turn.setdefault(turn, {})
        if "query" not in slot:
            slot["query"] = chunk.get("content", "")
        else:
            slot["reply"] = chunk.get("content", "")
    for turn in sorted(by_turn):
        entry = by_turn[turn]
        lines.append(f"## Turn {turn}")
        lines.append(f"**User:** {html.escape(entry.get('query', ''))}")
        if entry.get("reply"):
            lines.append(f"\n**Assistant:** {html.escape(entry['reply'])}")
        lines.append("")
    return lines


def _agent_markdown(session_root: Path, conversation_id: str) -> list[str]:
    """Agent-side turns from the durable dsh session log:
    ``<session_root>/<cwd-slug>/agent-<id>/session.jsonl.zstd``."""
    lines: list[str] = []
    root = Path(session_root)
    if not root.is_dir():
        return lines
    session_files = list(root.glob(f"*/agent-{conversation_id}/session.jsonl.zstd"))
    for path in session_files:
        for raw in _read_zstd_lines(path):
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = record.get("type")
            if kind == "user/message":
                blocks = record.get("data", {}).get("content") or []
                text = "".join(b.get("text", "") for b in blocks
                               if isinstance(b, dict) and b.get("type") == "text")
                if text.strip() and "splinter-curated-context" not in text:
                    lines.append(f"**User:** {html.escape(text.strip())}\n")
            elif kind == "assistant/message":
                blocks = record.get("data", {}).get("message", {}).get("content") or []
                text = "".join(b.get("text", "") for b in blocks
                               if isinstance(b, dict) and b.get("type") == "text")
                if text.strip():
                    lines.append(f"**Assistant:** {html.escape(text.strip())}\n")
    return lines


def _agent_messages(session_root: Path, conversation_id: str) -> list[dict]:
    """The agent-side turns as structured messages (role + text)."""
    out: list[dict] = []
    root = Path(session_root)
    if not root.is_dir():
        return out
    for path in root.glob(f"*/agent-{conversation_id}/session.jsonl.zstd"):
        for raw in _read_zstd_lines(path):
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = record.get("type")
            if kind == "user/message":
                blocks = record.get("data", {}).get("content") or []
                text = "".join(b.get("text", "") for b in blocks
                               if isinstance(b, dict) and b.get("type") == "text")
                if text.strip() and "splinter-curated-context" not in text:
                    out.append({"role": "user", "text": text.strip()})
            elif kind == "assistant/message":
                blocks = record.get("data", {}).get("message", {}).get("content") or []
                text = "".join(b.get("text", "") for b in blocks
                               if isinstance(b, dict) and b.get("type") == "text")
                if text.strip():
                    out.append({"role": "assistant", "text": text.strip()})
    return out


def _read_zstd_lines(path: Path) -> list[str]:
    import zstandard

    dctx = zstandard.ZstdDecompressor()
    with open(path, "rb") as fh:
        reader = dctx.stream_reader(fh, read_across_frames=True)
        text = reader.read().decode("utf-8", errors="replace")
    return [ln for ln in text.splitlines() if ln.strip()]


# ---------------------------------------------------------------------------
# file attachment (/import)
# ---------------------------------------------------------------------------
#: the checkout this harness runs from (relative paths land here)
HARNESS_ROOT = Path(__file__).resolve().parents[1]

#: how much of a file the model gets to see (~2k tokens)
MAX_ATTACH_CHARS = 8000
#: the store caps a chunk at 4000 chars — keep a margin for the header
ATTACH_CHUNK_CHARS = 3500
#: a file bigger than this is refused before it is read
MAX_FILE_BYTES = 4_000_000

# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
Handler = Callable[[str, str], CommandResult]


@dataclass
class ConsoleCommands:
    """Descriptor registry + handlers; wired to the app's services."""

    st: object
    models: object
    agent: object
    transcripts_dir: Path

    def __post_init__(self) -> None:
        self._handlers: dict[str, Handler] = {
            "help": self._help,
            "new": self._new,
            "save": self._save,
            "model": self._model,
            "mode": self._mode,
            "provider": self._provider,
            "engine": self._engine,
            "bench": self._bench,
            "train": self._train,
            "status": self._status,
            "cascade": self._cascade,
            "sandbox": self._sandbox,
            "compact": self._compact,
            "dev": self._dev,
            "import": self._import,
        }
        self._descriptors = [
            CommandDescriptor("help", "list available commands"),
            CommandDescriptor("new", "start a fresh conversation (splinter + agent)"),
            CommandDescriptor("save", "export the conversation transcript as markdown",
                              input_hint="[name]"),
            CommandDescriptor("model", "list local models or load one",
                              input_hint="[name]"),
            CommandDescriptor("mode", "switch the chat pane transport",
                              input_hint="[splinter|agent]"),
            CommandDescriptor("provider", "list providers or set the default",
                              input_hint="[name]"),
            CommandDescriptor("engine", "list engines or set the default",
                              input_hint="[name]"),
            CommandDescriptor("bench", "run the mock HiveBench protocol",
                              input_hint="[max_convs]"),
            CommandDescriptor("train", "list training engines/recipes or show one",
                              input_hint="[engine]"),
            CommandDescriptor("status", "server / agent / provider summary"),
            CommandDescriptor("cascade", "list cascade runs, or start one",
                              input_hint="[tasks.json]"),
            CommandDescriptor("sandbox", "run a command in the sandbox",
                              input_hint="<shell command>"),
            CommandDescriptor("compact", "compact the conversation into a checkpoint",
                              input_hint=""),
            CommandDescriptor("dev", "developer mode: reveal the dev surface",
                              input_hint="[on|off]"),
            CommandDescriptor("import", "show the LLM a file: attach it to the chat",
                              input_hint="<path>"),
        ]

    # ------------------------------------------------------------------
    def descriptors(self) -> list[dict]:
        return [d.to_dict() for d in self._descriptors]

    def run(self, line: str, conversation_id: str) -> CommandResult:
        parsed = parse_command(line)
        if parsed is None:
            return CommandResult("error", "not a command line")
        name, raw_args = parsed
        if not name:
            return self._help("")
        handler = self._handlers.get(name)
        if handler is None:
            return CommandResult(
                "error", f"unknown command '/{name}' — try /help")
        try:
            return handler(raw_args.strip(), conversation_id)
        except Exception as exc:  # noqa: BLE001 - handlers settle as errors
            return CommandResult("error", f"/{name} failed: {exc}")

    # ------------------------------------------------------------------
    def _help(self, _args: str, _cid: str) -> CommandResult:
        lines = [f"/{d.name}{(' — ' + d.description) if d.description else ''}"
                 for d in self._descriptors]
        return CommandResult("success", "\n".join(lines))

    def _new(self, _args: str, _cid: str) -> CommandResult:
        new_id = f"console-{int(time.time() * 1000):x}"
        return CommandResult(
            "success", "Fresh conversation started (splinter store + agent session).",
            {"new_conversation_id": new_id})

    def _save(self, args: str, conversation_id: str) -> CommandResult:
        lines = ["# Conversation transcript — "
                 f"{conversation_id}", "",
                 f"_exported {time.strftime('%Y-%m-%d %H:%M:%S')}_", ""]
        splinter_lines = _hive_markdown(self.st, conversation_id)
        agent_lines = _agent_markdown(self.agent.session_root, conversation_id)
        if splinter_lines:
            lines += ["## Splinter conversation", ""] + splinter_lines
        if agent_lines:
            lines += ["## Agent (dsh) session", ""] + agent_lines
        if not splinter_lines and not agent_lines:
            return CommandResult("error",
                                 "nothing to save for this conversation yet")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = _safe_name(args) if args else _safe_name(conversation_id)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        out = self.transcripts_dir / f"{stamp}-{name}.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        return CommandResult("success", f"saved {out}")

    def _model(self, args: str, _cid: str) -> CommandResult:
        local = self.models.list_local()
        if not args:
            current = self.models.status().get("model") or "(none loaded)"
            lines = [f"loaded: {current}"] + [
                f"  {m['file']} — {m['size_gb']} GB" for m in local]
            return CommandResult("success", "\n".join(lines))
        resolved = self.models.resolve_model(args)
        if resolved is None:
            return CommandResult("error",
                                 f"no local model matches '{args}'")
        from harness.models import launch_extra_args

        load_options = self._last_load_options()
        info = self.models.start(model=str(resolved),
                                 ctx_size=load_options.get("context", 8192),
                                 ngl=load_options.get("gpu_layers", 999),
                                 extra_args=launch_extra_args(load_options))
        self.st_app().register_local(info, load_options=load_options)
        return CommandResult("success",
                             f"loaded {resolved.name} on port {info['port']}")

    def _launch_extra_args(self, load_options: dict) -> list[str]:
        from harness.models import launch_extra_args

        return launch_extra_args(load_options)

    def _last_load_options(self) -> dict:
        try:
            engine = self.st.engines.resolve("local")
            return dict(engine.load_options or {})
        except LookupError:
            return {}

    def st_app(self):
        """The FastAPI app (register_local lives there); set at wiring time."""
        return self._app

    def _mode(self, args: str, _cid: str) -> CommandResult:
        mode = args.lower()
        if mode not in ("splinter", "agent"):
            return CommandResult("error", "usage: /mode [splinter|agent]")
        return CommandResult("success", f"chat mode set to {mode}",
                             {"mode": mode})

    def _provider(self, args: str, _cid: str) -> CommandResult:
        reg = self.st.registry
        if not args:
            lines = [f"default: {reg.default or '(first)'}"] + [
                f"  {p.name} -> {p.base_url}" for p in reg.providers]
            return CommandResult("success", "\n".join(lines))
        for p in reg.providers:
            if p.name.lower() == args.lower():
                reg.default = p.name
                return CommandResult("success", f"default provider: {p.name}")
        return CommandResult("error", f"unknown provider '{args}'")

    def _engine(self, args: str, _cid: str) -> CommandResult:
        reg = self.st.engines
        if not args:
            lines = [f"default: {reg.default or '(first)'}"] + [
                f"  {e.name} ({e.kind})" for e in reg.engines]
            return CommandResult("success", "\n".join(lines))
        for e in reg.engines:
            if e.name.lower() == args.lower():
                reg.default = e.name
                return CommandResult("success", f"default engine: {e.name}")
        return CommandResult("error", f"unknown engine '{args}'")

    def _bench(self, args: str, _cid: str) -> CommandResult:
        max_convs = int(args) if args.isdigit() else 2
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = self.st.runs_root / f"protocol_{stamp}"
        import subprocess
        import sys as _sys
        from harness.app import REPO_ROOT

        run_dir.mkdir(parents=True, exist_ok=True)
        # the child inherits the log fd; the parent's copy must be closed
        log_fh = open(run_dir / "run_stdout.log", "ab")
        try:
            subprocess.Popen(
                [_sys.executable, "-m", "experiments.generate_data",
                 "--mock", "--protocol", "--max-convs", str(max_convs),
                 "--output", str(run_dir)],
                cwd=str(REPO_ROOT),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
        finally:
            log_fh.close()
        return CommandResult(
            "success",
            f"mock benchmark launched ({max_convs} convs) -> {run_dir.name}; "
            f"report at /view/{run_dir.name} when done")

    def _train(self, args: str, _cid: str) -> CommandResult:
        from harness.training import (
            engine_catalog,
            get_engine,
            ternary_python,
            ternary_runner,
        )

        if args:
            try:
                engine = get_engine(args.strip())
            except ValueError as exc:
                return CommandResult("error", str(exc))
            lines = [f"{engine.label} — {engine.description}"]
            for m in engine.methods:
                mark = "" if m.implemented else " (not implemented)"
                lines.append(f"  {m.id}{mark} — {m.label}")
            return CommandResult("success", "\n".join(lines))
        lines = ["training engines:"]
        for e in engine_catalog():
            ready = ", ".join(m.id for m in e.methods if m.implemented) or "—"
            lines.append(f"  {e.id} — {e.label} [ready: {ready}]")
        lines.append("runner: " + (" ".join(ternary_runner()) or
                                   "(unset — set $HIVE_TERNARY_RUNNER)"))
        lines.append(f"python: {ternary_python()}")
        lines.append("launch from the Training tab, or POST /v1/training/launch")
        return CommandResult("success", "\n".join(lines))

    def _cascade(self, args: str, _cid: str) -> CommandResult:
        """List cascade runs, or launch one on a task bundle."""
        runs_root = Path(getattr(self.st, "runs_root", "runs"))
        tasks = args.strip()
        if not tasks:
            rows = []
            if runs_root.is_dir():
                for d in sorted(runs_root.iterdir(), reverse=True):
                    if not d.is_dir():
                        continue
                    report = d / "report.json"
                    if not report.is_file():
                        continue
                    try:
                        data = json.loads(report.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        continue
                    summary = data.get("summary") or {}
                    rows.append(
                        f"{d.name} · {summary.get('tasks', '?')} tasks · "
                        f"cascade {summary.get('cascade_accuracy', '?')} · "
                        f"escalated {summary.get('escalation_rate', '?')}")
            return CommandResult(
                "success",
                "cascade runs:\n" + ("\n".join(rows) or "(none yet)") +
                "\nstart one: /cascade tasks-bench.json")
        from harness.app import REPO_ROOT

        if not tasks.endswith(".json"):
            return CommandResult("error", "usage: /cascade <tasks.json>")
        task_file = REPO_ROOT / "experiments" / "cascade" / tasks
        if not task_file.is_file():
            return CommandResult("error", f"no such task file: {tasks}")
        run_name = "cascade-" + time.strftime("%Y%m%d-%H%M%S")
        run_dir = runs_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(REPO_ROOT / "experiments" / "cascade" / "run_cascade.py"),
               "--tasks", str(task_file), "--out", str(runs_root),
               "--run-name", run_name]
        log = run_dir / "run_stdout.log"
        try:
            with open(log, "ab") as fh:
                proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=fh,
                                        stderr=subprocess.STDOUT)
        except OSError as exc:
            return CommandResult("error", f"could not launch: {exc}")
        return CommandResult("success",
                              f"cascade started: {run_name} (pid {proc.pid}) → {run_dir}")

    def _compact(self, args: str, conversation_id: str) -> CommandResult:
        """Compact the conversation into a checkpoint and offer the engine switch."""
        from harness.compaction.core import Message, Settings, Trigger, compact
        from harness.compaction.studio import make_summarizer

        raw = _agent_messages(self.agent.session_root, conversation_id)
        if len(raw) < 2:
            # no agent log (the chat is in Splinter mode): the visible
            # transcript is still compactable, so hand the job to the UI
            return CommandResult(
                "success",
                "compacting the visible transcript",
                {"compact_browser": True})
        messages = [Message(role=m["role"], text=m["text"]) for m in raw]
        events: list[dict] = []

        def call_model(prompt: str, older) -> str:
            try:
                provider = self.st.registry.resolve(None)
            except LookupError:
                return ""
            import requests

            msgs = [{"role": "user", "content": prompt}]
            for m in older[-12:]:
                msgs.append({"role": "user" if m.role == "user" else "assistant",
                             "content": m.text})
            payload = {"model": provider.model or "local", "messages": msgs,
                       "temperature": 0, "max_tokens": 1200}
            headers = {"content-type": "application/json"}
            if provider.api_key:
                headers["authorization"] = f"Bearer {provider.api_key}"
            resp = requests.post(f"{provider.base_url.rstrip('/')}/chat/completions",
                                 json=payload, headers=headers, timeout=180)
            resp.raise_for_status()
            data = resp.json()
            return (data["choices"][0]["message"].get("content") or "").strip()

        outcome = compact(messages, Trigger.manual(), Settings(auto=True),
                          summarize=make_summarizer(call_model),
                          emit=lambda kind, payload: events.append(kind))
        if outcome.status != "completed":
            return CommandResult("error", f"compaction {outcome.status}: {outcome.error or ''}")
        checkpoint = getattr(outcome, "checkpoint", None)
        if checkpoint is None:
            return CommandResult("error", "compaction produced no checkpoint")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        out = self.transcripts_dir / f"{stamp}-checkpoint-{_safe_name(conversation_id)}.md"
        out.write_text(checkpoint.rendered, encoding="utf-8")
        return CommandResult(
            "success",
            f"compacted {len(messages)} messages → {out}\n"
            f"the checkpoint is a safe switch point: /engine <name> to swap the local engine",
            {"compact_browser": True})

    def _import(self, args: str, conversation_id: str) -> CommandResult:
        """Show the LLM a file: attach it to this conversation.

        The file's text goes into the conversation's context store (Splinter
        mode) and rides along with the next agent turn (dsh mode), so the
        next reply can read it.  Text only: a binary file is refused rather
        than turned into noise.  Secrets in the file are redacted on the way
        in, the same way the store treats everything else.
        """
        raw = args.strip()
        if not raw:
            return CommandResult("error", "usage: /import <path>")
        path = self._resolve_path(raw)
        if path is None:
            return CommandResult("error", f"no such file: {raw}")
        if path.is_dir():
            return CommandResult("error", f"{path.name} is a directory")
        try:
            size = path.stat().st_size
            if size > MAX_FILE_BYTES:
                return CommandResult(
                    "error",
                    f"{path.name} is {size / 1e6:.1f} MB — too big to show the model")
            data = path.read_bytes()
        except OSError as exc:
            return CommandResult("error", f"cannot read {path.name}: {exc}")
        if b"\x00" in data[:8192]:
            return CommandResult(
                "error",
                f"{path.name} looks binary — only text can be shown to the model")
        text = data.decode("utf-8", errors="replace")

        from retention.hygiene import TRUNCATION_MARK, sanitize_for_storage

        safe = sanitize_for_storage(text, max_chars=MAX_ATTACH_CHARS)
        redacted = "[redacted-secret]" in safe or "[base64 blob stripped]" in safe
        trimmed = safe.endswith(TRUNCATION_MARK)
        header = (f"[attached file: {path.name} — {len(text)} chars"
                  + (f", first {MAX_ATTACH_CHARS} shown" if trimmed else "")
                  + "]")
        body = f"{header}\n{safe}"

        # the context store (Splinter mode) — chunked, because the store caps
        # a chunk at 4000 chars
        try:
            chunks = self._attach_to_store(conversation_id, body)
            store_note = f", {chunks} chunk(s) in the context store"
        except Exception as exc:  # noqa: BLE001 - the agent path still works
            chunks = 0
            store_note = f" (not stored: {exc})"
        # the next agent turn (dsh mode)
        self.agent.queue_context(conversation_id, body)

        notes = []
        if trimmed:
            notes.append(f"trimmed to the first {MAX_ATTACH_CHARS} chars")
        if redacted:
            notes.append("secrets redacted")
        suffix = (" — " + ", ".join(notes)) if notes else ""
        return CommandResult(
            "success",
            f"attached {path.name} ({len(text)} chars{store_note}){suffix}\n"
            f"the next reply can see it — ask about the file.",
            {"attachment": {"name": path.name, "path": str(path),
                            "text": body}})

    def _attach_to_store(self, conversation_id: str, body: str) -> int:
        """Put an attachment into the conversation's context store.

        The store is the conversation's memory, so the attachment is
        retrievable on any later turn — not just the next one.  The splinter
        is built without a backend (the chat's own route does the same: it
        curates here and generates upstream), so this works even before a
        provider is configured.
        """
        st = self.st
        with st.lock_for(conversation_id):
            splinter = st.splinter_for(conversation_id, None, with_backend=False)
            turn = splinter.turn
            count = 0
            for start in range(0, len(body), ATTACH_CHUNK_CHARS):
                piece = body[start:start + ATTACH_CHUNK_CHARS]
                if splinter.store.add_chunk(turn, piece):
                    count += 1
            st.save_conversation(conversation_id, splinter)
            return count

    def _resolve_path(self, raw: str) -> Optional[Path]:
        """A path the user can type: absolute, ~-relative, or relative to
        the working directory or the checkout."""
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            roots = [Path.cwd(), HARNESS_ROOT]
            for root in roots:
                probe = root / candidate
                if probe.exists():
                    candidate = probe
                    break
        try:
            candidate = candidate.resolve()
        except OSError:
            return None
        return candidate if candidate.exists() else None

    def _dev(self, args: str, _cid: str) -> CommandResult:
        """Turn the developer surface on or off.

        Dev mode reveals the deep-research queue, the Docker/VHDX setup tab
        and the raw server status.  It is off by default; the Studio hides
        those parts until it is on.
        """
        from harness.studio_settings import load_settings, save_settings

        state_dir = getattr(self.st, "state_dir", None)
        current = load_settings(state_dir)
        want = args.strip().lower()
        if want in ("on", "1", "true", "yes"):
            on = True
        elif want in ("off", "0", "false", "no"):
            on = False
        else:
            on = not current["dev_mode"]
        saved = save_settings({"dev_mode": on}, state_dir)
        return CommandResult(
            "success",
            f"developer mode {'ON' if saved['dev_mode'] else 'OFF'}"
            + ("\nreveals: the deep-research queue, the Docker/VHDX setup tab, the raw status."
               if saved["dev_mode"] else "\nhidden again."))

    def _sandbox(self, args: str, _cid: str) -> CommandResult:
        """Run a command in the sandbox (Docker when the daemon is up)."""
        command = args.strip()
        if not command:
            return CommandResult("error", "usage: /sandbox <shell command>")
        import shutil

        docker_bin = shutil.which("docker")
        daemon = False
        if docker_bin:
            try:
                subprocess.run([docker_bin, "info"], capture_output=True,
                               timeout=5, check=True)
                daemon = True
            except Exception:  # noqa: BLE001
                daemon = False
        if daemon:
            cmd = [docker_bin, "run", "--rm", "--network", "none",
                   "--memory", "512m", "--cpus", "1", "python:3.12-slim",
                   "sh", "-c", command]
        else:
            try:
                cmd = local_shell_command(command)
            except OSError as exc:
                return CommandResult("error", f"sandbox failed: {exc}")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return CommandResult("error", "sandbox command timed out (60s)")
        except OSError as exc:
            return CommandResult("error", f"sandbox failed: {exc}")
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        body = out + (("\nstderr: " + err) if err else "")
        if not body:
            body = f"(no output, exit {proc.returncode})"
        return CommandResult("success",
                              f"[{'docker' if daemon else 'local'}] exit {proc.returncode}\n{body}")

    def _status(self, _args: str, _cid: str) -> CommandResult:
        s = self.models.status()
        agent = "running" if self.agent.runtime_running else "not started"
        provider = self.st.registry.default or "(first)"
        engine = self.st.engines.default or "(first)"
        return CommandResult(
            "success",
            f"server: {'up' if s['running'] else 'down'}"
            f"{' / healthy' if s.get('healthy') else ''} · "
            f"model: {s.get('model') or '(none)'} · "
            f"agent runtime: {agent} · provider: {provider} · engine: {engine}")
