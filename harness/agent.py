"""dsh agent bridge — the DeepSeek Harness agent loop behind the console.

One persistent SDK runtime (``deepseek_harness.DeepSeekHarness``, JSON-RPC
stdio subprocess from the pinned fork's node carrier) serves agent sessions
keyed by conversation id. Each message runs the REAL agent loop: tools
(bash, fs, code-runtime, web, subagents), multi-step turns, the durable dsh
session log — with the model pointed at whatever the Studio currently has
loaded (provider ``local`` → managed llama-server).

The runtime is rebuilt automatically when the target (base_url, api_key,
model) changes, so reloading a different model in the Server tab is picked
up on the next agent message. Requires ``DSH_RUNTIME_MODE=node`` (the dev
carrier staged from the fork; Windows has no exe carrier).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import threading
import time
from itertools import count as _count
from pathlib import Path
from typing import Callable, Optional

from harness import agent_events

# The vendored dsh Python SDK ships inside this repo (vendor/deepseek_harness)
# so the sidecar runs from a clean checkout without an editable install of the
# fork's SDK. Inserted before the import below.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))

from deepseek_harness import DeepSeekHarness  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def _shape_notification(notification) -> Optional[dict]:
    """Map an SDK notification to a compact UI activity event."""
    method = getattr(notification, "method", "")
    payload = getattr(notification, "payload", None) or {}
    if method != "session.event":
        return None
    event = payload.get("event") if isinstance(payload, dict) else None
    if not isinstance(event, dict):
        return None
    kind = event.get("type", "")
    data = event.get("data") or {}
    if kind == "assistant/message":
        blocks = data.get("message", {}).get("content") or []
        text = "".join(b.get("text", "") for b in blocks
                       if isinstance(b, dict) and b.get("type") == "text")
        if text.strip():
            return {"type": "assistant", "text": text}
        return None
    if kind.startswith("tool/"):
        name = data.get("name") or data.get("tool") or kind
        return {"type": "tool", "tool": str(name), "phase": kind.split("/", 1)[1]}
    if kind in ("turn/start", "step/start", "turn/end", "step/end"):
        return {"type": "lifecycle", "event": kind}
    return None


def _content_text(blocks) -> str:
    """Text of an OpenAI-style content-block list (dicts or SDK objects)."""
    texts: list[str] = []
    for block in blocks or []:
        text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
        if isinstance(text, str) and text:
            texts.append(text)
    return "".join(texts)


def _json_or_text(value):
    """A tool-call argument blob as parsed JSON when it parses, else raw text."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value) if value.strip() else {}
    except json.JSONDecodeError:
        return value


class _ModelSpan:
    """One live child-session dispatch awaiting its §B3 ``model`` span."""

    __slots__ = ("model_id", "parent", "role", "model", "task", "started",
                 "output", "pending", "calls")

    def __init__(self, model_id: str, parent: str, started: float) -> None:
        self.model_id = model_id
        self.parent = parent
        self.role: Optional[str] = None
        self.model: Optional[str] = None
        self.task: Optional[str] = None
        self.started = started
        self.output = ""
        self.pending: list[dict] = []
        #: ``callId -> tool name`` for a ``tool/result``, whose own payload
        #: carries only the call id (in the message's source).
        self.calls: dict[str, str] = {}


class ActivityShaper:
    """Per-turn activity shaper: face events + §B3 model spans (T48).

    dsh runs the delegation, so the notifications carry the attribution:
    ``subagent.started``/``finished`` delimit a child session, and the child's
    ``request/context`` (or ``request/header.config``) names its provider route
    (``tier-worker``) and model.  That is the structural parent id for the
    §B3 ``model`` start/end pair — never inferred from timing — and the child's
    own ``tool/*`` events are forwarded parented to that model id so the cards
    nest (ADR-L6).
    """

    def __init__(self, session_id: str, on_event: Callable[[dict], None],
                 on_subagent: Callable[[str, str], None] | None = None) -> None:
        self._root = session_id
        self._on_event = on_event
        self._on_subagent = on_subagent
        self._spans: dict[str, _ModelSpan] = {}
        self._root_calls: dict[str, str] = {}
        self._seq = _count(1)

    def __call__(self, notification) -> None:
        method = getattr(notification, "method", "")
        payload = getattr(notification, "payload", None) or {}
        if not isinstance(payload, dict):
            return
        try:
            if method == "subagent.started":
                self._started(payload)
            elif method == "subagent.finished":
                self._finished(payload)
            elif method == "session.event":
                self._session_event(payload, notification)
        except Exception:  # noqa: BLE001 - a consumer queue must not kill the run
            pass

    # -- emission -----------------------------------------------------------
    def _emit(self, event: dict) -> None:
        try:
            self._on_event(event)
        except Exception:  # noqa: BLE001
            pass

    def _emit_child(self, span: _ModelSpan, event: dict) -> None:
        event = dict(event)
        event["tier"] = span.role or ""
        self._emit(agent_events.with_parent(event, span.model_id))

    def _model_start(self, span: _ModelSpan) -> None:
        self._emit(agent_events.model_event(
            tier=span.role or "unknown", model=span.model or "",
            phase=agent_events.PHASE_START, id=span.model_id, task=span.task,
        ))
        for event in span.pending:
            self._emit_child(span, event)
        span.pending = []

    def _child_tool(self, span: _ModelSpan, event: dict) -> None:
        if span.role is None:
            span.pending.append(event)  # the start event carries the parent id
        else:
            self._emit_child(span, event)

    # -- notifications ------------------------------------------------------
    def _started(self, payload: dict) -> None:
        child = payload.get("childSessionId")
        parent = payload.get("parentSessionId")
        if not isinstance(child, str) or not child:
            return
        if parent != self._root and parent not in self._spans:
            return  # another tree's child (or a nested one we do not own)
        if self._on_subagent is not None:
            try:
                self._on_subagent(str(parent or self._root), child)
            except Exception:  # noqa: BLE001 - activity must not break the run
                pass
        self._spans[child] = _ModelSpan(
            agent_events.new_model_id(next(self._seq)),
            str(parent or self._root),
            time.perf_counter(),
        )

    def _finished(self, payload: dict) -> None:
        child = payload.get("childSessionId")
        span = self._spans.pop(child, None) if isinstance(child, str) else None
        if span is None:
            return
        if span.role is None:
            # The child finished without a request event we understood; keep
            # the pair (an unknown-tier card) rather than losing the dispatch.
            span.role = "unknown"
            span.model = span.model or ""
            self._model_start(span)
        output = _content_text(payload.get("lastAssistantMessage")) or span.output
        self._emit(agent_events.model_event(
            tier=span.role, model=span.model or "", phase=agent_events.PHASE_END,
            id=span.model_id, task=span.task,
            duration_ms=int((time.perf_counter() - span.started) * 1000),
            output=output or None,
        ))

    def _session_event(self, payload: dict, notification) -> None:
        session_id = payload.get("sessionId")
        event = payload.get("event")
        if not isinstance(event, dict):
            return
        if session_id is None or session_id == self._root:
            # No sessionId is the legacy/fake shape: shape it as a face event.
            # Real notifications always carry the id.
            self._face_event(payload, notification)
            return
        span = self._spans.get(session_id) if isinstance(session_id, str) else None
        if span is None:
            return

        kind = event.get("type", "")
        data = event.get("data") or {}
        if kind in ("request/context", "request/header"):
            if kind == "request/context":
                provider, model = data.get("provider"), data.get("model")
            else:
                config = (data.get("header") or {}).get("config") or {}
                provider, model = config.get("provider"), config.get("model")
            self._resolve(span, provider, model)
        elif kind == "user/message":
            if span.task is None:
                # The event data *is* the user message; tolerate a wrapped one.
                message = data.get("message") or data
                text = _content_text(message.get("content") if isinstance(message, dict) else None)
                span.task = text[:400] or None
        elif kind == "assistant/message":
            text = _content_text((data.get("message") or {}).get("content"))
            if text:
                span.output = text
        elif kind == "tool/call":
            name = str(data.get("name") or "tool")
            call_id = str(data.get("callId") or "")
            if call_id:
                span.calls[call_id] = name
            self._child_tool(span, {
                "type": agent_events.EVENT_TOOL,
                "tool": name,
                "phase": "call",
                "args": _json_or_text(data.get("arguments")),
            })
        elif kind == "tool/result":
            message = data.get("message") or {}
            source = message.get("source") if isinstance(message, dict) else None
            call_id = str((source or {}).get("callId") or "") if isinstance(source, dict) else ""
            name = span.calls.pop(call_id) if call_id in span.calls else None
            result = _content_text(message.get("content")) if isinstance(message, dict) else ""
            self._child_tool(span, {
                "type": agent_events.EVENT_TOOL,
                "tool": name or "tool",
                "phase": "result",
                "result": result,
            })

    def _face_event(self, payload: dict, notification) -> None:
        """Face-level events, including the concrete tool input and result.

        The old face shape dropped ``args`` and ``result``; the activity feed
        needs them to show what the agent actually did.  Keep the old keys
        absent when the runtime did not provide them, so the frozen face
        shape still holds for the no-payload case.
        """
        event = payload.get("event")
        if not isinstance(event, dict):
            return
        kind = event.get("type", "")
        data = event.get("data") or {}
        if kind == "tool/call":
            name = str(data.get("name") or data.get("tool") or "tool")
            call_id = str(data.get("callId") or "")
            if call_id:
                self._root_calls[call_id] = name
            shaped = {"type": "tool", "tool": name, "phase": "call"}
            args = _json_or_text(data.get("arguments", data.get("input")))
            if args not in (None, "", {}, []):
                shaped["args"] = args
            self._emit(shaped)
            return
        if kind == "tool/result":
            message = data.get("message") or {}
            source = message.get("source") if isinstance(message, dict) else None
            call_id = str((source or {}).get("callId") or "") \
                if isinstance(source, dict) else ""
            name = self._root_calls.pop(call_id, None) if call_id else None
            result = _content_text(message.get("content")) \
                if isinstance(message, dict) else ""
            shaped = {"type": "tool", "tool": name or "tool", "phase": "result"}
            if result:
                shaped["result"] = result
            self._emit(shaped)
            return
        shaped = _shape_notification(notification)
        if shaped is not None:
            self._emit(shaped)

    def _resolve(self, span: _ModelSpan, provider, model) -> None:
        if span.role is not None:
            return
        route = str(provider or "").strip()
        role = route[len("tier-"):] if route.startswith("tier-") else route
        span.role = role or "unknown"
        span.model = str(model or "").strip()
        self._model_start(span)


def _windows_bash_path() -> Optional[str]:
    """Git for Windows ships bash but rarely puts it on PATH."""
    for candidate in (
        r"C:\Program Files\Git\bin",
        r"C:\Program Files\Git\usr\bin",
        r"C:\Program Files (x86)\Git\bin",
    ):
        if Path(candidate, "bash.exe").is_file():
            return candidate
    return None


def _runtime_env(session_root: str | None = None,
                 cordis: str | None = None) -> dict:
    """Extra env for the dsh runtime subprocess.

    The SDK used to inject DSH_SESSION_ROOT and DSH_CORDIS_CONFIG from its
    own config; it no longer does (the profile owns the persistence now), so
    the caller passes them through the environment.
    """
    env: dict[str, str] = {}
    if session_root:
        env["DSH_SESSION_ROOT"] = session_root
    if cordis:
        env["DSH_CORDIS_CONFIG"] = cordis
    if os.environ.get("HIVE_TOOL_URL"):
        env["HIVE_TOOL_URL"] = os.environ["HIVE_TOOL_URL"]
    # The DSH base composition already defaults to workspace-write; make the
    # safe default explicit for the Hive-owned runtime environment.
    env.setdefault("DSH_PERMISSION_MODE", "workspace-write")
    if os.name == "nt":
        git_bin = _windows_bash_path()
        if git_bin:
            # the bash tool spawns `bash`; Git for Windows' bash is the
            # standard local shell available on this machine
            env["PATH"] = git_bin + os.pathsep + os.environ.get("PATH", "")
    return env


class DshAgentService:
    """Persistent dsh runtime + per-conversation agent sessions.

    The SDK transport has no cancel method (initialize/session/prompt/
    shutdown only), so cancelling a run terminates the runtime subprocess:
    the blocked ``run()`` raises, the SSE reports ``cancelled``, and the
    conversation survives — the durable JSONL session is continued by the
    rebuilt runtime on the next message (same session id + session_root).
    """

    def __init__(self, default_cwd: Path, session_root: Path, *,
                 runtime_config: Optional[Callable[[], Optional[str]]] = None,
                 activity: Optional[object] = None) -> None:
        self.default_cwd = Path(default_cwd)
        self.session_root = Path(session_root)
        #: Optional harness.activity.ActivityHub.  Kept duck-typed so the
        #: agent module does not import the Studio state layer.
        self.activity = activity
        #: Called before each runtime build; returns the profile config path for
        #: the currently applied stack (``None`` = the runtime's own default).
        self._runtime_config = runtime_config
        self._cordis_path: Optional[str] = None
        self._lock = threading.Lock()
        self._harness = None
        self._target: Optional[tuple] = None
        self._run_lock = threading.Lock()  # one agent turn at a time (v1)
        self._cancel_requested = threading.Event()
        self._inflight = False
        self._generation = 0  # bumped on every runtime rebuild
        self._generation_initialized = False
        self._handoff_done: set[str] = set()
        #: context queued for a conversation's next turn (e.g. an attached
        #: file).  It rides along with the next message instead of being a
        #: turn the user has to ask for.
        self._pending_context: dict[str, list[str]] = {}
        self._context_lock = threading.Lock()
        self._hook_installed = False
        self._hook_disabled = False
        self._hook_reason = "not initialized"
        self._hook_paths: dict[str, str] = {}
        self._hook_preflight_key: tuple[str, str] | None = None
        self._hook_runtime_verified = False
        #: Called with (parent_session_id, child_session_id) for subagents.
        self.lineage_sink: Optional[Callable[[str, str], None]] = None

    def _init_generation(self) -> None:
        """Seed the generation counter from durable session dirs so a fresh
        sidecar never collides with logs written by a previous life."""
        if self._generation_initialized or not self.session_root.is_dir():
            self._generation_initialized = True
            return
        max_gen = -1
        for d in self.session_root.glob("*/agent-*"):
            m = re.match(r"agent-.*-g(\d+)$", d.name)
            gen = int(m.group(1)) if m else 0  # unsuffixed = generation 0
            max_gen = max(max_gen, gen)
        self._generation = max_gen + 1 if max_gen >= 0 else 0
        self._generation_initialized = True

    def _session_id(self, conversation_id: str) -> str:
        """Per-generation session id: the runtime rejects a persisted-id
        collision across restarts, so each runtime generation gets its own
        session; continuity comes from the transcript handoff instead."""
        return f"agent-{conversation_id}-g{self._generation}"

    def queue_context(self, conversation_id: str, text: str) -> None:
        """Queue context for this conversation's next agent turn."""
        if not text.strip():
            return
        with self._context_lock:
            self._pending_context.setdefault(conversation_id, []).append(text)

    def _take_context(self, conversation_id: str) -> str:
        """The queued context for this conversation (cleared as it is taken)."""
        with self._context_lock:
            items = self._pending_context.pop(conversation_id, [])
        if not items:
            return ""
        return "\n\n".join(items) + "\n\n"

    def _prior_transcript(self, conversation_id: str,
                          max_messages: int = 10) -> list[tuple[str, str]]:
        """User/assistant pairs from the previous generation's durable log."""
        import zstandard

        prior_gen = self._generation - 1
        candidates = [self.session_root.glob(
            f"*/agent-{conversation_id}-g{prior_gen}/session.jsonl.zstd")]
        if prior_gen == 0:
            # legacy unsuffixed sessions from before generations existed
            candidates.append(self.session_root.glob(
                f"*/agent-{conversation_id}/session.jsonl.zstd"))
        pairs: list[tuple[str, str]] = []
        for glob in candidates:
            for path in glob:
                dctx = zstandard.ZstdDecompressor()
                with open(path, "rb") as fh:
                    text = dctx.stream_reader(fh, read_across_frames=True) \
                        .read().decode("utf-8", errors="replace")
                pending_user: Optional[str] = None
                for line in text.splitlines():
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("type") != "assistant/message":
                        continue
                    data = record.get("data") or {}
                    message = data.get("message") or {}
                    blocks = message.get("content") or []
                    reply = "".join(b.get("text", "") for b in blocks
                                    if isinstance(b, dict)
                                    and b.get("type") == "text").strip()
                    if not reply:
                        continue
                    pairs.append((pending_user or "(continued)", reply[:400]))
                    pending_user = None
        return pairs[-max_messages:]

    def _handoff_prefix(self, conversation_id: str) -> str:
        if self._generation == 0 or conversation_id in self._handoff_done:
            return ""
        pairs = self._prior_transcript(conversation_id)
        self._handoff_done.add(conversation_id)
        if not pairs:
            return ""
        lines = ["[Context handoff — your previous session on this workspace, "
                 "for continuity:]"]
        for user, reply in pairs:
            lines.append(f"User: {user[:400]}")
            lines.append(f"You: {reply[:400]}")
        lines.append("[End handoff — continue naturally.]")
        return "\n".join(lines) + "\n\n"

    @property
    def runtime_running(self) -> bool:
        return self._harness is not None

    @property
    def permission_policy(self) -> str:
        if self._hook_installed:
            return "gateway-hook (PreToolUse requires a Hive approval decision)"
        if self._hook_disabled:
            return "full-access (HIVE_DSH_HOOK explicitly disabled)"
        return f"blocked (hook not configured: {self._hook_reason})"

    @property
    def gateway_configured(self) -> bool:
        return bool(self._hook_installed)

    @property
    def gateway_enforced(self) -> bool:
        """True only after a runtime has demonstrably loaded the hook."""
        return bool(self._hook_installed
                    and (self._hook_preflight_key
                         or self._hook_runtime_verified))

    @property
    def hook_status(self) -> dict:
        return {"installed": self._hook_installed,
                "disabled": self._hook_disabled,
                "verified": self._hook_preflight_key is not None,
                "reason": self._hook_reason, **self._hook_paths}

    def _ensure(self, base_url: str, api_key: str, model: str):
        cordis = self._runtime_cordis()
        key = (base_url, api_key, model, cordis)
        with self._lock:
            if self._harness is not None and self._target == key:
                return self._harness
            if self._harness is not None:
                try:
                    self._harness.close()
                except Exception:  # noqa: BLE001 - replacing a dead runtime
                    pass
                # new runtime generation: fresh session ids (the runtime
                # rejects persisted-id collisions across restarts)
                self._generation += 1
                self._handoff_done.clear()
            else:
                self._init_generation()
            # the dev node carrier is opt-in by design; the sidecar always
            # wants it (Windows has no exe carrier)
            os.environ["DSH_RUNTIME_MODE"] = "node"
            configured_home = os.environ.get("DSH_HOME", "").strip()
            dsh_home = (Path(configured_home).expanduser()
                        if configured_home else REPO_ROOT / ".dsh-home")
            os.environ.setdefault("DSH_HOME", str(dsh_home))
            dsh_home.mkdir(parents=True, exist_ok=True)
            self.session_root.mkdir(parents=True, exist_ok=True)
            self._hook_disabled = os.environ.get(
                "HIVE_DSH_HOOK", "").strip().lower() in (
                    "0", "off", "false", "no")
            if self._hook_disabled:
                self._hook_installed = False
                self._hook_reason = "disabled by HIVE_DSH_HOOK"
            else:
                try:
                    from harness.agent_hooks import ensure_hook_profile

                    hook = ensure_hook_profile(dsh_home, self.default_cwd)
                    self._hook_installed = bool(hook.get("installed"))
                    self._hook_reason = str(hook.get("reason") or "")
                    self._hook_paths = {
                        "hooks_file": str(hook.get("hooks") or ""),
                        "patch_file": str(hook.get("patch") or ""),
                    }
                except Exception as exc:  # noqa: BLE001 - report, don't hide
                    self._hook_installed = False
                    self._hook_reason = f"hook install failed: {exc}"
            if not self._hook_installed and not self._hook_disabled:
                raise RuntimeError(
                    "refusing the DSH agent: the PreToolUse approval hook is "
                    f"not installed ({self._hook_reason}); set HIVE_DSH_HOOK=off "
                    "only for an explicitly unguarded development run")
            dsh_bin = (os.environ.get("HIVE_DSH_BIN")
                       or shutil.which("dsh"))
            if dsh_bin and not self._hook_disabled:
                preflight_key = (
                    str(Path(dsh_bin).expanduser().resolve()),
                    str(dsh_home.expanduser().resolve()))
                if self._hook_preflight_key != preflight_key:
                    try:
                        from harness.agent_hooks import preflight_runtime

                        preflight_runtime(dsh_bin, dsh_home,
                                          self.default_cwd)
                    except Exception as exc:  # noqa: BLE001 - refuse, don't hide
                        raise RuntimeError(
                            "refusing the DSH agent: the approval hook did not "
                            f"load in the selected runtime ({exc})") from exc
                    self._hook_preflight_key = preflight_key
            self._hook_runtime_verified = False
            self._harness = DeepSeekHarness(
                provider="deepseek-official",
                base_url=base_url,
                api_key=api_key or "lm-studio",
                model=model,
                # A packaged Hive ships the runtime wheel; a development box
                # can point at an installed dsh (the npm package or a source
                # checkout) without changing the agent service.
                dsh_bin=dsh_bin,
                cwd=str(self.default_cwd),
                # the profile owns the runtime composition, the persistence
                # and the tools; the home is explicit (the SDK never guesses
                # ~/.dsh) and the cold profile handshake gets its own budget
                dsh_home=str(dsh_home),
                profile="sdk",
                initialize_timeout_seconds=120.0,
                env=_runtime_env(session_root=str(self.session_root),
                                 cordis=cordis),
            )
            if not dsh_bin and not self._hook_disabled:
                self._verify_packaged_runtime()
            self._target = key
            return self._harness

    def _verify_packaged_runtime(self) -> None:
        """Start the bundled runtime once and reject an inactive hook.

        The external-runtime path has ``preflight_runtime``.  A bundled
        runtime has no separate executable path, so its stderr is the only
        activation signal available before the first tool call.
        """
        starter = getattr(self._harness, "start", None)
        if not callable(starter):
            return
        try:
            starter()
        except Exception as exc:  # noqa: BLE001 - refuse, don't hide
            self._harness = None
            raise RuntimeError(
                f"refusing the DSH agent: the bundled runtime did not start "
                f"({exc})") from exc
        time.sleep(0.05)
        client = getattr(self._harness, "client", None)
        lines = getattr(client, "_stderr_lines", None)
        if lines is None:
            return
        from harness.agent_hooks import runtime_hook_error

        error = runtime_hook_error("\n".join(str(line) for line in list(lines)))
        if not error:
            self._hook_runtime_verified = True
            return
        try:
            self._harness.close()
        except Exception:  # noqa: BLE001 - preserve the original failure
            pass
        self._harness = None
        raise RuntimeError(
            f"refusing the DSH agent: the bundled runtime did not load the "
            f"approval hook ({error})")

    def _runtime_cordis(self) -> Optional[str]:
        """The applied-stack profile path, refreshed before each runtime build.

        A broken generator must never block chat: on error the previously
        resolved path (or ``None``) is reused.
        """
        if self._runtime_config is None:
            return None
        try:
            path = self._runtime_config()
        except Exception:  # noqa: BLE001
            return self._cordis_path
        self._cordis_path = str(path) if path else None
        return self._cordis_path

    def cancel(self, *, force: bool = False) -> dict:
        """Request cancellation of the in-flight run.

        ``force=False`` is the normal Stop: ask the runtime to shut down, then
        terminate it.  ``force=True`` is Cease: terminate the runtime and its
        child process tree immediately, without waiting for a graceful flush.
        """
        inflight = self._inflight
        if not inflight and not self.runtime_running:
            return {"ok": False, "note": "nothing in flight"}
        if inflight:
            self._cancel_requested.set()
        with self._lock:
            harness = self._harness
        if harness is None:
            if inflight:
                self._cancel_requested.clear()
            return {"ok": False, "note": "no runtime running"}
        target = self._force_stop_runtime if force else self.close
        threading.Thread(target=target, daemon=True).start()
        note = ("runtime process tree terminated"
                if force else "runtime terminated; run will settle as cancelled")
        return {"ok": True, "note": note}

    def cease_stop(self, *, conversation_id: str = "", session_id: str = "",
                   reason: str = "", force: bool = True) -> dict:
        """Callback used by ActivityHub.cease_now()."""
        result = self.cancel(force=force)
        result["reason"] = reason
        return result

    def _force_stop_runtime(self) -> None:
        """Kill the runtime and its descendants without a graceful wait."""
        with self._lock:
            harness = self._harness
            self._harness = None
            self._target = None
        proc = getattr(getattr(harness, "client", None), "_proc", None)
        if proc is not None and proc.poll() is None:
            try:
                import psutil

                parent = psutil.Process(proc.pid)
                tree = parent.children(recursive=True) + [parent]
                for child in tree:
                    try:
                        child.terminate()
                    except Exception:  # noqa: BLE001 - already gone
                        pass
                _, alive = psutil.wait_procs(tree, timeout=1.0)
                for child in alive:
                    try:
                        child.kill()
                    except Exception:  # noqa: BLE001 - already gone
                        pass
            except Exception:  # noqa: BLE001 - close() is the fallback
                pass
        if harness is not None:
            try:
                harness.close()
            except Exception:  # noqa: BLE001 - the process is already dead
                pass

    def close(self) -> None:
        with self._lock:
            if self._harness is not None:
                try:
                    self._harness.close()
                finally:
                    self._harness = None
                    self._target = None

    def run_turn(
        self,
        conversation_id: str,
        message: str,
        base_url: str,
        api_key: str,
        model: str,
        on_event: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        """Run one agent turn (blocking); stream activity via on_event."""
        activity = self.activity
        session_id = self._session_id(conversation_id)
        if activity is not None and activity.is_blocked(
                conversation_id=conversation_id, session_id=session_id):
            return {
                "final": "(ceased by user; resume before sending again)",
                "finish_reason": "ceased",
                "session_id": session_id,
                "events": 0,
            }

        harness = self._ensure(base_url, api_key, model)
        # ``_ensure`` can rebuild the runtime and bump the generation, so the
        # durable session id must be read after it.
        session_id = self._session_id(conversation_id)
        handoff = self._handoff_prefix(conversation_id)
        context = self._take_context(conversation_id)
        prefix = handoff + context
        message_to_send = prefix + message if prefix else message

        if activity is not None:
            activity.bind(conversation_id, session_id, sink=on_event)
            try:
                activity.record({
                    "type": "user",
                    "text": message_to_send,
                    "message": message,
                    "context": prefix,
                }, conversation_id=conversation_id, session_id=session_id)
            except Exception:  # noqa: BLE001 - activity must not block the turn
                pass
            if activity.is_blocked(conversation_id=conversation_id,
                                   session_id=session_id):
                activity.unbind(conversation_id, session_id)
                return {
                    "final": "(ceased by user; resume before sending again)",
                    "finish_reason": "ceased",
                    "session_id": session_id,
                    "events": 0,
                }

        def emit(event: dict) -> None:
            if activity is not None:
                try:
                    event = activity.record(
                        event, conversation_id=conversation_id,
                        session_id=session_id)
                except Exception:  # noqa: BLE001 - keep the stream alive
                    pass
            if on_event is not None:
                on_event(event)

        shaper = (ActivityShaper(session_id, emit,
                                 on_subagent=self.lineage_sink)
                  if (on_event is not None or activity is not None) else None)

        def on_notification(notification) -> None:
            if shaper is not None:
                shaper(notification)

        with self._run_lock:
            self._inflight = True
            try:
                result = harness.run(
                    message_to_send,
                    session_id=session_id,
                    on_notification=on_notification,
                )
            except Exception as exc:  # noqa: BLE001 - cancel/cease surface here
                if self._cancel_requested.is_set():
                    self._cancel_requested.clear()
                    ceased = (activity is not None and activity.is_blocked(
                        conversation_id=conversation_id,
                        session_id=session_id))
                    final = ("(ceased by user; resume before sending again)"
                             if ceased else "(cancelled by user)")
                    reason = "ceased" if ceased else "cancelled"
                    if activity is not None:
                        try:
                            activity.record({
                                "type": "turn", "phase": "end",
                                "finish_reason": reason, "final": final,
                            }, conversation_id=conversation_id,
                                session_id=session_id)
                        except Exception:  # noqa: BLE001
                            pass
                    return {
                        "final": final,
                        "finish_reason": reason,
                        "session_id": session_id,
                        "events": 0,
                    }
                if activity is not None:
                    try:
                        activity.record({
                            "type": "error", "error": str(exc),
                        }, conversation_id=conversation_id,
                            session_id=session_id)
                    except Exception:  # noqa: BLE001
                        pass
                raise
            finally:
                self._inflight = False
                if activity is not None:
                    activity.unbind(conversation_id, session_id)

        ceased = (activity is not None and activity.is_blocked(
            conversation_id=conversation_id, session_id=session_id))
        finish = "ceased" if ceased else result.finish_reason
        final = (result.final_response if not ceased
                 else "(ceased by user; resume before sending again)")
        if activity is not None:
            try:
                activity.record({
                    "type": "turn", "phase": "end",
                    "finish_reason": finish, "final": final,
                }, conversation_id=conversation_id, session_id=session_id)
            except Exception:  # noqa: BLE001
                pass
        # A cease can terminate the runtime after the model already returned;
        # do not let that flag cancel the next turn.
        self._cancel_requested.clear()
        return {
            "final": final,
            "finish_reason": finish,
            "session_id": result.session_id,
            "events": len(result.events or []),
        }


