"""Docker sandbox policy and the explicit local-fallback gate.

The sandbox is an execution boundary, so its image is a policy value, not a
per-request suggestion.  Docker is probed live; a missing daemon never turns
into a silent host execution.  The host path is available only when the policy
permits it *and* the caller marks this call as an explicit local-fallback
approval (or the policy is set to ``allow`` for a deliberate development box).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.paths import state_dir
from harness.state_store import AuditLog, JsonStore, StatePaths

DEFAULT_POLICY: dict[str, Any] = {
    "schema_version": 1,
    "mode": "docker",
    "docker_image": "hive-sandbox:1",
    "image_digest": "",
    "network": "none",
    # Empty uses the Docker daemon default (runc). Set runsc for gVisor or
    # kata-runtime for Kata Containers when the daemon has that runtime.
    "runtime": "",
    "memory": "512m",
    "cpus": "1",
    "pids_limit": 128,
    "timeout_s": 60,
    "local_fallback": "approval",  # approval | allow | deny
    "updated_at": None,
}
FALLBACK_MODES = ("approval", "allow", "deny")
#: The only host tree a child sandbox may mount.  The endpoint resolves the
#: requested path and rejects a root, parent, symlink escape, or Docker-hostile
#: injection before the container starts.
WORKSPACE_ROOT = Path("/srv/hive-ai/workspaces")
_NETWORKS = ("none", "bridge")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_IMAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,180}$")
_RUNTIME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")


class SandboxError(RuntimeError):
    """The sandbox policy cannot produce a safe execution command."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="seconds").replace("+00:00", "Z")


class SandboxPolicy:
    """Read/write ``settings/sandbox.json`` with a safe pinned-image contract."""

    def __init__(self, state: StatePaths | None = None) -> None:
        self.state = state or StatePaths(state_dir())
        self.store = JsonStore(self.state.sandbox_file(), default=DEFAULT_POLICY)

    def read(self) -> dict[str, Any]:
        data = {**DEFAULT_POLICY, **self.store.read()}
        if data.get("local_fallback") not in FALLBACK_MODES:
            data["local_fallback"] = "approval"
        return data

    def update(self, **fields: Any) -> dict[str, Any]:
        data = self.read()
        for key, value in fields.items():
            if value is not None:
                data[key] = value
        if data["local_fallback"] not in FALLBACK_MODES:
            raise SandboxError(
                f"local_fallback must be one of {', '.join(FALLBACK_MODES)}")
        if data.get("network") not in _NETWORKS:
            raise SandboxError(
                f"network must be one of {', '.join(_NETWORKS)}")
        runtime = str(data.get("runtime") or "").strip()
        if runtime and not _RUNTIME.match(runtime):
            raise SandboxError("runtime must be a Docker runtime name")
        data["runtime"] = runtime
        image = str(data.get("docker_image") or "").strip()
        if not _IMAGE.match(image):
            raise SandboxError(f"invalid sandbox image {image!r}")
        digest = str(data.get("image_digest") or "").strip().lower()
        if digest and not _DIGEST.match(digest):
            raise SandboxError("image_digest must be a 64-character sha256 hex")
        data["docker_image"] = image
        data["image_digest"] = digest
        data["updated_at"] = _utc()
        return self.store.write(data)

    def image_ref(self) -> str:
        """The exact image reference, refusing a mutable tag without a digest."""
        data = self.read()
        image = str(data["docker_image"]).strip()
        if "@sha256:" in image:
            tag, digest = image.rsplit("@sha256:", 1)
            if _DIGEST.match(digest):
                return image
            raise SandboxError("sandbox image has an invalid sha256 digest")
        digest = str(data.get("image_digest") or "").strip().lower()
        if not _DIGEST.match(digest):
            raise SandboxError(
                "sandbox image is not pinned; run `hive sandbox pin` or set "
                "image_digest to a 64-character sha256")
        return f"{image}@sha256:{digest}"

    def local_fallback(self, session_id: str | None, *,
                       approved: bool) -> tuple[bool, str]:
        mode = str(self.read().get("local_fallback") or "approval")
        if mode == "allow":
            return True, "policy allows local fallback"
        if mode == "deny":
            return False, "local fallback is disabled by sandbox policy"
        if approved and session_id:
            return True, "explicit local-fallback approval"
        if approved and not session_id:
            return False, "local fallback approval needs a session_id"
        return False, ("Docker is unavailable; local fallback needs an "
                       "explicit per-call approval")

    def record_fallback(self, session_id: str, command: str,
                        reason: str = "explicit local-fallback approval") -> None:
        AuditLog(self.state.tool_audit_file()).append({
            "event": "sandbox.local_fallback",
            "session_id": session_id,
            "command_sha256": _hash(command),
            "reason": reason,
            "ts": _utc(),
        })


def _hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def docker_host() -> str:
    """The daemon endpoint the Docker CLI will use."""
    return (os.environ.get("DOCKER_HOST", "").strip()
            or "unix:///var/run/docker.sock")


def is_rootless_host(host: str | None = None) -> bool:
    """True for a per-user rootless daemon socket, not the system socket."""
    value = str(host or docker_host()).lower()
    return "/run/user/" in value or "rootless" in value


def label_value(value: str) -> str:
    """A safe Docker label value for a session/conversation id."""
    return re.sub(r"[^A-Za-z0-9_.:-]", "_", str(value or ""))[:128]


def workspace_path(raw: str) -> str:
    """The canonical host path a child sandbox may mount at ``/workspace``.

    The path must resolve to an existing subdirectory of ``WORKSPACE_ROOT``.
    The root itself, a parent, a symlink escape, and a path Docker would parse
    as a volume option are all refused.  The caller passes the resolved string
    to ``docker run -v <path>:/workspace:rw``.
    """
    text = str(raw or "").strip()
    if not text:
        raise SandboxError("workspace is required")
    root = WORKSPACE_ROOT.resolve()
    try:
        path = Path(text).expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise SandboxError(f"workspace is not usable: {text}") from exc
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise SandboxError(f"workspace must be under {root}") from exc
    if not relative.parts:
        raise SandboxError(f"workspace must be a subdirectory of {root}")
    if not path.is_dir():
        raise SandboxError(f"workspace does not exist: {path}")
    rendered = str(path)
    if ":" in rendered or "," in rendered:
        raise SandboxError("workspace path contains a Docker-invalid character")
    return rendered


def kill_session_containers(session_id: str, *,
                            timeout: float = 10.0) -> dict[str, Any]:
    """Kill sandbox containers labelled for one agent session or its children.

    ``/v1/sandbox/exec`` labels every container with ``hive.session``; a child
    call also carries ``hive.parent``.  An emergency cease passes the parent
    session id and must stop both its own containers and any child container
    that outlived the in-process runtime.  A missing daemon or an empty id is
    reported, never silently ignored.
    """
    session_id = label_value(session_id)
    if not session_id:
        return {"killed": [], "error": "no session id"}
    status = docker_status()
    if not status["daemon"]:
        return {"killed": [], "error": status["detail"]}
    binary = str(status["binary"])
    labels = (f"hive.session={session_id}", f"hive.parent={session_id}")
    ids: list[str] = []
    for label in labels:
        try:
            listed = subprocess.run(
                [binary, "ps", "-q", "--filter", f"label={label}"],
                capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.SubprocessError) as exc:
            return {"killed": [], "error": f"cannot list containers: {exc}"}
        ids.extend(line.strip() for line in (listed.stdout or "").splitlines()
                   if line.strip())
    killed: list[str] = []
    errors: list[str] = []
    for container_id in dict.fromkeys(ids):
        try:
            proc = subprocess.run(
                [binary, "kill", container_id],
                capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f"{container_id}: {exc}")
            continue
        if proc.returncode == 0:
            killed.append(container_id)
        else:
            errors.append(
                f"{container_id}: {(proc.stderr or proc.stdout)[:200]}")
    return {"killed": killed, "errors": errors, "labels": list(labels)}


def docker_status() -> dict[str, Any]:
    """Probe Docker live; availability is never cached as policy truth."""
    docker_bin = shutil.which("docker")
    daemon = False
    detail = "docker is not installed"
    if docker_bin:
        detail = "docker daemon is not running"
        try:
            proc = subprocess.run(
                [docker_bin, "info"], capture_output=True, timeout=5, check=True)
            daemon = proc.returncode == 0
            if daemon:
                detail = "docker daemon is running"
        except (OSError, subprocess.SubprocessError):
            daemon = False
    return {"installed": bool(docker_bin), "daemon": daemon,
            "binary": docker_bin or "", "detail": detail,
            "host": docker_host(), "rootless": is_rootless_host()}


def pin_image(policy: SandboxPolicy, image: str | None = None,
              digest: str | None = None) -> dict[str, Any]:
    """Record a digest for the sandbox image, pulling it when necessary."""
    current = policy.read()
    image = str(image or current["docker_image"]).strip()
    if not _IMAGE.match(image):
        raise SandboxError(f"invalid sandbox image {image!r}")
    if digest:
        digest = digest.strip().lower()
        if not _DIGEST.match(digest):
            raise SandboxError("digest must be a 64-character sha256 hex")
        tag = image.split("@sha256:", 1)[0]
        return policy.update(docker_image=tag, image_digest=digest)

    if "@sha256:" in image:
        tag, found = image.rsplit("@sha256:", 1)
        if not _DIGEST.match(found):
            raise SandboxError("image has an invalid sha256 digest")
        return policy.update(docker_image=tag, image_digest=found)

    docker = docker_status()
    if not docker["daemon"]:
        raise SandboxError(docker["detail"] + "; cannot resolve an image digest")
    docker_bin = str(docker["binary"])
    fmt = "{{json .RepoDigests}}"
    proc = subprocess.run(
        [docker_bin, "image", "inspect", "--format", fmt, image],
        capture_output=True, text=True, timeout=30)
    import json

    digests: list[str] = []
    if proc.returncode == 0:
        try:
            digests = list(json.loads(proc.stdout or "[]"))
        except (TypeError, ValueError):
            digests = []
    if not digests:
        pull = subprocess.run(
            [docker_bin, "pull", image], capture_output=True, text=True,
            timeout=600)
        if pull.returncode != 0:
            raise SandboxError(
                f"cannot pull {image}: {(pull.stderr or pull.stdout)[:300]}")
        proc = subprocess.run(
            [docker_bin, "image", "inspect", "--format", fmt, image],
            capture_output=True, text=True, timeout=30)
        try:
            digests = list(json.loads(proc.stdout or "[]"))
        except (TypeError, ValueError):
            digests = []
    if not digests:
        raise SandboxError(f"Docker reports no RepoDigests for {image}")
    found = str(digests[0])
    if "@sha256:" not in found:
        raise SandboxError(f"unexpected Docker digest {found!r}")
    tag, hexdigest = found.rsplit("@sha256:", 1)
    if not _DIGEST.match(hexdigest):
        raise SandboxError(f"unexpected Docker digest {found!r}")
    return policy.update(docker_image=tag, image_digest=hexdigest)


def status(policy: SandboxPolicy | None = None) -> dict[str, Any]:
    policy = policy or SandboxPolicy()
    data = policy.read()
    docker = docker_status()
    pinned = False
    pin_error = ""
    try:
        policy.image_ref()
        pinned = True
    except SandboxError as exc:
        pin_error = str(exc)
    return {
        **docker,
        "policy": data,
        "pinned": pinned,
        "pin_error": pin_error,
        "fallback_requires_approval": data["local_fallback"] == "approval",
    }
