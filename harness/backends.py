"""What each llama-server backend build is, where it lives, and whether it is installed.

The stack schema's ``backend`` field is the executable choice; this module is
the declaration behind it.  It exists so ``hive serve`` can say *which* builds
are installed, *why* it picked one, and what a user is missing — instead of
probing a directory convention and failing with "no llama-server binary for
backend 'rocm'".

The shape follows LocalAI's backend gallery (``backend/index.yaml``): one entry
per build with its acceleration variants and a note.  It deliberately does not
follow LocalAI's gRPC transport — our engines speak OpenAI over HTTP.
"""

from __future__ import annotations

import fnmatch
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Optional

from harness import paths

REPO_ROOT = paths.SOURCE_ROOT
MANIFEST_PATH = paths.manifest_path()
GITHUB_API = "https://api.github.com/repos"
_USER_AGENT = "hivebench-backend-fetch"

#: Which build to prefer for an accelerator, best first.  A vendor build beats
#: the portable one; the fork is a deliberate choice, so it is not auto-picked
#: ahead of rocm/cuda; CPU is last and only matches ``cpu``.
_PREFERENCE: tuple[str, ...] = ("rocm", "cuda", "sycl", "vulkan", "scion", "cpu")


def load(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Read the manifest.  Raises ``FileNotFoundError``/``ValueError``."""
    target = Path(path) if path else MANIFEST_PATH
    data = json.loads(target.read_text(encoding="utf-8"))
    entries = data.get("backends")
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"backend manifest {target} has no 'backends' object")
    return entries


def names(path: str | Path | None = None) -> tuple[str, ...]:
    """The declared backend names, in manifest order."""
    return tuple(load(path))


def resolve(name: str, *, root: str | Path | None = None,
            path: str | Path | None = None) -> dict[str, Any]:
    """One backend with its absolute binary path and whether it is installed."""
    entries = load(path)
    key = str(name).strip().lower()
    if key not in entries:
        raise KeyError(f"unknown backend {name!r}; known: {', '.join(entries)}")
    entry = dict(entries[key])
    entry["name"] = key
    base = Path(root) if root else paths.home()
    exe = "llama-server.exe" if os.name == "nt" else "llama-server"
    binary = base / str(entry.get("binary_dir") or f"tools/backends/{key}") / exe
    entry["binary"] = str(binary)
    entry["installed"] = binary.is_file()
    return entry


def installed(*, root: str | Path | None = None,
              path: str | Path | None = None) -> list[str]:
    """Declared backends whose binary is present, in manifest order."""
    return [name for name in names(path)
            if resolve(name, root=root, path=path)["installed"]]


def for_accelerator(kind: str, *, root: str | Path | None = None,
                    path: str | Path | None = None) -> Optional[str]:
    """The installed backend to use for ``amd``/``nvidia``/``intel``/``cpu``.

    ``None`` means nothing suitable is installed; the caller should fall back
    to the default llama-server binary rather than guess.
    """
    kind = str(kind or "").strip().lower()
    order = {name: index for index, name in enumerate(_PREFERENCE)}
    candidates = [
        name for name in names(path)
        if kind in (resolve(name, root=root, path=path).get("accelerators") or [])
        and resolve(name, root=root, path=path)["installed"]
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda name: order.get(name, len(order)))
    return candidates[0]


def platform_key() -> str:
    """The release asset family for this host, e.g. ``linux-x64``."""
    system = platform.system().strip().lower()
    machine = platform.machine().strip().lower()
    arch = {
        "x86_64": "x64", "amd64": "x64",
        "aarch64": "arm64", "arm64": "arm64",
    }.get(machine, machine)
    if system == "linux":
        return f"linux-{arch}"
    if system == "windows":
        return f"win-{arch}"
    return f"{system}-{arch}"


def _github_json(url: str, timeout: float = 30.0) -> Any:
    request = urllib.request.Request(
        url, headers={"User-Agent": _USER_AGENT,
                      "Accept": "application/vnd.github+json"})
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(300).decode("utf-8", "replace")
        raise RuntimeError(f"GitHub API {exc.code}: {detail}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"cannot reach GitHub: {exc}") from exc


def github_release(repo: str, tag: str | None = None,
                   timeout: float = 30.0) -> dict[str, Any]:
    """One GitHub release and its assets (latest build by default).

    ``/releases/latest`` is not used: llama.cpp's moving build releases are
    not GitHub's "latest" release, so ask for the newest release entry.
    """
    if tag:
        url = f"{GITHUB_API}/{repo}/releases/tags/{urllib.parse.quote(tag)}"
        data = _github_json(url, timeout=timeout)
    else:
        url = f"{GITHUB_API}/{repo}/releases?per_page=1"
        data = _github_json(url, timeout=timeout)
        if not isinstance(data, list) or not data:
            raise RuntimeError(f"{repo} has no releases")
        data = data[0]
    if not isinstance(data, dict):
        raise RuntimeError(f"unexpected release response for {repo}")
    assets = [
        {"name": str(asset.get("name") or ""),
         "size": int(asset.get("size") or 0),
         "url": str(asset.get("browser_download_url") or "")}
        for asset in (data.get("assets") or [])
        if asset.get("name") and asset.get("browser_download_url")
    ]
    return {"tag": str(data.get("tag_name") or tag or ""), "assets": assets}


def select_asset(backend: str, assets: list[dict[str, Any]], *,
                 key: str | None = None,
                 path: str | Path | None = None) -> dict[str, Any]:
    """Pick the release asset for this platform from the backend manifest."""
    entry = resolve(backend, path=path)
    release = entry.get("release")
    if not isinstance(release, dict):
        raise RuntimeError(
            f"backend {backend!r} has no downloadable release in the manifest")
    key = key or platform_key()
    pattern = (release.get("assets") or {}).get(key)
    if not pattern:
        raise RuntimeError(
            f"backend {backend!r} has no release asset for {key} "
            f"(known: {', '.join(sorted(release.get('assets') or {}))})")
    matches = [asset for asset in assets
               if fnmatch.fnmatch(asset["name"], pattern)]
    if not matches:
        raise RuntimeError(
            f"release has no asset matching {pattern!r} for {key}")
    # Stable choice when several wildcard matches exist: the lexicographically
    # newest name is the higher build/version in llama.cpp's naming.
    return sorted(matches, key=lambda asset: asset["name"])[-1]


def _download(url: str, destination: Path,
              progress: Callable[[int, int], None] | None = None,
              timeout: float = 300.0) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with destination.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"download failed: {exc}") from exc


def _safe_extract(archive: Path, destination: Path) -> None:
    """Extract an archive without allowing paths outside ``destination``."""
    destination = destination.resolve()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                target = (destination / member.filename).resolve()
                if destination not in target.parents and target != destination:
                    raise RuntimeError(f"archive member escapes: {member.filename}")
            bundle.extractall(destination)
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                target = (destination / member.name).resolve()
                if destination not in target.parents and target != destination:
                    raise RuntimeError(f"archive member escapes: {member.name}")
            try:
                bundle.extractall(destination, filter="data")
            except TypeError:  # Python < 3.11
                bundle.extractall(destination)
        return
    raise RuntimeError(f"unsupported release archive: {archive.name}")


def _find_binary(root: Path) -> Path:
    executable = "llama-server.exe" if os.name == "nt" else "llama-server"
    found = sorted(root.rglob(executable))
    if not found:
        raise RuntimeError(f"release archive has no {executable}")
    return found[0]


def _verify(binary: Path) -> None:
    env = dict(os.environ)
    libdir = str(binary.parent)
    if os.name == "nt":
        env["PATH"] = libdir + os.pathsep + env.get("PATH", "")
    else:
        env["LD_LIBRARY_PATH"] = libdir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    try:
        result = subprocess.run(
            [str(binary), "--version"], cwd=str(binary.parent), env=env,
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"{binary.name} did not run: {exc}") from exc
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip()[-400:]
        raise RuntimeError(
            f"{binary.name} failed --version (exit {result.returncode}): {tail}")


def default_fetch_backend(*, root: str | Path | None = None,
                          path: str | Path | None = None) -> str:
    """The portable build to fetch when the user names none.

    Vulkan is the cross-vendor default; a CPU-only user can ask for ``cpu``.
    If Vulkan is already installed the call is a no-op rather than quietly
    installing a less-portable build.
    """
    return "vulkan"


def fetch_backend(
    name: str,
    *,
    tag: str | None = None,
    force: bool = False,
    root: str | Path | None = None,
    path: str | Path | None = None,
    progress: Callable[[int, int], None] | None = None,
    verify: bool = True,
    release: dict[str, Any] | None = None,
    download: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Download and install one declared backend into ``<HIVE_HOME>/tools/backends/<name>``.

    The release archive is staged and verified before it replaces anything, so
    a failed download never leaves a half-installed engine.  ``release`` and
    ``download`` are injectable for tests; production uses the manifest's
    repository and GitHub's newest build.
    """
    entry = resolve(name, root=root, path=path)
    target = Path(entry["binary"]).parent
    if entry["installed"] and not force:
        return {"name": name, "path": str(target), "binary": entry["binary"],
                "installed": True, "already": True, "tag": None, "asset": None}

    declared = entry.get("release")
    if release is None:
        if not isinstance(declared, dict):
            raise RuntimeError(
                f"backend {name!r} has no downloadable release in the manifest")
        declared_repo = str(declared.get("repo") or "")
        if not declared_repo:
            raise RuntimeError(f"backend {name!r} release has no repo")
        meta = github_release(declared_repo, tag)
        release = {"tag": meta["tag"], "assets": meta["assets"]}
    else:
        release = {"tag": str(release.get("tag") or tag or ""),
                   "assets": list(release.get("assets") or [])}

    asset = select_asset(name, release["assets"], path=path)
    url = asset.get("url") or ""
    if not url:
        raise RuntimeError(f"asset {asset['name']!r} has no download URL")

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hive-backend-") as tmp:
        tmpdir = Path(tmp)
        archive = tmpdir / asset["name"]
        (download or _download)(url, archive, progress)
        extract_dir = tmpdir / "extract"
        extract_dir.mkdir()
        _safe_extract(archive, extract_dir)
        binary = _find_binary(extract_dir)
        source_dir = binary.parent
        staging = target.with_name(f".{target.name}.fetch")
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(source_dir, staging)
        staged_binary = staging / binary.name
        try:
            os.chmod(staged_binary, 0o755)
        except OSError:
            pass
        if verify:
            _verify(staged_binary)

    if target.exists() or target.is_symlink():
        if not force:
            raise RuntimeError(
                f"backend directory {target} already exists; use --force to replace it")
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
    os.replace(staging, target)
    installed_entry = resolve(name, root=root, path=path)
    return {"name": name, "path": str(target),
            "binary": installed_entry["binary"], "installed": True,
            "already": False, "tag": release["tag"], "asset": asset["name"]}


def rows(*, root: str | Path | None = None,
         path: str | Path | None = None) -> list[dict[str, Any]]:
    """Every declared backend, for ``hive serve --backends``."""
    return [resolve(name, root=root, path=path) for name in names(path)]
