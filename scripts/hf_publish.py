#!/usr/bin/env python
"""Publish a folder to Hugging Face.

The Studio's releases (the white papers, the evaluation sets, the tuned
encoder) live in this repository and are published to the Hub from here.
The tool is deliberately small: it creates the repo if it does not exist,
uploads a folder, and prints the URL.  Nothing else - no commits to a
branch that is not yours, no deletions.

**Always show the operator the file list and get their go before an
upload.**  Publishing to the Hub is not a git push: it is a release, and
it is their call.  Use ``--dry-run`` to produce the list.

Token resolution, in order:

  1. ``--token``
  2. ``$HF_TOKEN`` / ``$HUGGING_FACE_HUB_TOKEN``
  3. ``~/.hf-token``
  4. ``~/.cache/huggingface/token``

Usage::

    python scripts/hf_publish.py --repo SkyIsNotGreen/splinter-memory \\
        --folder docs/splinter-memory-hf --type model --dry-run
    python scripts/hf_publish.py --repo SkyIsNotGreen/splinter-memory \\
        --folder docs/splinter-memory-hf --type model
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def resolve_token(args) -> str | None:
    """The token to publish with, or None."""
    if args.token:
        return args.token.strip()
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    for path in (Path.home() / ".hf-token",
                 Path.home() / ".cache" / "huggingface" / "token"):
        try:
            if path.exists():
                value = path.read_text(encoding="utf-8").strip()
                if value:
                    return value
        except OSError:
            continue
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True,
                        help="the repo id, e.g. SkyIsNotGreen/splinter-memory")
    parser.add_argument("--folder", required=True, type=Path,
                        help="the local folder to upload (its contents, not the folder itself)")
    parser.add_argument("--type", default="model", choices=("model", "dataset", "space"))
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token", default="")
    parser.add_argument("--message", default="")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be uploaded and stop")
    args = parser.parse_args(argv)

    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        print(f"no such folder: {folder}", file=sys.stderr)
        return 2
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    if not files:
        print(f"the folder is empty: {folder}", file=sys.stderr)
        return 2
    print(f"repo    : {args.repo} ({args.type})")
    print(f"folder  : {folder}")
    print(f"files   : {len(files)}")
    for path in files:
        print(f"  {path.relative_to(folder)}  ({path.stat().st_size} bytes)")
    if args.dry_run:
        print("\n(dry run - nothing was uploaded)")
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is not installed", file=sys.stderr)
        return 2
    token = resolve_token(args)
    if not token:
        print("no HF token found (--token, $HF_TOKEN, ~/.hf-token, "
              "~/.cache/huggingface/token)", file=sys.stderr)
        return 2

    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=args.repo, repo_type=args.type,
                        exist_ok=True, private=args.private)
    except Exception as exc:  # noqa: BLE001 - the upload may still work
        print(f"create_repo failed (it may already exist): {exc}", file=sys.stderr)
    message = args.message or f"Update {args.repo} from the HiveBench Studio"
    api.upload_folder(
        repo_id=args.repo,
        repo_type=args.type,
        folder_path=str(folder),
        token=token,
        commit_message=message,
    )
    url = f"https://huggingface.co/{args.repo}"
    if args.type == "dataset":
        url += "/datasets"
    print(f"\nuploaded to {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
