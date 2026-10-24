"""Terminal presentation for the Hive CLI.

The CLI has to be useful in a pipe and pleasant in a terminal.  Everything in
this module is presentation only: JSON never goes through it, and colour is
disabled when the stream is not a TTY, when ``NO_COLOR`` is set, or when
``TERM=dumb``.  ``FORCE_COLOR=1`` and ``--no-color`` override the detection.

The palette is the Studio's black and yellow: #FFDD00 for headers, keys and
highlights, #FFB703 for amber/secondary emphasis, white for body text, grey for
secondary detail, and red only for a failure.  The Hive mark is an ASCII
wordmark printed at startup; there is no glyph lockup in command output.
"""
from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from typing import Iterable, Sequence, TextIO

RESET = "\x1b[0m"


def _truecolor() -> bool:
    return os.environ.get("COLORTERM", "").strip().lower() in ("truecolor", "24bit")


# The Studio palette: #FFDD00 is the yellow used for text on black; #FFB703 is
# its amber/secondary.  Use exact RGB where the terminal advertises truecolor
# and the nearest xterm-256 pair (220 / 214) otherwise.
_ACCENT = "38;2;255;221;0" if _truecolor() else "38;5;220"
_WARN = "38;2;255;183;3" if _truecolor() else "38;5;214"
_RED = "38;5;203"      # failures only
_GRAY = "38;5;245"
_DIM = "2"
_BOLD = "1"

_FORCE: bool | None = None

_UNICODE = {
    "hex": "⬡",
    "ok": "✔",
    "warn": "!",
    "fail": "✘",
    "dot": "●",
    "arrow": "›",
    "rule": "─",
    "bullet": "•",
}
_ASCII = {
    "hex": "#",
    "ok": "[ok]",
    "warn": "[!]",
    "fail": "[x]",
    "dot": "*",
    "arrow": ">",
    "rule": "-",
    "bullet": "-",
}

_LOGO = [
    r"  ██╗  ██╗██╗██╗   ██╗███████╗",
    r"  ██║  ██║██║██║   ██║██╔════╝",
    r"  ███████║██║██║   ██║█████╗  ",
    r"  ██╔══██║██║╚██╗ ██╔╝██╔══╝  ",
    r"  ██║  ██║██║ ╚████╔╝ ███████╗",
    r"  ╚═╝  ╚═╝╚═╝  ╚═══╝  ╚══════╝",
]
_LOGO_ASCII = [
    r"  #   #  ###  #   #  ####",
    r"  #   #   #   #   #  #   ",
    r"  #####   #   #   #  ### ",
    r"  #   #   #    # #   #   ",
    r"  #   #  ###    #    ####",
]


def set_color(enabled: bool | None) -> None:
    """Force colour on/off; ``None`` restores automatic detection."""
    global _FORCE
    _FORCE = enabled


def interactive(stream: TextIO | None = None) -> bool:
    """Whether this stream is a terminal (spinners and animation allowed)."""
    stream = stream or sys.stdout
    if os.environ.get("TERM") == "dumb":
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def symbols(stream: TextIO | None = None) -> dict[str, str]:
    """Unicode symbols when the stream can encode them, ASCII otherwise."""
    stream = stream or sys.stdout
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        "⬡ ✔ ✘ ● ─".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return _ASCII
    return _UNICODE


def logo(stream: TextIO | None = None) -> list[str]:
    """The startup wordmark, Unicode when the stream can encode it."""
    return list(_LOGO if symbols(stream) is _UNICODE else _LOGO_ASCII)


def color_enabled(stream: TextIO | None = None) -> bool:
    if _FORCE is not None:
        return _FORCE
    if os.environ.get("NO_COLOR"):
        return False
    force = os.environ.get("FORCE_COLOR")
    if force is not None and force.strip() not in ("", "0", "false", "no"):
        return True
    if os.environ.get("TERM") == "dumb":
        return False
    stream = stream or sys.stdout
    return bool(getattr(stream, "isatty", lambda: False)())


def paint(text: object, code: str, stream: TextIO | None = None) -> str:
    text = str(text)
    if not color_enabled(stream):
        return text
    return f"\x1b[{code}m{text}{RESET}"


def accent(text: object, stream: TextIO | None = None) -> str:
    return paint(text, _ACCENT, stream)


def bold(text: object, stream: TextIO | None = None) -> str:
    return paint(text, _BOLD, stream)


def dim(text: object, stream: TextIO | None = None) -> str:
    return paint(text, _GRAY, stream)


def _mark(kind: str, text: object, stream: TextIO | None = None) -> str:
    code = {"ok": _ACCENT, "warn": _WARN, "fail": _RED}[kind]
    sign = symbols(stream)[kind]
    return f"{paint(sign, code, stream)} {text}"


def ok(text: object, stream: TextIO | None = None) -> str:
    return _mark("ok", text, stream)


def warn(text: object, stream: TextIO | None = None) -> str:
    return _mark("warn", text, stream)


def fail(text: object, stream: TextIO | None = None) -> str:
    return _mark("fail", text, stream)


def info(text: object, stream: TextIO | None = None) -> str:
    return f"{paint(symbols(stream)['arrow'], _ACCENT, stream)} {text}"


def header(title: str, subtitle: str | None = None,
           stream: TextIO | None = None) -> None:
    """A small text lockup: ``hive · title`` (no glyph in the line)."""
    stream = stream or sys.stdout
    line = f"  {accent(bold('hive', stream), stream)}"
    if title:
        line += f"  {dim('·', stream)} {title}"
    print(line, file=stream)
    if subtitle:
        print(f"     {dim(subtitle, stream)}", file=stream)


def rule(width: int = 64, stream: TextIO | None = None) -> None:
    stream = stream or sys.stdout
    print(f"  {dim(symbols(stream)['rule'] * width, stream)}", file=stream)


def banner(version: str, stats: Iterable[tuple[str, str]],
           commands: Sequence[str], stream: TextIO | None = None) -> None:
    """The startup page for a bare ``hive``."""
    stream = stream or sys.stdout
    for line in logo(stream):
        print(f"  {accent(line, stream)}", file=stream)
    print(f"  {accent('hive', stream)} {dim(version, stream)}", file=stream)
    print(f"     {dim('local GGUF stacks · one engine per role', stream)}",
          file=stream)
    print(file=stream)
    for key, value in stats:
        print(f"  {accent(key.ljust(9), stream)} {value}", file=stream)
    print(file=stream)
    print(f"  {dim('get started', stream)}", file=stream)
    arrow = symbols(stream)["arrow"]
    for command in commands:
        print(f"    {accent(arrow, stream)} {command}", file=stream)
    print(file=stream)
    print(f"  {dim('hive --help for every command · docs/CLI.md', stream)}",
          file=stream)


def table(headers: Sequence[str], rows: Sequence[Sequence[object]],
          *, indent: str = "  ", stream: TextIO | None = None) -> None:
    """A plain table with a yellow header and a hairline rule."""
    stream = stream or sys.stdout
    data = [[("" if cell is None else str(cell)) for cell in row] for row in rows]
    widths: list[int] = []
    for index, head in enumerate(headers):
        width = len(head)
        for row in data:
            if index < len(row):
                width = max(width, len(row[index]))
        widths.append(width)

    def line(cells: Sequence[str]) -> str:
        return indent + "  ".join(
            cells[index].ljust(widths[index]) for index in range(len(headers))
        ).rstrip()

    head = [headers[index].upper() for index in range(len(headers))]
    if color_enabled(stream):
        print(line([bold(accent(cell, stream), stream) for cell in head]),
              file=stream)
        print(f"{indent}{dim(symbols(stream)['rule'] * (sum(widths) + 2 * (len(widths) - 1)), stream)}",
              file=stream)
    else:
        print(line(head), file=stream)
    for row in data:
        cells = [row[index] if index < len(row) else "" for index in range(len(headers))]
        print(line(cells), file=stream)


def kv(rows: Iterable[tuple[str, object]], *, indent: str = "  ",
       stream: TextIO | None = None) -> None:
    stream = stream or sys.stdout
    pairs = [(str(key), str(value)) for key, value in rows]
    width = max((len(key) for key, _ in pairs), default=0)
    for key, value in pairs:
        print(f"{indent}{dim(key.ljust(width), stream)}  {value}", file=stream)


@contextmanager
def spinner(label: str, stream: TextIO | None = None):
    """A one-line spinner for a blocking operation, only on a real terminal."""
    stream = stream or sys.stderr
    if not interactive(stream) or os.environ.get("HIVE_NO_SPINNER"):
        yield
        return
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    if symbols(stream) is _ASCII:
        frames = "|/-\\"
    stop = threading.Event()

    def run() -> None:
        index = 0
        while not stop.is_set():
            frame = frames[index % len(frames)]
            stream.write(f"\r  {dim(frame, stream)} {label}")
            stream.flush()
            index += 1
            stop.wait(0.1)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=0.5)
        stream.write("\r" + " " * (len(label) + 6) + "\r")
        stream.flush()
