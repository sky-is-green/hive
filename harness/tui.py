"""Interactive terminal dashboard for Hive.

Bare ``hive`` on a TTY opens this; piped or ``HIVE_NO_TUI=1`` falls back to the
static dashboard in :mod:`harness.cli`.  It is deliberately small: four tabs
(Status, Models, Stacks, Runs), arrow-key navigation, a detail pane, the
supervisor log, and stop/refresh actions.  All reads go through the same
modules as the CLI, so the two surfaces cannot disagree.

No third-party dependency: the app is written against the standard-library
``curses`` module.  A terminal that cannot run curses, or a Windows build
without it, gets the static dashboard instead.
"""
from __future__ import annotations

import json
import locale
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

try:  # Windows builds without windows-curses take the static fallback
    import curses  # type: ignore
except ImportError:  # pragma: no cover - platform dependent
    curses = None  # type: ignore

from harness import cli as ops
from harness import serve_state, termui
from harness.stack import schema

TABS = ("Status", "Models", "Stacks", "Runs")


@dataclass
class Row:
    cells: list[str]
    data: dict[str, Any]
    selectable: bool = True
    kind: str = "row"


def _ns() -> Any:
    import argparse
    return argparse.Namespace(models_dir=None, stacks_dir=None)


class App:
    def __init__(self, stdscr: Any) -> None:
        self.screen = stdscr
        self.tab = 0
        self.sel = 0
        self.top = 0
        self.rows: list[Row] = []
        self.message = ""
        self.mode = "main"          # main | detail | logs | help
        self.overlay_title = ""
        self.overlay: list[str] = []
        self.offset = 0
        self.color = bool(termui.color_enabled())
        self._init_colors()
        self.sel_attr = (self.pairs.get("sel", 0) if self.color
                         else curses.A_REVERSE)
        if self.color:
            try:
                self.screen.bkgd(" ", curses.color_pair(1))
            except curses.error:
                pass
        self._load()

    # ------------------------------------------------------------------
    # colours / drawing
    # ------------------------------------------------------------------
    def _init_colors(self) -> None:
        self.pairs: dict[str, int] = {}
        if curses is None or not curses.has_colors():
            self.color = False
            return
        try:
            curses.start_color()
            # Explicit black background: the whole TUI is black + yellow.
            white = curses.COLOR_WHITE
            # xterm 220 is #FFD700, the nearest 256-colour step to the Studio's
            # #FFDD00; 214 is the amber #FFB703.
            yellow = 220 if curses.COLORS >= 256 else curses.COLOR_YELLOW
            gray = 245 if curses.COLORS >= 256 else curses.COLOR_WHITE
            red = 203 if curses.COLORS >= 256 else curses.COLOR_RED
            self.pairs = {
                "text": curses.color_pair(1) if self._pair(1, white, curses.COLOR_BLACK) else 0,
                "amber": curses.color_pair(2) if self._pair(2, yellow, curses.COLOR_BLACK) else 0,
                "gray": curses.color_pair(3) if self._pair(3, gray, curses.COLOR_BLACK) else 0,
                "red": curses.color_pair(4) if self._pair(4, red, curses.COLOR_BLACK) else 0,
                "sel": curses.color_pair(5) if self._pair(5, curses.COLOR_BLACK, yellow) else 0,
            }
        except curses.error:
            self.color = False

    def _pair(self, index: int, fg: int, bg: int) -> bool:
        try:
            curses.init_pair(index, fg, bg)
            return True
        except curses.error:
            return False

    def attr(self, name: str = "") -> int:
        if not self.color:
            return 0
        value = self.pairs.get(name, 0)
        if name == "amber":
            value |= curses.A_BOLD
        elif name == "gray":
            value |= curses.A_DIM
        return value

    def put(self, y: int, x: int, text: str, attr: int = 0) -> None:
        if curses is None:
            return
        height, width = self.screen.getmaxyx()
        if y < 0 or y >= height or x >= width:
            return
        text = str(text).replace("\t", " ")
        try:
            self.screen.addnstr(y, x, text, max(0, width - x - 1), attr)
        except (curses.error, UnicodeEncodeError):
            pass

    def _rule(self, y: int) -> None:
        _, width = self.screen.getmaxyx()
        self.put(y, 0, termui.symbols()["rule"] * (width - 1), self.attr("gray"))

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------
    def _load(self) -> None:
        self.rows = []
        self.sel = 0
        self.top = 0
        if self.tab == 0:
            self._load_status()
        elif self.tab == 1:
            self._load_models()
        elif self.tab == 2:
            self._load_stacks()
        else:
            self._load_runs()
        self._select_first()

    def _load_status(self) -> None:
        try:
            records = serve_state.list_records()
        except OSError:
            records = []
        for record in records:
            live = serve_state.pid_matches(record)
            uptime = serve_state.format_duration(
                max(0.0, time.time() - float(record.get("started_at") or time.time())))
            state = "running" if live else "stale"
            self.rows.append(Row(
                [str(record.get("name")), str(record.get("pid")),
                 str(record.get("mode") or "foreground"), uptime, state],
                {"kind": "serve", "record": record}, True, "serve"))
            for tier in list(record.get("tiers") or []):
                endpoint = ops._tier_endpoint(record, tier)
                probe = ops._probe(endpoint, timeout=0.35) if live and endpoint else {}
                tier_state = ("up" if probe.get("up") else
                              "down" if probe else "-")
                self.rows.append(Row(
                    [f"  {tier.get('role') or '?'}", endpoint,
                     f"ctx {tier.get('ctx') or '?'}", str(tier.get("backend") or ""),
                     tier_state],
                    {"kind": "tier", "record": record, "tier": tier}, False, "tier"))
        if not self.rows:
            self.message = "Nothing is running.  Start one with: hive serve <stack> --detach"

    def _load_models(self) -> None:
        try:
            manager = ops._models_manager(_ns())
            rows = manager.list_local(use_cache=False)
        except (OSError, ValueError) as exc:
            self.message = f"Cannot read the model library: {exc}"
            return
        for row in rows:
            self.rows.append(Row(
                [str(row.get("name")), f"{float(row.get('size_gb') or 0):.2f} GiB",
                 str(row.get("quantization") or "-"),
                 str(row.get("context_length") or "-")],
                row, True, "model"))
        if not self.rows:
            self.message = "No GGUF models yet.  Download one with: hive models download <repo>"

    def _load_stacks(self) -> None:
        try:
            directory = ops._stacks_directory(None)
            rows = schema.list_stacks_in(directory)
        except (OSError, ValueError) as exc:
            self.message = f"Cannot read stacks: {exc}"
            return
        self._stack_dir = directory
        for row in rows:
            self.rows.append(Row(
                [str(row.get("name")), str(row.get("tiers")),
                 ",".join(row.get("roles") or [])],
                {"kind": "stack", "name": row.get("name")}, True, "stack"))
        if not self.rows:
            self.message = f"No saved stacks in {directory}"

    def _load_runs(self) -> None:
        runs_dir = ops.CASCADE_RUNS
        if not runs_dir.is_dir():
            self.message = f"No cascade runs in {runs_dir}"
            return
        for path in sorted(runs_dir.iterdir(), key=lambda p: p.name, reverse=True):
            report = path / "report.json"
            if not path.is_dir() or not report.is_file():
                continue
            summary: dict = {}
            try:
                summary = dict(json.loads(report.read_text(encoding="utf-8")).get("summary") or {})
            except (OSError, ValueError, AttributeError):
                pass
            acc = summary.get("cascade_accuracy")
            esc = summary.get("escalation_rate")
            self.rows.append(Row(
                [path.name, str(summary.get("tasks", "-")),
                 f"{acc:.0%}" if isinstance(acc, (int, float)) else "-",
                 f"{esc:.0%}" if isinstance(esc, (int, float)) else "-"],
                {"kind": "run", "path": str(report), "summary": summary}, True, "run"))
        if not self.rows:
            self.message = f"No cascade runs in {runs_dir}"

    def _select_first(self) -> None:
        for index, row in enumerate(self.rows):
            if row.selectable:
                self.sel = index
                return
        self.sel = 0

    def current(self) -> Row | None:
        if 0 <= self.sel < len(self.rows):
            return self.rows[self.sel]
        return None

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------
    def draw(self) -> None:
        if curses is None:
            return
        self.screen.erase()
        height, width = self.screen.getmaxyx()
        if height < 12 or width < 50:
            self.put(0, 0, "terminal too small (need at least 50x12)", curses.A_BOLD)
            self.screen.refresh()
            return
        self.put(0, 2, f"hive {ops._version()}", self.attr("amber"))
        stamp = time.strftime("%H:%M:%S")
        self.put(0, max(2, width - len(stamp) - 2), stamp, self.attr("gray"))

        x = 2
        for index, name in enumerate(TABS):
            label = f" {index + 1} {name} "
            if index == self.tab:
                self.put(1, x, label, self.sel_attr)
            else:
                self.put(1, x, label, self.attr("gray"))
            x += len(label) + 1
        self._rule(2)

        if self.mode == "main":
            self._draw_main(height, width)
        else:
            self._draw_overlay(height, width)

        if self.message:
            self.put(height - 2, 2, self.message[:width - 4], self.attr("gray"))
        self._footer(height, width)
        self.screen.refresh()

    def _draw_main(self, height: int, width: int) -> None:
        if self.tab == 0:
            self._draw_status(height, width)
        elif self.tab == 1:
            self._draw_table(["NAME", "SIZE", "QUANT", "CTX"],
                             [34, 10, 10, 8], 3, height - 3)
        elif self.tab == 2:
            self._draw_table(["NAME", "TIERS", "ROLES"],
                             [28, 7, 30], 3, height - 3)
        else:
            self._draw_table(["RUN", "TASKS", "CASCADE", "ESCALATE"],
                             [40, 7, 10, 10], 3, height - 3)

    def _draw_status(self, height: int, width: int) -> None:
        head = ["NAME", "PID", "MODE", "UPTIME", "STATE"]
        widths = [26, 8, 12, 10, 8]
        self.put(3, 3, "  ".join(h.ljust(w) for h, w in zip(head, widths)),
                 self.attr("amber"))
        y = 4
        view = height - 7
        for offset in range(self.top, min(len(self.rows), self.top + view)):
            row = self.rows[offset]
            if y >= height - 3:
                break
            attr = self.attr("text")
            if row.kind == "serve" and offset == self.sel:
                attr = self.sel_attr
            elif row.kind == "serve" and row.cells[-1] == "stale":
                attr = self.attr("red")
            elif row.kind == "tier":
                attr = self.attr("red") if row.cells[-1] == "down" else self.attr("gray")
            marker = ">" if offset == self.sel else " "
            if row.kind == "serve":
                text = "  ".join(
                    [str(row.cells[0])[:widths[0]], row.cells[1].rjust(widths[1]),
                     row.cells[2][:widths[2]], row.cells[3].rjust(widths[3]),
                     row.cells[4]])
            else:
                endpoint = row.cells[1]
                room = max(10, width - widths[0] - 34)
                text = (f"  {row.cells[0][:widths[0]]}  {endpoint[:room].ljust(room)}  "
                        f"{row.cells[2]}  {row.cells[3]:<10} {row.cells[4]}")
            self.put(y, 2, f"{marker} {text}", attr)
            y += 1
        if not self.rows:
            self.put(5, 4, "No serves are running.", self.attr("gray"))
            self.put(6, 4, "Start one:  hive serve <stack> --detach",
                     self.attr("amber"))

    def _draw_table(self, headers: list[str], widths: list[int],
                    y: int, bottom: int) -> None:
        self.put(y, 3, "  ".join(
            header.ljust(widths[index]) for index, header in enumerate(headers)),
            self.attr("amber"))
        view = max(1, bottom - y - 1)
        for offset in range(self.top, min(len(self.rows), self.top + view)):
            row = self.rows[offset]
            attr = self.sel_attr if offset == self.sel else self.attr("text")
            marker = ">" if offset == self.sel else " "
            cells = []
            for index, cell in enumerate(row.cells):
                width = widths[index] if index < len(widths) else 20
                cells.append(str(cell)[:width].ljust(width))
            self.put(y + 1 + (offset - self.top), 2,
                     f"{marker} " + "  ".join(cells), attr)
        if not self.rows:
            self.put(y + 2, 4, "Nothing here yet.", self.attr("gray"))

    def _footer(self, height: int, width: int) -> None:
        context = {
            0: "enter details  l log  x stop",
            1: "enter details",
            2: "enter details",
            3: "enter report",
        }[self.tab]
        line = f" {context}  ·  r refresh  ·  ? help  ·  q quit "
        self.put(height - 1, 0, line[-width:], self.sel_attr)

    def _draw_overlay(self, height: int, width: int) -> None:
        top, bottom = 4, height - 3
        self.put(top - 1, 2, f" {self.overlay_title} ", self.sel_attr)
        view = bottom - top
        for index in range(view):
            source = self.offset + index
            if source >= len(self.overlay):
                break
            self.put(top + index, 3, self.overlay[source][:width - 6],
                     self.attr("text"))
        self.put(height - 2, 2,
                 "↑↓ scroll · r reload · q back"[:width - 4],
                 self.attr("gray"))

    # ------------------------------------------------------------------
    # actions
    # ------------------------------------------------------------------
    def _detail(self) -> None:
        row = self.current()
        if row is None:
            return
        if row.kind == "serve":
            record = row.data["record"]
            lines = [
                f"name      {record.get('name')}",
                f"pid       {record.get('pid')}",
                f"mode      {record.get('mode')}",
                f"stack     {record.get('stack')}",
                f"started   {record.get('started_iso') or record.get('started_at')}",
                f"log       {record.get('log_path') or '(foreground terminal)'}",
                "",
            ]
            for tier in list(record.get("tiers") or []):
                lines += [
                    f"tier {tier.get('role')}",
                    f"  endpoint  {ops._tier_endpoint(record, tier)}",
                    f"  model     {tier.get('model')}",
                    f"  ctx       {tier.get('ctx')}",
                    f"  backend   {tier.get('backend')}",
                ]
            self._open_overlay(f"serve {record.get('name')}", lines)
        elif row.kind == "model":
            data = row.data
            lines = [f"{key}: {data[key]}" for key in
                     ("name", "file", "size_gb", "quantization",
                      "context_length", "architecture")]
            self._open_overlay(f"model {data.get('name')}", lines)
        elif row.kind == "stack":
            try:
                stack = schema.load_stack_in(row.data["name"], self._stack_dir)
                lines = json.dumps(stack.to_dict(), indent=2).splitlines()
            except (OSError, ValueError) as exc:
                lines = [f"cannot read stack: {exc}"]
            self._open_overlay(f"stack {row.data['name']}", lines)
        elif row.kind == "run":
            summary = row.data.get("summary") or {}
            lines = [f"{key}: {ops._metric(key, value)}"
                     for key, value in summary.items()]
            self._open_overlay(f"run {row.data['path'].split('/')[-2]}", lines)

    def _open_overlay(self, title: str, lines: list[str]) -> None:
        self.mode = "detail"
        self.overlay_title = title
        self.overlay = lines or ["(empty)"]
        self.offset = 0

    def _logs(self) -> None:
        row = self.current()
        if row is None or row.kind != "serve":
            return
        record = row.data["record"]
        path = record.get("log_path")
        if not path:
            self.message = "This serve is in the foreground; its log is on its terminal."
            return
        try:
            self.overlay = serve_state.tail_lines(path, 0)
        except OSError as exc:
            self.message = f"Cannot read {path}: {exc}"
            return
        self.mode = "logs"
        self.overlay_title = f"log {record.get('name')}"
        self.offset = max(0, len(self.overlay) - 1)

    def _stop(self) -> None:
        row = self.current()
        if row is None or row.kind != "serve":
            return
        record = row.data["record"]
        name = str(record.get("name"))
        if not serve_state.pid_matches(record):
            serve_state.remove_record(name)
            self.message = f"removed stale record {name}"
            self._load()
            return
        self.message = f"stopping {name} ..."
        self.draw()
        ok, detail = serve_state.terminate(record["pid"], force=False, timeout=8.0)
        if ok:
            serve_state.remove_record(name)
            self.message = f"stopped {name}"
        else:
            self.message = f"could not stop {name}: {detail}"
        self._load()

    def _tab(self, delta: int) -> None:
        self.tab = (self.tab + delta) % len(TABS)
        self._load()

    def _move(self, delta: int) -> None:
        if not self.rows:
            return
        index = self.sel + delta
        while 0 <= index < len(self.rows):
            if self.rows[index].selectable:
                self.sel = index
                break
            index += delta
        self._clamp_top()

    def _clamp_top(self) -> None:
        if curses is None:
            return
        height, _ = self.screen.getmaxyx()
        view = max(1, height - 8)
        if self.sel < self.top:
            self.top = self.sel
        elif self.sel >= self.top + view:
            self.top = self.sel - view + 1

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------
    def handle(self, key: int) -> bool:
        if self.mode in ("detail", "logs"):
            if key in (ord("q"), 27):
                self.mode = "main"
                return False
            if key in (curses.KEY_UP, ord("k")):
                self.offset = max(0, self.offset - 1)
            elif key in (curses.KEY_DOWN, ord("j")):
                self.offset = min(max(0, len(self.overlay) - 1), self.offset + 1)
            elif key == curses.KEY_PPAGE:
                self.offset = max(0, self.offset - 10)
            elif key == curses.KEY_NPAGE:
                self.offset = min(max(0, len(self.overlay) - 1), self.offset + 10)
            elif key == ord("r") and self.mode == "logs":
                row = self.current()
                if row and row.kind == "serve":
                    path = row.data["record"].get("log_path")
                    if path:
                        try:
                            self.overlay = serve_state.tail_lines(path, 0)
                        except OSError:
                            pass
            return False
        if self.mode == "help":
            if key in (ord("q"), 27, ord("?")):
                self.mode = "main"
            return False
        if key in (ord("q"),):
            return True
        if key in (curses.KEY_LEFT, ord("h")):
            self._tab(-1)
        elif key in (curses.KEY_RIGHT, 9):  # TAB
            self._tab(1)
        elif key in (curses.KEY_UP, ord("k")):
            self._move(-1)
        elif key in (curses.KEY_DOWN, ord("j")):
            self._move(1)
        elif key == curses.KEY_RESIZE:
            pass
        elif key == ord("r"):
            self._load()
            self.message = "refreshed"
        elif key in (10, 13, curses.KEY_ENTER):
            self._detail()
        elif key == ord("?") or key == curses.KEY_F1:
            self.mode = "help"
            self.overlay_title = "keys"
            self.overlay = [
                "1-4, ←/→, TAB   switch tab",
                "↑/↓, j/k          move",
                "ENTER            details",
                "l                supervisor log (Status)",
                "x                stop selected serve (Status)",
                "r                refresh",
                "q                quit",
                "",
                "Data is read through the same modules as the CLI.",
            ]
            self.offset = 0
        elif self.tab == 0 and key == ord("l"):
            self._logs()
        elif self.tab == 0 and key == ord("x"):
            self._stop()
        elif ord("1") <= key <= ord("4"):
            index = key - ord("1")
            if index != self.tab:
                self.tab = index
                self._load()
        return False

    def splash(self) -> None:
        """The fish-like greeting: ASCII wordmark, then the dashboard."""
        if os.environ.get("HIVE_NO_SPLASH") or curses is None:
            return
        height, width = self.screen.getmaxyx()
        lines = termui.logo()
        widest = max((len(line) for line in lines), default=0)
        if height < len(lines) + 9 or width < widest + 4:
            return
        top = max(1, (height - len(lines) - 6) // 2)
        for index, line in enumerate(lines):
            self.put(top + index, max(0, (width - len(line)) // 2),
                     line, self.attr("amber"))
        tagline = "local GGUF stacks · one engine per role"
        version = f"hive {ops._version()}"
        hint = "press any key"
        self.put(top + len(lines) + 1,
                 max(0, (width - len(tagline)) // 2), tagline, self.attr("gray"))
        self.put(top + len(lines) + 2,
                 max(0, (width - len(version)) // 2), version, self.attr("amber"))
        self.put(top + len(lines) + 4,
                 max(0, (width - len(hint)) // 2), hint, self.attr("gray"))
        self.screen.refresh()
        try:
            self.screen.timeout(1200)
            self.screen.getch()
        except curses.error:
            pass
        finally:
            try:
                self.screen.timeout(-1)
            except curses.error:
                pass

    def loop(self) -> None:
        while True:
            self.draw()
            key = self.screen.getch()
            if key == -1:
                continue
            if self.handle(key):
                return


def run() -> int:
    """Run the interactive dashboard; return 0 on quit.

    Returns 1 without touching the terminal when there is no TTY or curses is
    unavailable, so the caller can fall back to the static dashboard.
    """
    if curses is None or os.environ.get("HIVE_NO_TUI"):
        return 1
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return 1
    if os.environ.get("TERM") == "dumb":
        return 1
    try:
        locale.setlocale(locale.LC_ALL, "")
    except locale.Error:
        pass

    def _main(screen: Any) -> None:
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        screen.keypad(True)
        app = App(screen)
        app.splash()
        app.loop()

    try:
        curses.wrapper(_main)
    except KeyboardInterrupt:
        return 0
    except Exception:  # noqa: BLE001 - restore the terminal, then fall back
        if os.environ.get("HIVE_DEBUG"):
            raise
        return 1
    return 0
