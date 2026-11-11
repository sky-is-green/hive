/*
 * HiveProcessesPanel — process manager, engine log and raw server status.
 *
 * Read + one explicit mutation (kill). The kill path is the same API the
 * legacy Studio used:
 *
 *   GET  /api/v1/processes         spawned servers + child shells + sidecar
 *   GET  /api/v1/server/log?tail=  llama-server log tail
 *   GET  /api/v1/server/status     instances, loaded models, hardware
 *   GET  /api/v1/server/memory     sidecar RSS, threads, conversations
 *   POST /api/v1/processes/kill    {pid} or {key}
 *
 * Killing is always gated on a confirm() and never runs on mount/refresh.
 *
 * window.HiveProcessesPanel.mount(el, opts?) -> { refresh, state }
 *   opts.base     : API prefix (default "/api/v1")
 *   opts.confirm  : confirm function (injectable for tests)
 *   opts.fetchJSON / opts.postJSON : injectable for tests
 */
(function (global) {
  "use strict";

  var BASE = "/api/v1";

  function esc(value) {
    return String(value == null ? "" : value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function fmtMb(value) {
    var n = Number(value);
    if (!isFinite(n)) return "—";
    return Math.round(n) + " MB";
  }

  function fmtPct(value) {
    var n = Number(value);
    if (!isFinite(n)) return "—";
    return n.toFixed(1) + "%";
  }

  function shortCmd(cmdline) {
    if (!Array.isArray(cmdline) || !cmdline.length) return "";
    var text = cmdline.join(" ");
    return text.length > 120 ? text.slice(0, 117) + "…" : text;
  }

  async function defaultFetchJSON(path) {
    var res = await fetch(path, { cache: "no-store" });
    var body = null;
    try { body = await res.json(); } catch (_) { /* keep null */ }
    if (!res.ok) {
      var detail = body && (body.detail || body.error || body.message);
      throw new Error(detail || (res.status + " " + path));
    }
    return body;
  }

  async function defaultPostJSON(path, payload) {
    var res = await fetch(path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload || {}),
    });
    var body = null;
    try { body = await res.json(); } catch (_) { /* keep null */ }
    if (!res.ok) {
      var detail = body && (body.detail || body.error || body.message);
      throw new Error(detail || (res.status + " " + path));
    }
    return body;
  }

  function node(tag, cls, value) {
    var el = document.createElement(tag);
    if (cls) el.className = cls;
    if (value !== undefined && value !== null) el.textContent = String(value);
    return el;
  }

  function card(label, value, hint, tone) {
    var el = node("div", "card" + (tone ? " " + tone : ""));
    el.append(
      node("div", "label", label),
      node("div", "value", value),
      node("div", "hint", hint || "")
    );
    return el;
  }

  function emptyRow(cols, message) {
    var tr = document.createElement("tr");
    var td = document.createElement("td");
    td.colSpan = cols;
    td.className = "note";
    td.textContent = message;
    tr.appendChild(td);
    return tr;
  }

  function createPanel(rootEl, opts) {
    opts = opts || {};
    var base = opts.base || BASE;
    var fetchJSON = opts.fetchJSON || defaultFetchJSON;
    var postJSON = opts.postJSON || defaultPostJSON;
    var confirmFn = opts.confirm || (global.confirm ? global.confirm.bind(global) : function () { return false; });
    var state = {
      status: "idle",
      processes: null,
      log: null,
      server: null,
      memory: null,
      errors: {},
      notice: "",
      killPending: {},
    };
    var ui = {};

    function build() {
      rootEl.innerHTML = "";
      var root = node("section", "hive-processes");

      ui.message = node("div", "note");
      ui.message.style.margin = "0 0 10px";

      ui.cards = node("div", "cards");

      var procPanel = node("div", "panel");
      var procHead = node("div", "panel-head");
      procHead.append(node("h2", "", "Process manager"), node("span", "muted", "CPU / RAM · kill"));
      ui.table = document.createElement("table");
      var thead = document.createElement("thead");
      var hr = document.createElement("tr");
      ["PID", "Kind", "Name", "CPU", "RAM", "Port", "Status", ""].forEach(function (label) {
        var th = document.createElement("th");
        th.textContent = label;
        hr.appendChild(th);
      });
      thead.appendChild(hr);
      ui.table.appendChild(thead);
      ui.tbody = document.createElement("tbody");
      ui.table.appendChild(ui.tbody);
      var procWrap = node("div", "table-wrap");
      procWrap.appendChild(ui.table);
      procPanel.append(procHead, procWrap);

      var logPanel = node("div", "panel");
      var logHead = node("div", "panel-head");
      logHead.append(node("h2", "", "Engine log"), node("span", "muted", "tail 200"));
      ui.log = document.createElement("pre");
      ui.log.style.margin = "0";
      ui.log.style.padding = "12px 15px";
      ui.log.style.background = "#0d0f12";
      ui.log.style.color = "#c9d1d9";
      ui.log.style.maxHeight = "280px";
      ui.log.style.overflow = "auto";
      ui.log.style.fontSize = "11.5px";
      ui.log.style.whiteSpace = "pre-wrap";
      logPanel.append(logHead, ui.log);

      var rawPanel = node("div", "panel");
      var rawHead = node("div", "panel-head");
      rawHead.append(node("h2", "", "Server status"), node("span", "muted", "raw JSON"));
      ui.raw = document.createElement("pre");
      ui.raw.style.margin = "0";
      ui.raw.style.padding = "12px 15px";
      ui.raw.style.background = "#0d0f12";
      ui.raw.style.color = "#c9d1d9";
      ui.raw.style.maxHeight = "300px";
      ui.raw.style.overflow = "auto";
      ui.raw.style.fontSize = "11.5px";
      ui.raw.style.whiteSpace = "pre-wrap";
      var details = document.createElement("details");
      details.appendChild(node("summary", "note", "Show raw /v1/server/status"));
      details.appendChild(ui.raw);
      details.style.padding = "10px 15px";
      rawPanel.append(rawHead, details);

      root.append(ui.message, ui.cards, procPanel, logPanel, rawPanel);
      rootEl.appendChild(root);

      ui.tbody.addEventListener("click", function (event) {
        var button = event.target.closest("button[data-kill-pid]");
        if (button) killProcess(Number(button.getAttribute("data-kill-pid")));
      });
    }

    function renderMessage() {
      var errors = Object.keys(state.errors).filter(function (k) { return state.errors[k]; });
      if (state.status === "loading") {
        ui.message.textContent = "Reading processes…";
        ui.message.style.color = "var(--muted)";
      } else if (state.notice) {
        ui.message.textContent = state.notice;
        ui.message.style.color = "var(--good)";
      } else if (errors.length) {
        ui.message.textContent = errors.map(function (k) { return k + ": " + state.errors[k]; }).join(" · ");
        ui.message.style.color = "var(--bad)";
      } else {
        ui.message.textContent = "";
      }
    }

    function renderCards() {
      ui.cards.replaceChildren();
      var memory = state.memory || {};
      var server = state.server || {};
      var proc = state.processes || {};
      var loaded = Array.isArray(server.loaded) ? server.loaded.length : 0;
      ui.cards.append(
        card("Sidecar RSS", fmtMb(memory.rss_mb), "pid " + (proc.sidecar ? proc.sidecar.pid : "—")),
        card("Threads", memory.threads == null ? "—" : String(memory.threads), "sidecar process"),
        card(
          "Conversations",
          memory.conversations_in_memory == null ? "—" : String(memory.conversations_in_memory),
          memory.max_conversations ? "max " + memory.max_conversations : "in memory"
        ),
        card("Loaded models", String(loaded), loaded ? "see Lifecycle" : "none")
      );
    }

    function renderTable() {
      ui.tbody.replaceChildren();
      if (!state.processes) {
        ui.tbody.appendChild(emptyRow(8, state.errors.processes || "process list unavailable"));
        return;
      }
      var rows = Array.isArray(state.processes.processes) ? state.processes.processes : [];
      if (!rows.length) {
        ui.tbody.appendChild(emptyRow(8, "No spawned servers or child shells."));
        return;
      }
      rows.forEach(function (row) {
        var tr = document.createElement("tr");
        var pid = row.pid;
        var cells = [
          String(pid == null ? "—" : pid),
          row.kind || "—",
          row.name || "—",
          fmtPct(row.cpu_percent),
          fmtMb(row.memory_mb),
          row.port == null ? "—" : String(row.port),
          row.status || "—",
        ];
        cells.forEach(function (value, index) {
          var td = document.createElement("td");
          if (index === 0 || index === 5) td.className = "mono";
          td.textContent = value;
          tr.appendChild(td);
        });
        var nameTd = tr.children[2];
        var cmd = shortCmd(row.cmdline);
        if (cmd) {
          var hint = document.createElement("div");
          hint.className = "muted mono";
          hint.style.fontSize = "10.5px";
          hint.textContent = cmd;
          nameTd.appendChild(hint);
        }
        var action = document.createElement("td");
        if (pid != null && row.status !== "not-found") {
          var button = node("button", "ghost", state.killPending[pid] ? "Killing…" : "Kill");
          button.type = "button";
          button.disabled = !!state.killPending[pid];
          button.setAttribute("data-kill-pid", String(pid));
          action.appendChild(button);
        }
        tr.appendChild(action);
        ui.tbody.appendChild(tr);
      });
      var sidecar = state.processes.sidecar;
      if (sidecar) {
        var footer = emptyRow(8, "sidecar " + (sidecar.name || "process") +
          " · pid " + sidecar.pid + " · " + fmtPct(sidecar.cpu_percent) +
          " · " + fmtMb(sidecar.memory_mb) + " · " + (sidecar.status || "—"));
        footer.className = "muted";
        ui.tbody.appendChild(footer);
      }
    }

    function renderLog() {
      if (!state.log) {
        ui.log.textContent = state.errors.log || "log unavailable";
        return;
      }
      var lines = Array.isArray(state.log.lines) ? state.log.lines : [];
      ui.log.textContent = lines.length ? lines.join("\n") : "log: no lines yet";
    }

    function renderRaw() {
      if (!state.server) {
        ui.raw.textContent = state.errors.server || "server status unavailable";
        return;
      }
      ui.raw.textContent = JSON.stringify(state.server, null, 2);
    }

    function render() {
      renderMessage();
      renderCards();
      renderTable();
      renderLog();
      renderRaw();
    }

    async function refresh() {
      state.status = "loading";
      state.notice = "";
      state.errors = {};
      render();
      var results = await Promise.allSettled([
        fetchJSON(base + "/processes"),
        fetchJSON(base + "/server/log?tail=200"),
        fetchJSON(base + "/server/status"),
        fetchJSON(base + "/server/memory"),
      ]);
      var names = ["processes", "log", "server", "memory"];
      results.forEach(function (result, index) {
        var name = names[index];
        if (result.status === "fulfilled") {
          state[name] = result.value;
        } else {
          state[name] = null;
          state.errors[name] = String(result.reason && result.reason.message ? result.reason.message : result.reason);
        }
      });
      state.status = results.some(function (r) { return r.status === "fulfilled"; }) ? "ready" : "error";
      if (state.status === "error") {
        state.notice = "";
        state.errors.all = "all process endpoints failed";
      }
      render();
      return state;
    }

    async function killProcess(pid) {
      var rows = state.processes && Array.isArray(state.processes.processes)
        ? state.processes.processes : [];
      var row = rows.find(function (item) { return Number(item.pid) === pid; });
      var label = row
        ? (row.kind || "process") + " " + (row.name || "") + " (pid " + pid + ")"
        : "pid " + pid;
      if (!confirmFn("Kill " + label + "? This stops the process; a managed model is unloaded.")) return;
      state.killPending[pid] = true;
      state.notice = "";
      render();
      try {
        var payload = { pid: pid };
        if (row && row.key) payload.key = row.key;
        await postJSON(base + "/processes/kill", payload);
        state.notice = "Killed " + label + ".";
      } catch (err) {
        state.errors.kill = String(err && err.message ? err.message : err);
      } finally {
        delete state.killPending[pid];
        await refresh();
      }
    }

    build();
    render();
    refresh();
    return { refresh: refresh, state: state };
  }

  global.HiveProcessesPanel = {
    mount: function (el, opts) {
      if (!el) throw new Error("HiveProcessesPanel.mount: element required");
      return createPanel(el, opts);
    },
  };
})(typeof window !== "undefined" ? window : globalThis);
