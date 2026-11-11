/*
 * HiveAbPanel — live engine A/B compare for the Engines page.
 *
 * This is the real bench, not the static Benchmark matrix. It loads two
 * engine profiles on basePort and basePort+1, runs the same three prompts
 * through each, and reports tokens/sec plus a noise-aware winner.
 *
 *   GET  /api/v1/engines              profiles + default
 *   POST /api/v1/engines/ab/bench     {profile_a, profile_b, basePort}
 *
 * The run loads models and can take minutes, so it only starts from an
 * explicit click after a confirm(). Nothing is loaded on mount.
 *
 * window.HiveAbPanel.mount(el, opts?) -> { refresh, state }
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

  function fmtTps(value) {
    var n = Number(value);
    if (!isFinite(n)) return "—";
    return n.toFixed(1) + " tok/s";
  }

  function winnerOf(result) {
    if (result && result.winner) return String(result.winner).toUpperCase();
    var a = Number(result && (result.a_tok_per_sec ?? result.tokens_per_sec_a) || 0);
    var b = Number(result && (result.b_tok_per_sec ?? result.tokens_per_sec_b) || 0);
    if (a > b) return "A";
    if (b > a) return "B";
    return "TIE";
  }

  function tpsOf(side, result) {
    if (!result) return 0;
    if (side === "a") return Number(result.a_tok_per_sec ?? result.a?.tok_per_sec ?? result.tokens_per_sec_a ?? 0);
    return Number(result.b_tok_per_sec ?? result.b?.tok_per_sec ?? result.tokens_per_sec_b ?? 0);
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

  function createPanel(rootEl, opts) {
    opts = opts || {};
    var base = opts.base || BASE;
    var fetchJSON = opts.fetchJSON || defaultFetchJSON;
    var postJSON = opts.postJSON || defaultPostJSON;
    var confirmFn = opts.confirm || (global.confirm ? global.confirm.bind(global) : function () { return false; });
    var state = {
      status: "idle",
      engines: [],
      default: "",
      error: "",
      notice: "",
      running: false,
      result: null,
    };
    var ui = {};

    function build() {
      rootEl.innerHTML = "";
      var panel = node("div", "panel");
      var head = node("div", "panel-head");
      head.append(node("h2", "", "Live A/B compare"), node("span", "muted", "loads both · 3 prompts"));

      var body = node("div");
      body.style.padding = "12px 15px";

      var note = node("p", "note");
      note.style.marginTop = "0";
      note.textContent =
        "Loads both profiles on basePort and basePort+1, runs the same three prompts, " +
        "and compares tokens/sec. The first run can take minutes while the models load. " +
        "This is separate from the static Benchmark page.";

      var grid = node("div", "engine-grid");
      var aLabel = node("label", "", "A profile");
      ui.a = document.createElement("select");
      aLabel.appendChild(ui.a);
      var bLabel = node("label", "", "B profile");
      ui.b = document.createElement("select");
      bLabel.appendChild(ui.b);
      var portLabel = node("label", "", "base port");
      ui.port = document.createElement("input");
      ui.port.type = "number";
      ui.port.min = "1024";
      ui.port.max = "65534";
      ui.port.value = "1234";
      portLabel.appendChild(ui.port);
      grid.append(aLabel, bLabel, portLabel);

      var actions = node("div", "engine-actions");
      ui.bench = node("button", "approve", "Bench A vs B");
      ui.bench.type = "button";
      ui.bench.addEventListener("click", bench);
      ui.status = node("span", "muted", "");
      actions.append(ui.bench, ui.status);

      ui.result = node("div");
      ui.result.style.marginTop = "14px";

      body.append(note, grid, actions, ui.result);
      panel.append(head, body);
      rootEl.appendChild(panel);
    }

    function renderSelects() {
      if (!ui.a || !ui.b) return;
      var keepA = ui.a.value;
      var keepB = ui.b.value;
      [ui.a, ui.b].forEach(function (sel) {
        sel.replaceChildren();
        state.engines.forEach(function (engine) {
          var option = document.createElement("option");
          option.value = engine.name;
          option.textContent = engine.name;
          sel.appendChild(option);
        });
      });
      if (state.engines.length) {
        ui.a.value = state.engines.some(function (e) { return e.name === keepA; })
          ? keepA
          : (state.engines.find(function (e) { return e.name === state.default; }) || state.engines[0]).name;
        ui.b.value = state.engines.some(function (e) { return e.name === keepB; })
          ? keepB
          : (state.engines[1] || state.engines[0]).name;
      }
    }

    function render() {
      var count = state.engines.length;
      var canBench = count >= 2 && !state.running && !state.error;
      ui.bench.disabled = !canBench;
      ui.bench.textContent = state.running ? "Benching…" : "Bench A vs B";
      ui.status.textContent = state.error ? state.error
        : state.running ? "Running the bench; the models load first."
        : count < 2 ? "Two engine profiles are required."
        : state.notice ? state.notice
        : "";
      ui.status.style.color = state.error ? "var(--bad)" : "var(--muted)";

      ui.result.replaceChildren();
      if (!state.result) return;
      var result = state.result;
      var winner = winnerOf(result);
      var tone = winner === "TIE" ? "" : "ok";
      var label = winner === "TIE" ? "tie" : winner + " wins";

      var resultHead = node("div", "engine-actions");
      resultHead.append(
        node("span", "pill " + tone, label),
        node("span", "muted", "base port " + (result.base_port || result.basePort || "—")
          + " · " + ((result.prompts || []).length || 3) + " prompts")
      );

      var cards = node("div", "cards");
      cards.append(
        card(
          "A · " + (result.a?.profile || result.profile_a || "A"),
          fmtTps(tpsOf("a", result)),
          "port " + (result.a?.port || result.port_a || "—") + " · " + (result.a?.base_url || ""),
          winner === "A" ? "ok" : ""
        ),
        card(
          "B · " + (result.b?.profile || result.profile_b || "B"),
          fmtTps(tpsOf("b", result)),
          "port " + (result.b?.port || result.port_b || "—") + " · " + (result.b?.base_url || ""),
          winner === "B" ? "ok" : ""
        )
      );
      ui.result.append(resultHead, cards);

      if (result.a && result.b) {
        var detail = node("details");
        detail.appendChild(node("summary", "note", "Per-profile details"));
        var pre = document.createElement("pre");
        pre.style.margin = "8px 0 0";
        pre.style.padding = "10px";
        pre.style.background = "#0d0f12";
        pre.style.color = "#c9d1d9";
        pre.style.maxHeight = "220px";
        pre.style.overflow = "auto";
        pre.style.fontSize = "11.5px";
        pre.style.whiteSpace = "pre-wrap";
        pre.textContent = JSON.stringify({ a: result.a, b: result.b }, null, 2);
        detail.appendChild(pre);
        ui.result.appendChild(detail);
      }
    }

    async function refresh() {
      state.status = "loading";
      state.error = "";
      render();
      try {
        var data = await fetchJSON(base + "/engines");
        state.engines = Array.isArray(data.engines) ? data.engines : [];
        state.default = data.default || "";
        state.status = "ready";
        renderSelects();
      } catch (err) {
        state.status = "error";
        state.error = "Engine profiles unavailable: " + (err && err.message ? err.message : err);
      }
      render();
      return state;
    }

    async function bench() {
      if (state.running) return;
      var a = ui.a.value;
      var b = ui.b.value;
      var basePort = Number(ui.port.value || 1234);
      if (!a || !b) {
        state.error = "Select two engine profiles.";
        render();
        return;
      }
      if (a === b) {
        state.error = "Pick two different profiles.";
        render();
        return;
      }
      if (!(basePort >= 1024 && basePort <= 65534)) {
        state.error = "Base port must be between 1024 and 65534.";
        render();
        return;
      }
      if (!confirmFn(
        "Bench " + a + " vs " + b + " on ports " + basePort + " and " + (basePort + 1) + "?\n\n" +
        "This loads both models and runs three prompts through each; it can take minutes."
      )) return;
      state.running = true;
      state.error = "";
      state.notice = "";
      state.result = null;
      render();
      try {
        var result = await postJSON(base + "/engines/ab/bench", {
          profile_a: a,
          profile_b: b,
          basePort: basePort,
        });
        state.result = result;
        state.notice = "Bench complete.";
      } catch (err) {
        state.error = "A/B bench failed: " + (err && err.message ? err.message : err);
      } finally {
        state.running = false;
        render();
      }
    }

    build();
    render();
    refresh();
    return { refresh: refresh, state: state };
  }

  global.HiveAbPanel = {
    mount: function (el, opts) {
      if (!el) throw new Error("HiveAbPanel.mount: element required");
      return createPanel(el, opts);
    },
  };
})(typeof window !== "undefined" ? window : globalThis);
