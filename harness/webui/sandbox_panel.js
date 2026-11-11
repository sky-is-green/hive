/*
 * HiveSandboxPanel — read-only sandbox status for the new shell.
 *
 * Read-only by design: the write paths stay in the CLI until the policy
 * contract is exposed through the API. This panel answers the operational
 * questions a user has before an agent runs code:
 *
 *   GET /api/v1/sandbox/status
 *     docker installed/daemon/binary/host/rootless
 *     policy: image, digest, runtime, network, limits, fallback
 *     pinned / pin_error / fallback string
 *
 * window.HiveSandboxPanel.mount(el, opts?) -> { refresh, state }
 *   opts.base    : API prefix (default "/api/v1")
 *   opts.fetchJSON : injectable fetch function (tests)
 */
(function (global) {
  "use strict";

  var BASE = "/api/v1";
  var FALLBACK_LABELS = {
    approval: "explicit approval required",
    allow: "allowed by policy",
    deny: "disabled by policy",
  };

  function esc(value) {
    return String(value == null ? "" : value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function text(value, fallback) {
    if (value === undefined || value === null || value === "") return fallback || "—";
    return String(value);
  }

  function shortDigest(value) {
    var s = String(value || "");
    if (!s) return "—";
    return s.length > 16 ? s.slice(0, 12) + "…" + s.slice(-4) : s;
  }

  function formatFallback(value) {
    return FALLBACK_LABELS[value] || text(value, "unknown");
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

  function row(label, value, mono) {
    var tr = document.createElement("tr");
    var th = document.createElement("th");
    th.scope = "row";
    th.textContent = label;
    var td = document.createElement("td");
    if (mono) td.className = "mono";
    td.textContent = text(value);
    tr.append(th, td);
    return tr;
  }

  function createPanel(rootEl, opts) {
    opts = opts || {};
    var base = opts.base || BASE;
    var fetchJSON = opts.fetchJSON || defaultFetchJSON;
    var state = { status: "idle", data: null, error: "" };
    var ui = {};

    function build() {
      rootEl.innerHTML = "";
      var root = node("section", "hive-sandbox");

      ui.message = node("div", "note");
      ui.message.style.margin = "0 0 10px";

      ui.cards = node("div", "cards");

      var grid = node("div", "grid two");
      var dockerPanel = node("div", "panel");
      var dockerHead = node("div", "panel-head");
      dockerHead.append(node("h2", "", "Docker"), node("span", "muted", "live probe"));
      ui.docker = document.createElement("table");
      var dockerWrap = node("div", "table-wrap");
      dockerWrap.appendChild(ui.docker);
      dockerPanel.append(dockerHead, dockerWrap);

      var policyPanel = node("div", "panel");
      var policyHead = node("div", "panel-head");
      policyHead.append(node("h2", "", "Policy"), node("span", "muted", "settings/sandbox.json"));
      ui.policy = document.createElement("table");
      var policyWrap = node("div", "table-wrap");
      policyWrap.appendChild(ui.policy);
      policyPanel.append(policyHead, policyWrap);
      grid.append(dockerPanel, policyPanel);

      var fallbackPanel = node("div", "panel");
      var fallbackHead = node("div", "panel-head");
      fallbackHead.append(node("h2", "", "Local fallback"));
      ui.fallback = node("div", "note");
      ui.fallback.style.padding = "12px 15px";
      fallbackPanel.append(fallbackHead, ui.fallback);

      var hint = node("div", "note");
      hint.style.marginTop = "8px";
      hint.innerHTML =
        "Write paths stay in the CLI for now: " +
        "<code>hive sandbox status --json</code>, " +
        "<code>hive sandbox pin</code>, " +
        "<code>hive sandbox policy --local-fallback deny</code>.";

      root.append(ui.message, ui.cards, grid, fallbackPanel, hint);
      rootEl.appendChild(root);
    }

    function render() {
      var data = state.data || {};
      var policy = data.policy || {};
      var loading = state.status === "loading";
      var failed = state.status === "error";

      ui.message.textContent = failed ? state.error
        : loading ? "Reading sandbox status…"
        : "";
      ui.message.style.color = failed ? "var(--bad)" : "var(--muted)";

      ui.cards.replaceChildren();
      if (data && Object.keys(data).length) {
        ui.cards.append(
          card(
            "Docker",
            data.docker_daemon ? "running" : data.docker_installed ? "installed" : "missing",
            data.detail || "",
            data.docker_daemon ? "ok" : "bad"
          ),
          card(
            "Image",
            data.pinned ? "pinned" : "unpinned",
            data.pinned ? text(policy.docker_image) : (data.pin_error || "run: hive sandbox pin"),
            data.pinned ? "ok" : "bad"
          ),
          card(
            "Runtime",
            text(policy.runtime, "runc (default)"),
            policy.runtime ? "explicit runtime" : "daemon default"
          ),
          card(
            "Local fallback",
            formatFallback(policy.local_fallback),
            data.fallback_requires_approval ? "per-call approval" : "see policy",
            policy.local_fallback === "deny" ? "ok" : policy.local_fallback === "allow" ? "bad" : ""
          )
        );
      }

      ui.docker.replaceChildren();
      if (data && Object.keys(data).length) {
        ui.docker.append(
          row("installed", data.docker_installed ? "yes" : "no", true),
          row("daemon", data.docker_daemon ? "running" : "not running", true),
          row("binary", data.binary, true),
          row("host", data.host, true),
          row("rootless", data.rootless ? "yes" : "no", true),
          row("detail", data.detail)
        );
      }

      ui.policy.replaceChildren();
      if (data && Object.keys(data).length) {
        ui.policy.append(
          row("image", policy.docker_image, true),
          row("digest", shortDigest(policy.image_digest), true),
          row("runtime", text(policy.runtime, "daemon default"), true),
          row("network", policy.network, true),
          row("memory", policy.memory, true),
          row("cpus", policy.cpus, true),
          row("pids limit", policy.pids_limit, true),
          row("timeout", policy.timeout_s ? policy.timeout_s + " s" : "—", true),
          row("fallback", formatFallback(policy.local_fallback)),
          row("updated", policy.updated_at, true)
        );
      }

      var fallback = data.fallback || formatFallback(policy.local_fallback);
      ui.fallback.textContent = fallback +
        " · Local fallback executions are audited. Set `local_fallback=deny` to forbid them entirely.";
    }

    async function refresh() {
      state.status = "loading";
      state.error = "";
      render();
      try {
        state.data = await fetchJSON(base + "/sandbox/status");
        state.status = "ready";
      } catch (err) {
        state.status = "error";
        state.error = "Sandbox status unavailable: " + (err && err.message ? err.message : err);
      }
      render();
      return state;
    }

    build();
    render();
    refresh();
    return { refresh: refresh, state: state };
  }

  global.HiveSandboxPanel = {
    mount: function (el, opts) {
      if (!el) throw new Error("HiveSandboxPanel.mount: element required");
      return createPanel(el, opts);
    },
  };
})(typeof window !== "undefined" ? window : globalThis);
