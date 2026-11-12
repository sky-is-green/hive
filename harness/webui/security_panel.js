/*
 * HiveSecurityPanel — tool policy tiers, master switches and grants.
 *
 *   GET  /api/v1/tools/policy     policy + grants
 *   PUT  /api/v1/tools/policy     tier + master switches
 *   POST /api/v1/tools/revoke     one identity or all grants
 *
 * Pending approvals and the audit stay on the Risk page; this panel owns the
 * policy that decides them.
 *
 * window.HiveSecurityPanel.mount(el, opts?) -> { refresh, state }
 */
(function (global) {
  "use strict";

  var BASE = "/api/v1";
  var TIERS = ["strict", "guarded", "trusted", "open"];

  function esc(value) {
    return String(value == null ? "" : value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
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

  async function defaultPostJSON(path, payload, method) {
    var res = await fetch(path, {
      method: method || "POST",
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

  function createPanel(rootEl, opts) {
    opts = opts || {};
    var base = opts.base || BASE;
    var fetchJSON = opts.fetchJSON || defaultFetchJSON;
    var postJSON = opts.postJSON || defaultPostJSON;
    var confirmFn = opts.confirm || (global.confirm ? global.confirm.bind(global) : function () { return false; });
    var state = { status: "idle", policy: null, grants: [], error: "", notice: "", busy: false };
    var ui = {};

    function build() {
      rootEl.innerHTML = "";
      var panel = node("div", "panel");
      var head = node("div", "panel-head");
      head.append(node("h2", "", "Tool policy & grants"), node("span", "muted", "gateway"));
      var body = node("div");
      body.style.padding = "12px 15px";

      var row = node("div", "engine-actions");
      var tierLabel = node("label", "", "tier");
      ui.tier = document.createElement("select");
      TIERS.forEach(function (tier) {
        var option = document.createElement("option");
        option.value = tier;
        option.textContent = tier;
        ui.tier.appendChild(option);
      });
      tierLabel.appendChild(ui.tier);
      ui.master = {};

      function master(key, label) {
        var lab = node("label", "inline-check");
        var input = document.createElement("input");
        input.type = "checkbox";
        input.setAttribute("data-master", key);
        lab.append(input, node("span", "", label));
        ui.master[key] = input;
        return lab;
      }
      var approval = master("approval_prompts", "approval prompts");
      var sandbox = master("sandbox", "sandbox");
      var privateNet = master("private_network", "private-network gate");

      ui.save = node("button", "approve", "Save policy");
      ui.save.type = "button";
      ui.save.addEventListener("click", save);
      ui.revokeAll = node("button", "ghost danger", "Revoke all grants");
      ui.revokeAll.type = "button";
      ui.revokeAll.addEventListener("click", revokeAll);
      row.append(tierLabel, approval, sandbox, privateNet, ui.save, ui.revokeAll);

      ui.message = node("div", "note");
      ui.message.style.margin = "8px 0";

      ui.grants = document.createElement("table");
      var wrap = node("div", "table-wrap");
      wrap.appendChild(ui.grants);
      body.append(row, ui.message, wrap);
      panel.append(head, body);
      rootEl.appendChild(panel);

      ui.grants.addEventListener("click", function (event) {
        var button = event.target.closest("button[data-revoke-identity]");
        if (button) revokeOne(
          button.getAttribute("data-revoke-identity"),
          button.getAttribute("data-revoke-session") || "",
          button.getAttribute("data-revoke-resource") || ""
        );
      });
    }

    function renderMessage() {
      if (state.error) {
        ui.message.textContent = state.error;
        ui.message.style.color = "var(--bad)";
      } else if (state.notice) {
        ui.message.textContent = state.notice;
        ui.message.style.color = "var(--good)";
      } else if (state.status === "loading") {
        ui.message.textContent = "Reading policy…";
        ui.message.style.color = "var(--muted)";
      } else {
        ui.message.textContent = "";
      }
    }

    function renderPolicy() {
      var policy = state.policy || {};
      var master = policy.master || {};
      ui.tier.value = TIERS.indexOf(policy.tier) >= 0 ? policy.tier : "guarded";
      Object.keys(ui.master).forEach(function (key) {
        ui.master[key].checked = !!master[key];
      });
      ui.save.disabled = state.busy;
      ui.revokeAll.disabled = state.busy || !state.grants.length;
    }

    function renderGrants() {
      ui.grants.replaceChildren();
      var thead = document.createElement("thead");
      var hr = document.createElement("tr");
      ["Identity", "Scope", "Resource", "Session", ""].forEach(function (label) {
        var th = document.createElement("th");
        th.textContent = label;
        hr.appendChild(th);
      });
      thead.appendChild(hr);
      ui.grants.appendChild(thead);
      var tbody = document.createElement("tbody");
      if (!state.grants.length) {
        var tr = document.createElement("tr");
        var td = document.createElement("td");
        td.colSpan = 5;
        td.className = "note";
        td.textContent = "No tool grants.";
        tr.appendChild(td);
        tbody.appendChild(tr);
      } else {
        state.grants.forEach(function (row) {
          var tr = document.createElement("tr");
          var identity = document.createElement("td");
          identity.className = "mono";
          identity.textContent = row.tool_identity || "—";
          var scope = document.createElement("td");
          scope.textContent = row.scope || "—";
          var resource = document.createElement("td");
          resource.textContent = row.resource || "—";
          var session = document.createElement("td");
          session.className = "mono";
          session.textContent = row.session_id || "—";
          var action = document.createElement("td");
          var button = node("button", "ghost danger", "Revoke");
          button.type = "button";
          button.disabled = state.busy;
          button.setAttribute("data-revoke-identity", row.tool_identity || "");
          if (row.session_id) button.setAttribute("data-revoke-session", row.session_id);
          if (row.resource) button.setAttribute("data-revoke-resource", row.resource);
          action.appendChild(button);
          tr.append(identity, scope, resource, session, action);
          tbody.appendChild(tr);
        });
      }
      ui.grants.appendChild(tbody);
    }

    function render() {
      renderMessage();
      renderPolicy();
      renderGrants();
    }

    async function refresh() {
      state.status = "loading";
      state.error = "";
      state.notice = "";
      render();
      try {
        var data = await fetchJSON(base + "/tools/policy");
        state.policy = data.policy || {};
        state.grants = Array.isArray(data.grants) ? data.grants : [];
        state.status = "ready";
      } catch (err) {
        state.status = "error";
        state.error = "Policy unavailable: " + (err && err.message ? err.message : err);
      }
      render();
      return state;
    }

    async function save() {
      if (state.busy) return;
      var tier = ui.tier.value;
      var master = {};
      Object.keys(ui.master).forEach(function (key) {
        master[key] = !!ui.master[key].checked;
      });
      state.busy = true;
      state.error = "";
      state.notice = "";
      render();
      try {
        var data = await postJSON(base + "/tools/policy", {
          tier: tier,
          master: master,
        }, "PUT");
        state.policy = data.policy || state.policy;
        state.notice = "Policy saved.";
      } catch (err) {
        state.error = "Could not save policy: " + (err && err.message ? err.message : err);
      }
      state.busy = false;
      render();
    }

    async function revokeAll() {
      if (state.busy || !state.grants.length) return;
      if (!confirmFn("Revoke every global tool grant? Pending approvals are not affected.")) return;
      state.busy = true;
      state.error = "";
      state.notice = "";
      render();
      try {
        await postJSON(base + "/tools/revoke", { all: true });
        state.notice = "All tool grants revoked.";
        state.busy = false;
        await refresh();
        state.notice = "All tool grants revoked.";
      } catch (err) {
        state.busy = false;
        state.error = "Revoke failed: " + (err && err.message ? err.message : err);
      }
      render();
    }

    async function revokeOne(identity, sessionId, resource) {
      if (state.busy || !identity) return;
      var label = identity + (resource ? " on " + resource : "");
      if (!confirmFn("Revoke the grant for " + label + "?")) return;
      state.busy = true;
      state.error = "";
      state.notice = "";
      render();
      try {
        var payload = { identity: identity };
        if (sessionId) payload.session_id = sessionId;
        await postJSON(base + "/tools/revoke", payload);
        state.notice = "Revoked " + label + ".";
        state.busy = false;
        await refresh();
        state.notice = "Revoked " + label + ".";
      } catch (err) {
        state.busy = false;
        state.error = "Revoke failed: " + (err && err.message ? err.message : err);
      }
      render();
    }

    build();
    render();
    refresh();
    return { refresh: refresh, state: state };
  }

  global.HiveSecurityPanel = {
    mount: function (el, opts) {
      if (!el) throw new Error("HiveSecurityPanel.mount: element required");
      return createPanel(el, opts);
    },
  };
})(typeof window !== "undefined" ? window : globalThis);
