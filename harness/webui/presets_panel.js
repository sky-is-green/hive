/* HivePresetsPanel — read and duplicate agent presets in the new shell.
 *
 * window.HivePresetsPanel.mount(el) -> { refresh, state }
 * No API request is made until mount; presets are never applied or edited here.
 */
(function (global) {
  "use strict";

  var BASE = "/api/v1/agent-presets";

  function node(tag, className, text) {
    var el = document.createElement(tag);
    if (className) el.className = className;
    if (text != null) el.textContent = String(text);
    return el;
  }

  function clear(el) {
    while (el.firstChild) el.removeChild(el.firstChild);
  }

  function errorText(data, response) {
    if (data && typeof data === "object") {
      var detail = data.detail != null ? data.detail
        : data.error != null ? data.error : data.message;
      if (detail != null) {
        if (typeof detail === "string") return detail;
        try { return JSON.stringify(detail); } catch (_) { /* fall through */ }
      }
    }
    return response ? "HTTP " + response.status : "Request failed";
  }

  function mount(el, opts) {
    if (!el || typeof el.appendChild !== "function") {
      throw new Error("HivePresetsPanel.mount(el) needs a DOM element");
    }
    opts = opts || {};
    var base = String(opts.base || BASE).replace(/\/+$/, "");
    var fetchImpl = opts.fetch || (typeof global.fetch === "function" ? global.fetch.bind(global) : null);
    var confirmFn = opts.confirm || (typeof global.confirm === "function"
      ? global.confirm.bind(global) : function () { return false; });
    var clipboard = opts.clipboard || (global.navigator && global.navigator.clipboard);
    var state = {
      status: "idle", // idle | loading | ready | mutating | error
      presets: [],
      selectedId: "",
      content: "",
      readStatus: "idle", // idle | loading | ready | error
      readError: null,
      error: null,
      notice: null,
      copyFrom: "",
      newId: "",
      dirty: false
    };
    var ui = {};

    function request(method, path, body) {
      if (!fetchImpl) return Promise.reject(new Error("fetch unavailable"));
      var init = {
        method: method,
        cache: "no-store",
        headers: { accept: "application/json" }
      };
      if (body !== undefined) {
        init.headers["content-type"] = "application/json";
        init.body = JSON.stringify(body);
      }
      return fetchImpl(base + path, init).then(function (res) {
        return res.json().catch(function () { return null; }).then(function (data) {
          if (!res.ok) throw new Error(errorText(data, res));
          return data;
        });
      });
    }

    function build() {
      var root = node("div", "hive-presets");
      var heading = node("div", "hpr-heading");
      heading.append(
        node("div", "", "Agent presets"),
        node("p", "hpr-sub", "Browse system and user presets. Read or duplicate configurations; nothing is applied automatically.")
      );
      ui.refresh = node("button", "hpr-btn", "Refresh");
      ui.refresh.type = "button";
      ui.refresh.addEventListener("click", function () { refresh(); });
      heading.appendChild(ui.refresh);
      ui.status = node("div", "hpr-status");
      ui.error = node("div", "hpr-message hpr-error");
      ui.notice = node("div", "hpr-message hpr-notice");
      ui.groups = node("div", "hpr-groups");

      var copyPanel = node("form", "hpr-copy-panel");
      copyPanel.addEventListener("submit", function (ev) {
        ev.preventDefault();
        duplicate();
      });
      copyPanel.append(node("h3", "", "Duplicate preset to your user presets"));
      var fromLabel = node("label", "hpr-field");
      fromLabel.append(node("span", "hpr-label", "Source preset"));
      ui.copyFrom = node("select", "hpr-select");
      ui.copyFrom.addEventListener("change", function () {
        state.copyFrom = ui.copyFrom.value;
        renderControls();
      });
      fromLabel.append(ui.copyFrom);
      var idLabel = node("label", "hpr-field");
      idLabel.append(node("span", "hpr-label", "New preset id"));
      ui.newId = node("input", "hpr-input");
      ui.newId.type = "text";
      ui.newId.autocomplete = "off";
      ui.newId.spellcheck = false;
      ui.newId.placeholder = "my-preset";
      ui.newId.addEventListener("input", function () {
        state.newId = ui.newId.value;
        state.dirty = !!state.newId.trim();
        renderControls();
      });
      idLabel.append(ui.newId);
      ui.duplicate = node("button", "hpr-btn hpr-primary", "Duplicate");
      ui.duplicate.type = "submit";
      ui.dirty = node("span", "hpr-dirty", "unsaved duplicate details");
      copyPanel.append(fromLabel, idLabel, ui.duplicate, ui.dirty);

      ui.detail = node("section", "hpr-detail");
      ui.detail.hidden = true;
      var detailHead = node("div", "hpr-detail-head");
      ui.detailTitle = node("h3", "", "");
      ui.copyContent = node("button", "hpr-btn", "Copy to clipboard");
      ui.copyContent.type = "button";
      ui.copyContent.addEventListener("click", copyToClipboard);
      detailHead.append(ui.detailTitle, ui.copyContent);
      ui.readMessage = node("div", "hpr-read-message");
      ui.content = node("pre", "hpr-content");
      ui.detail.append(detailHead, ui.readMessage, ui.content);

      root.append(heading, ui.status, ui.error, ui.notice, ui.groups, copyPanel, ui.detail);
      clear(el);
      el.appendChild(root);
    }

    function trustOf(preset) {
      return String(preset && preset.trust != null ? preset.trust : "system");
    }

    function presetName(preset) {
      return String((preset && preset.name) || (preset && preset.id) || "(unnamed preset)");
    }

    function renderControls() {
      if (!ui.copyFrom) return;
      var selected = state.copyFrom;
      clear(ui.copyFrom);
      if (!state.presets.length) {
        var empty = node("option", "", "No presets available");
        empty.value = "";
        ui.copyFrom.appendChild(empty);
      } else {
        state.presets.forEach(function (preset) {
          var option = node("option", "", presetName(preset) + " (" + String(preset.id || "") + ")");
          option.value = String(preset.id || "");
          ui.copyFrom.appendChild(option);
        });
      }
      if (state.presets.some(function (p) { return String(p.id || "") === selected; })) {
        ui.copyFrom.value = selected;
      } else {
        state.copyFrom = state.presets.length ? String(state.presets[0].id || "") : "";
        ui.copyFrom.value = state.copyFrom;
      }
      ui.newId.value = state.newId;
      ui.dirty.hidden = !state.dirty;
      ui.duplicate.disabled = state.status === "loading" || state.status === "mutating" ||
        !state.copyFrom || !state.newId.trim();
      ui.copyFrom.disabled = state.status === "loading" || state.status === "mutating" || !state.presets.length;
      ui.newId.disabled = state.status === "loading" || state.status === "mutating";
    }

    function renderGroups() {
      clear(ui.groups);
      if (state.status === "loading" && !state.presets.length) {
        ui.groups.appendChild(node("div", "hpr-empty", "Loading agent presets…"));
        return;
      }
      if (!state.presets.length) {
        ui.groups.appendChild(node("div", "hpr-empty", "No agent presets found."));
        return;
      }
      ["system", "user"].forEach(function (kind) {
        var presets = state.presets.filter(function (preset) {
          return (trustOf(preset) === "user" ? "user" : "system") === kind;
        });
        if (!presets.length) return;
        var section = node("section", "hpr-group hpr-group-" + kind);
        section.appendChild(node("h3", "hpr-group-title", kind === "system" ? "System presets" : "User presets"));
        var cards = node("div", "hpr-cards");
        presets.forEach(function (preset) {
          var id = String(preset.id == null ? "" : preset.id);
          var trust = trustOf(preset);
          var card = node("article", "hpr-card");
          if (id === state.selectedId) card.classList.add("hpr-selected");
          var title = node("div", "hpr-card-title");
          title.append(node("strong", "hpr-name", presetName(preset)), node("code", "hpr-id", id || "—"));
          var description = node("p", "hpr-description", preset.description == null ? "No description provided." : preset.description);
          var meta = node("div", "hpr-meta");
          meta.append(node("span", "hpr-trust", "Trust: " + trust));
          if (preset.broken) meta.append(node("span", "hpr-broken", "Broken: " + String(preset.broken)));
          var actions = node("div", "hpr-actions");
          var read = node("button", "hpr-btn", id === state.selectedId ? "Selected / read" : "Read");
          read.type = "button";
          read.disabled = state.status === "loading" || state.status === "mutating" || !id;
          read.addEventListener("click", function () { readPreset(preset); });
          actions.appendChild(read);
          if (trust === "user") {
            var remove = node("button", "hpr-btn hpr-danger", "Remove");
            remove.type = "button";
            remove.disabled = state.status === "loading" || state.status === "mutating";
            remove.addEventListener("click", function () { removePreset(preset); });
            actions.appendChild(remove);
          }
          card.append(title, description, meta, actions);
          cards.appendChild(card);
        });
        section.appendChild(cards);
        ui.groups.appendChild(section);
      });
    }

    function renderDetail() {
      ui.detail.hidden = !state.selectedId;
      ui.detailTitle.textContent = state.selectedId
        ? "agent.cordis.yml — " + state.selectedId : "";
      ui.copyContent.disabled = state.readStatus !== "ready";
      ui.readMessage.textContent = "";
      ui.readMessage.className = "hpr-read-message";
      if (state.readStatus === "loading") {
        ui.readMessage.textContent = "Loading agent.cordis.yml…";
        ui.content.textContent = "";
      } else if (state.readStatus === "error") {
        ui.readMessage.classList.add("hpr-read-error");
        ui.readMessage.textContent = state.readError || "Could not read this preset.";
        ui.content.textContent = "";
      } else if (state.readStatus === "ready") {
        ui.content.textContent = state.content;
      } else {
        ui.content.textContent = "";
      }
    }

    function render() {
      ui.status.textContent = state.status === "loading" ? "Loading presets…"
        : state.status === "mutating" ? "Updating presets…"
        : state.status === "error" ? "Preset list unavailable"
        : state.presets.length + (state.presets.length === 1 ? " preset" : " presets");
      ui.refresh.disabled = state.status === "loading" || state.status === "mutating";
      ui.error.hidden = !state.error;
      ui.error.textContent = state.error || "";
      ui.notice.hidden = !state.notice;
      ui.notice.textContent = state.notice || "";
      renderGroups();
      renderControls();
      renderDetail();
    }

    function refresh() {
      state.status = "loading";
      state.error = null;
      state.notice = null;
      render();
      return request("GET", "/list").then(function (data) {
        var presets = Array.isArray(data) ? data : (data && Array.isArray(data.presets) ? data.presets : []);
        state.presets = presets.filter(function (item) { return item && typeof item === "object"; });
        if (!state.presets.some(function (p) { return String(p.id || "") === state.selectedId; })) {
          state.selectedId = "";
          state.content = "";
          state.readStatus = "idle";
        }
        state.status = "ready";
      }).catch(function (err) {
        state.status = "error";
        state.error = String(err && err.message ? err.message : err);
      }).then(function () {
        render();
        return state;
      });
    }

    function readPreset(preset) {
      var id = String(preset.id == null ? "" : preset.id);
      if (!id || state.status === "loading" || state.status === "mutating") return Promise.resolve(state);
      state.selectedId = id;
      state.content = "";
      state.readStatus = "loading";
      state.readError = null;
      state.error = null;
      render();
      return request("POST", "/read", { agentPreset: id }).then(function (data) {
        state.content = data && typeof data.content === "string" ? data.content : "";
        state.readStatus = "ready";
      }).catch(function (err) {
        state.content = "";
        state.readStatus = "error";
        state.readError = String(err && err.message ? err.message : err);
      }).then(function () {
        render();
        return state;
      });
    }

    function copyToClipboard() {
      if (state.readStatus !== "ready") return Promise.resolve(false);
      if (!clipboard || typeof clipboard.writeText !== "function") {
        state.notice = "Clipboard access is unavailable in this browser.";
        render();
        return Promise.resolve(false);
      }
      return Promise.resolve(clipboard.writeText(state.content)).then(function () {
        state.notice = "agent.cordis.yml copied to clipboard.";
        state.error = null;
        render();
        return true;
      }).catch(function (err) {
        state.error = "Clipboard copy failed: " + String(err && err.message ? err.message : err);
        render();
        return false;
      });
    }

    function duplicate() {
      if (state.status === "loading" || state.status === "mutating") return Promise.resolve(state);
      var from = String(state.copyFrom || "").trim();
      var id = String(state.newId || "").trim();
      if (!from || !id) return Promise.resolve(state);
      if (state.presets.some(function (preset) { return String(preset.id || "") === id; })) {
        state.error = "A preset with id \"" + id + "\" is already listed.";
        render();
        return Promise.resolve(state);
      }
      state.status = "mutating";
      state.error = null;
      state.notice = null;
      render();
      return request("POST", "/copy", { from: from, agentPreset: id }).then(function () {
        state.newId = "";
        state.dirty = false;
        return refresh().then(function () {
          state.notice = "Created user preset " + id + ".";
          render();
        });
      }).catch(function (err) {
        state.status = "ready";
        state.error = String(err && err.message ? err.message : err);
      }).then(function () {
        render();
        return state;
      });
    }

    function removePreset(preset) {
      var id = String(preset.id == null ? "" : preset.id);
      if (trustOf(preset) !== "user" || !id || state.status === "loading" || state.status === "mutating") {
        return Promise.resolve(state);
      }
      if (!confirmFn("Remove user preset \"" + id + "\"? This cannot be undone.")) {
        return Promise.resolve(state);
      }
      state.status = "mutating";
      state.error = null;
      state.notice = null;
      render();
      return request("POST", "/remove", { agentPreset: id }).then(function () {
        if (state.selectedId === id) {
          state.selectedId = "";
          state.content = "";
          state.readStatus = "idle";
        }
        return refresh().then(function () {
          state.notice = "Removed user preset " + id + ".";
          render();
        });
      }).catch(function (err) {
        state.status = "ready";
        state.error = String(err && err.message ? err.message : err);
      }).then(function () {
        render();
        return state;
      });
    }

    build();
    render();
    var ready = refresh();
    return {
      refresh: refresh,
      state: state,
      ready: ready,
      destroy: function () { clear(el); }
    };
  }

  global.HivePresetsPanel = { mount: mount };
})(typeof window !== "undefined" ? window : globalThis);
