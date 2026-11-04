/* HiveSettingsPanel — edit the Global Studio settings. */
(function (global) {
  "use strict";

  var BASE = "/api/v1/studio/settings";
  var BOOL_KEYS = ["sandbox_enabled", "tools_enabled", "auto_compact", "dev_mode", "harvest_enabled"];
  var STR_KEYS = ["strata_base", "strata_repo"];
  var JSON_KEYS = ["memory_defaults", "quality_defaults", "mcp", "runtime_defaults", "encoder"];
  var KNOWN_KEYS = BOOL_KEYS.concat(STR_KEYS, JSON_KEYS);
  var DEFAULTS = {
    sandbox_enabled: true,
    tools_enabled: true,
    auto_compact: true,
    dev_mode: false,
    harvest_enabled: false,
    strata_base: "http://127.0.0.1:8081/v1",
    strata_repo: "",
    memory_defaults: {},
    quality_defaults: {},
    mcp: { enabled: true, tools: ["remember", "search"], read_only: false },
    runtime_defaults: {},
    encoder: {
      embedding_model: "sentence-transformers/paraphrase-MiniLM-L3-v2",
      embedding_backend: "local",
      embedding_url: "",
      medium_model: "microsoft/graphcodebert-base",
      controller_idle_unload: true,
      controller_idle_seconds: 300
    }
  };

  function node(tag, className, text) {
    var el = document.createElement(tag);
    if (className) el.className = className;
    if (text != null) el.textContent = String(text);
    return el;
  }

  function clear(el) {
    while (el.firstChild) el.removeChild(el.firstChild);
  }

  function isObject(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
  }

  function clone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function equal(a, b) {
    return JSON.stringify(a) === JSON.stringify(b);
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

  function normalize(data) {
    var source = isObject(data) ? data : {};
    var values = {};
    BOOL_KEYS.forEach(function (key) {
      values[key] = typeof source[key] === "boolean" ? source[key] : DEFAULTS[key];
    });
    STR_KEYS.forEach(function (key) {
      values[key] = typeof source[key] === "string" ? source[key] : DEFAULTS[key];
    });
    JSON_KEYS.forEach(function (key) {
      values[key] = isObject(source[key]) ? clone(source[key]) : clone(DEFAULTS[key]);
    });
    return values;
  }

  function mount(el, opts) {
    if (!el || typeof el.appendChild !== "function") {
      throw new Error("HiveSettingsPanel.mount(el) needs a DOM element");
    }
    opts = opts || {};
    var endpoint = String(opts.url || BASE);
    var fetchImpl = opts.fetch || (typeof global.fetch === "function" ? global.fetch.bind(global) : null);
    var state = {
      status: "idle", // idle | loading | ready | saving | error | empty
      dirty: false,
      changedKeys: [],
      validationErrors: {},
      error: null,
      notice: null,
      values: clone(DEFAULTS),
      original: clone(DEFAULTS),
      serverSettings: {}
    };
    var ui = { bools: {}, strings: {}, json: {}, jsonErrors: {} };

    function request(method, body) {
      if (!fetchImpl) return Promise.reject(new Error("fetch unavailable"));
      var init = { method: method, cache: "no-store", headers: { accept: "application/json" } };
      if (body !== undefined) {
        init.headers["content-type"] = "application/json";
        init.body = JSON.stringify(body);
      }
      return fetchImpl(endpoint, init).then(function (res) {
        return res.json().catch(function () { return null; }).then(function (data) {
          if (!res.ok) throw new Error(errorText(data, res));
          return data;
        });
      });
    }

    function readJsonInputs() {
      var parsed = {};
      var errors = {};
      JSON_KEYS.forEach(function (key) {
        try {
          var value = JSON.parse(ui.json[key].value);
          if (!isObject(value)) throw new Error("Value must be a JSON object.");
          parsed[key] = value;
        } catch (err) {
          errors[key] = err && err.message ? err.message : "Invalid JSON.";
        }
      });
      state.validationErrors = errors;
      return { values: parsed, errors: errors };
    }

    function syncDirty() {
      var json = readJsonInputs();
      var changed = [];
      BOOL_KEYS.forEach(function (key) {
        state.values[key] = !!ui.bools[key].checked;
        if (!equal(state.values[key], state.original[key])) changed.push(key);
      });
      STR_KEYS.forEach(function (key) {
        state.values[key] = ui.strings[key].value;
        if (!equal(state.values[key], state.original[key])) changed.push(key);
      });
      JSON_KEYS.forEach(function (key) {
        if (Object.prototype.hasOwnProperty.call(json.values, key)) state.values[key] = json.values[key];
        if (json.errors[key] || !equal(state.values[key], state.original[key])) changed.push(key);
      });
      state.changedKeys = changed;
      state.dirty = changed.length > 0;
      renderStatus();
      renderActions();
      renderValidation();
    }

    function setForm(values) {
      state.values = clone(values);
      state.original = clone(values);
      BOOL_KEYS.forEach(function (key) { ui.bools[key].checked = values[key]; });
      STR_KEYS.forEach(function (key) { ui.strings[key].value = values[key]; });
      JSON_KEYS.forEach(function (key) {
        ui.json[key].value = JSON.stringify(values[key], null, 2);
      });
      state.dirty = false;
      state.changedKeys = [];
      state.validationErrors = {};
      renderValidation();
      renderActions();
    }

    function build() {
      var root = node("section", "hive-settings");
      var header = node("header", "hsg-header");
      var title = node("div", "hsg-title", "Global settings");
      header.append(title, node("p", "hsg-subtitle", "Set the defaults and global controls used by Studio."));
      ui.refresh = node("button", "hsg-btn", "Reload");
      ui.refresh.type = "button";
      ui.refresh.addEventListener("click", function () { refresh(); });
      header.appendChild(ui.refresh);
      ui.status = node("div", "hsg-status", "Not loaded");
      ui.error = node("div", "hsg-message hsg-error");
      ui.notice = node("div", "hsg-message hsg-notice");
      ui.empty = node("div", "hsg-empty", "The server returned no settings. Showing built-in defaults.");

      var toggleGroup = node("section", "hsg-card");
      toggleGroup.appendChild(node("h2", "hsg-section-title", "Global controls"));
      var labels = {
        sandbox_enabled: "Sandbox enabled",
        tools_enabled: "Tools enabled",
        auto_compact: "Auto-compact conversations",
        dev_mode: "Developer mode",
        harvest_enabled: "Harvest chat turns"
      };
      BOOL_KEYS.forEach(function (key) {
        var row = node("label", "hsg-toggle");
        var input = node("input", "hsg-checkbox");
        input.type = "checkbox";
        input.setAttribute("data-setting-key", key);
        input.addEventListener("change", syncDirty);
        ui.bools[key] = input;
        row.append(input, node("span", "", labels[key]));
        toggleGroup.appendChild(row);
      });

      var stringGroup = node("section", "hsg-card hsg-string-grid");
      stringGroup.appendChild(node("h2", "hsg-section-title hsg-wide", "Strata"));
      STR_KEYS.forEach(function (key) {
        var label = node("label", "hsg-field");
        var titleText = key === "strata_base" ? "Strata base URL" : "Strata repository path";
        label.appendChild(node("span", "hsg-label", titleText));
        var input = node("input", "hsg-input");
        input.type = "text";
        input.autocomplete = "off";
        input.setAttribute("data-setting-key", key);
        input.addEventListener("input", syncDirty);
        ui.strings[key] = input;
        label.appendChild(input);
        stringGroup.appendChild(label);
      });

      var jsonGroup = node("section", "hsg-json-grid");
      jsonGroup.appendChild(node("h2", "hsg-section-title hsg-wide", "Configuration blocks"));
      JSON_KEYS.forEach(function (key) {
        var field = node("label", "hsg-card hsg-json-field");
        field.appendChild(node("span", "hsg-label", key));
        var area = node("textarea", "hsg-textarea");
        area.spellcheck = false;
        area.setAttribute("data-json-key", key);
        area.setAttribute("aria-label", key + " JSON");
        area.addEventListener("input", syncDirty);
        ui.json[key] = area;
        var error = node("span", "hsg-field-error");
        error.hidden = true;
        ui.jsonErrors[key] = error;
        field.append(area, error);
        jsonGroup.appendChild(field);
      });

      var actions = node("footer", "hsg-actions");
      ui.dirty = node("span", "hsg-dirty", "Unsaved changes");
      ui.dirty.hidden = true;
      ui.save = node("button", "hsg-btn hsg-primary", "Save settings");
      ui.save.type = "button";
      ui.save.disabled = true;
      ui.save.addEventListener("click", function () { save(); });
      actions.append(ui.dirty, ui.save);
      root.append(header, ui.status, ui.error, ui.notice, ui.empty, toggleGroup, stringGroup, jsonGroup, actions);
      clear(el);
      el.appendChild(root);
    }

    function renderStatus() {
      ui.status.textContent = state.status === "loading" ? "Loading settings…"
        : state.status === "saving" ? "Saving settings…"
        : state.status === "ready" ? (state.dirty ? "Unsaved changes" : "Settings loaded")
        : state.status === "empty" ? "No settings returned"
        : state.status === "error" ? "Settings unavailable" : "Not loaded";
      ui.empty.hidden = state.status !== "empty";
      ui.error.hidden = !state.error;
      ui.error.textContent = state.error || "";
      ui.notice.hidden = !state.notice;
      ui.notice.textContent = state.notice || "";
    }

    function renderActions() {
      var busy = state.status === "loading" || state.status === "saving";
      ui.refresh.disabled = busy;
      ui.dirty.hidden = !state.dirty;
      ui.save.disabled = busy || !state.dirty || Object.keys(state.validationErrors).length > 0;
      KNOWN_KEYS.forEach(function (key) {
        var input = ui.bools[key] || ui.strings[key] || ui.json[key];
        if (input) input.disabled = busy;
      });
    }

    function renderValidation() {
      JSON_KEYS.forEach(function (key) {
        var message = state.validationErrors[key] || "";
        ui.jsonErrors[key].textContent = message;
        ui.jsonErrors[key].hidden = !message;
        if (message) ui.json[key].setAttribute("aria-invalid", "true");
        else ui.json[key].removeAttribute("aria-invalid");
      });
    }

    function refresh() {
      state.status = "loading";
      state.error = null;
      state.notice = null;
      renderStatus();
      renderActions();
      return request("GET").then(function (data) {
        var responseData = isObject(data) ? data : {};
        state.serverSettings = clone(responseData); // retain unknown keys locally; never submit them
        var values = normalize(responseData);
        setForm(values);
        state.status = Object.keys(responseData).length ? "ready" : "empty";
      }).catch(function (err) {
        state.status = "error";
        state.error = String(err && err.message ? err.message : err);
      }).then(function () {
        renderStatus();
        renderActions();
        return state;
      });
    }

    function save() {
      if (state.status === "loading" || state.status === "saving") return Promise.resolve(state);
      var parsed = readJsonInputs();
      state.validationErrors = parsed.errors;
      var changed = [];
      var payload = {};
      BOOL_KEYS.forEach(function (key) {
        state.values[key] = !!ui.bools[key].checked;
        if (!equal(state.values[key], state.original[key])) changed.push(key);
      });
      STR_KEYS.forEach(function (key) {
        state.values[key] = ui.strings[key].value;
        if (!equal(state.values[key], state.original[key])) changed.push(key);
      });
      JSON_KEYS.forEach(function (key) {
        if (parsed.errors[key]) return;
        state.values[key] = parsed.values[key];
        if (!equal(state.values[key], state.original[key])) changed.push(key);
      });
      state.changedKeys = changed;
      state.dirty = changed.length > 0;
      renderValidation();
      if (Object.keys(parsed.errors).length || !changed.length) {
        renderStatus();
        renderActions();
        return Promise.resolve(state);
      }
      changed.forEach(function (key) {
        if (KNOWN_KEYS.indexOf(key) !== -1) payload[key] = clone(state.values[key]);
      });
      state.status = "saving";
      state.error = null;
      state.notice = null;
      renderStatus();
      renderActions();
      return request("PUT", payload).then(function (data) {
        var responseData = isObject(data) ? data : {};
        // Some compatible APIs may return only their effective object. Use submitted values
        // for changed keys omitted from that response while keeping all server extras intact.
        var effective = Object.assign({}, responseData);
        Object.keys(payload).forEach(function (key) {
          if (!Object.prototype.hasOwnProperty.call(effective, key)) effective[key] = payload[key];
        });
        state.serverSettings = clone(effective);
        setForm(normalize(effective));
        state.status = Object.keys(effective).length ? "ready" : "empty";
        state.notice = "Settings saved.";
      }).catch(function (err) {
        state.status = "ready";
        state.error = String(err && err.message ? err.message : err);
      }).then(function () {
        renderStatus();
        renderActions();
        return state;
      });
    }

    build();
    renderStatus();
    renderActions();
    var ready = refresh();
    return { refresh: refresh, state: state, ready: ready, save: save };
  }

  global.HiveSettingsPanel = { mount: mount };
})(typeof window !== "undefined" ? window : globalThis);
