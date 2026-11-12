/*
 * HiveSetupPanel — hardware detection and the first-model recommendation.
 *
 * Ports the useful part of the legacy Home wizard:
 *   GET  /api/v1/server/status       hardware + local models + loaded
 *   GET  /api/v1/studio/strata       optional Strata serve
 *   GET  /api/v1/engines/preset      per-model launch preset
 *   POST /api/v1/engines             save the profile
 *   POST /api/v1/server/start        load the model
 *   GET/POST /api/v1/provider/config to register Strata as a provider
 *
 * Read-only until the operator clicks "Set up" or "Use Strata"; both actions
 * confirm first because they load weights / change the default provider.
 *
 * window.HiveSetupPanel.mount(el, opts?) -> { refresh, state }
 */
(function (global) {
  "use strict";

  var BASE = "/api/v1";
  var KV_GB_PER_KTOK = 0.25; // legacy wizard estimate, kept for parity

  function esc(value) {
    return String(value == null ? "" : value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function num(value, digits) {
    var n = Number(value);
    if (!isFinite(n)) return "—";
    return n.toFixed(digits == null ? 1 : digits);
  }

  function gb(value) {
    var n = Number(value);
    if (!isFinite(n)) return "—";
    return n.toFixed(1) + " GB";
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
      hardware: null,
      strata: null,
      models: [],
      loaded: [],
      error: "",
      notice: "",
      busy: "",
    };
    var ui = {};

    function build() {
      rootEl.innerHTML = "";
      var root = node("section", "hive-setup");

      var intro = node("div", "panel");
      var introHead = node("div", "panel-head");
      introHead.append(node("h2", "", "First model"), node("span", "muted", "detect · recommend · load"));
      var introBody = node("div");
      introBody.style.padding = "12px 15px";
      var note = node("p", "note");
      note.style.margin = "0 0 8px";
      note.textContent =
        "Detect the box, rank the local GGUFs by what fits at 8k context, and " +
        "create + load an engine profile in one step. Nothing loads until you click a model.";
      ui.refresh = node("button", "ghost", "Detect hardware & recommend");
      ui.refresh.type = "button";
      ui.refresh.addEventListener("click", refresh);
      ui.message = node("span", "muted", "");
      introBody.append(note, ui.refresh, ui.message);
      intro.append(introHead, introBody);

      ui.cards = node("div", "cards");

      ui.strataPanel = node("div", "panel");
      var strataHead = node("div", "panel-head");
      strataHead.append(node("h2", "", "Strata backend"), node("span", "muted", "optional"));
      ui.strata = node("div", "note");
      ui.strata.style.padding = "12px 15px";
      ui.strataPanel.append(strataHead, ui.strata);

      ui.modelsPanel = node("div", "panel");
      var modelsHead = node("div", "panel-head");
      modelsHead.append(node("h2", "", "Local models"), node("span", "muted", "8k context estimate"));
      ui.models = node("div", "table-wrap");
      ui.modelsTable = document.createElement("table");
      ui.models.appendChild(ui.modelsTable);
      ui.modelsPanel.append(modelsHead, ui.models);

      root.append(intro, ui.cards, ui.strataPanel, ui.modelsPanel);
      rootEl.appendChild(root);
    }

    function renderMessage() {
      if (state.error) {
        ui.message.textContent = state.error;
        ui.message.style.color = "var(--bad)";
      } else if (state.notice) {
        ui.message.textContent = state.notice;
        ui.message.style.color = "var(--good)";
      } else if (state.status === "loading") {
        ui.message.textContent = "Detecting…";
        ui.message.style.color = "var(--muted)";
      } else {
        ui.message.textContent = "";
      }
    }

    function renderCards() {
      ui.cards.replaceChildren();
      var hw = state.hardware || {};
      if (!Object.keys(hw).length) return;
      ui.cards.append(
        card("VRAM", gb(hw.vram_gb), hw.available_gb != null ? gb(hw.available_gb) + " free" : ""),
        card("RAM", gb(hw.total_ram_gb), hw.available_ram_gb != null ? gb(hw.available_ram_gb) + " free" : ""),
        card("CPU", hw.cpu_count == null ? "—" : String(hw.cpu_count), "cores"),
        card("Loaded", String((state.loaded || []).length || 0), "models")
      );
    }

    function renderStrata() {
      ui.strata.replaceChildren();
      var strata = state.strata || {};
      if (!strata.alive) {
        ui.strata.textContent = "Strata is not running" +
          (strata.base_url ? " on " + strata.base_url : "") +
          ". Start the 176B serve separately if you want it as a backend.";
        return;
      }
      var row = node("div", "engine-actions");
      var label = node("span", "");
      label.append(
        node("b", "", "Strata is running"),
        node("span", "muted", " · " + (strata.model || "local 176B") +
          " · " + (strata.base_url || ""))
      );
      var use = node("button", "ghost", "Use Strata as default");
      use.type = "button";
      use.addEventListener("click", useStrata);
      row.append(label, use);
      ui.strata.appendChild(row);
    }

    function renderModels() {
      ui.modelsTable.replaceChildren();
      var thead = document.createElement("thead");
      var hr = document.createElement("tr");
      ["Model", "Size", "Needs at 8k", "Fit", ""].forEach(function (label) {
        var th = document.createElement("th");
        th.textContent = label;
        hr.appendChild(th);
      });
      thead.appendChild(hr);
      ui.modelsTable.appendChild(thead);
      var tbody = document.createElement("tbody");
      if (!state.models.length) {
        var tr = document.createElement("tr");
        var td = document.createElement("td");
        td.colSpan = 5;
        td.className = "note";
        td.textContent = state.status === "loading"
          ? "Scanning the local library…"
          : "No local GGUF models found. Import one on the Models page.";
        tr.appendChild(td);
        tbody.appendChild(tr);
      } else {
        state.models.forEach(function (model) {
          var row = document.createElement("tr");
          var name = document.createElement("td");
          name.textContent = model.name || model.file || "—";
          var size = document.createElement("td");
          size.className = "mono";
          size.textContent = gb(model.size_gb);
          var need = document.createElement("td");
          need.className = "mono";
          need.textContent = gb(model.need_gb);
          var fit = document.createElement("td");
          var tag = node("span", "tag " + (model.fits ? "allow" : "deny"),
            model.fits ? "fits" : "too large");
          fit.appendChild(tag);
          var action = document.createElement("td");
          var button = node("button", "ghost", state.busy === model.file ? "Setting up…" : "Set up");
          button.type = "button";
          button.disabled = !!state.busy;
          button.addEventListener("click", function () { setupModel(model); });
          action.appendChild(button);
          row.append(name, size, need, fit, action);
          tbody.appendChild(row);
        });
      }
      ui.modelsTable.appendChild(tbody);
    }

    function render() {
      ui.refresh.disabled = state.status === "loading" || !!state.busy;
      renderMessage();
      renderCards();
      renderStrata();
      renderModels();
    }

    async function refresh() {
      state.status = "loading";
      state.error = "";
      state.notice = "";
      render();
      try {
        var results = await Promise.allSettled([
          fetchJSON(base + "/server/status"),
          fetchJSON(base + "/studio/strata"),
        ]);
        var status = results[0].status === "fulfilled" ? results[0].value : null;
        if (!status) throw new Error(results[0].reason || "server status unavailable");
        state.hardware = status.hardware || {};
        state.loaded = status.loaded || [];
        var models = status.local_models;
        if (!Array.isArray(models)) {
          var local = await fetchJSON(base + "/models/local");
          models = local.models || [];
        }
        var available = Number(state.hardware.available_gb);
        if (!isFinite(available)) available = 0;
        state.models = (models || [])
          .filter(function (model) { return Number(model.size_gb || 0) > 0.2; })
          .map(function (model) {
            var size = Number(model.size_gb) || 0;
            var need = size + (8192 / 1024) * KV_GB_PER_KTOK;
            return {
              file: model.file || model.name || "",
              name: model.name || String(model.file || "").split("/").pop(),
              size_gb: size,
              need_gb: need,
              fits: need <= available,
            };
          })
          .sort(function (a, b) {
            if (a.fits !== b.fits) return a.fits ? -1 : 1;
            return b.size_gb - a.size_gb;
          })
          .slice(0, 8);
        state.strata = results[1].status === "fulfilled" ? results[1].value : {};
        state.status = "ready";
      } catch (err) {
        state.status = "error";
        state.error = "Setup detection failed: " + (err && err.message ? err.message : err);
      }
      render();
      return state;
    }

    async function useStrata() {
      if (state.busy) return;
      var strata = state.strata || {};
      if (!strata.alive) return;
      if (!confirmFn("Register Strata as the default provider? This changes the provider new conversations use.")) return;
      state.busy = "strata";
      state.error = "";
      state.notice = "";
      render();
      try {
        var current = await fetchJSON(base + "/provider/config");
        var providers = (current.providers || []).filter(function (p) { return p.name !== "strata"; });
        var root = String(strata.base_url || "").replace(/\/+$/, "");
        if (root.endsWith("/v1")) root = root.slice(0, -3);
        providers.push({
          name: "strata",
          base_url: root,
          api_key: "",
          model: strata.model || "",
          headers: {},
        });
        await postJSON(base + "/provider/config", {
          providers: providers,
          default: "strata",
          persist: true,
          expectedRevision: current.revision,
        });
        state.notice = "Strata is now the default provider.";
      } catch (err) {
        state.error = "Could not register Strata: " + (err && err.message ? err.message : err);
      }
      state.busy = "";
      render();
    }

    async function setupModel(model) {
      if (state.busy) return;
      var label = model.name || model.file;
      if (!confirmFn(
        "Create an engine profile and load " + label +
        " at 8k context?\n\nThis loads the weights and can take minutes."
      )) return;
      state.busy = model.file;
      state.error = "";
      state.notice = "";
      render();
      try {
        var preset = await fetchJSON(
          base + "/engines/preset?file=" + encodeURIComponent(model.file));
        var load = preset.load_options || preset.preset || {};
        var name = String(model.file || label).split("/").pop().replace(/\.gguf$/i, "");
        var current = await fetchJSON(base + "/engines");
        var engines = (current.engines || []).filter(function (e) { return e.name !== name; });
        engines.push({
          name: name,
          kind: "llama_cpp",
          model: model.file,
          base_url: "http://127.0.0.1:1234/v1",
          load_options: load,
          capabilities: ["streaming"],
          sampling: {},
        });
        await postJSON(base + "/engines", {
          engines: engines,
          default: current.default || name,
          persist: true,
          expectedRevision: current.revision,
        });
        await postJSON(base + "/server/start", {
          model: model.file,
          ctx_size: load.context || 8192,
          ngl: load.gpu_layers == null ? 999 : load.gpu_layers,
          flash_attn: !!load.flash_attn,
          register_provider: true,
          claim_default: true,
        });
        state.notice = "Started " + name + ". Open Chat or Lifecycle to use it.";
        state.busy = "";
        await refresh();
        state.notice = "Started " + name + ". Open Chat or Lifecycle to use it.";
      } catch (err) {
        state.busy = "";
        state.error = "Setup failed: " + (err && err.message ? err.message : err);
      }
      render();
    }

    build();
    render();
    refresh();
    return { refresh: refresh, state: state };
  }

  global.HiveSetupPanel = {
    mount: function (el, opts) {
      if (!el) throw new Error("HiveSetupPanel.mount: element required");
      return createPanel(el, opts);
    },
  };
})(typeof window !== "undefined" ? window : globalThis);
