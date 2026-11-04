/*
 * HiveLifecyclePanel — model / engine lifecycle panel (HIVE-WEBUI F3).
 *
 * Self-contained read + mutate panel for the live Hive sidecar.  QUEEN mounts
 * it into the new shell; nothing here is wired to a page on its own.
 *
 * Read-only (fetched on mount/refresh only, never mutated automatically):
 *   GET /api/v1/models/local       local GGUF library
 *   GET /api/v1/server/status      loaded instances + hardware + health
 *   GET /api/v1/stacks             saved stack definitions
 *   GET /api/v1/stacks/status      applied stack + per-tier health
 *   GET /api/v1/engines            engine profiles + default (allocation)
 *
 * Mutations (every one is gated on confirm() and disabled while pending, and
 * NEVER invoked on mount/refresh — only from an explicit button click):
 *   POST /api/v1/server/start      load a local model
 *   POST /api/v1/server/unload     unload one instance by key
 *   POST /api/v1/stacks/{name}/validate
 *   POST /api/v1/stacks/{name}/apply
 *   POST /api/v1/stacks/{name}/unload
 *
 * window.HiveLifecyclePanel.mount(el, opts?) -> { refresh, state }
 *   opts.base    : API prefix (default "/api/v1")
 *   opts.confirm : confirm function (default window.confirm) — injectable for tests
 */
(function (global) {
  "use strict";

  var BASE = "/api/v1";

  // ---------------------------------------------------------------- helpers
  function esc(value) {
    return String(value == null ? "" : value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function fmtGb(value) {
    var n = Number(value);
    if (!isFinite(n)) return "—";
    return n.toFixed(1) + " GB";
  }

  function fmtInt(value) {
    var n = Number(value);
    if (!isFinite(n)) return "—";
    return String(Math.round(n));
  }

  function timeNow() {
    return new Date().toLocaleTimeString([], { hour12: false });
  }

  // Pull an error string out of a FastAPI JSON error body or a raw response.
  function extractError(data, res) {
    if (data && typeof data === "object") {
      if (typeof data.detail === "string") return data.detail;
      if (typeof data.error === "string") return data.error;
      if (typeof data.message === "string") return data.message;
    }
    if (data && typeof data.detail === "object") {
      try { return JSON.stringify(data.detail); } catch (_) { /* ignore */ }
    }
    return (res && res.status ? res.status + " " + res.statusText : "request failed");
  }

  // Compact, stable summary of a mutation result for the message banner.
  function summarize(data) {
    if (data == null) return "";
    if (typeof data !== "object") return String(data);
    var parts = [];
    if (typeof data.ok === "boolean") parts.push(data.ok ? "ok" : "failed");
    if (data.model) parts.push("model " + data.model);
    if (data.key) parts.push("key " + data.key);
    if (data.port) parts.push("port " + data.port);
    if (Array.isArray(data.unloaded)) parts.push("unloaded " + data.unloaded.length);
    if (Array.isArray(data.tiers)) parts.push(data.tiers.length + " tier(s)");
    if (Array.isArray(data.warnings) && data.warnings.length)
      parts.push(data.warnings.length + " warning(s)");
    if (parts.length === 0) {
      var keys = Object.keys(data).slice(0, 4);
      if (keys.length) {
        parts = keys.map(function (k) {
          var v = data[k];
          if (typeof v === "object") v = JSON.stringify(v);
          return k + "=" + String(v).slice(0, 40);
        });
      }
    }
    return parts.join(" · ");
  }

  // ------------------------------------------------------------------ panel
  function createPanel(rootEl, opts) {
    opts = opts || {};
    var base = opts.base || BASE;
    var confirmFn = opts.confirm || (typeof global.confirm === "function" ? global.confirm : function () { return false; });

    // The live state object handed back to the integrator.
    var state = {
      status: null,       // GET /server/status
      models: null,       // GET /models/local  { models_dir, models }
      stacks: [],         // GET /stacks        [ { name, tiers, roles } ]
      stackStatus: null,  // GET /stacks/status { ok, stack, tiers, warnings }
      engines: null,      // GET /engines       { default, engines, file, revision }
      errors: {},         // per-endpoint read error text
      pending: "",        // human label of the in-flight mutation ("" = idle)
      message: null,      // { kind: "ok"|"error", text }
      lastUpdated: null,
    };

    var els = {};

    function buildDom() {
      rootEl.innerHTML =
        '<div class="lcp">' +
        '  <div class="lcp-toolbar">' +
        '    <div class="lcp-toolbar-left">' +
        '      <span class="dot" data-lcp-dot></span>' +
        '      <span class="lcp-status" data-lcp-status>loading…</span>' +
        '      <span class="lcp-updated muted" data-lcp-updated></span>' +
        '    </div>' +
        '    <button type="button" class="ghost" data-lcp-refresh>Refresh</button>' +
        '  </div>' +
        '  <div class="lcp-message" data-lcp-message hidden></div>' +

        '  <section class="lcp-section">' +
        '    <div class="lcp-head"><h2>Loaded instances</h2>' +
        '      <span class="muted" data-lcp-hw></span></div>' +
        '    <div data-lcp-loaded></div>' +
        '  </section>' +

        '  <section class="lcp-section">' +
        '    <div class="lcp-head"><h2>Local model library</h2>' +
        '      <span class="muted" data-lcp-models-meta></span></div>' +
        '    <div data-lcp-models></div>' +
        '  </section>' +

        '  <section class="lcp-section">' +
        '    <div class="lcp-head"><h2>Stacks</h2>' +
        '      <span class="muted" data-lcp-stacks-meta></span></div>' +
        '    <div data-lcp-stacks></div>' +
        '  </section>' +

        '  <section class="lcp-section">' +
        '    <div class="lcp-head"><h2>Engine profiles</h2>' +
        '      <span class="muted" data-lcp-engines-meta></span></div>' +
        '    <div data-lcp-engines></div>' +
        '  </section>' +
        '</div>';

      els.dot = rootEl.querySelector("[data-lcp-dot]");
      els.status = rootEl.querySelector("[data-lcp-status]");
      els.updated = rootEl.querySelector("[data-lcp-updated]");
      els.message = rootEl.querySelector("[data-lcp-message]");
      els.hw = rootEl.querySelector("[data-lcp-hw]");
      els.loaded = rootEl.querySelector("[data-lcp-loaded]");
      els.modelsMeta = rootEl.querySelector("[data-lcp-models-meta]");
      els.models = rootEl.querySelector("[data-lcp-models]");
      els.stacksMeta = rootEl.querySelector("[data-lcp-stacks-meta]");
      els.stacks = rootEl.querySelector("[data-lcp-stacks]");
      els.enginesMeta = rootEl.querySelector("[data-lcp-engines-meta]");
      els.engines = rootEl.querySelector("[data-lcp-engines]");
    }

    // ------------------------------------------------------------- fetch IO
    function apiGet(path) {
      return fetch(base + path, { cache: "no-store", headers: { accept: "application/json" } })
        .then(function (res) {
          return res.text().then(function (text) {
            var data = null;
            try { data = text ? JSON.parse(text) : null; } catch (_) { data = null; }
            if (!res.ok) {
              var err = new Error(extractError(data, res));
              err.status = res.status;
              throw err;
            }
            return data;
          });
        });
    }

    function apiPost(path, body) {
      return fetch(base + path, {
        method: "POST",
        cache: "no-store",
        headers: { "content-type": "application/json", accept: "application/json" },
        body: body ? JSON.stringify(body) : undefined,
      }).then(function (res) {
        return res.text().then(function (text) {
          var data = null;
          try { data = text ? JSON.parse(text) : null; } catch (_) { data = null; }
          if (!res.ok) {
            var err = new Error(extractError(data, res));
            err.status = res.status;
            throw err;
          }
          return data;
        });
      });
    }

    // ------------------------------------------------------------ read-only
    function refresh() {
      var fetches = [
        apiGet("/server/status").then(function (d) { state.status = d; delete state.errors.status; },
          function (e) { state.status = null; state.errors.status = e.message; }),
        apiGet("/models/local").then(function (d) { state.models = d; delete state.errors.models; },
          function (e) { state.models = null; state.errors.models = e.message; }),
        apiGet("/stacks").then(function (d) { state.stacks = Array.isArray(d) ? d : (d && d.stacks) || []; delete state.errors.stacks; },
          function (e) { state.stacks = []; state.errors.stacks = e.message; }),
        apiGet("/stacks/status").then(function (d) { state.stackStatus = d; delete state.errors.stackStatus; },
          function (e) { state.stackStatus = null; state.errors.stackStatus = e.message; }),
        apiGet("/engines").then(function (d) { state.engines = d; delete state.errors.engines; },
          function (e) { state.engines = null; state.errors.engines = e.message; }),
      ];
      return Promise.allSettled(fetches).then(function () {
        state.lastUpdated = timeNow();
        renderAll();
        return state;
      });
    }

    // ------------------------------------------------------------- mutations
    // Every mutation is gated on confirm() and runs only from a click handler.
    // state.pending disables all mutation buttons while one is in flight, and
    // no mutation is ever triggered on mount/refresh.
    function runMutation(kind, label, path, body, confirmText) {
      if (state.pending) return Promise.resolve(null);
      if (!confirmFn(confirmText)) return Promise.resolve(null);

      state.pending = label;
      state.message = null;
      syncPendingUI();
      return apiPost(path, body).then(function (data) {
        state.message = { kind: "ok", text: label + " → " + (summarize(data) || "done") };
        return data;
      }).catch(function (err) {
        state.message = { kind: "error", text: label + " failed → " + err.message };
        return null;
      }).then(function (data) {
        state.pending = "";
        syncPendingUI();
        return refresh().then(function () { return data; });
      });
    }

    function confirmLoadModel(m) {
      return runMutation(
        "load",
        "Load " + (m.file || m.name),
        "/server/start",
        { model: m.file || m.name, register_provider: true },
        "Load model \"" + (m.file || m.name) + "\"? This starts a live llama-server " +
          "and may claim the default provider slot. Proceed?"
      );
    }

    function confirmUnloadInstance(item) {
      return runMutation(
        "unload",
        "Unload " + (item.key || item.model),
        "/server/unload",
        { key: item.key },
        "Unload instance \"" + (item.key || item.model) + "\"? This stops the running " +
          "server and frees its VRAM. Proceed?"
      );
    }

    function confirmStackAction(action, name) {
      var verb = action === "apply" ? "Apply"
        : action === "unload" ? "Unload stack"
        : "Validate";
      var confirmText;
      if (action === "apply") {
        confirmText = "Apply stack \"" + name + "\"? This loads its tiers (unloading the " +
          "currently applied stack first) and starts live model servers. Proceed?";
      } else if (action === "unload") {
        confirmText = "Unload stack \"" + name + "\"? This stops every launched tier. Proceed?";
      } else {
        confirmText = "Validate stack \"" + name + "\"? Read-only: checks the stack against " +
          "local hardware. Proceed?";
      }
      return runMutation(
        "stack-" + action,
        verb + " " + name,
        "/stacks/" + encodeURIComponent(name) + "/" + action,
        null,
        confirmText
      );
    }

    // --------------------------------------------------------------- render
    function renderAll() {
      renderToolbar();
      renderLoaded();
      renderModels();
      renderStacks();
      renderEngines();
      renderMessage();
    }

    function renderToolbar() {
      var online = !!state.status && !state.errors.status;
      els.dot.className = "dot " + (online ? "ok" : "bad");
      var parts = [];
      if (state.errors.status) parts.push("status: " + state.errors.status);
      if (state.errors.models) parts.push("models: " + state.errors.models);
      if (state.errors.stacks) parts.push("stacks: " + state.errors.stacks);
      if (state.errors.stackStatus) parts.push("stacks/status: " + state.errors.stackStatus);
      if (state.errors.engines) parts.push("engines: " + state.errors.engines);

      if (state.pending) {
        els.status.textContent = "pending: " + state.pending + " …";
        els.status.className = "lcp-status lcp-pending";
      } else if (parts.length) {
        els.status.textContent = "partial: " + parts.join("; ");
        els.status.className = "lcp-status lcp-warn";
      } else {
        els.status.textContent = "Hive reachable";
        els.status.className = "lcp-status lcp-ok";
      }
      els.updated.textContent = state.lastUpdated ? "updated " + state.lastUpdated : "";

      var hw = state.status && state.status.hardware;
      if (hw) {
        var bits = [];
        if (hw.vram_gb != null) bits.push("VRAM " + fmtGb(hw.vram_gb));
        if (hw.available_gb != null) bits.push("avail " + fmtGb(hw.available_gb));
        if (hw.available_ram_gb != null) bits.push("RAM " + fmtGb(hw.available_ram_gb));
        els.hw.textContent = bits.length ? bits.join(" · ") : "";
      } else {
        els.hw.textContent = "";
      }
    }

    function renderLoaded() {
      var box = els.loaded;
      if (state.errors.status) {
        box.innerHTML = '<div class="empty-line muted">server status unavailable: ' +
          esc(state.errors.status) + "</div>";
        return;
      }
      var st = state.status || {};
      var loaded = Array.isArray(st.loaded) ? st.loaded : (Array.isArray(st.instances) ? st.instances : []);

      if (!loaded.length) {
        box.innerHTML = '<div class="empty-line muted">No loaded instances. ' +
          "Load a model from the library below, or apply a stack.</div>";
        return;
      }

      var rows = loaded.map(function (item) {
        var backend = item.backend || "—";
        var isStrata = String(backend).toLowerCase() === "strata";
        var health = item.healthy ? '<span class="tag allow">healthy</span>'
          : '<span class="tag deny">down</span>';
        var cache = item.cache_share != null
          ? '<span class="lcp-cache" title="page-cache residency share">cache ' +
            Math.round(Number(item.cache_share) * 100) + "%</span>"
          : "";
        var actions = "";
        if (!isStrata && item.key) {
          actions =
            '<button type="button" class="lcp-btn danger" data-action="unload" ' +
            'data-key="' + esc(item.key) + '" data-model="' + esc(item.model || "") + '">Unload</button>';
        } else {
          actions = '<span class="muted">external</span>';
        }
        return "<tr>" +
          "<td class=\"lcp-strong\">" + esc(item.model || item.key || "—") + "</td>" +
          "<td class=\"mono\">" + esc(backend) + "</td>" +
          "<td class=\"mono\">" + (item.port != null ? esc(item.port) : "—") + "</td>" +
          "<td>" + health + " " + cache + "</td>" +
          "<td class=\"lcp-actions\">" + actions + "</td>" +
          "</tr>";
      }).join("");

      box.innerHTML =
        '<div class="table-wrap"><table class="lcp-table">' +
        "<thead><tr><th>model</th><th>backend</th><th>port</th><th>health</th><th>actions</th></tr></thead>" +
        "<tbody>" + rows + "</tbody></table></div>" +
        '<p class="note lcp-destructive-note">Unload is destructive: it stops the ' +
        "running server and frees its VRAM.</p>";
    }

    function modelMeta(m) {
      var gguf = m && (m.gguf_metadata || m.ggufMetadata) || {};
      var arch = m && (m.architecture != null ? m.architecture
        : (gguf.architecture != null ? gguf.architecture : gguf["general.architecture"]));
      var quant = m && (m.quantization != null ? m.quantization : gguf.quantization);
      var ctx = m && (m.context_length != null ? m.context_length : gguf.context_length);
      return { arch: arch, quant: quant, ctx: ctx };
    }

    function renderModels() {
      var box = els.models;
      var meta = els.modelsMeta;
      if (state.errors.models) {
        meta.textContent = "unavailable";
        box.innerHTML = '<div class="empty-line muted">model library unavailable: ' +
          esc(state.errors.models) + "</div>";
        return;
      }
      var models = (state.models && state.models.models) || [];
      meta.textContent = models.length + " model" + (models.length === 1 ? "" : "s");

      if (!models.length) {
        box.innerHTML = '<div class="empty-line muted">No local models.</div>';
        return;
      }

      var rows = models.map(function (m) {
        var mm = modelMeta(m);
        var quant = mm.quant ? '<span class="tag">' + esc(mm.quant) + "</span>" : '<span class="muted">—</span>';
        var arch = mm.arch ? '<span class="tag host">' + esc(mm.arch) + "</span>" : '<span class="muted">—</span>';
        var ctx = mm.ctx != null ? '<span class="muted">ctx ' + fmtInt(mm.ctx) + "</span>" : "";
        return "<tr>" +
          "<td class=\"lcp-strong\">" + esc(m.name || m.file || "—") + "</td>" +
          "<td class=\"mono muted\">" + esc(m.file || "—") + "</td>" +
          "<td class=\"mono\">" + fmtGb(m.size_gb != null ? m.size_gb : m.sizeGb) + "</td>" +
          "<td>" + quant + "</td>" +
          "<td>" + arch + " " + ctx + "</td>" +
          "<td class=\"lcp-actions\">" +
          '<button type="button" class="lcp-btn" data-action="load" ' +
          'data-file="' + esc(m.file || "") + '" data-name="' + esc(m.name || "") + '">Load</button>' +
          "</td>" +
          "</tr>";
      }).join("");

      box.innerHTML =
        '<div class="table-wrap"><table class="lcp-table">' +
        "<thead><tr><th>name</th><th>file</th><th>size</th><th>quant</th><th>arch</th><th>actions</th></tr></thead>" +
        "<tbody>" + rows + "</tbody></table></div>" +
        '<p class="note lcp-destructive-note">Load starts a live llama-server for that ' +
        "model on a free port.</p>";
    }

    function renderStacks() {
      var box = els.stacks;
      var meta = els.stacksMeta;
      if (state.errors.stacks) {
        meta.textContent = "unavailable";
        box.innerHTML = '<div class="empty-line muted">stacks unavailable: ' +
          esc(state.errors.stacks) + "</div>";
        return;
      }
      var stacks = state.stacks || [];
      meta.textContent = stacks.length + " saved";

      var status = state.stackStatus || {};
      var applied = status.stack || null;
      var tiers = Array.isArray(status.tiers) ? status.tiers : [];
      var statusErr = state.errors.stackStatus;

      if (!stacks.length && !tiers.length && !statusErr) {
        box.innerHTML = '<div class="empty-line muted">No saved stacks.</div>';
        return;
      }

      var chunks = [];

      if (statusErr) {
        chunks.push('<div class="lcp-note warn">stacks/status unavailable: ' +
          esc(statusErr) + "</div>");
      } else if (applied) {
        chunks.push(renderAppliedTierStrip(applied, tiers, status.warnings));
      } else {
        chunks.push('<div class="lcp-note muted">No stack applied — tier health empty.</div>');
      }

      chunks.push(stacks.map(function (s) {
        var name = s.name || "—";
        var roles = Array.isArray(s.roles) ? s.roles : [];
        var isApplied = applied === name;
        var badge = isApplied
          ? '<span class="tag allow">applied</span>'
          : '<span class="tag">saved</span>';
        var roleList = roles.length ? roles.map(function (r) {
          return '<span class="tag host">' + esc(r) + "</span>";
        }).join(" ") : '<span class="muted">—</span>';

        return '<div class="lcp-card">' +
          '<div class="lcp-card-head">' +
          '<span class="lcp-strong">' + esc(name) + "</span> " + badge +
          '<span class="muted">' + fmtInt(s.tiers) + " tier(s)</span>" +
          "</div>" +
          '<div class="lcp-card-roles">' + roleList + "</div>" +
          '<div class="lcp-actions">' +
          '<button type="button" class="lcp-btn" data-action="stack-validate" data-name="' + esc(name) + '">Validate</button>' +
          '<button type="button" class="lcp-btn warn" data-action="stack-apply" data-name="' + esc(name) + '">Apply</button>' +
          '<button type="button" class="lcp-btn danger" data-action="stack-unload" data-name="' + esc(name) + '">Unload-stack</button>' +
          "</div>" +
          "</div>";
      }).join(""));

      box.innerHTML = chunks.join("");
    }

    function renderAppliedTierStrip(applied, tiers, warnings) {
      var warnHtml = Array.isArray(warnings) && warnings.length
        ? '<div class="lcp-note warn">warnings: ' +
          esc(warnings.map(String).join("; ")) + "</div>"
        : "";

      var rows = tiers.map(function (t) {
        var resident = t.resident ? '<span class="tag allow">resident</span>'
          : '<span class="tag deny">not resident</span>';
        var managed = t.managed ? '<span class="tag host">managed</span>'
          : '<span class="muted">adopted</span>';
        var tok = t.tok_s != null ? fmtInt(t.tok_s) + " tok/s" : "—";
        return "<tr>" +
          "<td class=\"lcp-strong\">" + esc(t.role || "—") + "</td>" +
          "<td class=\"mono\">" + esc(t.model || t.key || "—") + "</td>" +
          "<td class=\"mono\">" + (t.port != null ? esc(t.port) : "—") + "</td>" +
          "<td class=\"mono\">ctx " + fmtInt(t.ctx) + "</td>" +
          "<td class=\"mono\">" + fmtGb(t.vram_gb) + "</td>" +
          "<td class=\"mono\">" + tok + "</td>" +
          "<td>" + resident + " " + managed + "</td>" +
          "</tr>";
      }).join("");

      return '<div class="lcp-applied">' +
        '<div class="lcp-applied-head"><span class="tag allow">applied</span>' +
        '<span class="lcp-strong">' + esc(applied) + "</span></div>" +
        warnHtml +
        '<div class="table-wrap"><table class="lcp-table">' +
        "<thead><tr><th>role</th><th>model</th><th>port</th><th>ctx</th><th>VRAM</th><th>throughput</th><th>state</th></tr></thead>" +
        "<tbody>" + (rows || '<tr><td colspan="7" class="muted">no tier rows</td></tr>') + "</tbody></table></div>" +
        "</div>";
    }

    function renderEngines() {
      var box = els.engines;
      var meta = els.enginesMeta;
      if (state.errors.engines) {
        meta.textContent = "unavailable";
        box.innerHTML = '<div class="empty-line muted">engines unavailable: ' +
          esc(state.errors.engines) + "</div>";
        return;
      }
      var d = state.engines || {};
      var engines = Array.isArray(d.engines) ? d.engines : [];
      var def = d.default || "";
      meta.textContent = engines.length + " profile" + (engines.length === 1 ? "" : "s");

      if (!engines.length) {
        box.innerHTML = '<div class="empty-line muted">No engine profiles.</div>';
        return;
      }

      var rows = engines.map(function (e) {
        var lo = e.load_options || {};
        var isDefault = String(e.name || "").toLowerCase() === String(def).toLowerCase();
        var badge = isDefault
          ? '<span class="tag allow">default</span>'
          : '<span class="tag">profile</span>';
        var where = e.base_url || e.provider || "";
        var ctx = lo.context != null ? "ctx " + fmtInt(lo.context) : "ctx —";
        var gpu = lo.gpu_layers != null ? "gpu " + fmtInt(lo.gpu_layers) : "gpu —";
        var caps = Array.isArray(e.capabilities) && e.capabilities.length
          ? e.capabilities.slice(0, 4).map(function (c) {
            return '<span class="tag">' + esc(c) + "</span>";
          }).join(" ")
          : '<span class="muted">—</span>';
        return "<tr>" +
          "<td class=\"lcp-strong\">" + esc(e.name || "—") + " " + badge + "</td>" +
          "<td class=\"mono\">" + esc(e.kind || "—") + "</td>" +
          "<td class=\"mono muted\">" + esc(where || "—") + "</td>" +
          "<td class=\"mono\">" + ctx + " · " + gpu + "</td>" +
          "<td>" + caps + "</td>" +
          "</tr>";
      }).join("");

      box.innerHTML =
        '<div class="table-wrap"><table class="lcp-table">' +
        "<thead><tr><th>profile</th><th>kind</th><th>base url / provider</th><th>allocation</th><th>capabilities</th></tr></thead>" +
        "<tbody>" + rows + "</tbody></table></div>" +
        '<p class="note">Engine profiles are read-only here; allocation is the ' +
        "context / GPU-layer load budget each profile advertises.</p>";
    }

    function renderMessage() {
      var box = els.message;
      if (!state.message) { box.hidden = true; return; }
      box.hidden = false;
      box.className = "lcp-message " + (state.message.kind === "error" ? "error" : "ok");
      box.textContent = state.message.text;
    }

    function syncPendingUI() {
      var buttons = rootEl.querySelectorAll("[data-action]");
      for (var i = 0; i < buttons.length; i++) {
        buttons[i].disabled = !!state.pending;
      }
      var refreshBtn = rootEl.querySelector("[data-lcp-refresh]");
      if (refreshBtn) refreshBtn.disabled = !!state.pending;
      renderToolbar();
      renderMessage();
    }

    // ------------------------------------------------------------ event wire
    function onClick(ev) {
      var t = ev.target;
      while (t && t !== rootEl && !(t.getAttribute && t.getAttribute("data-action"))) {
        t = t.parentElement;
      }
      if (!t || t === rootEl) {
        if (ev.target && ev.target.getAttribute && ev.target.getAttribute("data-lcp-refresh") != null) {
          refresh();
        }
        return;
      }
      var action = t.getAttribute("data-action");
      if (!action) return;
      if (state.pending) return;

      if (action === "load") {
        confirmLoadModel({ file: t.getAttribute("data-file"), name: t.getAttribute("data-name") });
      } else if (action === "unload") {
        confirmUnloadInstance({ key: t.getAttribute("data-key"), model: t.getAttribute("data-model") });
      } else if (action === "stack-validate" || action === "stack-apply" || action === "stack-unload") {
        confirmStackAction(action.slice("stack-".length), t.getAttribute("data-name"));
      }
    }

    // Wire the DOM and kick off the initial read-only load.  No mutation is
    // ever triggered here — only the five GET endpoints.
    buildDom();
    rootEl.addEventListener("click", onClick);
    refresh();

    return {
      refresh: function () { return refresh(); },
      state: state,
    };
  }

  // -------------------------------------------------------------- public API
  global.HiveLifecyclePanel = {
    mount: function (el, opts) {
      if (!el) throw new Error("HiveLifecyclePanel.mount: element required");
      return createPanel(el, opts);
    },
  };
})(typeof window !== "undefined" ? window : globalThis);
