/*
 * HiveTrainingPanel — Studio Training tab ported into the new shell (T1).
 *
 * Self-contained module; QUEEN owns the page wiring, so this file only defines
 * window.HiveTrainingPanel. It performs no network work at import time: reads
 * start in mount() / refresh(), and the only side-effecting call (launch) is
 * gated on confirm() and triggered exclusively by a button click.
 *
 *   const panel = window.HiveTrainingPanel.mount(el, opts);
 *   await panel.refresh();   // re-fetch engines + run history (read-only)
 *   panel.state;             // live state object (see createPanel)
 *
 * opts.base    : API prefix (default "/api/v1").
 * opts.confirm : confirm function (default window.confirm); injectable for tests.
 * opts.fetch   : fetch implementation (default global.fetch); injectable for tests.
 *
 * Live APIs used (proxied by the shell's dev_server to /v1/...):
 *   GET  /api/v1/training/engines            {engines, python, runner}
 *   POST /api/v1/training/launch             {engine, method, name, fields}
 *   GET  /api/v1/training/status/{run_dir}   {state, progress, log}
 *   POST /api/v1/training/report/{run_dir}   report bundle
 *   GET  /api/v1/runs                        run history
 *   GET  /api/v1/trainer/evidence            evidence + summary
 *   POST /api/v1/trainer/draft               {baseline} -> candidate
 *
 * Polling runs every 3s only while a run is active and stops on finished/
 * stopped. The preset trainer never auto-launches or promotes.
 */
(function (global) {
  "use strict";

  var BASE = "/api/v1";
  var POLL_MS = 3000;
  var MAX_POLL_ERRORS = 5;
  var REQUEST_TIMEOUT_MS = 20000;

  // ---------------------------------------------------------------- helpers

  function esc(value) {
    return String(value == null ? "" : value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function timeNow() {
    return new Date().toLocaleTimeString([], { hour12: false });
  }

  function fmtLoss(value) {
    var n = Number(value);
    if (!isFinite(n)) return "—";
    if (n === 0) return "0";
    return String(Math.round(n * 10000) / 10000);
  }

  function fmtRatio(value) {
    var n = Number(value);
    if (!isFinite(n)) return "—";
    // deployed_ratio / init_ratio arrive as fractions in [0, 1]; anything
    // larger is already a percentage-like or raw number, so echo it verbatim.
    if (n > 0 && n <= 1) return (n * 100).toFixed(1) + "%";
    return String(Math.round(n * 1000) / 1000);
  }

  function fmtElapsed(value) {
    var n = Number(value);
    if (!isFinite(n) || n < 0) return "—";
    if (n < 60) return Math.round(n) + "s";
    if (n < 3600) {
      return Math.floor(n / 60) + "m " + Math.round(n % 60) + "s";
    }
    return Math.floor(n / 3600) + "h " + Math.floor((n % 3600) / 60) + "m";
  }

  function fmtInt(value) {
    var n = Number(value);
    if (!isFinite(n)) return "—";
    return String(Math.round(n));
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

  function withTimeout(promise, ms, label) {
    return new Promise(function (resolve, reject) {
      var timer = setTimeout(function () {
        reject(new Error("timed out after " + Math.round(ms / 1000) + "s: " + label));
      }, ms);
      Promise.resolve(promise).then(function (value) {
        clearTimeout(timer);
        resolve(value);
      }, function (err) {
        clearTimeout(timer);
        reject(err);
      });
    });
  }

  // ------------------------------------------------------------------ panel
  function createPanel(rootEl, opts) {
    opts = opts || {};
    var base = String(opts.base == null ? BASE : opts.base).replace(/\/+$/, "");
    var confirmFn = typeof opts.confirm === "function"
      ? opts.confirm
      : (typeof global.confirm === "function" ? global.confirm : function () { return false; });
    var fetcher = typeof opts.fetch === "function"
      ? opts.fetch
      : (typeof global.fetch === "function" ? global.fetch.bind(global) : null);

    // Live state object handed back to the integrator.
    var state = {
      engines: null,        // { engines, python, runner }
      runs: [],             // run history from GET /runs
      run: null,            // active run { run_dir, pid, engine, method, name }
      status: null,         // last GET /training/status payload
      report: null,         // last POST /training/report bundle
      trainerSummary: null, // last evidence summary
      trainerEvidence: [],  // evidence items
      trainerDraft: null,   // last draft result
      trainerLogText: "",   // pretty-printed draft JSON
      errors: {},           // per-source error text
      pending: "",          // in-flight mutation label ("" = idle)
      message: null,        // { kind: "ok"|"error", text }
      lastUpdated: null,
    };

    var els = {};
    var lastEnginesKey = null;
    var pollTimer = null;
    var pollInFlight = false;
    var pollErrors = 0;
    var doc = rootEl.ownerDocument || (typeof document !== "undefined" ? document : null);

    function buildDom() {
      rootEl.innerHTML =
        '<div class="trp">' +
        '  <div class="trp-toolbar">' +
        '    <div class="trp-toolbar-left">' +
        '      <span class="dot" data-trp-dot></span>' +
        '      <span class="trp-status" data-trp-status>loading…</span>' +
        '      <span class="trp-updated muted" data-trp-updated></span>' +
        '    </div>' +
        '    <button type="button" class="ghost" data-trp-refresh>Refresh</button>' +
        '  </div>' +
        '  <div class="trp-message" data-trp-message hidden></div>' +

        '  <section class="trp-section">' +
        '    <div class="trp-head"><h2>New training run</h2>' +
        '      <span class="muted" data-trp-engine-meta></span></div>' +
        '    <div class="trp-form">' +
        '      <label class="trp-field">Engine' +
        '        <select data-trp-engine></select></label>' +
        '      <label class="trp-field">Recipe' +
        '        <select data-trp-method></select></label>' +
        '      <label class="trp-field">Run name' +
        '        <input data-trp-name type="text" placeholder="ste-rotate 20k"></label>' +
        '      <label class="trp-field trp-field--wide">Model dir' +
        '        <input data-trp-model type="text" placeholder="/path/to/hf base model"></label>' +
        '      <label class="trp-field trp-field--wide">Corpus' +
        '        <input data-trp-corpus type="text" placeholder="/path/to/corpus.txt"></label>' +
        '      <label class="trp-field">steps' +
        '        <input data-trp-steps type="number" value="20000" min="1" step="1"></label>' +
        '      <label class="trp-field">seq' +
        '        <input data-trp-seq type="number" value="512" min="1" step="1"></label>' +
        '      <label class="trp-field">lr' +
        '        <input data-trp-lr type="text" value="5e-5"></label>' +
        '      <label class="trp-field">device' +
        '        <input data-trp-device type="text" value="cuda:1"></label>' +
        '      <label class="trp-field">seed' +
        '        <input data-trp-seed type="number" value="1337" step="1"></label>' +
        '      <label class="trp-field">eval regions' +
        '        <input data-trp-regions type="number" value="8" min="0" step="1"></label>' +
        '      <label class="trp-field">eval windows' +
        '        <input data-trp-windows type="number" value="8" min="0" step="1"></label>' +
        '      <div class="trp-form-actions">' +
        '        <button type="button" class="trp-btn trp-btn--primary" data-action="launch">Start training</button>' +
        '      </div>' +
        '    </div>' +
        '    <div class="trp-run-meta mono" data-trp-run-meta hidden></div>' +
        '  </section>' +

        '  <section class="trp-section">' +
        '    <div class="trp-head"><h2>Run status</h2>' +
        '      <span class="muted" data-trp-status-meta></span></div>' +
        '    <div class="trp-stat-grid" data-trp-status-cards></div>' +
        '    <div class="trp-log-wrap">' +
        '      <div class="trp-log-head"><span class="muted">Streamed log (tail)</span></div>' +
        '      <pre class="trp-log" data-trp-log></pre>' +
        '    </div>' +
        '  </section>' +

        '  <section class="trp-section">' +
        '    <div class="trp-head"><h2>Run history</h2>' +
        '      <span class="muted">GET /api/v1/runs</span></div>' +
        '    <div data-trp-runs></div>' +
        '  </section>' +

        '  <section class="trp-section">' +
        '    <div class="trp-head"><h2>Preset trainer</h2>' +
        '      <span class="muted">evidence-driven composition iteration</span></div>' +
        '    <div class="trp-form">' +
        '      <label class="trp-field trp-field--wide">Baseline preset' +
        '        <input data-trp-baseline type="text" placeholder="/path/to/baseline-preset.yml"></label>' +
        '      <div class="trp-form-actions">' +
        '        <button type="button" class="trp-btn" data-action="mine">Mine evidence</button>' +
        '        <button type="button" class="trp-btn" data-action="draft">Draft candidate</button>' +
        '      </div>' +
        '      <div class="trp-trainer-status" data-trp-trainer-status></div>' +
        '    </div>' +
        '    <pre class="trp-log" data-trp-trainer-log hidden></pre>' +
        '  </section>' +
        '</div>';

      els.dot = rootEl.querySelector("[data-trp-dot]");
      els.status = rootEl.querySelector("[data-trp-status]");
      els.updated = rootEl.querySelector("[data-trp-updated]");
      els.refresh = rootEl.querySelector("[data-trp-refresh]");
      els.message = rootEl.querySelector("[data-trp-message]");

      els.engineMeta = rootEl.querySelector("[data-trp-engine-meta]");
      els.engineSelect = rootEl.querySelector("[data-trp-engine]");
      els.methodSelect = rootEl.querySelector("[data-trp-method]");
      els.nameInput = rootEl.querySelector("[data-trp-name]");
      els.modelInput = rootEl.querySelector("[data-trp-model]");
      els.corpusInput = rootEl.querySelector("[data-trp-corpus]");
      els.stepsInput = rootEl.querySelector("[data-trp-steps]");
      els.seqInput = rootEl.querySelector("[data-trp-seq]");
      els.lrInput = rootEl.querySelector("[data-trp-lr]");
      els.deviceInput = rootEl.querySelector("[data-trp-device]");
      els.seedInput = rootEl.querySelector("[data-trp-seed]");
      els.regionsInput = rootEl.querySelector("[data-trp-regions]");
      els.windowsInput = rootEl.querySelector("[data-trp-windows]");
      els.runMeta = rootEl.querySelector("[data-trp-run-meta]");

      els.statusMeta = rootEl.querySelector("[data-trp-status-meta]");
      els.statusCards = rootEl.querySelector("[data-trp-status-cards]");
      els.log = rootEl.querySelector("[data-trp-log]");

      els.runs = rootEl.querySelector("[data-trp-runs]");

      els.baselineInput = rootEl.querySelector("[data-trp-baseline]");
      els.trainerStatus = rootEl.querySelector("[data-trp-trainer-status]");
      els.trainerLog = rootEl.querySelector("[data-trp-trainer-log]");
    }

    function option(value, text) {
      var o = doc.createElement("option");
      o.value = value;
      o.textContent = text;
      return o;
    }

    // ------------------------------------------------------------- fetch IO
    function apiGet(path) {
      if (!fetcher) return Promise.reject(new Error("fetch is not available in this environment"));
      return withTimeout(
        Promise.resolve(fetcher(base + path, { cache: "no-store", headers: { accept: "application/json" } }))
          .then(function (res) {
            return Promise.resolve(res.text()).then(function (text) {
              var data = null;
              try { data = text ? JSON.parse(text) : null; } catch (_) { data = null; }
              if (!res.ok) {
                var err = new Error(extractError(data, res));
                err.status = res.status;
                throw err;
              }
              return data;
            });
          }),
        REQUEST_TIMEOUT_MS,
        "GET " + path
      );
    }

    function apiPost(path, body) {
      if (!fetcher) return Promise.reject(new Error("fetch is not available in this environment"));
      return withTimeout(
        Promise.resolve(fetcher(base + path, {
          method: "POST",
          cache: "no-store",
          headers: { "content-type": "application/json", accept: "application/json" },
          body: body ? JSON.stringify(body) : undefined,
        })).then(function (res) {
          return Promise.resolve(res.text()).then(function (text) {
            var data = null;
            try { data = text ? JSON.parse(text) : null; } catch (_) { data = null; }
            if (!res.ok) {
              var err = new Error(extractError(data, res));
              err.status = res.status;
              throw err;
            }
            return data;
          });
        }),
        REQUEST_TIMEOUT_MS,
        "POST " + path
      );
    }

    function setMessage(kind, text) {
      if (kind == null) {
        state.message = null;
        return;
      }
      state.message = { kind: kind, text: text };
    }

    // ------------------------------------------------------------ read-only
    function refresh() {
      var fetches = [
        apiGet("/training/engines").then(function (d) {
          state.engines = d;
          delete state.errors.engines;
        }, function (e) {
          state.engines = null;
          state.errors.engines = e.message;
        }),
        apiGet("/runs").then(function (d) {
          state.runs = Array.isArray(d) ? d : (d && Array.isArray(d.runs) ? d.runs : []);
          delete state.errors.runs;
        }, function (e) {
          state.runs = [];
          state.errors.runs = e.message;
        }),
      ];
      return Promise.allSettled(fetches).then(function () {
        state.lastUpdated = timeNow();
        renderAll();
        return state;
      });
    }

    function refreshRuns() {
      return apiGet("/runs").then(function (d) {
        state.runs = Array.isArray(d) ? d : (d && Array.isArray(d.runs) ? d.runs : []);
        delete state.errors.runs;
      }, function (e) {
        state.errors.runs = e.message;
      }).then(function () {
        renderRuns();
      });
    }

    // ------------------------------------------------------------ catalogue
    function enginesList() {
      return (state.engines && state.engines.engines) || [];
    }

    function defaultEngineId() {
      var list = enginesList();
      for (var i = 0; i < list.length; i += 1) {
        if (list[i].methods && list[i].methods.some(function (m) { return m.implemented; })) {
          return list[i].id;
        }
      }
      return list.length ? list[0].id : "";
    }

    function currentEngine() {
      var id = els.engineSelect.value;
      var list = enginesList();
      for (var i = 0; i < list.length; i += 1) {
        if (list[i].id === id) return list[i];
      }
      return null;
    }

    function selectedMethod() {
      var engine = currentEngine();
      if (!engine) return null;
      var mid = els.methodSelect.value;
      for (var i = 0; i < (engine.methods || []).length; i += 1) {
        if (engine.methods[i].id === mid) return engine.methods[i];
      }
      return null;
    }

    function canLaunch() {
      var method = selectedMethod();
      var runner = (state.engines && state.engines.runner) || [];
      return !!method && method.implemented === true && runner.length > 0;
    }

    function syncEngineOptions() {
      var prev = els.engineSelect.value;
      var list = enginesList();
      els.engineSelect.innerHTML = "";
      if (!list.length) {
        els.engineSelect.disabled = true;
        els.engineSelect.appendChild(option("", "— no engines —"));
        syncMethodOptions();
        return;
      }
      els.engineSelect.disabled = false;
      list.forEach(function (e) {
        els.engineSelect.appendChild(option(e.id, e.label + (e.external ? " (external)" : "")));
      });
      var prevStillThere = list.some(function (e) { return e.id === prev; });
      els.engineSelect.value = prevStillThere ? prev : defaultEngineId();
      syncMethodOptions();
    }

    function syncMethodOptions() {
      var engine = currentEngine();
      var prev = els.methodSelect.value;
      els.methodSelect.innerHTML = "";
      if (!engine || !(engine.methods || []).length) {
        els.methodSelect.disabled = true;
        els.methodSelect.appendChild(option("", "— no recipes —"));
        return;
      }
      els.methodSelect.disabled = false;
      engine.methods.forEach(function (m) {
        var o = option(m.id, m.label + (m.implemented ? "" : " — not implemented"));
        o.disabled = !m.implemented;
        els.methodSelect.appendChild(o);
      });
      var prevValid = engine.methods.some(function (m) { return m.id === prev && m.implemented; });
      if (prevValid) {
        els.methodSelect.value = prev;
      } else {
        var first = engine.methods.filter(function (m) { return m.implemented; })[0];
        els.methodSelect.value = first ? first.id : "";
      }
    }

    function enginesKey() {
      return enginesList().map(function (e) {
        return e.id + ":" + (e.methods || []).map(function (m) {
          return m.id + ":" + (m.implemented ? 1 : 0);
        }).join(",");
      }).join("|");
    }

    // ------------------------------------------------------------- launch
    function fieldText(input) {
      var t = String(input.value || "").trim();
      return t === "" ? undefined : t;
    }

    function fieldNumber(input) {
      var t = String(input.value || "").trim();
      if (t === "") return undefined;
      var n = Number(t);
      return Number.isFinite(n) ? n : t;
    }

    function startLaunch() {
      var method = selectedMethod();
      var engine = currentEngine();
      if (!engine || !method) {
        setMessage("error", "pick a recipe first");
        renderMessage();
        return;
      }
      if (!method.implemented) {
        setMessage("error", "recipe \"" + method.label + "\" is not implemented yet");
        renderMessage();
        return;
      }
      var modelDir = fieldText(els.modelInput);
      var corpus = fieldText(els.corpusInput);
      if (!modelDir || !corpus) {
        setMessage("error", "model dir and corpus are both required");
        renderMessage();
        return;
      }

      var confirmText = "Start a \"" + method.label + "\" training run on " +
        engine.label + "?\n\nThis launches a live training job: a new process is " +
        "spawned and may use the GPU. Proceed?";
      if (!confirmFn(confirmText)) return;

      var fields = {
        model_dir: modelDir,
        corpus: corpus,
        steps: fieldNumber(els.stepsInput),
        seq: fieldNumber(els.seqInput),
        lr: fieldText(els.lrInput),
        device: fieldText(els.deviceInput),
        seed: fieldNumber(els.seedInput),
        eval_regions: fieldNumber(els.regionsInput),
        eval_windows: fieldNumber(els.windowsInput),
      };
      Object.keys(fields).forEach(function (k) {
        if (fields[k] === undefined) delete fields[k];
      });

      var body = {
        engine: engine.id,
        method: method.id,
        name: String(els.nameInput.value || "").trim(),
        fields: fields,
      };

      state.pending = "launch";
      setMessage(null, null);
      syncPendingUI();

      apiPost("/training/launch", body).then(function (r) {
        state.run = {
          run_dir: r.run_dir,
          pid: r.pid,
          engine: engine.id,
          method: method.id,
          name: body.name,
        };
        state.status = null;
        state.report = null;
        delete state.errors.status;
        delete state.errors.report;
        setMessage("ok", "training started: " + r.run_dir + " (pid " + r.pid + ")");
        startPolling();
        refreshRuns();
      }).catch(function (e) {
        setMessage("error", "launch failed: " + e.message);
      }).then(function () {
        state.pending = "";
        syncPendingUI();
        renderAll();
      });
    }

    // -------------------------------------------------------------- polling
    function isTerminal(s) {
      return s === "finished" || s === "stopped";
    }

    function startPolling() {
      stopPolling();
      pollErrors = 0;
      pollTick();
    }

    function stopPolling() {
      if (pollTimer) {
        clearTimeout(pollTimer);
        pollTimer = null;
      }
    }

    function scheduleNextPoll() {
      if (!state.run) return;
      stopPolling();
      pollTimer = setTimeout(pollTick, POLL_MS);
    }

    function pollTick() {
      if (!state.run || pollInFlight) return;
      pollInFlight = true;

      apiGet("/training/status/" + encodeURIComponent(state.run.run_dir)).then(function (s) {
        state.status = s;
        delete state.errors.status;
        pollErrors = 0;
        state.lastUpdated = timeNow();
        renderAll();
        if (isTerminal(s.state)) {
          stopPolling();
          onTerminal(s.state);
        } else {
          scheduleNextPoll();
        }
      }).catch(function (e) {
        state.errors.status = e.message;
        pollErrors += 1;
        renderAll();
        if (pollErrors >= MAX_POLL_ERRORS) {
          stopPolling();
          setMessage("error", "status polling gave up after repeated errors: " + e.message);
          renderMessage();
        } else {
          scheduleNextPoll();
        }
      }).then(function () {
        pollInFlight = false;
      });
    }

    function onTerminal(terminalState) {
      if (terminalState === "finished") {
        fetchReport();
        refreshRuns();
      } else if (terminalState === "stopped") {
        setMessage("error", "run stopped before finishing");
        renderMessage();
      }
    }

    function fetchReport() {
      if (!state.run) return;
      apiPost("/training/report/" + encodeURIComponent(state.run.run_dir), null).then(function (r) {
        state.report = r;
        delete state.errors.report;
      }).catch(function (e) {
        state.errors.report = e.message;
      }).then(function () {
        renderAll();
      });
    }

    // -------------------------------------------------------- preset trainer
    function mineEvidence() {
      state.pending = "mine";
      setMessage(null, null);
      syncPendingUI();
      els.trainerStatus.textContent = "mining session logs…";

      apiGet("/trainer/evidence").then(function (r) {
        state.trainerSummary = (r && r.summary) || {};
        state.trainerEvidence = (r && Array.isArray(r.evidence)) ? r.evidence : [];
        delete state.errors.trainer;
        els.trainerStatus.textContent = "Mined " + state.trainerEvidence.length +
          " evidence item(s) from " + (state.trainerSummary.sessions || 0) + " session(s).";
        setMessage("ok", "mined " + state.trainerEvidence.length + " evidence item(s)");
      }).catch(function (e) {
        state.errors.trainer = e.message;
        els.trainerStatus.textContent = "evidence mining failed: " + e.message;
        setMessage("error", "evidence mining failed: " + e.message);
      }).then(function () {
        state.pending = "";
        syncPendingUI();
        renderTrainer();
      });
    }

    function draftCandidate() {
      var baseline = String(els.baselineInput.value || "").trim();
      if (!baseline) {
        setMessage("error", "set a baseline preset path first");
        renderMessage();
        return;
      }
      state.pending = "draft";
      setMessage(null, null);
      syncPendingUI();
      els.trainerStatus.textContent = "drafting candidate…";

      apiPost("/trainer/draft", { baseline: baseline }).then(function (r) {
        state.trainerDraft = r;
        state.trainerLogText = JSON.stringify(r, null, 2);
        delete state.errors.trainer;
        els.trainerStatus.textContent = "Drafted " + (r.name || "candidate") +
          " — evaluate it with POST /api/v1/trainer/evaluate, then promote from the candidate directory.";
        setMessage("ok", "drafted candidate " + (r.name || ""));
      }).catch(function (e) {
        state.errors.trainer = e.message;
        els.trainerStatus.textContent = "draft failed: " + e.message;
        setMessage("error", "draft failed: " + e.message);
      }).then(function () {
        state.pending = "";
        syncPendingUI();
        renderTrainer();
      });
    }

    // --------------------------------------------------------------- render
    function renderAll() {
      renderToolbar();
      renderMessage();
      renderForm();
      renderStatus();
      renderLog();
      renderRuns();
      renderTrainer();
      renderButtons();
    }

    function renderToolbar() {
      var errs = [];
      if (state.errors.engines) errs.push("engines: " + state.errors.engines);
      if (state.errors.runs) errs.push("runs: " + state.errors.runs);

      var online = !state.errors.engines && !state.errors.runs && !!state.engines;
      els.dot.className = "dot " + (online ? "ok" : "bad");

      if (state.pending) {
        els.status.textContent = "pending: " + state.pending + " …";
        els.status.className = "trp-status trp-pending";
      } else if (errs.length) {
        els.status.textContent = "partial: " + errs.join("; ");
        els.status.className = "trp-status trp-warn";
      } else if (!state.engines) {
        els.status.textContent = "loading…";
        els.status.className = "trp-status";
      } else {
        els.status.textContent = "Hive reachable";
        els.status.className = "trp-status trp-ok";
      }
      els.updated.textContent = state.lastUpdated ? "updated " + state.lastUpdated : "";
    }

    function renderMessage() {
      var box = els.message;
      if (!state.message) { box.hidden = true; return; }
      box.hidden = false;
      box.className = "trp-message " + (state.message.kind === "error" ? "error" : "ok");
      box.textContent = state.message.text;
    }

    function renderForm() {
      var key = enginesKey();
      if (key !== lastEnginesKey) {
        lastEnginesKey = key;
        syncEngineOptions();
      }

      if (state.errors.engines) {
        els.engineMeta.textContent = "unavailable";
      } else {
        var list = enginesList();
        var runner = (state.engines && state.engines.runner) || [];
        els.engineMeta.textContent = list.length + " engine(s)" +
          (runner.length ? " · runner ready" : " · no runner resolved");
      }

      if (state.run) {
        els.runMeta.hidden = false;
        els.runMeta.textContent = "run " + state.run.run_dir + " · pid " + state.run.pid +
          (state.run.engine ? " · " + state.run.engine + "/" + state.run.method : "");
      } else {
        els.runMeta.hidden = true;
        els.runMeta.textContent = "";
      }
    }

    function renderStatus() {
      var box = els.statusCards;
      var meta = els.statusMeta;

      if (!state.run) {
        meta.textContent = "no active run";
        box.innerHTML = '<div class="trp-empty">No active run — configure a run above and ' +
          'Start training, or browse run history below.</div>';
        return;
      }

      meta.textContent = "polling every 3s" +
        (state.lastUpdated ? " · updated " + state.lastUpdated : "");

      if (!state.status && state.errors.status) {
        box.innerHTML = '<div class="trp-empty">status unavailable: ' +
          esc(state.errors.status) + "</div>";
        return;
      }

      var s = state.status || {};
      var p = s.progress || {};
      var st = s.state || "launched";
      var tone = st === "finished" ? "good"
        : st === "stopped" ? "warn"
          : st === "running" ? "accent" : "muted";

      var stepText = fmtInt(p.step) + " / " + (p.total_steps != null ? fmtInt(p.total_steps) : "?");

      var cards = [
        ["state", st, tone],
        ["step / total", stepText, ""],
        ["loss", fmtLoss(p.loss), ""],
        ["deployed ratio", fmtRatio(p.deployed_ratio), ""],
        ["init ratio", fmtRatio(p.init_ratio), ""],
        ["elapsed", fmtElapsed(p.elapsed_s), ""],
      ];

      var html = "";
      if (st === "finished") {
        html += '<div class="trp-report-link">' +
          '<a href="/view/' + encodeURIComponent(state.run.run_dir) + '">open report →</a>' +
          (state.report && state.report.retention != null
            ? '<span class="muted">retention ' + fmtInt(state.report.retention) + "</span>"
            : "") +
          "</div>";
      }
      html += cards.map(function (c) {
        return '<div class="trp-stat">' +
          '<span class="trp-stat-label">' + esc(c[0]) + "</span>" +
          '<b class="trp-stat-value' + (c[2] ? " " + c[2] : "") + '">' + esc(c[1]) + "</b>" +
          "</div>";
      }).join("");
      box.innerHTML = html;
    }

    function renderLog() {
      if (!state.run) {
        els.log.textContent = "";
        return;
      }
      var log = (state.status && state.status.log) || "";
      els.log.textContent = log || "— waiting for the first log line —";
    }

    function renderRuns() {
      var box = els.runs;
      if (state.errors.runs) {
        box.innerHTML = '<div class="trp-empty">run history unavailable: ' +
          esc(state.errors.runs) + "</div>";
        return;
      }
      var runs = state.runs || [];
      if (!runs.length) {
        box.innerHTML = '<div class="trp-empty">No runs yet.</div>';
        return;
      }
      var rows = runs.map(function (r) {
        var name = esc(r.name || "—");
        var kind = r.kind ? '<span class="trp-tag">' + esc(r.kind) + "</span>" : "";
        var mod = esc(r.modified || "—");
        var link = r.has_report
          ? '<a class="trp-link" href="/view/' + encodeURIComponent(r.name || "") + '">view</a>'
          : '<span class="trp-muted">no report</span>';
        return "<tr><td class=\"trp-strong\">" + name + "</td><td>" + kind +
          '</td><td class="trp-muted">' + mod + "</td><td>" + link + "</td></tr>";
      }).join("");
      box.innerHTML = '<div class="trp-table-wrap"><table class="trp-table">' +
        "<thead><tr><th>run</th><th>kind</th><th>modified</th><th></th></tr></thead>" +
        "<tbody>" + rows + "</tbody></table></div>";
    }

    function renderTrainer() {
      if (!els.trainerStatus.textContent) {
        els.trainerStatus.textContent = "Mine session evidence, then draft a candidate preset from a baseline.";
      }
      if (state.trainerLogText) {
        els.trainerLog.hidden = false;
        els.trainerLog.textContent = state.trainerLogText;
      } else {
        els.trainerLog.hidden = true;
      }
    }

    function renderButtons() {
      var actions = rootEl.querySelectorAll("[data-action]");
      for (var i = 0; i < actions.length; i += 1) {
        var action = actions[i].getAttribute("data-action");
        actions[i].disabled = !!state.pending || (action === "launch" && !canLaunch());
      }
      if (els.refresh) els.refresh.disabled = !!state.pending;
    }

    function syncPendingUI() {
      renderToolbar();
      renderMessage();
      renderButtons();
    }

    // ------------------------------------------------------------ event wire
    function onClick(ev) {
      var target = ev.target;
      if (target && target.getAttribute && target.getAttribute("data-trp-refresh") != null) {
        refresh();
        return;
      }
      var t = target;
      while (t && t !== rootEl && !(t.getAttribute && t.getAttribute("data-action"))) {
        t = t.parentElement;
      }
      if (!t || t === rootEl) return;
      if (state.pending) return;
      var action = t.getAttribute("data-action");
      if (action === "launch") startLaunch();
      else if (action === "mine") mineEvidence();
      else if (action === "draft") draftCandidate();
    }

    // Wire the DOM, then kick off the read-only load. No launch, no polling,
    // no preset-trainer side effects happen here.
    buildDom();
    rootEl.addEventListener("click", onClick);
    els.engineSelect.addEventListener("change", function () {
      syncMethodOptions();
      renderButtons();
    });
    renderAll();
    refresh();

    return {
      refresh: function () { return refresh(); },
      state: state,
    };
  }

  // -------------------------------------------------------------- public API
  global.HiveTrainingPanel = {
    mount: function (el, opts) {
      if (!el) throw new Error("HiveTrainingPanel.mount: element required");
      return createPanel(el, opts);
    },
    version: "1.0.0",
  };
})(typeof window !== "undefined" ? window : globalThis);
