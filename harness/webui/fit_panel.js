/*!
 * Hive memory calculator panel — F1 (Round 1)
 *
 * Self-contained module. QUEEN owns the page wiring; this file only defines
 * window.HiveFitPanel. It does not fetch at import time: network work starts
 * in mount() and in the refresh() it returns.
 *
 *   const panel = window.HiveFitPanel.mount(el, { modelGb, ctx, kv });
 *   await panel.refresh();   // re-fetch
 *   panel.state();           // last normalized data (plain object)
 *
 * APIs:
 *   GET /api/v1/server/fit?model_gb=<gb>&ctx=<tokens>&kv=<type>
 *   GET /api/v1/server/status        ({hardware:{...}} or flat shape)
 *   GET /api/v1/models/local
 */
(function (global) {
  "use strict";

  var KV_TYPES = ["f16", "q8_0", "q4_0"];
  var KV_GB_PER_1K = { f16: 0.14, q8_0: 0.07, q4_0: 0.035 };
  var DEFAULT_KV = "q8_0";
  var DEFAULT_CTX = 8192;
  var MIN_CTX = 0;
  var MAX_CTX = 1048576;

  var IO_WARN_CTX_FALLBACK = 131072;

  // ---------------------------------------------------------------------
  // small value helpers (no DOM, no network)
  // ---------------------------------------------------------------------

  function asObject(value) {
    return value && typeof value === "object" && !Array.isArray(value)
      ? value : null;
  }

  function pick(obj, keys) {
    if (!obj) return undefined;
    for (var i = 0; i < keys.length; i += 1) {
      if (obj[keys[i]] !== undefined && obj[keys[i]] !== null) {
        return obj[keys[i]];
      }
    }
    return undefined;
  }

  function numberOr(value, fallback) {
    if (value === null || value === undefined || value === "") return fallback;
    var n = typeof value === "number" ? value : parseFloat(value);
    return Number.isFinite(n) ? n : fallback;
  }

  function round2(value) {
    return Math.round((numberOr(value, 0) + Number.EPSILON) * 100) / 100;
  }

  function clampCtx(value) {
    var n = Math.round(numberOr(value, DEFAULT_CTX));
    if (n < MIN_CTX) return MIN_CTX;
    if (n > MAX_CTX) return MAX_CTX;
    return n;
  }

  function kvRate(kv) {
    var key = String(kv || "").toLowerCase();
    return KV_GB_PER_1K[key] || KV_GB_PER_1K[DEFAULT_KV];
  }

  function formatGb(value) {
    var n = numberOr(value, null);
    if (n === null) return "—";
    if (n === 0) return "0 GB";
    if (n >= 100) return n.toFixed(0) + " GB";
    if (n >= 10) return n.toFixed(1) + " GB";
    return n.toFixed(2) + " GB";
  }

  function formatCtx(tokens) {
    var n = numberOr(tokens, 0);
    if (n <= 0) return "—";
    var k = n / 1024;
    if (k >= 1024) {
      var m = k / 1024;
      return (m >= 10 ? m.toFixed(0) : m.toFixed(1)) + "M";
    }
    return Math.round(k) + "K";
  }

  function errorMessage(err) {
    var msg = err && err.message ? String(err.message) : String(err || "unknown error");
    return msg.length > 200 ? msg.slice(0, 197) + "…" : msg;
  }

  // ---------------------------------------------------------------------
  // response normalization — tolerant, never throws
  // ---------------------------------------------------------------------

  function normalizeFit(raw) {
    var root = asObject(raw);
    if (!root) return null;
    var metrics = asObject(root.metrics) || root;
    var flags = asObject(root.flags) || {};

    var recognised = [
      "total_gb", "totalGb", "model_gb", "modelGb", "kv_gb", "kvGb",
      "tier1_vram_gb", "tier1VramGb", "tier2_ram_gb", "tier3_disk_gb",
    ];
    var hasAny = recognised.some(function (key) {
      return metrics[key] !== undefined;
    });
    if (!hasAny) return null;

    var modelGb = numberOr(pick(metrics, ["model_gb", "modelGb"]), 0);
    var kvGb = numberOr(pick(metrics, ["kv_gb", "kvGb"]), 0);
    var totalGb = numberOr(pick(metrics, ["total_gb", "totalGb"]), modelGb + kvGb);

    return {
      metrics: {
        totalGb: round2(totalGb),
        modelGb: round2(modelGb),
        kvGb: round2(kvGb),
        vramGb: round2(numberOr(pick(metrics, ["tier1_vram_gb", "tier1VramGb"]), 0)),
        ramGb: round2(numberOr(pick(metrics, ["tier2_ram_gb", "tier2RamGb"]), 0)),
        diskGb: round2(numberOr(pick(metrics, ["tier3_disk_gb", "tier3DiskGb"]), 0)),
      },
      flags: {
        ioWarning: Boolean(pick(flags, ["io_latency_warning", "ioLatencyWarning"])),
        diskFull: Boolean(pick(flags, ["disk_full", "diskFull"])),
        recommendCap: Math.max(
          0,
          Math.round(numberOr(pick(flags, ["recommend_cap", "recommendCap"]), 0))
        ),
      },
      estimate: false,
    };
  }

  function normalizeHardware(raw) {
    var root = asObject(raw);
    if (!root) return null;
    var hw = asObject(root.hardware) || root;

    var vramGb = numberOr(
      pick(hw, ["combined_vram_gb", "combinedVramGb", "vram_gb", "vramGb"]),
      null
    );
    var ramGb = numberOr(
      pick(hw, ["available_ram_gb", "availableRamGb", "total_ram_gb", "totalRamGb"]),
      null
    );
    if (ramGb === null) {
      // flat fallback payloads only carry available_gb
      ramGb = numberOr(pick(hw, ["available_gb", "availableGb"]), null);
    }
    if (vramGb === null && ramGb === null) return null;

    var disk = asObject(hw.disk);
    var repo = disk ? asObject(disk.repo) : null;
    var diskFreeGb = numberOr(
      pick(repo || {}, ["free_gb", "freeGb"]),
      numberOr(pick(hw, ["disk_free_gb", "diskFreeGb"]), null)
    );
    var diskTotalGb = numberOr(
      pick(repo || {}, ["total_gb", "totalGb"]),
      numberOr(pick(hw, ["disk_total_gb", "diskTotalGb"]), null)
    );

    var devices = Array.isArray(hw.devices) ? hw.devices.slice(0, 8).map(function (d) {
      var dev = asObject(d) || {};
      return {
        name: String(pick(dev, ["name", "card"]) || "GPU"),
        memoryGb: numberOr(pick(dev, ["memory_gb", "memoryGb"]), null),
        freeGb: numberOr(pick(dev, ["free_gb", "freeGb"]), null),
      };
    }) : [];

    return {
      vramGb: vramGb,
      ramGb: ramGb,
      diskFreeGb: diskFreeGb,
      diskTotalGb: diskTotalGb,
      vramSource: pick(hw, ["vram_source", "vramSource"]) || null,
      devices: devices,
    };
  }

  function normalizeModels(raw) {
    var root = asObject(raw);
    var list = Array.isArray(root) ? root : (root && Array.isArray(root.models) ? root.models : []);
    return list.map(function (item) {
      var m = asObject(item) || {};
      var sizeGb = numberOr(pick(m, ["size_gb", "sizeGb"]), 0);
      var file = String(pick(m, ["file", "name"]) || "");
      return {
        file: file,
        name: String(pick(m, ["name", "file"]) || "model"),
        sizeGb: round2(sizeGb),
        contextLength: Math.round(numberOr(pick(m, ["context_length", "contextLength"]), 0)) || null,
      };
    }).filter(function (m) {
      return Boolean(m.file || m.name);
    });
  }

  function capacityOf(state) {
    var hw = state.hardware;
    if (!hw) return null;
    var vram = numberOr(hw.vramGb, 0);
    var ram = numberOr(hw.ramGb, 0);
    var total = vram + ram;
    if (total <= 0) return null;
    return {
      vramGb: round2(vram),
      ramGb: round2(ram),
      totalGb: round2(total),
      diskFreeGb: hw.diskFreeGb,
    };
  }

  function currentPlan(state) {
    if (state.fit) return state.fit;
    var rate = kvRate(state.kv);
    var kvGb = (state.ctx / 1024) * rate;
    var totalGb = state.modelGb + kvGb;
    var cap = capacityOf(state);
    var ramGb = cap ? cap.ramGb : 0;
    var recommendCap = rate > 0 && ramGb > 0 ? Math.floor((ramGb * 1024) / rate) : 0;
    return {
      metrics: {
        totalGb: round2(totalGb),
        modelGb: round2(state.modelGb),
        kvGb: round2(kvGb),
        vramGb: 0,
        ramGb: 0,
        diskGb: 0,
      },
      flags: {
        ioWarning: false,
        diskFull: false,
        recommendCap: recommendCap,
      },
      // local estimate: tier split is unknown without the fit API
      estimate: true,
    };
  }

  function cloneFit(fit) {
    if (!fit) return null;
    return {
      metrics: {
        totalGb: fit.metrics.totalGb,
        modelGb: fit.metrics.modelGb,
        kvGb: fit.metrics.kvGb,
        vramGb: fit.metrics.vramGb,
        ramGb: fit.metrics.ramGb,
        diskGb: fit.metrics.diskGb,
      },
      flags: {
        ioWarning: fit.flags.ioWarning,
        diskFull: fit.flags.diskFull,
        recommendCap: fit.flags.recommendCap,
      },
      estimate: Boolean(fit.estimate),
    };
  }

  function snapshotOf(state) {
    var plan = currentPlan(state);
    var cap = capacityOf(state);
    return {
      status: state.status,
      error: state.error,
      updatedAt: state.updatedAt,
      modelFile: state.modelFile,
      modelGb: state.modelGb,
      ctx: state.ctx,
      kv: state.kv,
      fit: cloneFit(state.fit),
      hardware: state.hardware ? {
        vramGb: state.hardware.vramGb,
        ramGb: state.hardware.ramGb,
        diskFreeGb: state.hardware.diskFreeGb,
        diskTotalGb: state.hardware.diskTotalGb,
        vramSource: state.hardware.vramSource,
        devices: state.hardware.devices.map(function (d) { return { name: d.name, memoryGb: d.memoryGb, freeGb: d.freeGb }; }),
      } : null,
      models: state.models.map(function (m) {
        return { file: m.file, name: m.name, sizeGb: m.sizeGb, contextLength: m.contextLength };
      }),
      capacityGb: cap ? cap.totalGb : null,
      largestContext: plan.flags.recommendCap,
      estimate: Boolean(plan.estimate),
    };
  }

  // ---------------------------------------------------------------------
  // network
  // ---------------------------------------------------------------------

  async function getJSON(fetcher, url) {
    var res = await withTimeout(
      Promise.resolve(fetcher(url, { cache: "no-store", headers: { accept: "application/json" } })),
      20000,
      url
    );
    if (!res || typeof res !== "object") {
      throw new Error("empty response from " + url);
    }
    if (res.ok === false) {
      throw new Error((res.status || "HTTP") + " " + url);
    }
    if (typeof res.json !== "function") {
      throw new Error("invalid JSON response from " + url);
    }
    return res.json();
  }

  function withTimeout(promise, ms, label) {
    return new Promise(function (resolve, reject) {
      var timer = setTimeout(function () {
        reject(new Error("timed out after " + Math.round(ms / 1000) + "s: " + label));
      }, ms);
      promise.then(function (value) {
        clearTimeout(timer);
        resolve(value);
      }, function (err) {
        clearTimeout(timer);
        reject(err);
      });
    });
  }

  // ---------------------------------------------------------------------
  // mount
  // ---------------------------------------------------------------------

  function mount(el, opts) {
    var options = asObject(opts) || {};
    var state = {
      status: "idle",
      error: null,
      updatedAt: null,
      modelFile: typeof (options.modelFile || options.model) === "string"
        ? String(options.modelFile || options.model) : "",
      modelGb: Math.max(0, numberOr(options.modelGb, 0)),
      ctx: clampCtx(numberOr(options.ctx, DEFAULT_CTX)),
      kv: KV_TYPES.indexOf(String(options.kv || "").toLowerCase()) >= 0
        ? String(options.kv).toLowerCase() : DEFAULT_KV,
      fit: null,
      fitKey: null,
      hardware: null,
      models: [],
    };

    var host = el;
    if (typeof host === "string" && typeof document !== "undefined") {
      host = document.querySelector(host);
    }
    if (!host || typeof host.appendChild !== "function") {
      state.status = "error";
      state.error = "mount target not found";
      return {
        refresh: function () { return Promise.resolve(snapshotOf(state)); },
        state: function () { return snapshotOf(state); },
      };
    }

    var doc = host.ownerDocument || (typeof document !== "undefined" ? document : null);
    var fetcher = typeof options.fetch === "function"
      ? options.fetch
      : (typeof global.fetch === "function" ? global.fetch.bind(global) : null);
    var base = String(options.apiBase || "").replace(/\/+$/, "");

    // ---- DOM helpers (scoped to this mount) --------------------------
    function h(tag, cls, text) {
      var node = doc.createElement(tag);
      if (cls) node.className = cls;
      if (text !== undefined && text !== null) node.textContent = String(text);
      return node;
    }

    function field(labelText, control) {
      var label = h("label", "hf-field");
      label.appendChild(h("span", "hf-field-label", labelText));
      label.appendChild(control);
      return label;
    }

    function stat(label, value, hint, tone) {
      var card = h("div", "hf-stat" + (tone ? " hf-stat--" + tone : ""));
      card.appendChild(h("div", "hf-stat-label", label));
      card.appendChild(h("div", "hf-stat-value", value));
      card.appendChild(h("div", "hf-stat-hint", hint));
      return card;
    }

    function flagItem(tone, title, detail) {
      var row = h("div", "hf-flag hf-flag--" + tone);
      var icon = tone === "bad" || tone === "warn" ? "!" : (tone === "good" ? "✓" : "…");
      row.appendChild(h("span", "hf-flag-icon", icon));
      var body = h("div", "hf-flag-body");
      body.appendChild(h("strong", "hf-flag-title", title));
      body.appendChild(h("span", "hf-flag-detail", detail));
      row.appendChild(body);
      return row;
    }

    // ---- skeleton -----------------------------------------------------
    host.textContent = "";
    if (host.classList && typeof host.classList.add === "function") {
      host.classList.add("hive-fit");
    }

    var root = h("div", "hf");
    root.setAttribute("role", "region");
    root.setAttribute("aria-label", "Memory calculator");

    var head = h("div", "hf-head");
    var titles = h("div", "hf-titles");
    titles.appendChild(h("h3", "hf-title", "Memory calculator"));
    titles.appendChild(h("p", "hf-sub", "Where a model and its KV cache fit: VRAM, RAM, disk."));
    var actions = h("div", "hf-actions");
    var pill = h("span", "hf-pill hf-pill--loading", "loading");
    var refreshBtn = h("button", "hf-btn hf-btn--ghost", "Refresh");
    refreshBtn.type = "button";
    actions.appendChild(pill);
    actions.appendChild(refreshBtn);
    head.appendChild(titles);
    head.appendChild(actions);

    var banner = h("div", "hf-banner hf-banner--bad");
    banner.hidden = true;
    banner.setAttribute("role", "status");
    banner.setAttribute("aria-live", "polite");
    var bannerTitle = h("strong", "hf-banner-title", "Fit API unavailable");
    var bannerDetail = h("span", "hf-banner-detail", "Showing a local estimate.");
    banner.appendChild(bannerTitle);
    banner.appendChild(bannerDetail);

    var modelSelect = h("select", "hf-select");
    modelSelect.setAttribute("aria-label", "Local model");
    modelSelect.title = "GET /api/v1/models/local";
    var ctxRange = h("input", "hf-range");
    ctxRange.type = "range";
    ctxRange.min = "0";
    ctxRange.max = String(MAX_CTX);
    ctxRange.step = "512";
    ctxRange.setAttribute("aria-label", "Context tokens");
    var ctxNumber = h("input", "hf-number");
    ctxNumber.type = "number";
    ctxNumber.min = "0";
    ctxNumber.max = String(MAX_CTX);
    ctxNumber.step = "512";
    ctxNumber.inputMode = "numeric";
    ctxNumber.setAttribute("aria-label", "Context tokens, exact");
    var ctxWrap = h("div", "hf-ctx");
    ctxWrap.appendChild(ctxRange);
    ctxWrap.appendChild(ctxNumber);
    ctxWrap.appendChild(h("span", "hf-unit", "tokens"));

    var kvSelect = h("select", "hf-select hf-select--kv");
    kvSelect.setAttribute("aria-label", "KV cache type");
    kvSelect.title = "KV size scales with the cache type";
    KV_TYPES.forEach(function (kv) {
      var option = h("option", null, kv);
      option.value = kv;
      kvSelect.appendChild(option);
    });

    var controls = h("div", "hf-controls");
    controls.appendChild(field("Model", modelSelect));
    controls.appendChild(field("Context", ctxWrap));
    controls.appendChild(field("KV cache", kvSelect));

    var summary = h("div", "hf-summary");

    var util = h("div", "hf-util");
    var utilText = h("div", "hf-util-text");
    var utilTrack = h("div", "hf-util-track");
    var utilFill = h("i", "hf-util-fill is-good");
    utilTrack.appendChild(utilFill);
    util.appendChild(utilText);
    util.appendChild(utilTrack);

    var capStrip = h("div", "hf-cap");
    var capLabel = h("span", "hf-cap-label", "Largest context that fits");
    var capValue = h("strong", "hf-cap-value", "—");
    var capHint = h("span", "hf-cap-hint", "");
    capStrip.appendChild(capLabel);
    capStrip.appendChild(capValue);
    capStrip.appendChild(capHint);

    var bar = h("div", "hf-bar");
    bar.setAttribute("role", "img");
    bar.setAttribute("aria-label", "Placement across VRAM, RAM and disk");

    var plan = h("div", "hf-plan");
    var flags = h("div", "hf-flags");
    flags.setAttribute("role", "status");
    flags.setAttribute("aria-live", "polite");
    var footer = h("div", "hf-footer");

    root.appendChild(head);
    root.appendChild(banner);
    root.appendChild(controls);
    root.appendChild(summary);
    root.appendChild(util);
    root.appendChild(capStrip);
    root.appendChild(bar);
    root.appendChild(plan);
    root.appendChild(flags);
    root.appendChild(footer);
    host.appendChild(root);

    // ---- fetching -----------------------------------------------------
    var inflight = null;
    var dirty = false;
    var lastModelsKey = null;
    var ctxTimer = null;

    function fitKey() {
      return [round2(state.modelGb), state.ctx, state.kv].join("|");
    }

    function fitUrl() {
      return base + "/api/v1/server/fit?model_gb=" + encodeURIComponent(round2(state.modelGb)) +
        "&ctx=" + encodeURIComponent(state.ctx) +
        "&kv=" + encodeURIComponent(state.kv);
    }

    function refresh() {
      // Joining an in-flight request is intentional: callers can await the
      // current fetch without queueing another one. Parameter changes go
      // through reload(), which queues a follow-up.
      if (inflight) return inflight;
      inflight = runRefresh().then(function (snap) {
        return snap;
      }, function () {
        // runRefresh handles its own failures; keep the API promise safe
        return snapshotOf(state);
      }).then(function (snap) {
        inflight = null;
        if (dirty) {
          dirty = false;
          refresh();
        }
        return snap;
      });
      return inflight;
    }

    function reload() {
      if (inflight) {
        dirty = true;
        return inflight;
      }
      return refresh();
    }

    async function runRefresh() {
      state.status = "loading";
      state.error = null;
      render();

      if (!fetcher) {
        state.fit = null;
        state.fitKey = null;
        state.error = "fetch is not available in this environment";
        state.status = "error";
        state.updatedAt = new Date().toISOString();
        render();
        return snapshotOf(state);
      }

      try {
        var early = await Promise.allSettled([
          getJSON(fetcher, base + "/api/v1/server/status"),
          getJSON(fetcher, base + "/api/v1/models/local"),
        ]);

        if (early[0].status === "fulfilled") {
          var hardware = normalizeHardware(early[0].value);
          if (hardware) state.hardware = hardware;
        }
        if (early[1].status === "fulfilled") {
          var models = normalizeModels(early[1].value);
          state.models = models;
        }

        if (!state.modelFile && state.modelGb <= 0 && state.models.length > 0) {
          state.modelFile = state.models[0].file;
          state.modelGb = state.models[0].sizeGb;
        }

        var requestedKey = fitKey();
        var rawFit = await getJSON(fetcher, fitUrl());
        var fit = normalizeFit(rawFit);
        if (!fit) throw new Error("unrecognized fit payload");

        if (requestedKey === fitKey()) {
          state.fit = fit;
          state.fitKey = requestedKey;
          state.status = "ready";
        } else {
          // context/model changed while this request was in flight
          state.fit = null;
          state.fitKey = null;
          dirty = true;
        }
      } catch (err) {
        state.fit = null;
        state.fitKey = null;
        state.error = errorMessage(err);
        state.status = "error";
      }

      state.updatedAt = new Date().toISOString();
      render();
      return snapshotOf(state);
    }

    function scheduleRefresh() {
      if (ctxTimer) clearTimeout(ctxTimer);
      ctxTimer = setTimeout(function () {
        ctxTimer = null;
        reload();
      }, 200);
    }

    // ---- rendering ----------------------------------------------------
    function syncControls() {
      var active = doc && doc.activeElement;
      if (active !== ctxNumber) ctxNumber.value = String(state.ctx);
      ctxRange.value = String(state.ctx);
      kvSelect.value = state.kv;
      modelSelect.value = state.modelFile;
    }

    function renderModelOptions() {
      var key = state.models.map(function (m) {
        return m.file + ":" + m.sizeGb;
      }).join("|") + "|" + state.modelFile;
      if (key === lastModelsKey) return;
      lastModelsKey = key;

      modelSelect.textContent = "";
      var placeholder = h("option", null,
        state.models.length ? "— choose a local model —" : "— no local models —");
      placeholder.value = "";
      modelSelect.appendChild(placeholder);
      state.models.forEach(function (m) {
        var option = h("option", null, m.name + " · " + formatGb(m.sizeGb));
        option.value = m.file;
        modelSelect.appendChild(option);
      });
      modelSelect.value = state.modelFile;
      modelSelect.disabled = state.models.length === 0;
    }

    function renderPill() {
      // a ready panel without a fresh fit is estimating while it waits
      var effective = state.status === "ready" && !state.fit ? "loading" : state.status;
      var tone = effective === "ready" ? "ok"
        : effective === "error" ? "bad"
          : effective === "loading" ? "loading" : "idle";
      var text = effective === "ready" ? "live"
        : effective === "error" ? "estimate"
          : effective === "loading" ? "refreshing" : "idle";
      pill.className = "hf-pill hf-pill--" + tone;
      pill.textContent = text;
      refreshBtn.disabled = effective === "loading";
      refreshBtn.textContent = effective === "loading" ? "Refreshing…" : "Refresh";
    }

    function renderSummary(planData, cap) {
      var m = planData.metrics;
      var totalTone = "";
      if (cap) {
        var ratio = m.totalGb / cap.totalGb;
        totalTone = ratio > 1 ? "bad" : ratio > 0.85 ? "warn" : "good";
      }
      summary.textContent = "";
      summary.appendChild(stat(
        "Model size",
        formatGb(m.modelGb),
        state.modelFile || "no local model selected",
        ""
      ));
      summary.appendChild(stat(
        "KV cache",
        formatGb(m.kvGb),
        state.kv + " · " + kvRate(state.kv).toFixed(3) + " GB / 1K tokens",
        ""
      ));
      summary.appendChild(stat(
        "Total need",
        formatGb(m.totalGb),
        "weights + KV cache",
        totalTone
      ));
      summary.appendChild(stat(
        "Available",
        cap ? formatGb(cap.totalGb) : "—",
        cap
          ? "VRAM " + formatGb(cap.vramGb) + " · RAM " + formatGb(cap.ramGb)
          : "hardware unknown",
        ""
      ));
    }

    function renderUtil(planData, cap) {
      var total = planData.metrics.totalGb;
      if (!cap) {
        utilText.textContent = "Total need " + formatGb(total) + " — host capacity unknown";
        utilFill.style.width = "0%";
        utilFill.className = "hf-util-fill is-good";
        return;
      }
      var pct = cap.totalGb > 0 ? (total / cap.totalGb) * 100 : 0;
      utilText.textContent = "Total need " + formatGb(total) + " of " +
        formatGb(cap.totalGb) + " in VRAM+RAM (" + Math.round(pct) + "%)";
      utilFill.style.width = Math.min(100, pct).toFixed(1) + "%";
      utilFill.className = "hf-util-fill " +
        (pct > 100 ? "is-bad" : pct > 85 ? "is-warn" : "is-good");
    }

    function renderCap(planData) {
      var capTokens = planData.flags.recommendCap;
      capValue.textContent = formatCtx(capTokens);
      capHint.textContent = planData.estimate
        ? "estimate from RAM only — the fit API has not confirmed it"
        : "KV cache stays in RAM up to this context";
    }

    function renderBar(planData) {
      bar.textContent = "";
      var m = planData.metrics;
      var segments = [
        ["VRAM", m.vramGb, "hf-seg--vram"],
        ["RAM", m.ramGb, "hf-seg--ram"],
        ["Disk", m.diskGb, "hf-seg--disk"],
      ];
      var sum = segments.reduce(function (acc, seg) {
        return acc + Math.max(0, numberOr(seg[1], 0));
      }, 0);

      if (planData.estimate) {
        bar.appendChild(h("span", "hf-bar-empty", "VRAM / RAM / disk split needs the fit API"));
        bar.setAttribute("aria-label", "Memory split unavailable; local estimate only");
        return;
      }
      if (sum <= 0) {
        bar.appendChild(h("span", "hf-bar-empty", "nothing to place"));
        bar.setAttribute("aria-label", "Plan is empty");
        return;
      }

      segments.forEach(function (seg) {
        var value = Math.max(0, numberOr(seg[1], 0));
        if (value <= 0) return;
        var piece = h("span", "hf-seg " + seg[2]);
        piece.style.width = ((value / sum) * 100).toFixed(2) + "%";
        piece.setAttribute("title", seg[0] + ": " + formatGb(value));
        bar.appendChild(piece);
      });
      bar.setAttribute("aria-label",
        "Placement: VRAM " + formatGb(m.vramGb) +
        ", RAM " + formatGb(m.ramGb) +
        ", disk " + formatGb(m.diskGb));
    }

    function renderPlan(planData) {
      plan.textContent = "";
      if (planData.estimate) {
        plan.appendChild(h("div", "hf-plan-note",
          "Tier split unavailable — showing a local estimate from model size and KV constants."));
        return;
      }
      [
        ["VRAM", planData.metrics.vramGb, "hf-dot--vram", "weights that fit in VRAM"],
        ["RAM", planData.metrics.ramGb, "hf-dot--ram", "weights + KV that fit in RAM"],
        ["Disk", planData.metrics.diskGb, "hf-dot--disk", "spills to disk (slower)"],
      ].forEach(function (row) {
        var line = h("div", "hf-plan-row");
        var name = h("div", "hf-plan-name");
        name.appendChild(h("span", "hf-dot " + row[2]));
        var text = h("span", null, row[0]);
        name.appendChild(text);
        name.appendChild(h("span", "hf-plan-desc", row[3]));
        line.appendChild(name);
        line.appendChild(h("span", "hf-plan-gb", formatGb(row[1])));
        plan.appendChild(line);
      });
    }

    function renderFlags(planData, cap) {
      flags.textContent = "";
      var items = [];
      var m = planData.metrics;
      var f = planData.flags;

      if (state.error) {
        items.push([
          "bad",
          "Fit API unavailable",
          state.error + " — the numbers below are a local estimate.",
        ]);
      }
      if (planData.estimate && !state.error) {
        items.push(["muted", "Waiting for the fit API", "Tier split and warnings are not confirmed yet."]);
      }
      if (!planData.estimate) {
        if (f.ioWarning) {
          items.push([
            "warn",
            "Disk will be hit",
            "Part of the plan streams from disk" +
              (state.ctx > IO_WARN_CTX_FALLBACK
                ? " (context is above 128K, so prompt reads slow down)"
                : " (KV cache spills out of RAM)") + ".",
          ]);
        }
        if (f.diskFull) {
          items.push([
            "bad",
            "Disk nearly full",
            "The spill needs " + formatGb(m.diskGb) + " but only " +
              (cap && typeof cap.diskFreeGb === "number"
                ? formatGb(cap.diskFreeGb)
                : "an unknown amount") +
              " is free on the repo disk.",
          ]);
        }
        if (m.diskGb > 0 && !f.ioWarning && !f.diskFull) {
          items.push([
            "warn",
            "Weights spill to disk",
            formatGb(m.diskGb) + " of weights do not fit in VRAM+RAM.",
          ]);
        }
        if (cap && m.totalGb > cap.totalGb && m.diskGb <= 0) {
          items.push([
            "warn",
            "Over VRAM+RAM",
            formatGb(m.totalGb - cap.totalGb) + " more than the host can hold in memory.",
          ]);
        }
        if (f.recommendCap > 0 && state.ctx > f.recommendCap) {
          items.push([
            "warn",
            "Context above the RAM ceiling",
            formatCtx(state.ctx) + " requested; KV fits in RAM up to " +
              formatCtx(f.recommendCap) + ".",
          ]);
        }
      }
      if (items.length === 0) {
        items.push(["good", "Fits cleanly", "Weights and KV cache fit in VRAM and RAM."]);
      }

      items.forEach(function (item) {
        flags.appendChild(flagItem(item[0], item[1], item[2]));
      });
    }

    function renderFooter(planData) {
      footer.textContent = "";
      footer.appendChild(h("span", "hf-foot-item",
        "Fits up to " + formatCtx(planData.flags.recommendCap)));
      footer.appendChild(h("span", "hf-foot-sep", "·"));
      footer.appendChild(h("span", "hf-foot-item",
        state.updatedAt ? "Updated " + state.updatedAt.slice(11, 19) + " UTC" : "Not fetched yet"));
      footer.appendChild(h("span", "hf-foot-sep", "·"));
      footer.appendChild(h("span", "hf-foot-item",
        state.error ? "local estimate"
          : planData.estimate ? "estimating…"
            : "GET /api/v1/server/fit"));
    }

    function render() {
      renderModelOptions();
      syncControls();
      renderPill();

      banner.hidden = !state.error;
      if (state.error) {
        bannerDetail.textContent = state.error + " — showing a local estimate.";
      }

      var planData = currentPlan(state);
      var cap = capacityOf(state);
      renderSummary(planData, cap);
      renderUtil(planData, cap);
      renderCap(planData);
      renderBar(planData);
      renderPlan(planData);
      renderFlags(planData, cap);
      renderFooter(planData);
    }

    // ---- events -------------------------------------------------------
    refreshBtn.addEventListener("click", function () {
      refresh();
    });

    modelSelect.addEventListener("change", function (ev) {
      var file = ev && ev.target ? String(ev.target.value) : "";
      var entry = state.models.filter(function (m) { return m.file === file; })[0] || null;
      state.modelFile = entry ? entry.file : "";
      state.modelGb = entry ? entry.sizeGb : 0;
      state.fit = null;
      state.fitKey = null;
      reload();
    });

    kvSelect.addEventListener("change", function (ev) {
      var next = ev && ev.target ? String(ev.target.value).toLowerCase() : DEFAULT_KV;
      state.kv = KV_TYPES.indexOf(next) >= 0 ? next : DEFAULT_KV;
      state.fit = null;
      state.fitKey = null;
      reload();
    });

    function applyCtx(raw, immediate) {
      state.ctx = clampCtx(raw);
      state.fit = null;
      state.fitKey = null;
      syncControls();
      if (immediate) {
        reload();
      } else {
        render();
        scheduleRefresh();
      }
    }

    ctxRange.addEventListener("input", function (ev) {
      applyCtx(ev && ev.target ? ev.target.value : state.ctx, false);
    });
    ctxRange.addEventListener("change", function (ev) {
      applyCtx(ev && ev.target ? ev.target.value : state.ctx, true);
    });
    ctxNumber.addEventListener("change", function (ev) {
      applyCtx(ev && ev.target ? ev.target.value : state.ctx, true);
    });

    // ---- go -----------------------------------------------------------
    render();
    refresh();

    return {
      refresh: refresh,
      state: function () {
        return snapshotOf(state);
      },
    };
  }

  global.HiveFitPanel = {
    mount: mount,
    version: "1.0.0",
    kvTypes: KV_TYPES.slice(),
    estimateKvGb: function (ctx, kv) {
      return round2((Math.max(0, numberOr(ctx, 0)) / 1024) * kvRate(kv));
    },
  };
})(typeof window !== "undefined" ? window : globalThis);
