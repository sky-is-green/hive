/*
 * HiveLibraryPanel — model folder + Hugging Face finder (new shell).
 *
 * One module, no frameworks:
 *   window.HiveLibraryPanel.mount(el) -> { refresh, state }
 * APIs (proxied by dev_server to live Hive /v1):
 *   GET    /api/v1/models/local
 *   POST   /api/v1/models/local/path          {folder}
 *   POST   /api/v1/models/local/import-path   {folder}
 *   DELETE /api/v1/models/local?file=...
 *   GET    /api/v1/models/hub?q=&limit=
 *   GET    /api/v1/models/hub/files/{repo}
 *   POST   /api/v1/models/hub/download        {repo,file}
 *   GET    /api/v1/models/hub/downloads
 */
(function (global) {
  "use strict";
  var BASE = "/api/v1";
  var HUB_LIMIT = 12;

  function esc(value) {
    return String(value == null ? "" : value)
      .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;").replaceAll('"', "&quot;");
  }
  function fmtGb(value) {
    var n = Number(value);
    return isFinite(n) ? n.toFixed(1) + " GB" : "—";
  }
  function errText(err) { return String(err && err.message ? err.message : err); }

  function createPanel(root, opts) {
    opts = opts || {};
    var base = String(opts.base || BASE).replace(/\/+$/, "");
    var fetchImpl = opts.fetch || (typeof global.fetch === "function" ? global.fetch.bind(global) : null);
    var confirmFn = opts.confirm || (typeof global.confirm === "function" ? global.confirm.bind(global) : function () { return false; });
    var state = {
      models: [], modelsDir: "", downloads: [], hub: [], files: {},
      busy: "", message: null, poll: null
    };
    var el = {};

    function h(tag, cls, text) {
      var node = document.createElement(tag);
      if (cls) node.className = cls;
      if (text != null) node.textContent = String(text);
      return node;
    }
    function api(path, init) {
      if (!fetchImpl) return Promise.reject(new Error("fetch unavailable"));
      return fetchImpl(base + path, init || {}).then(function (res) {
        return res.json().catch(function () { return {}; }).then(function (data) {
          if (!res.ok) throw new Error(data.detail || data.error || (res.status + " " + path));
          return data;
        });
      });
    }
    function setMessage(text, bad) {
      state.message = text ? { text: text, bad: !!bad } : null;
      renderMessage();
    }

    function build() {
      root.innerHTML = "";
      var wrap = h("div", "hive-lib");
      var head = h("div", "hl-head");
      head.innerHTML = '<div><h2 class="hl-title">Model library</h2>' +
        '<p class="hl-sub">One folder for local GGUFs and Hugging Face downloads. ' +
        'Selectors read this folder.</p></div>';
      var pathBox = h("div", "hl-pathbox");
      pathBox.innerHTML = '<label class="hl-label">Library folder</label>' +
        '<div class="hl-pathrow"><input class="hl-input hl-path" spellcheck="false" placeholder="/path/to/models">' +
        '<button class="hl-btn hl-use" type="button">Use folder</button>' +
        '<button class="hl-btn" type="button" data-hl="link">Link folder</button>' +
        '<button class="hl-btn" type="button" data-hl="refresh">Refresh</button></div>' +
        '<p class="hl-note">“Use folder” switches Hive’s library. “Link folder” keeps the files where they are and registers the folder.</p>';
      var msg = h("div", "hl-message");
      var modelsPanel = h("div", "hl-panel");
      modelsPanel.innerHTML = '<div class="hl-panel-head"><h3>Local models</h3><span class="hl-count"></span></div>' +
        '<div class="hl-table-wrap"><table class="hl-table"><thead><tr>' +
        '<th>model</th><th>file</th><th>size</th><th>quant</th><th>arch</th><th></th>' +
        '</tr></thead><tbody class="hl-models"></tbody></table></div>';
      var hubPanel = h("div", "hl-panel");
      hubPanel.innerHTML = '<div class="hl-panel-head"><h3>Hugging Face finder</h3><span class="hl-note">GGUF repos</span></div>' +
        '<div class="hl-searchrow"><input class="hl-input hl-query" placeholder="search GGUF repos…" spellcheck="false">' +
        '<button class="hl-btn hl-primary" type="button" data-hl="search">Search</button></div>' +
        '<div class="hl-hub-results"></div>' +
        '<div class="hl-downloads"><h4>Downloads</h4><div class="hl-download-list"></div></div>';
      var sources = [
        ["LFM3 collection", "https://huggingface.co/collections/LiquidAI/lfm3", "text, VL, audio, omni, d1"],
        ["LFM3-1.7B", "https://huggingface.co/LiquidAI/LFM3-1.7B", "text · 256k · local agent"],
        ["LFM3-1.7B-Base", "https://huggingface.co/LiquidAI/LFM3-1.7B-Base", "fine-tune base"],
        ["LFM3-VL-2B", "https://huggingface.co/LiquidAI/LFM3-VL-2B", "vision / video"],
        ["LFM3-Audio-2B", "https://huggingface.co/LiquidAI/LFM3-Audio-2B", "speech in/out"],
        ["LFM3-Omni-2B", "https://huggingface.co/LiquidAI/LFM3-Omni-2B", "text + vision + audio"],
        ["d1-2B", "https://huggingface.co/LiquidAI/d1-2B", "calibrated decisions"],
        ["d1-omni-2B", "https://huggingface.co/LiquidAI/d1-omni-2B", "decisions + audio"],
        ["LFM3 fine-tuning cookbook", "https://github.com/Liquid4All/cookbook/tree/main/finetuning/notebooks", "SFT · DPO · GRPO · CPT"],
        ["Ternary Bonsai 27B", "https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf", "Prism ternary 27B"],
        ["Qwen3.8-35B-A3B Distill", "https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill-GGUF", "Qwen 35B GGUF"],
        ["Scion-35B-A3B", "https://huggingface.co/SkyIsNotGreen/Scion-35B-A3B", "our 27B+ controller"],
        ["Scion-FlashNext-176B-A6B", "https://huggingface.co/SkyIsNotGreen/Scion-FlashNext-176B-A6B", "our heavy planner"],
      ];
      var sourcesPanel = h("div", "hl-panel hl-sources");
      sourcesPanel.innerHTML = '<div class="hl-panel-head"><h3>Upstream sources</h3>' +
        '<span class="hl-note">model cards and fine-tuning recipes</span></div>' +
        '<div class="hl-source-list">' + sources.map(function (row) {
          return '<a class="hl-source" href="' + esc(row[1]) + '" target="_blank" rel="noreferrer">' +
            '<b>' + esc(row[0]) + '</b><span>' + esc(row[2]) + '</span></a>';
        }).join("") + '</div>';
      wrap.append(head, pathBox, msg, modelsPanel, hubPanel, sourcesPanel);
      root.appendChild(wrap);
      el = {
        path: wrap.querySelector(".hl-path"),
        message: msg,
        modelsBody: wrap.querySelector(".hl-models"),
        count: wrap.querySelector(".hl-count"),
        query: wrap.querySelector(".hl-query"),
        hub: wrap.querySelector(".hl-hub-results"),
        downloads: wrap.querySelector(".hl-download-list")
      };
      wrap.addEventListener("click", onClick);
      el.query.addEventListener("keydown", function (ev) {
        if (ev.key === "Enter") { ev.preventDefault(); searchHub(); }
      });
      renderAll();
    }

    function renderMessage() {
      if (!el.message) return;
      el.message.hidden = !state.message;
      el.message.className = "hl-message" + (state.message && state.message.bad ? " bad" : "");
      el.message.textContent = state.message ? state.message.text : "";
    }
    function renderModels() {
      el.count.textContent = state.models.length + " model(s)";
      el.modelsBody.innerHTML = state.models.map(function (m) {
        return '<tr><td class="hl-strong">' + esc(m.name || m.file) + '</td>' +
          '<td class="hl-mono">' + esc(m.file || "") + '</td>' +
          '<td class="hl-mono">' + fmtGb(m.size_gb) + '</td>' +
          '<td>' + esc(m.quantization || "—") + '</td>' +
          '<td>' + esc(m.architecture || "—") + '</td>' +
          '<td><button class="hl-del" type="button" data-file="' + esc(m.file || "") + '">remove</button></td></tr>';
      }).join("") || '<tr><td colspan="6" class="hl-empty">No models in this folder yet.</td></tr>';
    }
    function renderDownloads() {
      var jobs = state.downloads || [];
      el.downloads.innerHTML = jobs.length ? jobs.map(function (j) {
        var label = (j.repo || "") + "/" + (j.filename || j.file || "");
        var pct = Number(j.progress || j.percent || 0);
        return '<div class="hl-download"><span class="hl-state ' + esc(j.state || "?") + '">' +
          esc(j.state || "?") + '</span><span class="hl-mono">' + esc(label) + '</span>' +
          (j.error ? '<span class="hl-err">' + esc(j.error) + '</span>' : '') +
          (pct > 0 ? '<span class="hl-pct">' + Math.round(pct) + '%</span>' : '') + '</div>';
      }).join("") : '<p class="hl-empty">No downloads yet.</p>';
    }
    function renderHub() {
      el.hub.innerHTML = (state.hub || []).map(function (r) {
        var key = "hl-files-" + String(r.repo || "").replace(/[^A-Za-z0-9]/g, "_");
        return '<div class="hl-repo"><div class="hl-repo-head"><b>' + esc(r.repo) + '</b>' +
          '<span class="hl-note">↓' + esc(r.downloads || 0) + ' · ★' + esc(r.likes || 0) + '</span></div>' +
          (r.pipeline_tag ? '<div class="hl-note">' + esc(r.pipeline_tag) + '</div>' : '') +
          '<div class="hl-files" id="' + key + '"></div>' +
          '<button class="hl-btn" type="button" data-repo="' + esc(r.repo) + '">Show files</button></div>';
      }).join("") || (state.hub.length ? "" : '<p class="hl-empty">Search Hugging Face for GGUF models.</p>');
      Object.keys(state.files).forEach(function (repo) {
        var target = root.querySelector("#hl-files-" + repo.replace(/[^A-Za-z0-9]/g, "_"));
        if (!target) return;
        var files = state.files[repo] || [];
        target.innerHTML = files.length ? files.map(function (f) {
          return '<div class="hl-file"><code>' + esc(f.file) + '</code><span class="hl-note">' + fmtGb(f.size_gb) + '</span>' +
            '<button class="hl-btn hl-primary" type="button" data-repo="' + esc(repo) + '" data-file="' + esc(f.file) + '">Download</button></div>';
        }).join("") : '<p class="hl-empty">No .gguf files in this repo.</p>';
      });
    }
    function renderAll() {
      if (el.path) el.path.value = state.modelsDir || "";
      renderMessage(); renderModels(); renderDownloads(); renderHub();
    }

    function refresh() {
      state.busy = "refresh"; renderAll();
      return Promise.all([
        api("/models/local"),
        api("/models/hub/downloads").catch(function () { return { downloads: [] }; })
      ]).then(function (res) {
        state.modelsDir = res[0].models_dir || "";
        state.models = Array.isArray(res[0].models) ? res[0].models : [];
        state.downloads = res[1].downloads || res[1].jobs || [];
        state.busy = ""; renderAll();
        if (state.downloads.some(function (j) { return j.state === "queued" || j.state === "downloading"; })) startPoll();
      }).catch(function (err) {
        state.busy = ""; setMessage("Library unavailable: " + errText(err), true); renderAll();
      });
    }
    function startPoll() {
      if (state.poll) return;
      state.poll = setTimeout(function () {
        state.poll = null;
        api("/models/hub/downloads").then(function (d) {
          state.downloads = d.downloads || d.jobs || [];
          renderDownloads();
          if (state.downloads.some(function (j) { return j.state === "queued" || j.state === "downloading"; })) startPoll();
          else refresh();
        }).catch(function () {});
      }, 1800);
    }
    function setPath(link) {
      var folder = (el.path.value || "").trim();
      if (!folder) { setMessage("Enter a folder path.", true); return; }
      state.busy = link ? "link" : "use"; renderAll();
      api(link ? "/models/local/import-path" : "/models/local/path", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ folder: folder })
      }).then(function (d) {
        state.modelsDir = d.models_dir || folder;
        setMessage((link ? "Linked " : "Using ") + state.modelsDir);
        return refresh();
      }).catch(function (err) { state.busy = ""; setMessage("Folder change failed: " + errText(err), true); renderAll(); });
    }
    function removeModel(file) {
      if (!file || !confirmFn('Remove "' + file + '" from the library? The GGUF file is deleted.')) return;
      api("/models/local?file=" + encodeURIComponent(file), { method: "DELETE" })
        .then(function () { setMessage("Removed " + file); return refresh(); })
        .catch(function (err) { setMessage("Remove failed: " + errText(err), true); });
    }
    function searchHub() {
      var q = (el.query.value || "").trim();
      if (q.length < 2) { setMessage("Type at least two characters.", true); return; }
      state.busy = "search"; renderAll();
      api("/models/hub?q=" + encodeURIComponent(q) + "&limit=" + HUB_LIMIT)
        .then(function (d) { state.hub = d.results || []; state.busy = ""; renderAll(); })
        .catch(function (err) { state.busy = ""; setMessage("HF search failed: " + errText(err), true); renderAll(); });
    }
    function showFiles(repo) {
      if (!repo) return;
      api("/models/hub/files/" + encodeURIComponent(repo))
        .then(function (d) { state.files[repo] = d.files || []; renderHub(); })
        .catch(function (err) { setMessage("File list failed: " + errText(err), true); });
    }
    function download(repo, file) {
      if (!repo || !file) return;
      setMessage("Queued " + file + " — downloading in the background.");
      api("/models/hub/download", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ repo: repo, file: file })
      }).then(function () { return refresh(); })
        .catch(function (err) { setMessage("Download failed: " + errText(err), true); });
    }
    function onClick(ev) {
      var target = ev.target;
      if (!target || !target.getAttribute) return;
      if (target.classList.contains("hl-use")) { setPath(false); return; }
      if (target.classList.contains("hl-del")) { removeModel(target.getAttribute("data-file")); return; }
      var action = target.getAttribute("data-hl");
      if (action === "link") setPath(true);
      else if (action === "refresh") refresh();
      else if (action === "search") searchHub();
      else if (target.getAttribute("data-repo") && target.getAttribute("data-file")) {
        download(target.getAttribute("data-repo"), target.getAttribute("data-file"));
      } else if (target.getAttribute("data-repo")) {
        showFiles(target.getAttribute("data-repo"));
      }
    }

    build();
    refresh();
    return { refresh: refresh, state: state };
  }

  global.HiveLibraryPanel = {
    mount: function (el, opts) {
      if (!el) throw new Error("HiveLibraryPanel.mount: element required");
      return createPanel(el, opts);
    },
    version: "1.0.0"
  };
})(typeof window !== "undefined" ? window : globalThis);
