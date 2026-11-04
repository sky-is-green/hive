/*
 * HiveRunsPanel — read-only run history and report summaries.
 *
 * Importing this file only defines the public API. Network requests begin in
 * mount() (initial list load) or when refresh()/a run selection is triggered.
 *
 *   const panel = window.HiveRunsPanel.mount(el);
 *   await panel.refresh();
 *   panel.state; // current list, selection, report, and errors
 */
(function (global) {
  "use strict";

  var DEFAULT_BASE = "/api/v1";

  function esc(value) {
    return String(value == null ? "" : value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function asObject(value) {
    return value && typeof value === "object" && !Array.isArray(value) ? value : null;
  }

  function displayValue(value) {
    if (value == null || value === "") return "—";
    if (typeof value === "object") {
      try { return JSON.stringify(value); } catch (_) { return String(value); }
    }
    return String(value);
  }

  function requestJSON(fetcher, url) {
    if (!fetcher) return Promise.reject(new Error("fetch is not available in this environment"));
    return Promise.resolve(fetcher(url, {
      method: "GET",
      cache: "no-store",
      headers: { accept: "application/json" },
    })).then(function (res) {
      if (!res || typeof res !== "object") throw new Error("empty response from " + url);
      if (res.ok === false) {
        return Promise.resolve(typeof res.text === "function" ? res.text() : "").then(function (text) {
          var message = "";
          try {
            var body = text ? JSON.parse(text) : null;
            message = body && (body.detail || body.error || body.message);
          } catch (_) { /* use HTTP status below */ }
          throw new Error(message || ((res.status || "HTTP") + " " + (res.statusText || "request failed")));
        });
      }
      if (typeof res.json === "function") return res.json();
      if (typeof res.text === "function") {
        return res.text().then(function (text) { return text ? JSON.parse(text) : null; });
      }
      throw new Error("invalid response from " + url);
    });
  }

  function reportMetrics(report) {
    var sources = [
      ["metrics", asObject(report.metrics)],
      ["summary", asObject(report.summary)],
      ["aggregate", asObject(report.aggregate)],
    ];
    var seen = Object.create(null);
    var entries = [];
    sources.forEach(function (source) {
      if (!source[1]) return;
      Object.keys(source[1]).forEach(function (key) {
        var value = source[1][key];
        if (value == null || typeof value === "object" || seen[key]) return;
        seen[key] = true;
        entries.push([key, value]);
      });
    });
    return entries;
  }

  function reportTables(report) {
    var tables = [];
    if (Array.isArray(report.tables)) {
      report.tables.forEach(function (table, index) {
        var obj = asObject(table);
        if (obj && Array.isArray(obj.rows)) {
          tables.push({ title: obj.title || obj.name || ("Table " + (index + 1)), rows: obj.rows });
        } else if (Array.isArray(table)) {
          tables.push({ title: "Table " + (index + 1), rows: table });
        }
      });
    } else if (asObject(report.tables)) {
      Object.keys(report.tables).forEach(function (name) {
        var rows = report.tables[name];
        if (Array.isArray(rows)) tables.push({ title: name, rows: rows });
      });
    }
    ["regions", "results", "verdicts", "checks"].forEach(function (key) {
      if (Array.isArray(report[key]) && report[key].length) {
        tables.push({ title: key, rows: report[key] });
      }
    });
    return tables;
  }

  function renderTable(table) {
    var rows = table.rows.slice(0, 100);
    var objectRows = rows.map(asObject).filter(Boolean);
    if (!objectRows.length) {
      return '<section class="hrp-report-table"><h4>' + esc(table.title) + '</h4>' +
        '<pre>' + esc(rows.map(displayValue).join("\n")) + '</pre></section>';
    }
    var columns = [];
    objectRows.forEach(function (row) {
      Object.keys(row).forEach(function (key) {
        if (columns.indexOf(key) === -1 && columns.length < 12) columns.push(key);
      });
    });
    if (!columns.length) return "";
    var body = objectRows.map(function (row) {
      return "<tr>" + columns.map(function (key) {
        return "<td>" + esc(displayValue(row[key])) + "</td>";
      }).join("") + "</tr>";
    }).join("");
    var head = columns.map(function (key) { return "<th>" + esc(key) + "</th>"; }).join("");
    var more = table.rows.length > rows.length
      ? '<p class="hrp-muted">Showing ' + rows.length + " of " + table.rows.length + " rows.</p>" : "";
    return '<section class="hrp-report-table"><h4>' + esc(table.title) + '</h4>' +
      '<div class="hrp-table-wrap"><table><thead><tr>' + head + '</tr></thead><tbody>' +
      body + '</tbody></table></div>' + more + '</section>';
  }

  function mount(el, opts) {
    if (!el) throw new Error("HiveRunsPanel.mount: element required");
    opts = opts || {};
    var base = String(opts.base == null ? DEFAULT_BASE : opts.base).replace(/\/+$/, "");
    var fetcher = typeof opts.fetch === "function"
      ? opts.fetch
      : (typeof global.fetch === "function" ? global.fetch.bind(global) : null);
    var state = {
      runs: [],
      selectedRun: null,
      report: null,
      loadingRuns: false,
      loadingReport: false,
      errors: {},
      lastUpdated: null,
    };
    var requestId = 0;

    el.innerHTML =
      '<div class="hrp">' +
      '  <div class="hrp-toolbar">' +
      '    <div><h2>Runs &amp; reports</h2><span class="hrp-muted" data-hrp-updated></span></div>' +
      '    <button type="button" class="hrp-refresh" data-hrp-refresh>Refresh</button>' +
      '  </div>' +
      '  <div class="hrp-content">' +
      '    <section class="hrp-runs" aria-label="Runs"><div data-hrp-runs></div></section>' +
      '    <section class="hrp-detail" aria-live="polite"><div data-hrp-detail>' +
      '      <div class="hrp-empty">Select a run to inspect its report summary.</div>' +
      '    </div></section>' +
      '  </div>' +
      '</div>';
    var runsEl = el.querySelector("[data-hrp-runs]");
    var detailEl = el.querySelector("[data-hrp-detail]");
    var updatedEl = el.querySelector("[data-hrp-updated]");
    var refreshEl = el.querySelector("[data-hrp-refresh]");

    function renderRuns() {
      if (state.errors.runs) {
        runsEl.innerHTML = '<div class="hrp-empty hrp-error">Runs unavailable: ' + esc(state.errors.runs) + "</div>";
        return;
      }
      if (!state.runs.length) {
        runsEl.innerHTML = '<div class="hrp-empty">No runs found.</div>';
        return;
      }
      runsEl.innerHTML = '<ul class="hrp-run-list">' + state.runs.map(function (run) {
        var name = run.name || run.run_dir || "Unnamed run";
        var selected = state.selectedRun && state.selectedRun.name === name;
        return '<li><button type="button" class="hrp-run' + (selected ? " is-selected" : "") +
          '" data-hrp-run="' + esc(name) + '" aria-pressed="' + (selected ? "true" : "false") + '">' +
          '<span class="hrp-run-name">' + esc(name) + '</span>' +
          '<span class="hrp-run-meta"><span class="hrp-kind">' + esc(run.kind || "unknown") + '</span>' +
          '<span>' + esc(run.modified || "—") + '</span></span>' +
          '<span class="hrp-presence ' + (run.has_report ? "has-report" : "no-report") + '">' +
          (run.has_report ? "Report available" : "No report") + '</span></button></li>';
      }).join("") + "</ul>";
    }

    function renderDetail() {
      if (state.errors.report) {
        detailEl.innerHTML = '<div class="hrp-empty hrp-error">Report unavailable: ' + esc(state.errors.report) + "</div>";
        return;
      }
      if (state.loadingReport) {
        detailEl.innerHTML = '<div class="hrp-empty">Loading report…</div>';
        return;
      }
      if (!state.selectedRun) {
        detailEl.innerHTML = '<div class="hrp-empty">Select a run to inspect its report summary.</div>';
        return;
      }
      if (!state.report) {
        detailEl.innerHTML = '<div class="hrp-empty">No report data for this run.</div>';
        return;
      }
      var report = asObject(state.report) || {};
      var runName = state.selectedRun.name || state.selectedRun.run_dir || "";
      var path = encodeURIComponent(String(state.selectedRun.run_dir || runName));
      var metrics = reportMetrics(report);
      var tables = reportTables(report);
      var html = '<header class="hrp-report-head"><div><h3>' + esc(runName) + '</h3>' +
        '<p><span class="hrp-kind">' + esc(report.kind || state.selectedRun.kind || "unknown") + '</span>' +
        '<span class="hrp-status">' + esc(report.status || "status unavailable") + '</span></p></div>' +
        '<a class="hrp-report-link" href="/view/' + path + '">Open full HTML report ↗</a></header>';
      if (metrics.length) {
        html += '<section class="hrp-metrics" aria-label="Report metrics">' + metrics.map(function (entry) {
          return '<div class="hrp-metric"><span>' + esc(entry[0].replaceAll("_", " ")) +
            '</span><strong>' + esc(displayValue(entry[1])) + '</strong></div>';
        }).join("") + '</section>';
      }
      if (tables.length) html += tables.map(renderTable).join("");
      if (!metrics.length && !tables.length) {
        html += '<div class="hrp-empty">Report loaded; no summary metrics or tables were present.</div>';
      }
      detailEl.innerHTML = html;
    }

    function render() {
      renderRuns();
      renderDetail();
      if (updatedEl) updatedEl.textContent = state.lastUpdated ? "Updated " + state.lastUpdated : "";
      if (refreshEl) {
        refreshEl.disabled = state.loadingRuns;
        refreshEl.textContent = state.loadingRuns ? "Refreshing…" : "Refresh";
      }
    }

    function refresh() {
      state.loadingRuns = true;
      delete state.errors.runs;
      render();
      return requestJSON(fetcher, base + "/runs").then(function (payload) {
        var runs = Array.isArray(payload) ? payload : (payload && Array.isArray(payload.runs) ? payload.runs : []);
        state.runs = runs.filter(function (run) { return run && typeof run === "object"; });
        delete state.errors.runs;
        if (state.selectedRun) {
          var selectedName = state.selectedRun.name || state.selectedRun.run_dir;
          state.selectedRun = state.runs.filter(function (run) {
            return (run.name || run.run_dir) === selectedName;
          })[0] || null;
          if (!state.selectedRun) {
            state.report = null;
            delete state.errors.report;
          }
        }
      }).catch(function (error) {
        state.runs = [];
        state.errors.runs = error && error.message ? error.message : String(error);
      }).then(function () {
        state.loadingRuns = false;
        state.lastUpdated = new Date().toLocaleTimeString([], { hour12: false });
        render();
        return state;
      });
    }

    function selectRun(name) {
      var run = state.runs.filter(function (candidate) {
        return (candidate.name || candidate.run_dir) === name;
      })[0];
      if (!run) return Promise.resolve(state);
      state.selectedRun = run;
      state.report = null;
      delete state.errors.report;
      state.loadingReport = true;
      var currentRequest = ++requestId;
      render();
      var runDir = String(run.run_dir || run.name || "");
      return requestJSON(fetcher, base + "/report/" + encodeURIComponent(runDir)).then(function (report) {
        if (currentRequest !== requestId) return state;
        state.report = report;
      }).catch(function (error) {
        if (currentRequest !== requestId) return state;
        state.errors.report = error && error.message ? error.message : String(error);
      }).then(function () {
        if (currentRequest === requestId) {
          state.loadingReport = false;
          render();
        }
        return state;
      });
    }

    el.addEventListener("click", function (event) {
      var target = event.target;
      while (target && target !== el && !(target.getAttribute &&
          (target.getAttribute("data-hrp-refresh") != null || target.getAttribute("data-hrp-run") != null))) {
        target = target.parentElement;
      }
      if (!target || target === el) return;
      if (target.getAttribute("data-hrp-refresh") != null) refresh();
      else selectRun(target.getAttribute("data-hrp-run"));
    });

    render();
    refresh();
    return { refresh: refresh, state: state };
  }

  global.HiveRunsPanel = { mount: mount, version: "1.0.0" };
})(typeof window !== "undefined" ? window : globalThis);
