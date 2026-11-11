/*
 * HiveInspectorPanel — the prompt/curation inspector for Splinter conversations.
 *
 * The legacy Studio's Inspector called renderInspection()/fetchInspection(),
 * neither of which exists in the tree. This panel uses the working backend
 * contract instead:
 *
 *   GET /api/v1/splinter/state                  active conversations
 *   GET /api/v1/splinter/inspect/{conversation} last turn's full curation detail
 *
 * Read-only. No turn is created here; the conversation must already exist.
 *
 * window.HiveInspectorPanel.mount(el, opts?) -> { refresh, state }
 *   opts.base      : API prefix (default "/api/v1")
 *   opts.fetchJSON : injectable for tests
 */
(function (global) {
  "use strict";

  var BASE = "/api/v1";

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

  function panel(title, meta) {
    var el = node("div", "panel");
    var head = node("div", "panel-head");
    head.append(node("h2", "", title), node("span", "muted", meta || ""));
    el.appendChild(head);
    return el;
  }

  function emptyRow(cols, message) {
    var tr = document.createElement("tr");
    var td = document.createElement("td");
    td.colSpan = cols;
    td.className = "note";
    td.textContent = message;
    tr.appendChild(td);
    return tr;
  }

  function createPanel(rootEl, opts) {
    opts = opts || {};
    var base = opts.base || BASE;
    var fetchJSON = opts.fetchJSON || defaultFetchJSON;
    var state = {
      status: "idle",
      conversations: {},
      selected: "",
      inspection: null,
      error: "",
      notice: "",
    };
    var ui = {};

    function build() {
      rootEl.innerHTML = "";
      var root = node("section", "hive-inspector");

      var controls = node("div", "panel");
      var controlsHead = node("div", "panel-head");
      controlsHead.append(node("h2", "", "Conversation"), node("span", "muted", "active Splinter turns"));
      var controlsBody = node("div", "engine-actions");
      controlsBody.style.padding = "12px 15px";
      ui.select = document.createElement("select");
      ui.select.style.minWidth = "320px";
      ui.select.addEventListener("change", function () {
        loadInspection(ui.select.value).then(render);
      });
      ui.refresh = node("button", "ghost", "Refresh");
      ui.refresh.type = "button";
      ui.refresh.addEventListener("click", refresh);
      ui.message = node("span", "muted", "");
      controlsBody.append(ui.select, ui.refresh, ui.message);
      controls.append(controlsHead, controlsBody);

      ui.cards = node("div", "cards");

      var grid = node("div", "grid two");
      var selectedPanel = panel("Selected chunks", "in the context");
      ui.selected = document.createElement("table");
      var selectedWrap = node("div", "table-wrap");
      selectedWrap.appendChild(ui.selected);
      selectedPanel.appendChild(selectedWrap);

      var droppedPanel = panel("Dropped chunks", "top 20");
      ui.dropped = document.createElement("table");
      var droppedWrap = node("div", "table-wrap");
      droppedWrap.appendChild(ui.dropped);
      droppedPanel.appendChild(droppedWrap);
      grid.append(selectedPanel, droppedPanel);

      var assembledPanel = panel("Assembled context", "preview");
      ui.assembled = document.createElement("pre");
      ui.assembled.style.margin = "0";
      ui.assembled.style.padding = "12px 15px";
      ui.assembled.style.background = "#0d0f12";
      ui.assembled.style.color = "#c9d1d9";
      ui.assembled.style.maxHeight = "260px";
      ui.assembled.style.overflow = "auto";
      ui.assembled.style.fontSize = "11.5px";
      ui.assembled.style.whiteSpace = "pre-wrap";
      assembledPanel.appendChild(ui.assembled);

      var timingsPanel = panel("Stage timings", "last turn");
      ui.timings = document.createElement("table");
      var timingsWrap = node("div", "table-wrap");
      timingsWrap.appendChild(ui.timings);
      timingsPanel.appendChild(timingsWrap);

      root.append(controls, ui.cards, grid, assembledPanel, timingsPanel);
      rootEl.appendChild(root);
    }

    function renderSelect() {
      var ids = Object.keys(state.conversations || {});
      var keep = state.selected;
      ui.select.replaceChildren();
      if (!ids.length) {
        var none = document.createElement("option");
        none.value = "";
        none.textContent = "No active Splinter conversations";
        ui.select.appendChild(none);
        ui.select.disabled = true;
      } else {
        ui.select.disabled = false;
        ids.sort().forEach(function (id) {
          var info = state.conversations[id] || {};
          var option = document.createElement("option");
          option.value = id;
          option.textContent = id + " · turn " + (info.turn ?? "?") +
            " · " + (info.store_chunks ?? "?") + " chunks";
          ui.select.appendChild(option);
        });
        if (ids.indexOf(keep) >= 0) ui.select.value = keep;
        else {
          state.selected = ids[0];
          ui.select.value = state.selected;
        }
      }
    }

    function renderMessage() {
      if (state.error) {
        ui.message.textContent = state.error;
        ui.message.style.color = "var(--bad)";
      } else if (state.notice) {
        ui.message.textContent = state.notice;
        ui.message.style.color = "var(--muted)";
      } else if (state.status === "loading") {
        ui.message.textContent = "Reading conversation…";
        ui.message.style.color = "var(--muted)";
      } else {
        ui.message.textContent = "";
      }
    }

    function renderSummary() {
      ui.cards.replaceChildren();
      var insp = state.inspection;
      if (!insp) return;
      var budget = insp.budget || {};
      var util = Number(budget.utilization);
      var utilText = isFinite(util) ? (util * 100).toFixed(1) + "%" : "—";
      ui.cards.append(
        card("Mode", String(insp.mode || "—"), "last turn"),
        card("Turn", String(insp.turn ?? "—"), "conversation"),
        card("Route", String((insp.routing || {}).route_to || "—"), "drone decision"),
        card("Drift", insp.drift_detected ? "detected" : "clear",
          "topic drift", insp.drift_detected ? "bad" : "ok"),
        card("Budget", (budget.used ?? "—") + " / " + (budget.total ?? "—"),
          utilText + " used", isFinite(util) && util > 0.95 ? "bad" : ""),
        card("Top score", String(insp.top_raw_score ?? "—"), "raw retrieval"),
        card("Store", String(insp.store_chunks ?? "—"), "chunks"),
        card("Dropped", String(insp.dropped_count ?? "—"), "candidates not selected")
      );
      if (insp.error) {
        ui.message.textContent = insp.error;
        ui.message.style.color = "var(--bad)";
      }
    }

    function renderChunkTable(table, chunks, emptyText) {
      table.replaceChildren();
      var thead = document.createElement("thead");
      var hr = document.createElement("tr");
      ["ID", "Score", "Preview"].forEach(function (label) {
        var th = document.createElement("th");
        th.textContent = label;
        hr.appendChild(th);
      });
      thead.appendChild(hr);
      table.appendChild(thead);
      var tbody = document.createElement("tbody");
      if (!Array.isArray(chunks) || !chunks.length) {
        tbody.appendChild(emptyRow(3, emptyText));
      } else {
        chunks.forEach(function (chunk) {
          var tr = document.createElement("tr");
          var id = document.createElement("td");
          id.className = "mono";
          id.textContent = String(chunk.id ?? "—");
          var score = document.createElement("td");
          score.className = "mono";
          score.textContent = String(chunk.raw_score ?? "—");
          var preview = document.createElement("td");
          preview.textContent = String(chunk.preview ?? "");
          tr.append(id, score, preview);
          tbody.appendChild(tr);
        });
      }
      table.appendChild(tbody);
    }

    function renderTimings() {
      ui.timings.replaceChildren();
      var timings = (state.inspection && state.inspection.timings) || {};
      var keys = Object.keys(timings).sort();
      var thead = document.createElement("thead");
      var hr = document.createElement("tr");
      ["Stage", "ms"].forEach(function (label) {
        var th = document.createElement("th");
        th.textContent = label;
        hr.appendChild(th);
      });
      thead.appendChild(hr);
      ui.timings.appendChild(thead);
      var tbody = document.createElement("tbody");
      if (!keys.length) {
        tbody.appendChild(emptyRow(2, "No timings recorded for this turn."));
      } else {
        keys.forEach(function (key) {
          var tr = document.createElement("tr");
          var stage = document.createElement("td");
          stage.textContent = key;
          var value = document.createElement("td");
          value.className = "mono";
          value.textContent = String(timings[key]);
          tr.append(stage, value);
          tbody.appendChild(tr);
        });
      }
      ui.timings.appendChild(tbody);
    }

    function render() {
      renderSelect();
      renderMessage();
      renderSummary();
      var insp = state.inspection;
      renderChunkTable(ui.selected, insp && insp.selected_chunks,
        "No selected chunks for this turn.");
      renderChunkTable(ui.dropped, insp && insp.dropped_chunks,
        "No dropped chunks for this turn.");
      ui.assembled.textContent = insp && insp.assembled_preview
        ? insp.assembled_preview : "No assembled context recorded.";
      renderTimings();
    }

    async function loadInspection(id) {
      state.selected = String(id || "");
      state.inspection = null;
      state.error = "";
      if (!state.selected) {
        state.notice = "No conversation selected.";
        return state;
      }
      state.status = "loading";
      state.notice = "";
      render();
      try {
        state.inspection = await fetchJSON(
          base + "/splinter/inspect/" + encodeURIComponent(state.selected));
        state.notice = "Last turn: " + (state.inspection.turn ?? "?") + ".";
      } catch (err) {
        state.error = "Inspector unavailable: " +
          (err && err.message ? err.message : err);
      }
      state.status = "ready";
      return state;
    }

    async function refresh() {
      state.status = "loading";
      state.error = "";
      render();
      try {
        var data = await fetchJSON(base + "/splinter/state");
        state.conversations = data && data.conversations ? data.conversations : {};
        var ids = Object.keys(state.conversations);
        if (!ids.length) {
          state.selected = "";
          state.inspection = null;
          state.notice = "No active Splinter conversations.";
        } else if (!state.selected || ids.indexOf(state.selected) < 0) {
          state.selected = ids[0];
          await loadInspection(state.selected);
        } else {
          await loadInspection(state.selected);
        }
        state.status = "ready";
      } catch (err) {
        state.status = "error";
        state.error = "Conversation list unavailable: " +
          (err && err.message ? err.message : err);
      }
      render();
      return state;
    }

    build();
    render();
    refresh();
    return { refresh: refresh, state: state };
  }

  global.HiveInspectorPanel = {
    mount: function (el, opts) {
      if (!el) throw new Error("HiveInspectorPanel.mount: element required");
      return createPanel(el, opts);
    },
  };
})(typeof window !== "undefined" ? window : globalThis);
