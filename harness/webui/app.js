const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const state = { page: "overview", status: null, activityEvents: [] };
let activitySource = null;

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
function timeOf(ts) {
  if (!ts) return "—";
  const d = new Date(ts);
  return isNaN(d) ? String(ts).slice(11, 19) : d.toLocaleTimeString([], { hour12: false });
}
function short(value, n = 120) {
  const s = String(value ?? "");
  return esc(s.length > n ? s.slice(0, n) + "…" : s);
}

async function getJSON(path) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status} ${path}`);
  return res.json();
}

async function postJSON(path, body) {
  const res = await fetch(path, {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${path}`);
  return res.json();
}

async function suggestRoute(query) {
  try { return await postJSON("/api/route", { query }); } catch { return null; }
}

async function logTrace(trace) {
  try { await postJSON("/api/routing/traces", trace); } catch { /* tracing is best-effort */ }
}

const MODEL_TIER = {
  lfm3: "simple", lfm3vl: "simple", ds41: "complex",
  scion35: "medium", scion176: "complex",
};
const MODEL_WEIGHTS = {
  lfm3: 2.0, lfm3vl: 1.72, scion35: 10.56, scion176: 69.18,
  guards: 0, "drone-heuristic": 0, ds41: null, "d1-2b": 1.72,
};
const LIB = { models: [], instances: [], ts: 0, error: "" };

async function probe(port) {
  try {
    const r = await fetch(`/probe/${port}/health`, { cache: "no-store" });
    if (!r.ok) return false;
    const d = await r.json();
    return d.status === "ok" || d.loaded === true;
  } catch { return false; }
}

function setPill(el, ok, text) {
  el.className = "pill " + (ok ? "ok" : "bad");
  el.textContent = text;
}

async function loadGlobal() {
  try {
    const st = await getJSON("/api/v1/server/status");
    setPill($("#hive-pill"), true, "Hive reachable");
    $("#rail-dot").className = "dot ok";
    $("#rail-status").textContent = "Hive online";
    return st;
  } catch {
    setPill($("#hive-pill"), false, "Hive unreachable");
    $("#rail-dot").className = "dot bad";
    $("#rail-status").textContent = "Hive offline";
    return null;
  }
}

async function loadOverview() {
  const [statusR, pendingR, activityR, ds41, lfm3] = await Promise.allSettled([
    getJSON("/api/v1/server/status"),
    getJSON("/api/v1/tools/pending"),
    getJSON("/api/v1/activity?limit=8"),
    probe(8150), probe(8155),
  ]);
  const status = statusR.status === "fulfilled" ? statusR.value : null;
  state.status = status;
  const hiveOk = !!status;
  setPill($("#hive-pill"), hiveOk, hiveOk ? "Hive reachable" : "Hive unreachable");
  $("#rail-dot").className = "dot " + (hiveOk ? "ok" : "bad");
  $("#rail-status").textContent = hiveOk ? "Hive online" : "Hive offline";

  const svc = [
    ["Hive", hiveOk, "127.0.0.1:8765"],
    ["DS4.1", ds41.status === "fulfilled" && ds41.value, "127.0.0.1:8150"],
    ["LFM3 1.7B", lfm3.status === "fulfilled" && lfm3.value, "127.0.0.1:8155"],
  ];
  $("#service-cards").innerHTML = svc.map(([name, ok, addr]) => `
    <div class="card ${ok ? "ok" : "bad"}">
      <div class="label">${esc(name)}</div>
      <div class="value">${ok ? "up" : "down"}</div>
      <div class="hint">${esc(addr)}</div>
    </div>`).join("");

  const pending = pendingR.status === "fulfilled" ? (pendingR.value.requests || []) : [];
  $("#pending").innerHTML = pending.length
    ? pending.slice(0, 5).map(r => `
        <div class="approval">
          <div class="mono">${esc(r.tool_identity || "tool")}</div>
          <p>${short(r.action || r.reason || "", 160)}</p>
        </div>`).join("")
    : `<div class="empty-line">No pending approvals.</div>`;

  if (status?.hardware?.devices) {
    $("#gpu-note").textContent = `${status.hardware.devices.length} devices · sysfs`;
    $("#gpu-cards").innerHTML = status.hardware.devices.map(d => {
      const pct = d.memory_gb ? (d.used_gb / d.memory_gb) * 100 : 0;
      return `<div class="gpu">
        <div class="name">${esc(d.card)} · ${esc(d.name || "AMD GPU")}</div>
        <div class="big">${Number(d.used_gb).toFixed(1)} / ${Number(d.memory_gb).toFixed(1)} GB</div>
        <div class="bar"><i style="width:${pct.toFixed(1)}%"></i></div>
        <div class="small"><span>${pct.toFixed(0)}% used</span><span>${Number(d.free_gb).toFixed(1)} GB free</span></div>
      </div>`;
    }).join("");
  } else {
    $("#gpu-cards").innerHTML = `<div class="empty-line">No GPU data.</div>`;
  }

  if (activityR.status === "fulfilled") {
    renderActivityTable($("#recent"), activityR.value.events || [], 8);
  }
}

function renderActivityTable(table, events, limit) {
  table.innerHTML = `
    <thead><tr><th>time</th><th>type</th><th>tool</th><th>detail</th></tr></thead>
    <tbody>${events.slice(0, limit).map(e => {
      const detail = e.text || e.message || e.final || e.event || e.result || "";
      return `<tr>
        <td class="mono">${timeOf(e.ts)}</td>
        <td>${esc(e.type || "—")}</td>
        <td>${esc(e.tool || "—")}</td>
        <td>${short(detail, 150)}</td>
      </tr>`;
    }).join("") || `<tr><td colspan="4" class="muted">No activity.</td></tr>`}</tbody>`;
}

async function loadRisk() {
  const [auditR, pendingR] = await Promise.allSettled([
    getJSON("/api/v1/tools/audit?limit=30"),
    getJSON("/api/v1/tools/pending"),
  ]);
  const pending = pendingR.status === "fulfilled" ? (pendingR.value.requests || []) : [];
  $("#risk-pending-count").textContent = `${pending.length} waiting`;
  $("#risk-pending").innerHTML = "";
  if (pending.length) {
    for (const req of pending) {
      $("#risk-pending").appendChild(approvalCard(req, () => loadRisk()));
    }
  } else {
    $("#risk-pending").innerHTML = `<div class="empty-line">No pending approvals.</div>`;
  }
  const d = auditR.status === "fulfilled" ? auditR.value : {};
  const rows = d.records || [];
  $("#risk-table").innerHTML = `
    <thead><tr><th>time</th><th>event</th><th>tool</th><th>action</th><th>fs</th><th>risk</th><th>decision</th></tr></thead>
    <tbody>${rows.map(r => {
      const tag = r.decision ? `tag ${r.decision}` : "tag";
      return `<tr>
        <td class="mono">${timeOf(r.ts)}</td>
        <td>${esc((r.event || "").replace("tool.", ""))}</td>
        <td class="mono">${esc(r.tool_identity || "—")}</td>
        <td>${short(r.action || "—", 110)}</td>
        <td><span class="tag ${esc(r.filesystem || "")}">${esc(r.filesystem || "—")}</span></td>
        <td>${esc(r.risk || "—")}</td>
        <td><span class="${tag}">${esc(r.decision || "pending")}</span></td>
      </tr>`;
    }).join("") || `<tr><td colspan="7" class="muted">No audit records.</td></tr>`}</tbody>`;
}

function startActivityStream() {
  if (activitySource) return;
  activitySource = new EventSource("/api/v1/activity/stream");
  activitySource.onmessage = event => {
    let row;
    try { row = JSON.parse(event.data); } catch { return; }
    const seq = row.seq;
    state.activityEvents = [row, ...state.activityEvents.filter(e => seq == null || e.seq !== seq)].slice(0, 100);
    renderActivityTable($("#activity-table"), state.activityEvents, 100);
    $("#activity-note").textContent = `live · ${state.activityEvents.length} events`;
  };
  activitySource.onerror = () => {
    $("#activity-note").textContent = "live feed reconnecting…";
  };
}

function stopActivityStream() {
  if (activitySource) { activitySource.close(); activitySource = null; }
}

async function loadActivity() {
  const d = await getJSON("/api/v1/activity?limit=30");
  state.activityEvents = (d.events || []).slice().reverse();
  renderActivityTable($("#activity-table"), state.activityEvents, 30);
  $("#activity-note").textContent = `live · ${state.activityEvents.length} events`;
  startActivityStream();
}

async function loadModels() {
  const d = await getJSON("/api/v1/server/status");
  const models = d.local_models || [];
  $("#model-cards").innerHTML = models.map(m => `
    <div class="model">
      <div class="name">${esc(m.name || m.file)}</div>
      <div class="meta">${Number(m.size_gb || 0).toFixed(1)} GB · ${esc(m.architecture || "—")} · ctx ${esc(m.context_length || "—")}</div>
      <div class="meta mono">${esc(m.file || "")}</div>
    </div>`).join("") || `<div class="muted">No local models reported.</div>`;
}

const MODELS = {
  lfm3: { label: "LFM3 1.7B", base: "/model/lfm3", model: "lfm3", note: "8155 · 1.7B · long-context worker" },
  ds41: { label: "DS4.1", base: "/model/ds41", model: "deepseek-v4.1-flash-gsq", note: "8150 · frontier escalation" },
  lfm3vl: { label: "LFM3-VL-2B", base: "/model/lfm3vl", model: "LFM3-VL-2B-Q8_0", note: "8160 · vision", vision: true },
  scion35: { label: "Scion-35B-A3B", base: "/model/scion35", model: "scion-35b-a3b-pq2_0-corr", note: "8082 · controller" },
  scion176: { label: "Scion-FlashNext-176B", base: "/model/scion176", model: "scion-flashnext-176b-a6b-q2_0", note: "8081 · planner (heavy)" },
  controller17: { label: "Hive Conductor 1.7B", base: "/model/controller17", model: "conductor-1.7b", note: "8156 · controller + planner" },
  reader17: { label: "Hive Reader 1.7B", base: "/model/reader17", model: "reader-1.7b", note: "8159 · mechanics + RAG" },
  auditor17: { label: "Hive Auditor 1.7B", base: "/model/auditor17", model: "auditor-1.7b", note: "8162 · placement + critic" },
  dispatcher17: { label: "Hive Dispatcher 1.7B", base: "/model/dispatcher17", model: "dispatcher-1.7b", note: "8163 · dispatcher" },
  clerk17: { label: "Hive Clerk 1.7B", base: "/model/clerk17", model: "clerk-1.7b", note: "8164 · agency" },
};
const DEFAULT_ROLES = {
  planner: "scion35",
  controller: "controller17",
  router: "scion35",
  face: "scion35",
  agency: "clerk17",
  mechanics: "reader17",
  judge: "d1-2b",
  rag: "reader17",
  risk: "guards",
  vision: "lfm3vl",
  placement: "auditor17",
  critic: "auditor17",
  dispatcher: "dispatcher17",
};
const ROLE_OPTIONS = {
  planner: ["lfm3", "scion35", "scion176"],
  controller: ["controller17", "scion35", "scion176"],
  router: ["scion35", "d1-2b", "drone-heuristic"],
  face: ["scion35", "scion176", "lfm3", "ds41"],
  agency: ["clerk17", "scion35", "scion176", "lfm3", "ds41"],
  mechanics: ["reader17", "lfm3", "scion35", "ds41"],
  judge: ["d1-2b", "ds41"],
  rag: ["reader17", "lfm3", "scion35", "scion176", "ds41"],
  risk: ["guards", "guards+d1-3b", "d1-3b-cli"],
  vision: ["lfm3vl", "scion176", "scion35", "ds41"],
  placement: ["auditor17", "lfm3", "scion35"],
  critic: ["auditor17", "d1-2b", "scion35"],
  dispatcher: ["dispatcher17", "scion35", "lfm3"],
};
const ROLE_META = {
  planner: ["Planner", "Turns a goal into a task DAG and suggestions."],
  controller: ["Controller", "Accepts the plan, assigns workers, merges."],
  router: ["C1 router", "Decides ultra_small / medium / escalation."],
  face: ["Face / chat", "The model the human talks to."],
  agency: ["Agency / tools", "Tool use, agent loops, delegation."],
  mechanics: ["Mechanics / structured", "JSON, extraction, short deterministic work."],
  judge: ["Judge / escalation", "Second opinion or verification."],
  rag: ["RAG reader", "Long-context retrieval and exact answers."],
  risk: ["Command risk", "Guards plus the offline CLI oracle."],
  vision: ["Vision reader", "Images, screenshots, OCR and visual QA."],
  placement: ["Placement", "Chooses the card and KV budget for a model."],
  critic: ["Critic", "Validates a placement, schedule or assignment."],
  dispatcher: ["Dispatcher", "Assigns jobs to workers under dependencies and deadlines."],
};
const DEFAULT_PLACEMENTS = {
  planner: "non_display", controller: "display",
  router: "non_display", face: "non_display", agency: "non_display",
  mechanics: "display", judge: "display", rag: "display", risk: "",
  vision: "display", placement: "non_display", critic: "non_display",
  dispatcher: "non_display",
};
let rolesState = { ...DEFAULT_ROLES };
let placementsState = { ...DEFAULT_PLACEMENTS };
let devicesState = [];
let placementPlanState = null;
let placementPlanSignature = "";

let enginesState = [];
let engineRevision = "";
let engineDefault = "";
let engineIndex = 0;
let enginesLoaded = false;

function engineAtIndex(index) {
  return enginesState[index] || null;
}

function renderEngineSelect() {
  $("#engine-select").innerHTML = enginesState.map((e, i) =>
    `<option value="${i}">${esc(e.name || "(unnamed)")}</option>`).join("")
    || `<option value="">(no profiles)</option>`;
  $("#engine-select").value = String(engineIndex);
}

function engineFormData() {
  const num = id => {
    const raw = $(id).value.trim();
    return raw === "" ? null : Number(raw);
  };
  const sampling = {};
  const sampleMap = {
    temperature: "#engine-temp", top_p: "#engine-topp", top_k: "#engine-topk",
    min_p: "#engine-minp", repeat_penalty: "#engine-rep",
    presence_penalty: "#engine-pres", frequency_penalty: "#engine-freq",
    seed: "#engine-seed",
  };
  for (const [key, sel] of Object.entries(sampleMap)) {
    const value = num(sel);
    if (value !== null) sampling[key] = value;
  }
  const stop = $("#engine-stop").value.trim();
  if (stop) sampling.stop = stop.includes(",") ? stop.split(",").map(x => x.trim()) : stop;
  let load = {};
  const rawLoad = $("#engine-load").value.trim();
  if (rawLoad) load = JSON.parse(rawLoad);
  return {
    name: $("#engine-name").value.trim(),
    kind: $("#engine-kind").value,
    base_url: $("#engine-url").value.trim(),
    provider: engineAtIndex(engineIndex)?.provider || "",
    capabilities: $("#engine-caps").value.split(",").map(x => x.trim()).filter(Boolean),
    load_options: load,
    sampling,
  };
}

function fillEngineForm() {
  const e = engineAtIndex(engineIndex);
  if (!e) {
    ["#engine-name", "#engine-url", "#engine-caps", "#engine-stop", "#engine-load",
     "#engine-temp", "#engine-topp", "#engine-topk", "#engine-minp", "#engine-rep",
     "#engine-pres", "#engine-freq", "#engine-seed"].forEach(sel => { $(sel).value = ""; });
    $("#engine-default").checked = false;
    $("#engine-meta").textContent = "no profile — create one";
    $("#engine-cards").innerHTML = `
      <div class="card"><div class="label">profiles</div><div class="value">0</div><div class="hint">engine settings</div></div>`;
    return;
  }
  $("#engine-name").value = e.name || "";
  $("#engine-kind").value = e.kind || "llama_cpp";
  $("#engine-url").value = e.base_url || "";
  $("#engine-caps").value = (e.capabilities || []).join(", ");
  $("#engine-load").value = JSON.stringify(e.load_options || {}, null, 2);
  const s = e.sampling || {};
  $("#engine-temp").value = s.temperature ?? "";
  $("#engine-topp").value = s.top_p ?? "";
  $("#engine-topk").value = s.top_k ?? "";
  $("#engine-minp").value = s.min_p ?? "";
  $("#engine-rep").value = s.repeat_penalty ?? "";
  $("#engine-pres").value = s.presence_penalty ?? "";
  $("#engine-freq").value = s.frequency_penalty ?? "";
  $("#engine-seed").value = s.seed ?? "";
  $("#engine-stop").value = Array.isArray(s.stop) ? s.stop.join(",") : (s.stop || "");
  $("#engine-default").checked = engineDefault.toLowerCase() === (e.name || "").toLowerCase();
  $("#engine-meta").textContent = `${enginesState.length} profile${enginesState.length === 1 ? "" : "s"} · revision ${engineRevision || "—"}`;
  $("#engine-cards").innerHTML = `
    <div class="card"><div class="label">profiles</div><div class="value">${enginesState.length}</div><div class="hint">engine settings</div></div>
    <div class="card ok"><div class="label">default</div><div class="value">${esc(engineDefault || "first")}</div><div class="hint">new conversations</div></div>
    <div class="card"><div class="label">kind</div><div class="value">${esc(e.kind || "—")}</div><div class="hint">${esc(e.provider ? "provider " + e.provider : "base URL")}</div></div>
    <div class="card"><div class="label">context</div><div class="value">${esc((e.load_options || {}).context ?? "—")}</div><div class="hint">load option</div></div>`;
}

async function loadEngineModels() {
  try {
    const d = await getJSON("/api/v1/models/local");
    const models = d.models || [];
    $("#engine-model").innerHTML = models.map(m =>
      `<option value="${esc(m.file)}">${esc(m.name || m.file)} · ${Number(m.size_gb || 0).toFixed(1)} GB</option>`).join("")
      || `<option value="">(no local models)</option>`;
  } catch {
    $("#engine-model").innerHTML = `<option value="">(models unavailable)</option>`;
  }
}

async function loadEngines() {
  const d = await getJSON("/api/v1/engines");
  enginesLoaded = true;
  enginesState = d.engines || [];
  engineRevision = d.revision || "";
  engineDefault = d.default || "";
  const wanted = enginesState.findIndex(e => (e.name || "").toLowerCase() === engineDefault.toLowerCase());
  engineIndex = wanted >= 0 ? wanted : 0;
  renderEngineSelect();
  fillEngineForm();
  loadEngineModels();
}

async function saveEngines() {
  if (!enginesLoaded) {
    $("#engine-msg").textContent = "Profiles not loaded yet — click Reload before saving.";
    return;
  }
  let current;
  try { current = engineFormData(); }
  catch (err) { $("#engine-msg").textContent = "Load options JSON invalid: " + err.message; return; }
  if (!current.name) { $("#engine-msg").textContent = "Name is required."; return; }
  const duplicate = enginesState.findIndex((e, i) => i !== engineIndex && (e.name || "").toLowerCase() === current.name.toLowerCase());
  if (duplicate >= 0) { $("#engine-msg").textContent = "Another profile has that name."; return; }
  enginesState[engineIndex] = current;
  if ($("#engine-default").checked) engineDefault = current.name;
  else if ((engineDefault || "").toLowerCase() === current.name.toLowerCase()) engineDefault = "";
  try {
    const res = await fetch("/api/v1/engines", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({
        engines: enginesState, default: engineDefault,
        persist: true, expectedRevision: engineRevision,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || data.error || `${res.status} save failed`);
    enginesState = data.engines || enginesState;
    engineRevision = data.revision || engineRevision;
    engineDefault = data.default || engineDefault;
    renderEngineSelect();
    fillEngineForm();
    $("#engine-msg").textContent = "Saved · " + (data.persisted_to || "live") + " · " + new Date().toLocaleTimeString();
  } catch (err) {
    $("#engine-msg").textContent = "Save failed: " + err.message;
  }
}

function engineAdd() {
  const base = { name: "engine-" + (enginesState.length + 1), kind: "llama_cpp", base_url: "", provider: "", capabilities: ["streaming"], load_options: { context: 8192, gpu_layers: 99 }, sampling: {} };
  enginesState.push(base);
  engineIndex = enginesState.length - 1;
  renderEngineSelect();
  fillEngineForm();
  $("#engine-msg").textContent = "New profile — edit and Save (or Auto preset).";
}

function engineDuplicate() {
  const e = engineAtIndex(engineIndex);
  if (!e) return;
  const copy = JSON.parse(JSON.stringify(e));
  copy.name = (copy.name || "engine") + "-copy";
  enginesState.push(copy);
  engineIndex = enginesState.length - 1;
  renderEngineSelect();
  fillEngineForm();
  $("#engine-msg").textContent = "Duplicated — rename before saving.";
}

async function engineDelete() {
  const e = engineAtIndex(engineIndex);
  if (!e) return;
  if (!confirm(`Delete engine profile "${e.name}"?`)) return;
  if ((engineDefault || "").toLowerCase() === (e.name || "").toLowerCase()) engineDefault = "";
  enginesState.splice(engineIndex, 1);
  engineIndex = Math.max(0, Math.min(engineIndex, enginesState.length - 1));
  renderEngineSelect();
  fillEngineForm();
  await saveEngines();
}

async function engineAuto() {
  const file = $("#engine-model").value;
  if (!file) { $("#engine-msg").textContent = "Choose a local model first."; return; }
  $("#engine-msg").textContent = "computing preset…";
  try {
    const d = await getJSON("/api/v1/engines/preset?file=" + encodeURIComponent(file));
    const preset = d.preset || d.load_options || {};
    $("#engine-load").value = JSON.stringify(preset, null, 2);
    const hw = d.hardware || {};
    const model = d.model || {};
    $("#engine-msg").textContent = `Auto preset: ${model.size_gb || "?"} GB model on ${hw.available_gb || "?"} GB available · Save to apply.`;
  } catch (err) {
    $("#engine-msg").textContent = "Auto preset failed: " + err.message;
  }
}

function engineExport() {
  const blob = new Blob([JSON.stringify({ engines: enginesState, default: engineDefault, exported_at: new Date().toISOString(), version: 1 }, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "hive-engines.json";
  a.click();
  URL.revokeObjectURL(a.href);
}

async function engineImport(file) {
  try {
    const data = JSON.parse(await file.text());
    const incoming = data.engines || data.profiles || data;
    if (!Array.isArray(incoming)) throw new Error("missing engines array");
    for (const entry of incoming) {
      const at = enginesState.findIndex(e => (e.name || "").toLowerCase() === (entry.name || "").toLowerCase());
      if (at >= 0) enginesState[at] = entry;
      else enginesState.push(entry);
    }
    if (data.default) engineDefault = data.default;
    engineIndex = 0;
    renderEngineSelect();
    fillEngineForm();
    await saveEngines();
  } catch (err) {
    $("#engine-msg").textContent = "Import failed: " + err.message;
  }
}

let fitPanel = null;
let providersPanel = null;
let lifecyclePanel = null;
let trainingPanel = null;
let libraryPanel = null;
let runsPanel = null;
let suggestionsPanel = null;
let presetsPanel = null;
let settingsPanel = null;

async function mountFitPanel() {
  if (!window.HiveFitPanel || !$("#fit-panel")) return;
  if (!fitPanel) fitPanel = window.HiveFitPanel.mount($("#fit-panel"));
  await fitPanel.refresh();
}

async function mountProvidersPanel() {
  if (!window.HiveProvidersPanel || !$("#providers-panel")) return;
  if (!providersPanel) providersPanel = window.HiveProvidersPanel.mount($("#providers-panel"));
  await providersPanel.refresh();
}

async function mountLifecyclePanel() {
  if (!window.HiveLifecyclePanel || !$("#lifecycle-panel")) return;
  if (!lifecyclePanel) lifecyclePanel = window.HiveLifecyclePanel.mount($("#lifecycle-panel"));
  await lifecyclePanel.refresh();
}

async function mountTrainingPanel() {
  if (!window.HiveTrainingPanel || !$("#training-panel")) return;
  if (!trainingPanel) trainingPanel = window.HiveTrainingPanel.mount($("#training-panel"));
  await trainingPanel.refresh();
}

async function mountSettingsPanel() {
  if (!window.HiveSettingsPanel || !$("#settings-panel")) return;
  if (!settingsPanel) settingsPanel = window.HiveSettingsPanel.mount($("#settings-panel"));
  return settingsPanel;
}

async function mountPresetsPanel() {
  if (!window.HivePresetsPanel || !$("#presets-panel")) return;
  if (!presetsPanel) presetsPanel = window.HivePresetsPanel.mount($("#presets-panel"));
  return presetsPanel;
}

async function mountSuggestionsPanel() {
  if (!window.HiveSuggestionsPanel || !$("#suggestions-panel")) return;
  if (!suggestionsPanel) suggestionsPanel = window.HiveSuggestionsPanel.mount($("#suggestions-panel"));
  return suggestionsPanel;
}

async function mountRunsPanel() {
  if (!window.HiveRunsPanel || !$("#runs-panel")) return;
  if (!runsPanel) runsPanel = window.HiveRunsPanel.mount($("#runs-panel"));
  await runsPanel.refresh();
}

async function mountLibraryPanel() {
  if (!window.HiveLibraryPanel || !$("#library-panel")) return;
  if (!libraryPanel) libraryPanel = window.HiveLibraryPanel.mount($("#library-panel"));
  await libraryPanel.refresh();
}

async function loadRoles() {
  const [rolesR, catalogR] = await Promise.allSettled([
    getJSON("/api/roles"),
    getJSON("/api/v1/cascade/roles"),
  ]);
  if (rolesR.status === "fulfilled") {
    rolesState = { ...DEFAULT_ROLES, ...(rolesR.value.roles || {}) };
    placementsState = { ...DEFAULT_PLACEMENTS, ...(rolesR.value.placements || {}) };
  } else {
    rolesState = { ...DEFAULT_ROLES };
    placementsState = { ...DEFAULT_PLACEMENTS };
  }
  await refreshPlacementPlan();
  renderRoles();
  renderRoleCatalog(catalogR.status === "fulfilled" ? (catalogR.value.roles || []) : null);
}

function renderRoleCatalog(roles) {
  if (!roles) {
    $("#role-catalog-count").textContent = "unavailable";
    $("#role-catalog").innerHTML = '<tbody><tr><td class="muted">Cascade role catalog unavailable.</td></tr></tbody>';
    return;
  }
  const mapped = { A2: "vision", B1: "rag", C1: "router", D1: "judge", D2: "judge", E1: "face", E2: "face", E3: "mechanics", E4: "agency" };
  const assigned = roles.filter(r => mapped[r.id]).length;
  $("#role-catalog-count").textContent = `${roles.length} roles · ${assigned} mapped`;
  $("#role-catalog").innerHTML = `
    <thead><tr><th>id</th><th>role</th><th>kind</th><th>input → output</th><th>candidates</th><th>status</th></tr></thead>
    <tbody>${roles.map(r => `<tr>
      <td class="mono">${esc(r.id || "")}</td>
      <td><b>${esc(r.name || "")}</b><div class="muted">${esc(r.notes || "")}</div></td>
      <td class="mono">${esc(r.kind || "")}</td>
      <td class="muted">${esc(r.input || "")} → ${esc(r.output || "")}</td>
      <td class="muted">${esc((r.candidates || []).join(", ") || "—")}</td>
      <td>${mapped[r.id] ? `<span class="tag allow">${esc(mapped[r.id])}</span>` : '<span class="tag">unassigned</span>'}</td>
    </tr>`).join("")}</tbody>`;
}

function placementOptions(selected) {
  const opts = [
    ["", "No card policy"],
    ["auto", "Auto (best fit)"],
    ["non_display", "Non-display card"],
    ["display", "Display card (system + light)"],
  ];
  for (const d of devicesState) {
    const index = d.index ?? d.card ?? "?";
    const free = Number.isFinite(Number(d.free_gb)) ? Number(d.free_gb).toFixed(1) : "?";
    const total = Number.isFinite(Number(d.total_gb)) ? Number(d.total_gb).toFixed(1) : "?";
    opts.push([String(index), `#${index} ${d.card || "card"} · ${d.display ? "display" : "non-display"} · ${free}/${total} GB`]);
  }
  return opts.map(([value, label]) =>
    `<option value="${esc(value)}" ${String(selected ?? "") === String(value) ? "selected" : ""}>${esc(label)}</option>`
  ).join("");
}

function renderRoles() {
  const rows = Object.entries(ROLE_META).map(([key, meta]) => {
    const options = ROLE_OPTIONS[key].map(id => {
      const label = MODELS[id]?.label || id;
      return `<option value="${id}" ${rolesState[key] === id ? "selected" : ""}>${label}</option>`;
    }).join("");
    return `<tr>
      <td><span class="role-key">${key}</span><div class="role-desc">${meta[1]}</div></td>
      <td><select data-role="${key}">${options}</select></td>
      <td><select data-placement="${key}">${placementOptions(placementsState[key])}</select></td>
      <td class="muted">${meta[0]}</td>
    </tr>`;
  }).join("");
  $("#roles-table").innerHTML = `
    <thead><tr><th>role</th><th>model</th><th>card</th><th>purpose</th></tr></thead>
    <tbody>${rows}</tbody>`;
  $$("#roles-table select[data-role]").forEach(sel => {
    sel.addEventListener("change", async () => {
      rolesState[sel.dataset.role] = sel.value;
      await refreshPlacementPlan();
      await saveRoles();
    });
  });
  $$("#roles-table select[data-placement]").forEach(sel => {
    sel.addEventListener("change", async () => {
      placementsState[sel.dataset.placement] = sel.value;
      await refreshPlacementPlan();
      await saveRoles();
    });
  });
}

function roleConfigSignature() {
  return JSON.stringify({ roles: rolesState, placements: placementsState });
}

function placementPolicyLabel(policy) {
  if (!policy) return "No card policy";
  if (policy === "auto") return "Auto";
  if (policy === "display") return "Display";
  if (policy === "non_display") return "Non-display";
  return `Card #${policy}`;
}

function showPlacementError(message) {
  const el = $("#placement-error");
  el.textContent = message || "";
  el.hidden = !message;
}

function renderPlacementPlan(result) {
  const { plans = [], summary = {} } = result || {};
  const hasBlocked = plans.some(plan => !plan.fits);
  const blockedRoles = plans.filter(plan => !plan.fits).map(plan => plan.role);
  const warned = Number(summary.warn || 0);
  $("#placement-summary").textContent = `${summary.total || 0} roles · ${summary.ok || 0} OK · ${warned} warning${warned === 1 ? "" : "s"} · ${summary.blocked || 0} blocked`;
  if (hasBlocked) {
    showPlacementError(`Save blocked: placement cannot be proven for ${blockedRoles.join(", ")}.`);
  } else {
    showPlacementError("");
  }
  $("#placement-table").innerHTML = `
    <thead><tr><th>role</th><th>model</th><th>policy</th><th>resolved card</th><th>check</th><th>reason</th></tr></thead>
    <tbody>${plans.map(plan => {
      const size = MODEL_WEIGHTS[plan.model];
      const sizeLabel = size == null ? "unknown" : `${size} GB`;
      const modelLabel = MODELS[plan.model]?.label || plan.model;
      const card = plan.resident ? "resident" : plan.device == null ? "—" : (() => {
        const found = devicesState.find(d => String(d.index) === String(plan.device));
        return found ? `#${plan.device} · ${found.card || "card"}` : `#${plan.device}`;
      })();
      const status = !plan.fits ? "BLOCK" : plan.warnings?.length ? "WARN" : "OK";
      const statusClass = status === "BLOCK" ? "blocked" : status.toLowerCase();
      const explanation = [plan.reason, ...(plan.warnings || [])]
        .filter((value, index, all) => value && all.indexOf(value) === index).join(" · ");
      return `<tr>
        <td><span class="role-key">${esc(plan.role)}</span></td>
        <td>${esc(modelLabel)} <span class="muted">(${esc(sizeLabel)})</span></td>
        <td>${esc(placementPolicyLabel(plan.placement))}</td>
        <td>${esc(card)}</td>
        <td><span class="placement-status ${statusClass}">${status}</span></td>
        <td class="muted">${esc(explanation || "Placement fits.")}</td>
      </tr>`;
    }).join("")}</tbody>`;
}

async function refreshPlacementPlan() {
  placementPlanState = null;
  placementPlanSignature = "";
  $("#placement-summary").textContent = "Checking card policy…";
  $("#placement-table").innerHTML = "";
  showPlacementError("");
  try {
    const [status, scion35Up, scion176Up, ds41Up, lfm3Up, lfm3vlUp, d1Up] = await Promise.all([
      getJSON("/api/v1/server/status"),
      probe(8082), probe(8081), probe(8150), probe(8155), probe(8160), probe(8161),
    ]);
    const resident = {
      scion35: scion35Up, scion176: scion176Up, ds41: ds41Up, lfm3: lfm3Up,
      lfm3vl: lfm3vlUp, "d1-2b": d1Up,
    };
    const hardware = status.hardware || status;
    devicesState = Array.isArray(hardware.devices) ? hardware.devices.map(device => ({
      index: device.index,
      card: device.card,
      display: device.display,
      total_gb: Number(device.total_gb ?? device.memory_gb ?? 0),
      free_gb: Number(device.free_gb ?? 0),
    })) : [];
    const models = Object.fromEntries(Object.entries(MODEL_WEIGHTS).map(([id, weights_gb]) => [id, {
      weights_gb,
      resident: !!resident[id],
      label: MODELS[id]?.label || id,
    }]));
    const result = await postJSON("/api/placement/plan", {
      roles: rolesState,
      placements: placementsState,
      models,
      devices: devicesState,
    });
    placementPlanState = result;
    placementPlanSignature = roleConfigSignature();
    renderPlacementPlan(result);
    return result;
  } catch (error) {
    devicesState = [];
    $("#placement-summary").textContent = "Placement check unavailable";
    showPlacementError(`Placement check failed; saving is disabled. ${error.message}`);
    return null;
  }
}

async function saveRoles() {
  if (!placementPlanState || placementPlanSignature !== roleConfigSignature()) {
    showPlacementError("Placement check is missing or stale; save was not sent.");
    return false;
  }
  const blocked = placementPlanState.plans?.filter(plan => !plan.fits) || [];
  if (blocked.length) {
    showPlacementError(`Save blocked: placement cannot be proven for ${blocked.map(plan => plan.role).join(", ")}.`);
    return false;
  }
  const res = await fetch("/api/roles", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ roles: rolesState, placements: placementsState }),
  });
  if (!res.ok) throw new Error(`save failed ${res.status}`);
  const note = document.createElement("span");
  note.className = "save-state";
  note.textContent = " saved";
  $("#page-roles .panel-head").appendChild(note);
  setTimeout(() => note.remove(), 1200);
  return true;
}

let chatHistory = [];
let chatImage = null;
function libraryStem(file) {
  return String(file || "").split("/").pop().replace(/\.gguf$/i, "");
}

function instanceForFile(file) {
  const stem = libraryStem(file);
  const base = String(file || "").split("/").pop();
  return (LIB.instances || []).find(inst => {
    const model = String(inst.model || "");
    const key = String(inst.key || "");
    return model === file || key === file || model === base || key === base ||
      model === stem || key === stem;
  }) || null;
}

function resolveModel(id) {
  if (MODELS[id]) return { ...MODELS[id], id };
  if (!String(id || "").startsWith("lib:")) return null;
  const file = String(id).slice(4);
  const inst = instanceForFile(file);
  if (!inst || !inst.port) return null;
  const name = libraryStem(file);
  return {
    id, library: true,
    label: name,
    model: inst.model || name,
    base: `/probe/${inst.port}`,
    note: `port ${inst.port}`,
  };
}

function modelOptionGroups() {
  const endpoints = Object.entries(MODELS)
    .map(([id, m]) => `<option value="${esc(id)}">${esc(m.label)}</option>`).join("");
  const library = (LIB.models || []).filter(m => {
    const file = String(m.file || m.name || "").toLowerCase();
    const arch = String(m.architecture || "");
    // LFM3 text/VL/d1 need the newer LFM3 llama.cpp build; the manager's
    // current binary cannot load them. They remain visible on Models.
    return !file.includes("mmproj") && arch !== "clip" && arch !== "lfm3";
  }).map(m => {
    const file = m.file || m.name || "";
    const loaded = !!instanceForFile(file);
    const size = Number.isFinite(Number(m.size_gb)) ? ` · ${Number(m.size_gb).toFixed(1)} GB` : "";
    return `<option value="lib:${esc(file)}">${esc(m.name || libraryStem(file))}${size}${loaded ? " · loaded" : ""}</option>`;
  }).join("");
  return `<optgroup label="Serving endpoints">${endpoints}</optgroup>` +
    (library ? `<optgroup label="Library — load before use">${library}</optgroup>` : "");
}

function renderModelOptions() {
  const html = modelOptionGroups();
  const chat = $("#chat-model"), a = $("#ab-a"), b = $("#ab-b");
  const prevChat = chat.value, prevA = a.value, prevB = b.value;
  chat.innerHTML = html; a.innerHTML = html; b.innerHTML = html;
  const pick = (el, prev, fallback) => {
    el.value = [...el.options].some(o => o.value === prev) ? prev : fallback;
  };
  pick(chat, prevChat, "lfm3");
  pick(a, prevA, "lfm3");
  pick(b, prevB, "scion35");
  updateChatModelState();
}

async function populateChatModels() {
  renderModelOptions();
  try {
    const [local, status] = await Promise.all([
      getJSON("/api/v1/models/local"),
      getJSON("/api/v1/server/status"),
    ]);
    LIB.models = Array.isArray(local.models) ? local.models : [];
    LIB.instances = Array.isArray(status.instances) ? status.instances : [];
    LIB.ts = Date.now();
    LIB.error = "";
  } catch (err) {
    LIB.error = err.message;
  }
  renderModelOptions();
}

function updateChatModelState() {
  const id = $("#chat-model")?.value || "";
  const state = $("#chat-model-state");
  const load = $("#chat-load");
  if (!state || !load) return;
  if (!id.startsWith("lib:")) {
    state.textContent = MODELS[id]?.note || "";
    load.hidden = true;
    return;
  }
  const cfg = resolveModel(id);
  if (cfg) {
    state.textContent = `loaded · ${cfg.note}`;
    load.hidden = true;
  } else {
    state.textContent = "not loaded";
    load.hidden = false;
  }
}

async function loadSelectedLibraryModel() {
  const id = $("#chat-model")?.value || "";
  if (!id.startsWith("lib:")) return;
  const file = id.slice(4);
  const btn = $("#chat-load");
  btn.disabled = true;
  btn.textContent = "Loading…";
  $("#chat-note").textContent = `Starting ${file}…`;
  try {
    const inst = await postJSON("/api/v1/server/start", {
      model: file,
      register_provider: false,
    });
    LIB.instances = [...LIB.instances.filter(x => x.key !== inst.key), inst];
    renderModelOptions();
    $("#chat-model").value = id;
    updateChatModelState();
    $("#chat-note").textContent = `Loaded ${file} on port ${inst.port}.`;
  } catch (err) {
    $("#chat-note").textContent = `Load failed: ${err.message}`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Load";
    updateChatModelState();
  }
}

function addBubble(role, text, meta = "") {
  const empty = $(".chat-empty");
  if (empty) empty.remove();
  const div = document.createElement("div");
  div.className = `bubble ${role}`;
  const copy = document.createElement("span");
  copy.className = "bubble-body";
  copy.textContent = text;
  div.appendChild(copy);
  if (meta) {
    const span = document.createElement("span");
    span.className = "meta";
    span.textContent = meta;
    div.appendChild(span);
  }
  $("#chat-log").appendChild(div);
  $("#chat-log").scrollTop = $("#chat-log").scrollHeight;
  return div;
}

let chatMode = "direct";
let agentAbort = null;
let agentWatch = null;
let agentConv = "ui-" + Date.now();

function chatTranscript() {
  const rows = [];
  document.querySelectorAll("#chat-log .bubble").forEach(bubble => {
    const body = bubble.querySelector(".bubble-body");
    if (!body) return;
    const text = body.textContent || "";
    if (!text.trim()) return;
    rows.push({
      role: bubble.classList.contains("user") ? "user" : "assistant",
      text,
      meta: bubble.querySelector(".meta")?.textContent || "",
    });
  });
  return rows;
}

function chatCommandHelp() {
  return [
    "/help — show this list",
    "/save [name] — download the visible chat as Markdown",
    "/compact — summarize the visible chat into a checkpoint",
    "/compress — alias for /compact",
    "/new — start a fresh chat",
    "/clear — same as /new",
    "/model [id] — show or select a model",
    "/mode direct|agent|ab — switch chat mode",
    "/status — current model, mode, and message count",
  ].join("\n");
}

async function runChatCommand(line) {
  const [rawName, ...rest] = line.slice(1).trim().split(/\s+/);
  const name = (rawName || "help").toLowerCase();
  const args = rest.join(" ").trim();
  if (name === "help" || name === "commands") {
    addBubble("assistant", chatCommandHelp(), "command");
    return;
  }
  if (name === "save") {
    const rows = chatTranscript();
    if (!rows.length) { addBubble("assistant", "Nothing to save yet.", "command"); return; }
    const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    const safe = (args || "chat").replace(/[^A-Za-z0-9._-]+/g, "-");
    const filename = `hive-${safe}-${stamp}.md`;
    const md = [`# Hive chat — ${new Date().toISOString()}`, ""];
    for (const row of rows) {
      md.push(`**${row.role}:** ${row.text}`);
      if (row.meta) md.push(`_${row.meta}_`);
      md.push("");
    }
    const blob = new Blob([md.join("\n")], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    addBubble("assistant", `Saved ${rows.length} messages to ${filename}.`, "command");
    return;
  }
  if (name === "new" || name === "clear") {
    chatHistory = [];
    $("#chat-log").innerHTML = '<div class="chat-empty">Fresh chat. Pick a model and ask something.</div>';
    $("#chat-note").textContent = "New chat started.";
    return;
  }
  if (name === "compact" || name === "compress") {
    const rows = chatTranscript();
    if (rows.length < 2) { addBubble("assistant", "Need at least two messages to compact.", "command"); return; }
    const modelId = $("#chat-model").value;
    const cfg = resolveModel(modelId);
    if (!cfg) { addBubble("assistant", "Load a model first, then run /compact.", "command"); return; }
    const transcript = rows.map(r => `${r.role.toUpperCase()}: ${r.text}`).join("\n\n");
    const result = await callModel(modelId,
      "Compress this Hive chat into a compact checkpoint. Keep decisions, model/paths, " +
      "open questions, and next actions. Drop small talk.\n\n" + transcript);
    $("#chat-log").innerHTML = "";
    addBubble("assistant", "Compacted checkpoint:\n\n" + result.reply,
      `checkpoint · ${cfg.label}${result.ok ? "" : " · error"}`);
    $("#chat-note").textContent = "Conversation compacted.";
    return;
  }
  if (name === "model") {
    const options = [...document.querySelectorAll("#chat-model option")].map(o => ({
      value: o.value, label: o.textContent,
    }));
    if (!args) {
      addBubble("assistant", "Models:\n" + options.map(o => `  ${o.value} — ${o.label}`).join("\n"), "command");
      return;
    }
    const match = options.find(o => o.value === args || o.label.toLowerCase().includes(args.toLowerCase()));
    if (!match) { addBubble("assistant", `No model matches "${args}".`, "command"); return; }
    const sel = $("#chat-model");
    sel.value = match.value;
    sel.dispatchEvent(new Event("change", { bubbles: true }));
    addBubble("assistant", `Selected ${match.label}.`, "command");
    return;
  }
  if (name === "mode") {
    const mode = args.toLowerCase();
    if (!["direct", "agent", "ab"].includes(mode)) {
      addBubble("assistant", "Usage: /mode direct|agent|ab", "command");
      return;
    }
    const btn = [...document.querySelectorAll("#chat-mode button")].find(b => b.dataset.mode === mode);
    if (btn) btn.click();
    addBubble("assistant", `Chat mode: ${mode}.`, "command");
    return;
  }
  if (name === "status") {
    const modelId = $("#chat-model").value;
    const cfg = resolveModel(modelId);
    addBubble("assistant",
      `model: ${cfg?.label || modelId || "(none)"}\nmode: ${chatMode}\nmessages: ${chatTranscript().length}\n` +
      `endpoint: ${cfg?.base || "—"}`, "command");
    return;
  }
  addBubble("assistant", `Unknown command /${name}. Try /help.`, "command");
}

async function sendChat() {
  const raw = $("#chat-input").value.trim();
  if (raw.startsWith("/")) {
    $("#chat-input").value = "";
    await runChatCommand(raw);
    return;
  }
  if (chatMode === "agent") return sendAgent();
  if (chatMode === "ab") return sendAB();
  return sendDirect();
}

async function callModel(modelId, text) {
  const cfg = resolveModel(modelId);
  const started = Date.now();
  if (!cfg) {
    return { reply: "Error: this library model is not loaded. Load it in Direct chat first.", latency: 0, tokens: 0, ok: false };
  }
  const body = {
    model: cfg.model,
    messages: [{ role: "user", content: text }],
    max_tokens: 512, temperature: 0.2, stream: false,
  };
  if (modelId === "lfm3") body.chat_template_kwargs = { enable_thinking: false };
  try {
    const res = await fetch(cfg.base + "/v1/chat/completions", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    const data = await res.json();
    const msg = (data.choices || [{}])[0].message || {};
    return {
      reply: (msg.content || msg.reasoning_content || "").trim() || "(empty reply)",
      latency: (Date.now() - started) / 1000,
      tokens: (data.usage || {}).completion_tokens || 0,
      ok: true,
    };
  } catch (err) {
    return { reply: "Error: " + err.message, latency: (Date.now() - started) / 1000, tokens: 0, ok: false };
  }
}

async function readSSE(res, onEvent) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let done = false;
  const handle = block => {
    const data = block.split(/\r?\n/)
      .filter(line => line.startsWith("data:"))
      .map(line => line.slice(5).trimStart()).join("\n");
    if (!data || data === "[DONE]") return;
    try { onEvent(JSON.parse(data)); } catch { /* ignore malformed keepalives */ }
  };
  while (!done) {
    const chunk = await reader.read();
    done = chunk.done;
    buffer += decoder.decode(chunk.value || new Uint8Array(), { stream: !done });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      handle(block);
    }
  }
  if (buffer.trim()) handle(buffer);
}

async function sendDirect() {
  const input = $("#chat-input");
  const text = input.value.trim();
  if (!text) return;
  const modelId = $("#chat-model").value;
  const cfg = resolveModel(modelId);
  if (!cfg) {
    $("#chat-note").textContent = "This library model is not loaded. Click Load first.";
    return;
  }
  if (chatImage && !cfg.vision) {
    $("#chat-note").textContent = `${cfg.label} does not accept images. Pick LFM3-VL-2B.`;
    return;
  }
  input.value = "";
  const suggestion = await suggestRoute(text);
  addBubble("user", text, `direct · ${cfg.label}${chatImage ? " · image" : ""}${suggestion ? " · route " + suggestion.tier : ""}`);
  const bubble = addBubble("assistant", "", "streaming…");
  const body = bubble.querySelector(".bubble-body");
  const meta = bubble.querySelector(".meta");
  const started = Date.now();
  const controller = new AbortController();
  agentAbort = controller;
  $("#chat-send").disabled = true;
  $("#chat-stop").hidden = false;
  let answer = "", usage = null, failure = null;
  try {
    const userContent = chatImage
      ? [
          { type: "text", text },
          { type: "image_url", image_url: { url: chatImage.dataUrl } },
        ]
      : text;
    const payload = {
      model: cfg.model,
      messages: [{ role: "user", content: userContent }],
      max_tokens: 1024, temperature: 0.2, stream: true,
      stream_options: { include_usage: true },
    };
    if (modelId === "lfm3") payload.chat_template_kwargs = { enable_thinking: false };
    const res = await fetch(`${cfg.base}/v1/chat/completions`, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify(payload), signal: controller.signal,
    });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    await readSSE(res, chunk => {
      usage = chunk.usage || usage;
      const delta = (chunk.choices || [{}])[0].delta || {};
      const piece = delta.content || "";
      if (piece) {
        answer += piece;
        body.textContent = answer;
        $("#chat-log").scrollTop = $("#chat-log").scrollHeight;
      }
    });
    if (!answer.trim()) {
      answer = "(empty reply)";
      body.textContent = answer;
    }
    const seconds = (Date.now() - started) / 1000;
    const tokens = (usage || {}).completion_tokens || Math.round(answer.length / 4);
    meta.textContent = `${seconds.toFixed(2)}s · ${tokens} tok · ${cfg.label}`;
    $("#chat-note").textContent = "";
    if (chatImage) {
      chatImage = null;
      $("#chat-image").value = "";
      $("#chat-image-name").textContent = "";
    }
    logTrace({
      query: text, mode: "direct", actual_model: modelId,
      suggested_tier: suggestion?.tier || "?",
      actual_tier: MODEL_TIER[modelId] || "?",
      latency_s: seconds, tokens, ok: true,
    });
  } catch (err) {
    failure = err.name === "AbortError" ? "Stopped." : "Error: " + err.message;
    if (!answer) body.textContent = failure;
    meta.textContent = failure;
    logTrace({
      query: text, mode: "direct", actual_model: modelId,
      suggested_tier: suggestion?.tier || "?",
      actual_tier: MODEL_TIER[modelId] || "?",
      latency_s: (Date.now() - started) / 1000, tokens: 0,
      ok: false, error: failure,
    });
  } finally {
    agentAbort = null;
    $("#chat-send").disabled = false;
    $("#chat-stop").hidden = true;
  }
}

function approvalCard(req, onDecided) {
  const card = document.createElement("div");
  card.className = "inline-approval";
  card.innerHTML = `<div class="mono">${esc(req.tool_identity || "tool")} · risk ${esc(req.risk || "?")} · ${esc(req.filesystem || "?")}</div>
    <p>${esc(req.action || req.reason || "Tool approval requested")}</p>
    <div class="muted">${esc(req.reason || "")}</div>
    <button class="approve" data-choice="once">Allow once</button>
    <button class="ghost" data-choice="session">Allow session</button>
    <button class="ghost" data-choice="global">Always</button>
    <button class="deny" data-choice="deny">Deny</button>`;
  card.addEventListener("click", async e => {
    const btn = e.target.closest("button[data-choice]");
    if (!btn) return;
    const choice = btn.dataset.choice;
    card.querySelectorAll("button").forEach(x => x.disabled = true);
    try {
      const result = await postJSON("/api/v1/tools/decide", {
        session_id: req.session_id, request_id: req.request_id,
        choice: choice === "global" ? "always" : choice,
        scope: choice === "global" ? "global" : "session",
      });
      card.classList.add(choice === "deny" ? "denied" : "allowed");
      card.innerHTML = `<div class="mono">${esc(req.tool_identity || "tool")}</div>
        <p>${choice === "deny" ? "Denied." : "Approved (" + esc(result.reason || choice) + ") — the agent can continue."}</p>`;
      onDecided?.(choice, result);
    } catch (err) {
      card.querySelector("p").textContent = "Decision failed: " + err.message;
      card.querySelectorAll("button").forEach(x => x.disabled = false);
    }
  });
  return card;
}

async function sendAgent() {
  const input = $("#chat-input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  const suggestion = await suggestRoute(text);
  agentConv = "ui-" + Date.now();
  addBubble("user", text, `agent · conversation ${agentConv}`);
  const started = Date.now();
  const controller = new AbortController();
  agentAbort = controller;
  const seen = new Set();
  let agentSession = null;
  let sawAssistant = false;
  let finalText = "";
  let pollTimer = null;
  const button = $("#chat-send");
  button.disabled = true;
  $("#chat-stop").hidden = false;

  const pollApprovals = async () => {
    let pending = [];
    try {
      const d = await getJSON("/api/v1/tools/pending" + (agentSession ? "?session_id=" + encodeURIComponent(agentSession) : ""));
      pending = (d.requests || []).filter(r => agentSession || String(r.session_id || "").includes(agentConv));
    } catch { return; }
    for (const req of pending) {
      if (seen.has(req.request_id)) continue;
      seen.add(req.request_id);
      const card = approvalCard(req);
      $("#chat-log").appendChild(card);
      $("#chat-log").scrollTop = $("#chat-log").scrollHeight;
    }
  };
  pollTimer = setInterval(pollApprovals, 1200);

  try {
    const res = await fetch("/api/v1/agent/stream", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ message: text, conversation_id: agentConv }),
      signal: controller.signal,
    });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    await readSSE(res, event => {
      if (event.session_id) agentSession = event.session_id;
      if (event.type === "assistant" && event.text) {
        sawAssistant = true;
        addBubble("assistant", event.text, "agent");
      } else if (event.type === "tool") {
        const phase = event.phase || "call";
        const detail = phase === "call" ? event.args : event.result;
        const step = document.createElement("details");
        step.className = "toolstep";
        step.open = phase !== "result";
        step.innerHTML = `<summary>${phase === "call" ? "tool call" : "tool result"} · ${esc(event.tool || "tool")}</summary>
          <pre>${esc(typeof detail === "string" ? detail : JSON.stringify(detail ?? "", null, 2))}</pre>`;
        $("#chat-log").appendChild(step);
        $("#chat-log").scrollTop = $("#chat-log").scrollHeight;
      } else if (event.type === "model") {
        const line = document.createElement("div");
        line.className = "routecard";
        line.innerHTML = `<b>model</b> ${esc(event.tier || "?")} · ${esc(event.model || "?")} · ${esc(event.phase || "")}`;
        $("#chat-log").appendChild(line);
      } else if (event.type === "error") {
        addBubble("assistant", "Agent error: " + (event.error || "unknown"), "error");
      } else if (event.type === "done") {
        finalText = event.final || event.final_response || "";
        if (finalText && !sawAssistant) addBubble("assistant", finalText, "agent");
        $("#chat-note").textContent = `Agent finished · ${event.finish_reason || "completed"}`;
      }
      $("#chat-log").scrollTop = $("#chat-log").scrollHeight;
    });
    pollApprovals();
    const seconds = (Date.now() - started) / 1000;
    logTrace({
      query: text, mode: "agent", actual_model: "hive-agent",
      suggested_tier: suggestion?.tier || "?",
      actual_tier: "medium", latency_s: seconds,
      tokens: Math.round((finalText || "").length / 4), ok: true,
    });
  } catch (err) {
    addBubble("assistant", err.name === "AbortError" ? "Stopped." : "Agent error: " + err.message, "error");
    logTrace({
      query: text, mode: "agent", actual_model: "hive-agent",
      suggested_tier: suggestion?.tier || "?", actual_tier: "medium",
      latency_s: (Date.now() - started) / 1000, tokens: 0, ok: false,
      error: err.message,
    });
  } finally {
    if (pollTimer) clearInterval(pollTimer);
    agentAbort = null;
    button.disabled = false;
    $("#chat-stop").hidden = true;
  }
}

async function sendAB() {
  const input = $("#chat-input");
  const text = input.value.trim();
  if (!text) return;
  const a = $("#ab-a").value, b = $("#ab-b").value;
  const cfgA = resolveModel(a), cfgB = resolveModel(b);
  if (!cfgA || !cfgB) {
    $("#chat-note").textContent = "A/B needs both models loaded. Load library models in Direct chat first.";
    return;
  }
  input.value = "";
  const suggestion = await suggestRoute(text);
  addBubble("user", text, `A/B${suggestion ? " · route " + suggestion.tier : ""}`);
  const pair = document.createElement("div");
  pair.className = "ab-pair";
  const box = side => `<div class="ab-answer" data-side="${side}">
    <div class="head"><span>${side.toUpperCase()} · ${esc((side === "a" ? cfgA : cfgB).label)}</span><span>running…</span></div>
    <div class="body">…</div></div>`;
  pair.innerHTML = box("a") + box("b");
  $("#chat-log").appendChild(pair);
  $("#chat-send").disabled = true;
  const [ra, rb] = await Promise.all([callModel(a, text), callModel(b, text)]);
  const fill = (side, result) => {
    const el = pair.querySelector(`[data-side="${side}"]`);
    el.querySelector(".body").textContent = result.reply;
    el.querySelector(".head span:last-child").textContent =
      `${result.latency.toFixed(2)}s · ${result.tokens} tok`;
  };
  fill("a", ra); fill("b", rb);

  const pairData = {
    answer_a: ra.reply, answer_b: rb.reply,
    latency_a: ra.latency, latency_b: rb.latency,
    tokens_a: ra.tokens, tokens_b: rb.tokens,
    tier_a: MODEL_TIER[a] || "library", tier_b: MODEL_TIER[b] || "library",
  };
  let auto = null;
  try {
    const d1 = await postJSON("/api/decision", {
      state: `Prompt:\n${text}\n\nAnswer A:\n${ra.reply.slice(0, 4000)}\n\nAnswer B:\n${rb.reply.slice(0, 4000)}`,
      questions: {
        better: {
          type: "choice",
          instructions: "Which answer better satisfies the prompt?",
          criteria: { a: "Answer A is better", b: "Answer B is better", tie: "Both are equivalent" },
        },
      },
    });
    const ans = (d1.answers || {}).better;
    if (ans && ["a", "b", "tie"].includes(ans.choice)) {
      auto = { choice: ans.choice, confidence: ans.confidence ?? null, source: "d1" };
    }
  } catch { /* d1 is optional */ }
  if (!auto) {
    try {
      auto = await postJSON("/api/judge/predict", pairData);
      auto.source = auto.source || "logistic";
    } catch { /* judge is optional */ }
  }
  const call = document.createElement("div");
  call.className = "judge-call";
  if (auto) {
    const label = auto.choice === "tie"
      ? "Tie — no clear preference"
      : auto.choice === "a" ? `A (${cfgA.label})` : `B (${cfgB.label})`;
    const conf = auto.confidence != null ? ` · ${(auto.confidence * 100).toFixed(0)}%` : "";
    call.innerHTML = `<b>Auto-judge:</b> ${esc(label)} · ${esc(auto.source)}${conf}`;
    if (auto.choice === "a" || auto.choice === "b") {
      pair.querySelector(`[data-side="${auto.choice}"]`)?.classList.add("judge-pick");
    }
  } else {
    call.textContent = "Auto-judge unavailable.";
  }
  $("#chat-log").appendChild(call);

  const vote = document.createElement("div");
  vote.className = "ab-vote";
  vote.innerHTML = `<button data-vote="a">A better</button>
    <button data-vote="b">B better</button>
    <button data-vote="tie">Tie</button>
    <button data-vote="bad">Both bad</button>`;
  $("#chat-log").appendChild(vote);
  $("#chat-log").scrollTop = $("#chat-log").scrollHeight;
  $("#chat-send").disabled = false;
  vote.addEventListener("click", async e => {
    const btn = e.target.closest("button[data-vote]");
    if (!btn) return;
    const choice = btn.dataset.vote;
    vote.querySelectorAll("button").forEach(x => x.disabled = true);
    btn.classList.add("pick");
    pair.querySelectorAll(".ab-answer").forEach(el => {
      if (choice === el.dataset.side) el.classList.add("win");
    });
    if (auto && ["a", "b", "tie"].includes(auto.choice)) {
      call.classList.add(auto.choice === choice ? "agree" : "disagree");
      call.innerHTML += `<br>${auto.choice === choice ? "Agreed with the human." : "Disagreed with the human."}`;
    }
    let savedId = null;
    try {
      const saved = await postJSON("/api/preferences", {
        ...pairData, prompt: text, vote: choice,
        model_a: a, model_b: b,
        suggested_tier: suggestion?.tier || "?",
        confidence: suggestion?.confidence ?? null,
        reason: suggestion?.reason || "",
        auto_choice: auto?.choice || null,
        auto_confidence: auto?.confidence ?? null,
        auto_source: auto?.source || null,
      });
      savedId = saved.id;
      const note = document.createElement("div");
      note.className = "chat-note";
      note.textContent = "Preference saved. Optional: say why you picked it.";
      $("#chat-log").appendChild(note);
    } catch (err) {
      const fail = document.createElement("div");
      fail.className = "chat-note";
      fail.textContent = "Preference save failed: " + err.message;
      $("#chat-log").appendChild(fail);
    }
    if (savedId) showRationale(savedId);
  });
  $("#chat-note").textContent = "Both answers ready — pick the better one.";
}

function showRationale(id) {
  const tags = ["More accurate", "Clearer", "More concise", "Faster", "Safer", "Better formatted"];
  const box = document.createElement("div");
  box.className = "rationale";
  box.innerHTML = `<div class="title">Why did you pick that answer?</div>
    <div class="tags">${tags.map(t => `<button data-tag="${esc(t)}">${esc(t)}</button>`).join("")}</div>
    <textarea placeholder="Optional: one sentence about what made the difference."></textarea>
    <div class="actions"><button class="ghost skip">Skip</button><button class="approve save">Save reason</button></div>`;
  const chosen = new Set();
  box.querySelectorAll("[data-tag]").forEach(btn => btn.addEventListener("click", () => {
    const tag = btn.dataset.tag;
    if (chosen.has(tag)) { chosen.delete(tag); btn.classList.remove("on"); }
    else { chosen.add(tag); btn.classList.add("on"); }
  }));
  box.querySelector(".skip").addEventListener("click", () => box.remove());
  box.querySelector(".save").addEventListener("click", async () => {
    try {
      await postJSON("/api/preferences/rationale", {
        id, tags: [...chosen], rationale: box.querySelector("textarea").value.trim(),
      });
      box.innerHTML = `<div class="title">Reason saved.</div>`;
      setTimeout(() => box.remove(), 1200);
    } catch (err) {
      box.querySelector(".title").textContent = "Save failed: " + err.message;
    }
  });
  $("#chat-log").appendChild(box);
  $("#chat-log").scrollTop = $("#chat-log").scrollHeight;
}

async function loadJudge() {
  const d = await getJSON("/api/judge/status");
  const p = d.profile || {};
  const model = d.model || {};
  const pct = x => x == null ? "—" : (x * 100).toFixed(0) + "%";
  $("#judge-cards").innerHTML = `
    <div class="card"><div class="label">human votes</div><div class="value">${p.n || 0}</div><div class="hint">${p.ties || 0} tie${(p.ties || 0) === 1 ? "" : "s"} · A/B decisions</div></div>
    <div class="card ok"><div class="label">auto agreement</div><div class="value">${pct(p.auto_agreement)}</div><div class="hint">${p.auto_agreement_n || 0} judged votes</div></div>
    <div class="card"><div class="label">judge model</div><div class="value">${model.n || 0}</div><div class="hint">training votes</div></div>
    <div class="card"><div class="label">LOO accuracy</div><div class="value">${model.n >= 5 ? pct(model.loo_accuracy) : "—"}</div><div class="hint">${model.n >= 5 ? "leave-one-out, tiny sample" : "needs 5 votes"}</div></div>`;
  const w = model.weights || {};
  const fmtW = x => (Number(x) >= 0 ? "+" : "") + Number(x || 0).toFixed(2);
  $("#judge-weights").innerHTML = model.n
    ? `<span>learned weights:</span>` +
      ["length", "latency", "tokens", "tier"].map(k => `<span>${k} <b>${fmtW(w[k])}</b></span>`).join("") +
      `<span>bias <b>${fmtW(model.bias)}</b></span>` +
      `<span class="cap">Features are A-minus-B normalized differences; a positive weight means more of that trait makes the judge favor A. Baseline applies until 5 votes.</span>`
    : `<span class="cap">No model yet — the baseline (longer answer, or tie when equivalent) is used until 5 votes.</span>`;
  const rows = [
    ["longer answer chosen", p.longer],
    ["shorter answer chosen", p.shorter],
    ["faster answer chosen", p.faster],
    ["slower answer chosen", p.slower],
    ["higher tier chosen", p.higher_tier],
    ["lower tier chosen", p.lower_tier],
  ];
  const tags = Object.entries(p.tags || {}).map(([k, v]) => `${k} (${v})`).join(" · ") || "no reasons tagged yet";
  const lengthLean = (p.shorter || 0) >= 0.6 ? "concise answers"
    : (p.longer || 0) >= 0.6 ? "thorough answers" : "balanced on length";
  const speedLean = (p.faster || 0) >= 0.6 ? "speed-first"
    : (p.slower || 0) >= 0.6 ? "latency-tolerant" : "speed-neutral";
  const tierLean = (p.higher_tier || 0) >= 0.6 ? "prefers stronger tiers"
    : (p.lower_tier || 0) >= 0.6 ? "prefers cheaper tiers" : "tier-neutral";
  const summary = p.n
    ? `Leans ${lengthLean}; ${speedLean}; ${tierLean}.`
    : "No votes yet — run an A/B comparison to start the profile.";
  $("#judge-profile").innerHTML = `<div class="profile-summary">${esc(summary)}</div>`
    + rows.map(([label, value]) => `
    <div class="profile-row"><span>${label}</span>
      <div class="profile-bar"><i style="width:${((value || 0) * 100).toFixed(1)}%"></i></div>
      <b>${pct(value)}</b></div>`).join("")
    + `<div class="profile-row"><span>mean chosen / rejected length</span><span>${p.mean_chosen_len ?? "—"} / ${p.mean_rejected_len ?? "—"} chars</span><b></b></div>`
    + `<div class="profile-row"><span>mean chosen / rejected latency</span><span>${p.mean_chosen_latency ?? "—"} / ${p.mean_rejected_latency ?? "—"} s</span><b></b></div>`
    + `<div class="profile-row"><span>reason tags</span><span>${esc(tags)}</span><b></b></div>`;
  $("#judge-table").innerHTML = `
    <thead><tr><th>time</th><th>prompt</th><th>auto</th><th>human</th><th>why</th></tr></thead>
    <tbody>${(d.recent || []).slice().reverse().map(r => `<tr>
      <td class="mono">${timeOf(r.ts)}</td>
      <td>${short(r.prompt, 80)}</td>
      <td>${r.auto_choice ? `<span class="tag">${esc(r.auto_choice.toUpperCase())}</span> ${esc(r.auto_source || "")}` : "—"}</td>
      <td><span class="tag ${r.vote === "a" || r.vote === "b" ? "allow" : ""}">${esc((r.vote || "").toUpperCase())}</span></td>
      <td>${esc((r.tags || []).join(", ") || r.rationale || "—")}</td>
    </tr>`).join("") || `<tr><td colspan="5" class="muted">No votes yet.</td></tr>`}</tbody>`;
}


async function stopAgent() {
  try { await fetch("/api/v1/agent/cancel", { method: "POST" }); } catch {}
  if (agentAbort) agentAbort.abort();
}

const ROUTING = {
  models: [
    { name: "d1-2B", acc: 0.867, simple: "19/20", medium: "18/20", complex: "15/20" },
    { name: "DroneRouter", acc: 0.333, simple: "20/20", medium: "0/20", complex: "0/20" },
  ],
};
async function loadBench() {
  let d;
  try { d = await getJSON("/api/benchmark"); }
  catch (err) {
    $("#bench-cards").innerHTML = "";
    $("#bench-table").innerHTML = "";
    $("#bench-matrix").innerHTML = "";
    $("#bench-meta").textContent = "no run";
    $("#bench-note").textContent = "No completed matrix found: " + err.message;
    return;
  }
  const models = Object.entries(d.models || {});
  const alias = { "lfm3-1.7b": "lfm3" };
  const label = id => MODELS[id]?.label || MODELS[alias[id]]?.label || id;
  const acc = m => Number(m.accuracy || 0);
  const sorted = models.slice().sort((a, b) => acc(b[1]) - acc(a[1]));
  const leader = sorted[0];
  const fastest = models.slice().sort((a, b) => Number(a[1].median_s || 1e9) - Number(b[1].median_s || 1e9))[0];
  const pct = x => x == null ? "—" : (Number(x) * 100).toFixed(1) + "%";
  $("#bench-cards").innerHTML = `
    <div class="card"><div class="label">tasks</div><div class="value">${d.tasks || 0}</div><div class="hint">${d.samples || 1} sample · test split</div></div>
    <div class="card"><div class="label">models</div><div class="value">${models.length}</div><div class="hint">same prompt + budgets</div></div>
    <div class="card ok"><div class="label">leader</div><div class="value">${leader ? pct(leader[1].accuracy) : "—"}</div><div class="hint">${leader ? esc(label(leader[0])) : "no data"}</div></div>
    <div class="card"><div class="label">fastest median</div><div class="value">${fastest ? Number(fastest[1].median_s || 0).toFixed(1) + "s" : "—"}</div><div class="hint">${fastest ? esc(label(fastest[0])) : "no data"}</div></div>`;
  $("#bench-meta").textContent = `${d.file || "run"} · ${d.mtime ? d.mtime.slice(0, 10) : ""}`;
  $("#bench-table").innerHTML = `
    <thead><tr><th>model</th><th>accuracy</th><th>median</th><th>p95</th><th>completion tokens</th><th>errors</th></tr></thead>
    <tbody>${sorted.map(([id, m]) => `<tr>
      <td><b>${esc(label(id))}</b><div class="muted mono">${esc(id)}</div></td>
      <td><span class="tag ${acc(m) >= 0.9 ? "allow" : acc(m) >= 0.7 ? "ask" : "deny"}">${pct(m.accuracy)}</span></td>
      <td class="mono">${Number(m.median_s || 0).toFixed(2)}s</td>
      <td class="mono">${Number(m.p95_s || 0).toFixed(2)}s</td>
      <td class="mono">${Number(m.total_completion_tokens || 0).toLocaleString()}</td>
      <td class="mono">${m.errors || 0}</td>
    </tr>`).join("")}</tbody>`;
  const benchmarks = [...new Set(models.flatMap(([, m]) => Object.keys(m.by_benchmark || {})))];
  $("#bench-matrix").innerHTML = `
    <thead><tr><th>benchmark</th>${sorted.map(([id]) => `<th>${esc(label(id))}</th>`).join("")}</tr></thead>
    <tbody>${benchmarks.map(b => `<tr>
      <td class="mono">${esc(b)}</td>
      ${sorted.map(([, m]) => {
        const x = (m.by_benchmark || {})[b] || {};
        return `<td class="mono">${pct(x.accuracy)} <span class="muted">/ ${Number(x.median_s || 0).toFixed(1)}s</span></td>`;
      }).join("")}
    </tr>`).join("")}</tbody>`;
  $("#bench-note").textContent = "Single sample per task. DS4.1 used the generic sampling profile and its failures are dominated by repetition and truncation, so its 0.500 is not a fair final rank. Scion-35B is the 27B+ default; LFM3 is the cheap tier.";
}

let routeTargetsState = {
  simple: { model: "lfm3", engine: "lfm3-1.7b" },
  medium: { model: "scion35", engine: "scion-35b" },
  complex: { model: "ds41", engine: "" },
};

function routeEngineLabel(name, engines) {
  const e = engines.find(x => (x.name || "").toLowerCase() === (name || "").toLowerCase());
  if (!e) return name ? `${name} (missing)` : "no profile";
  const s = e.sampling || {};
  const bits = [];
  if (s.temperature != null) bits.push("t=" + s.temperature);
  if (s.top_p != null) bits.push("p=" + s.top_p);
  const ctx = (e.load_options || {}).context;
  if (ctx) bits.push("ctx " + ctx);
  return bits.join(" · ") || "profile defaults";
}

async function loadRouting() {
  const d = await getJSON("/api/routing/traces?limit=100");
  const sum = d.summary || {};
  const n = sum.n || 0;
  const pct = x => n ? ((x || 0) / n * 100).toFixed(0) + "%" : "—";
  $("#routing-cards").innerHTML = `
    <div class="card"><div class="label">live traces</div><div class="value">${n}</div><div class="hint">suggested vs actual</div></div>
    <div class="card ok"><div class="label">matched tier</div><div class="value">${pct(sum.match)}</div><div class="hint">${sum.match || 0} of ${n}</div></div>
    <div class="card"><div class="label">above suggestion</div><div class="value">${pct(sum.over)}</div><div class="hint">actual tier more expensive</div></div>
    <div class="card bad"><div class="label">below suggestion</div><div class="value">${pct(sum.under)}</div><div class="hint">actual tier cheaper</div></div>`;
  const rows = (d.traces || []).slice().reverse();
  $("#routing-table").innerHTML = `
    <thead><tr><th>time</th><th>query</th><th>suggested</th><th>actual</th><th>model</th><th>latency</th><th>ok</th></tr></thead>
    <tbody>${rows.map(r => `<tr>
      <td class="mono">${timeOf(r.ts)}</td>
      <td>${short(r.query, 90)}</td>
      <td><span class="tag">${esc(r.suggested_tier || "?")}</span></td>
      <td><span class="tag ${esc(r.actual_tier || "")}">${esc(r.actual_tier || "?")}</span></td>
      <td class="mono">${esc(r.actual_model || "—")}</td>
      <td class="mono">${Number(r.latency_s || 0).toFixed(2)}s</td>
      <td>${r.ok ? '<span class="tag allow">ok</span>' : '<span class="tag deny">fail</span>'}</td>
    </tr>`).join("") || `<tr><td colspan="7" class="muted">No live traces yet — send a chat or agent prompt.</td></tr>`}</tbody>`;
  try {
    const p = await getJSON("/api/preferences?limit=50");
    const ps = p.summary || {};
    const votes = ps.votes || {};
    $("#pref-cards").innerHTML = `
      <div class="card"><div class="label">votes</div><div class="value">${ps.n || 0}</div><div class="hint">A/B decisions</div></div>
      <div class="card ok"><div class="label">A wins</div><div class="value">${votes.a || 0}</div><div class="hint">first model</div></div>
      <div class="card ok"><div class="label">B wins</div><div class="value">${votes.b || 0}</div><div class="hint">second model</div></div>
      <div class="card"><div class="label">tie / bad</div><div class="value">${(votes.tie || 0) + (votes.bad || 0)}</div><div class="hint">no preference</div></div>`;
    $("#pref-table").innerHTML = `
      <thead><tr><th>time</th><th>prompt</th><th>A</th><th>B</th><th>vote</th><th>latency A/B</th></tr></thead>
      <tbody>${(p.preferences || []).slice().reverse().map(r => `<tr>
        <td class="mono">${timeOf(r.ts)}</td>
        <td>${short(r.prompt, 100)}</td>
        <td class="mono">${esc(r.model_a || "")} <span class="tag">${esc(r.tier_a || "")}</span></td>
        <td class="mono">${esc(r.model_b || "")} <span class="tag">${esc(r.tier_b || "")}</span></td>
        <td><span class="tag ${r.vote === "a" || r.vote === "b" ? "allow" : ""}">${esc(r.vote || "")}</span></td>
        <td class="mono">${Number(r.latency_a || 0).toFixed(2)} / ${Number(r.latency_b || 0).toFixed(2)}s</td>
      </tr>`).join("") || `<tr><td colspan="6" class="muted">No votes yet — use A/B mode in Chat.</td></tr>`}</tbody>`;
  } catch { /* preferences are optional */ }
  const note = document.querySelector("#page-routing .note");
  if (note) note.textContent = "Live suggestions use the DroneRouter heuristic as an observational preview; d1 replaces it when wired. "
    + "Above/below is structural until the cheaper tier is quality-checked. Calibration: d1 86.7% 3-class, DroneRouter 33.3%.";
  try {
    const [targets, engines] = await Promise.all([
      getJSON("/api/route-targets"),
      getJSON("/api/v1/engines"),
    ]);
    routeTargetsState = targets.targets || routeTargetsState;
    const profiles = engines.engines || [];
    const modelOptions = Object.entries(MODELS).map(([id, m]) =>
      `<option value="${id}">${esc(m.label)}</option>`).join("");
    const engineOptions = `<option value="">— no profile —</option>` +
      profiles.map(e => `<option value="${esc(e.name)}">${esc(e.name)}</option>`).join("");
    $("#route-targets-table").innerHTML = `
      <thead><tr><th>tier</th><th>model</th><th>engine profile</th><th>profile settings</th></tr></thead>
      <tbody>${["simple", "medium", "complex"].map(tier => {
        const row = routeTargetsState[tier] || { model: "", engine: "" };
        return `<tr>
          <td><span class="tag ${tier}">${esc(tier)}</span></td>
          <td><select data-tier="${tier}" data-kind="model">${modelOptions}</select></td>
          <td><select data-tier="${tier}" data-kind="engine">${engineOptions}</select></td>
          <td class="muted">${esc(routeEngineLabel(row.engine, profiles))}</td>
        </tr>`;
      }).join("")}</tbody>`;
    $$("#route-targets-table select").forEach(sel => {
      const row = routeTargetsState[sel.dataset.tier] || {};
      sel.value = sel.dataset.kind === "model" ? (row.model || "") : (row.engine || "");
      sel.addEventListener("change", () => {
        routeTargetsState[sel.dataset.tier] = {
          ...(routeTargetsState[sel.dataset.tier] || {}),
          [sel.dataset.kind]: sel.value,
        };
        $("#route-targets-msg").textContent = "Unsaved changes.";
      });
    });
  } catch (err) {
    $("#route-targets-table").innerHTML = `<tbody><tr><td class="muted">Route targets unavailable: ${esc(err.message)}</td></tr></tbody>`;
  }
}

async function saveRouteTargets() {
  try {
    await postJSON("/api/route-targets", { targets: routeTargetsState });
    $("#route-targets-msg").textContent = "Saved · new routing config";
    await loadRouting();
  } catch (err) {
    $("#route-targets-msg").textContent = "Save failed: " + err.message;
  }
}

function show(page) {
  state.page = page;
  $$("#nav button").forEach(b => b.classList.toggle("active", b.dataset.page === page));
  $$(".page").forEach(p => p.classList.toggle("active", p.id === `page-${page}`));
  const titles = {
    overview: ["Overview", "Live service and GPU state."],
    chat: ["Chat", "Prompt a model directly."],
    roles: ["Roles", "Choose which model serves which role."],
    bench: ["Benchmark", "Model matrix and role scorecards."],
    routing: ["Routing", "C1 tier selection, measured."],
    judge: ["Judge", "Auto-judge vs human preference."],
    risk: ["Risk & Approvals", "Tool gateway audit and decisions."],
    activity: ["Activity", "Recent tool and agent events."],
    engines: ["Engines", "Engine profiles, sampling and load options."],
    providers: ["Providers", "OpenAI-compatible endpoints and keys."],
    lifecycle: ["Lifecycle", "Load, unload and apply models and stacks."],
    training: ["Training", "Ternary training runs and preset evidence."],
    runs: ["Runs", "Run history and report summaries."],
    suggestions: ["Suggestions", "Planner ideas for human review."],
    presets: ["Agent presets", "System and user agent configurations."],
    settings: ["Settings", "Global Studio controls and defaults."],
    models: ["Models", "Library, download folder and Hugging Face finder."],
  };
  $("#page-title").textContent = titles[page][0];
  $("#page-sub").textContent = titles[page][1];
  if (page === "activity") startActivityStream(); else stopActivityStream();
  location.hash = page;
  refresh();
}

async function refresh() {
  try {
    await loadGlobal();
    if (state.page === "overview") await loadOverview();
    if (state.page === "chat") populateChatModels();
    if (state.page === "roles") await loadRoles();
    if (state.page === "risk") await loadRisk();
    if (state.page === "activity") await loadActivity();
    if (state.page === "engines") { await loadEngines(); await mountFitPanel(); }
    if (state.page === "providers") await mountProvidersPanel();
    if (state.page === "lifecycle") await mountLifecyclePanel();
    if (state.page === "training") await mountTrainingPanel();
    if (state.page === "runs") await mountRunsPanel();
    if (state.page === "suggestions") await mountSuggestionsPanel();
    if (state.page === "presets") await mountPresetsPanel();
    if (state.page === "settings") await mountSettingsPanel();
    if (state.page === "models") await mountLibraryPanel();
    if (state.page === "routing") await loadRouting();
    if (state.page === "judge") await loadJudge();
    if (state.page === "bench") await loadBench();
  } catch (err) {
    console.error(err);
  }
}

$("#refresh").addEventListener("click", refresh);
$("#judge-reset").addEventListener("click", async () => {
  if (!confirm("Archive the current votes, rationales and judge model, and start a fresh profile?")) return;
  try {
    await postJSON("/api/preferences/reset", {});
    await loadJudge();
  } catch (err) {
    alert("Reset failed: " + err.message);
  }
});
$("#chat-send").addEventListener("click", sendChat);
$("#chat-attach").addEventListener("click", () => $("#chat-image").click());
$("#chat-image").addEventListener("change", () => {
  const file = $("#chat-image").files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    chatImage = { name: file.name, dataUrl: String(reader.result || "") };
    $("#chat-image-name").textContent = file.name;
    $("#chat-note").textContent = "Image attached — LFM3-VL-2B can read it.";
  };
  reader.onerror = () => { $("#chat-note").textContent = "Could not read that image."; };
  reader.readAsDataURL(file);
});
$("#chat-model").addEventListener("change", updateChatModelState);
$("#chat-load").addEventListener("click", loadSelectedLibraryModel);
$("#chat-stop").addEventListener("click", stopAgent);
$("#chat-mode").addEventListener("click", e => {
  const b = e.target.closest("button[data-mode]");
  if (!b) return;
  chatMode = b.dataset.mode;
  $$("#chat-mode button").forEach(x => x.classList.toggle("active", x === b));
  $("#chat-model").style.display = chatMode === "direct" ? "" : "none";
  $("#ab-bar").hidden = chatMode !== "ab";
  $("#chat-note").textContent = chatMode === "agent"
    ? "Agent uses Hive's configured provider (DS4.1); role selection is not wired to the agent yet."
    : chatMode === "ab"
      ? "A/B runs both models concurrently; choose models on different cards for best latency."
      : "";
});
$("#chat-input").addEventListener("keydown", e => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); sendChat(); }
});
$("#decision-run").addEventListener("click", async () => {
  const msg = $("#decision-msg"), out = $("#decision-out");
  msg.textContent = "asking d1…";
  out.textContent = "";
  try {
    const questions = JSON.parse($("#decision-questions").value);
    const res = await postJSON("/api/decision", {
      state: $("#decision-state").value,
      questions,
    });
    out.textContent = JSON.stringify(res.answers || res, null, 2);
    msg.textContent = "answered";
  } catch (err) {
    msg.textContent = "failed: " + err.message;
  }
});
$("#route-targets-save").addEventListener("click", saveRouteTargets);
$("#engine-select").addEventListener("change", e => {
  engineIndex = Number(e.target.value) || 0;
  fillEngineForm();
});
$("#engine-add").addEventListener("click", engineAdd);
$("#engine-dup").addEventListener("click", engineDuplicate);
$("#engine-del").addEventListener("click", engineDelete);
$("#engine-save").addEventListener("click", saveEngines);
$("#engine-reload").addEventListener("click", async () => {
  await loadEngines();
  $("#engine-msg").textContent = "Reloaded from Hive.";
});
$("#engine-auto").addEventListener("click", engineAuto);
$("#engine-export").addEventListener("click", engineExport);
$("#engine-import-btn").addEventListener("click", () => $("#engine-import").click());
$("#engine-import").addEventListener("change", e => {
  const file = e.target.files && e.target.files[0];
  if (file) engineImport(file);
  e.target.value = "";
});
$("#engine-default").addEventListener("change", e => {
  const ecurrent = engineAtIndex(engineIndex);
  if (!ecurrent) return;
  engineDefault = e.target.checked ? ecurrent.name : "";
  $("#engine-msg").textContent = "Default changed — Save to apply.";
});
$("#chat-clear").addEventListener("click", () => {
  chatImage = null;
  $("#chat-image").value = "";
  $("#chat-image-name").textContent = "";
  chatHistory = [];
  $("#chat-log").innerHTML = '<div class="chat-empty">Pick a model and ask something.</div>';
  $("#chat-note").textContent = "";
});
populateChatModels();
$("#nav").addEventListener("click", e => {
  const b = e.target.closest("button[data-page]");
  if (b) show(b.dataset.page);
});

loadGlobal();
setInterval(() => {
  if (document.hidden) return;
  loadGlobal();
  if (state.page === "overview") loadOverview();
  if (state.page === "activity") loadActivity();
  if (state.page === "risk") loadRisk();
}, 10000);

const start = location.hash.slice(1) || "overview";
show(["overview","chat","roles","bench","routing","judge","risk","activity","engines","providers","lifecycle","training","models"].includes(start) ? start : "overview");
setInterval(() => { $("#clock").textContent = new Date().toLocaleString(); }, 1000);
$("#clock").textContent = new Date().toLocaleString();
