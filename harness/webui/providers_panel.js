/* Hive Console — providers panel (Round 1, F2)
 *
 * Self-contained, framework-free module for editing the LLM provider list.
 * It talks to GET/POST /api/v1/provider/config and owns no page wiring:
 *
 *   const panel = window.HiveProvidersPanel.mount(el);
 *   panel.refresh()   // async, re-reads the config
 *   panel.state       // live view of the panel state
 *
 * Contract notes:
 *  - The API returns api_key as "***" for every stored secret. A masked key
 *    is echoed back on save so the backend keeps the stored secret; "***"
 *    only ever means "leave the existing key alone".
 *  - Saves are optimistic-concurrency guarded: { providers, default,
 *    persist: true, expectedRevision }. A 409 is surfaced with a Reload
 *    action instead of silently overwriting someone else's change.
 */
(function (global) {
  "use strict";

  var ENDPOINT = "/api/v1/provider/config";
  var MASK = "***";
  var instanceSeq = 0;

  /* ---------------------------------------------------------------- DOM */

  function h(tag, attrs) {
    var node = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (key) {
        var value = attrs[key];
        if (value == null || value === false) return;
        if (key === "class") node.className = value;
        else if (key === "text") node.textContent = String(value);
        else if (key === "value") node.value = String(value);
        else if (key === "dataset") Object.assign(node.dataset, value);
        else if (key.slice(0, 2) === "on" && typeof value === "function") {
          node.addEventListener(key.slice(2), value);
        } else node.setAttribute(key, value === true ? "" : String(value));
      });
    }
    for (var i = 2; i < arguments.length; i += 1) {
      var child = arguments[i];
      if (child == null || child === false) continue;
      node.append(child instanceof Node ? child : String(child));
    }
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function label(text, field, hint) {
    var wrap = h("label", { class: "hp-field" },
      h("span", { class: "hp-label", text: text }));
    wrap.append(field);
    if (hint) wrap.append(h("small", { class: "hp-hint", text: hint }));
    return wrap;
  }

  /* ------------------------------------------------------------ headers */

  function headersToText(headers) {
    if (!headers || typeof headers !== "object") return "";
    return Object.keys(headers).sort().map(function (key) {
      return key + ": " + headers[key];
    }).join("\n");
  }

  function textToHeaders(text) {
    var headers = {};
    var lines = String(text == null ? "" : text).replace(/\r\n/g, "\n").split("\n");
    for (var i = 0; i < lines.length; i += 1) {
      var line = lines[i].trim();
      if (!line) continue;
      var at = line.indexOf(":");
      if (at <= 0) {
        throw new Error("Header line " + (i + 1) + ' must look like "Name: value".');
      }
      var name = line.slice(0, at).trim();
      var value = line.slice(at + 1).trim();
      if (!name) throw new Error("Header line " + (i + 1) + " has no name.");
      headers[name] = value;
    }
    return headers;
  }

  /* -------------------------------------------------------------- model */

  var idSeq = 0;

  function normalize(raw, index) {
    var key = raw && raw.api_key != null ? String(raw.api_key) : "";
    return {
      id: "p" + (++idSeq),
      name: String((raw && raw.name) || ""),
      base_url: String((raw && raw.base_url) || ""),
      model: String((raw && raw.model) || ""),
      api_key: key,
      keyMasked: key === MASK,
      keyEdited: false,
      headersText: headersToText(raw && raw.headers),
      origName: String((raw && raw.name) || ""),
      errors: {},
      row: null,
      order: index
    };
  }

  function blankProvider(index) {
    return {
      id: "p" + (++idSeq),
      name: "provider-" + index,
      base_url: "",
      model: "",
      api_key: "",
      keyMasked: false,
      keyEdited: true,
      headersText: "",
      origName: "",
      errors: {},
      row: null,
      order: index
    };
  }

  function createState() {
    return {
      status: "idle",      // idle | loading | ready | saving | error
      revision: null,
      default: "",
      file: "",
      providers: [],
      baseline: "",
      dirty: false,
      error: null,
      conflict: null,
      notice: null
    };
  }

  function detailOf(body) {
    if (!body || typeof body !== "object") return "";
    var detail = body.detail != null ? body.detail
      : body.error != null ? body.error : body.message;
    if (Array.isArray(detail)) {
      return detail.map(function (item) {
        return (item && (item.msg || item.message)) || JSON.stringify(item);
      }).join("; ");
    }
    if (detail && typeof detail === "object") return JSON.stringify(detail);
    return detail == null ? "" : String(detail);
  }

  /* --------------------------------------------------------------- mount */

  function mount(el, opts) {
    if (!el || typeof el.appendChild !== "function") {
      throw new Error("HiveProvidersPanel.mount(el) needs a DOM element");
    }
    var options = opts || {};
    var endpoint = String(options.endpoint || ENDPOINT);
    var uid = "hp" + (++instanceSeq);
    var state = createState();

    /* ---------------------------------------------------------- chrome */

    var revPill = h("span", { class: "hp-pill", text: "revision —" });
    var fileCode = h("code", { class: "hp-file", text: "" });
    var banner = h("div", { class: "hp-banner", role: "status", "aria-live": "polite" });
    var saveBtn = h("button", {
      class: "hp-btn hp-primary", type: "button", text: "Save changes"
    });
    var reloadBtn = h("button", {
      class: "hp-btn", type: "button", text: "Reload"
    });
    var addBtn = h("button", {
      class: "hp-btn hp-add", type: "button", text: "+ Add provider"
    });
    var dirtyFlag = h("span", { class: "hp-dirty", text: "unsaved changes" });
    var listEl = h("div", { class: "hp-list" });
    var footEl = h("p", { class: "hp-foot" });

    var root = h("div", { class: "hive-providers" },
      h("div", { class: "hp-head" },
        h("div", { class: "hp-heading" },
          h("h2", { class: "hp-title", text: "Providers" }),
          h("p", {
            class: "hp-sub",
            text: "OpenAI-compatible endpoints. API keys are write-only: the server returns them masked."
          })
        ),
        h("div", { class: "hp-meta" }, revPill, fileCode)
      ),
      banner,
      h("div", { class: "hp-toolbar" },
        saveBtn, reloadBtn, addBtn, dirtyFlag
      ),
      listEl,
      footEl
    );

    while (el.firstChild) el.removeChild(el.firstChild);
    el.appendChild(root);

    /* --------------------------------------------------------- helpers */

    function busy() {
      return state.status === "loading" || state.status === "saving";
    }

    function signature() {
      var providers = state.providers.map(function (p) {
        return [
          p.name.trim(),
          p.base_url.trim(),
          p.model.trim(),
          p.keyMasked && !p.keyEdited ? MASK : p.api_key,
          p.headersText.replace(/\r\n/g, "\n").trim()
        ];
      });
      return JSON.stringify({ default: state.default, providers: providers });
    }

    function syncDirty() {
      state.dirty = state.baseline !== "" && signature() !== state.baseline;
      syncChrome();
    }

    function defaultName() {
      return state.providers.length ? state.providers[0].name : "";
    }

    function toWire(p) {
      var apiKey = p.keyMasked && !p.keyEdited ? MASK : p.api_key;
      return {
        name: p.name.trim(),
        base_url: p.base_url.trim(),
        api_key: apiKey,
        model: p.model.trim(),
        headers: textToHeaders(p.headersText)
      };
    }

    /* ------------------------------------------------------ banner + chrome */

    function renderBanner() {
      clear(banner);
      banner.className = "hp-banner";
      banner.setAttribute("role", state.conflict || state.error ? "alert" : "status");
      if (state.conflict) {
        banner.classList.add("hp-conflict");
        banner.append(
          h("div", { class: "hp-banner-body" },
            h("strong", { text: "Someone else changed the provider config." }),
            h("span", {
              text: state.conflict.message ||
                "The revision on the server no longer matches the one this panel loaded."
            }),
            h("span", {
              class: "hp-banner-hint",
              text: "Reload to get the current revision, then re-apply your edits. Unsaved edits in this panel are kept until you reload."
            })
          ),
          h("button", {
            class: "hp-btn hp-primary", type: "button", text: "Reload",
            onclick: function () { refresh(true); }
          })
        );
        banner.hidden = false;
        return;
      }
      if (state.error) {
        banner.classList.add("hp-error");
        banner.append(
          h("div", { class: "hp-banner-body" },
            h("strong", { text: "Provider API error" }),
            h("span", { text: state.error })
          ),
          h("button", {
            class: "hp-btn", type: "button", text: "Retry",
            onclick: function () { refresh(true); }
          })
        );
        banner.hidden = false;
        return;
      }
      if (state.notice) {
        banner.classList.add("hp-ok");
        banner.append(h("div", { class: "hp-banner-body" },
          h("span", { text: state.notice })));
        banner.hidden = false;
        return;
      }
      banner.hidden = true;
    }

    function syncChrome() {
      root.dataset.state = state.status;
      var rev = state.revision;
      var revText = rev == null ? "revision —" : "revision " + String(rev);
      if (rev != null && String(rev).length > 14) {
        revText = "revision " + String(rev).slice(0, 12) + "…";
      }
      revPill.textContent = revText;
      revPill.title = rev == null ? "" : "revision " + String(rev);
      fileCode.textContent = state.file || "";
      fileCode.hidden = !state.file;

      addBtn.disabled = busy();
      reloadBtn.disabled = busy();
      saveBtn.disabled = busy() || !state.dirty;
      saveBtn.textContent = state.status === "saving" ? "Saving…" : "Save changes";
      dirtyFlag.hidden = !state.dirty;

      if (state.status === "loading" && !state.providers.length) {
        footEl.textContent = "Loading provider config…";
      } else if (state.error && !state.providers.length) {
        footEl.textContent = "Provider config unavailable. The panel will retry on Reload.";
      } else {
        var count = state.providers.length;
        footEl.textContent = count
          ? count + (count === 1 ? " provider" : " providers")
            + " · keys stay on the server; this panel never sees a stored secret."
          : "No providers configured yet.";
      }
    }

    /* ------------------------------------------------------------ rows */

    function rowErrors(p) {
      var box = p.row && p.row.errors;
      if (!box) return;
      clear(box);
      Object.keys(p.errors).forEach(function (field) {
        box.append(h("div", { class: "hp-row-error", text: p.errors[field] }));
      });
      box.hidden = !Object.keys(p.errors).length;
    }

    function markField(p, field, bad) {
      if (!p.row) return;
      var input = p.row.fields[field];
      if (!input) return;
      input.classList.toggle("hp-invalid", !!bad);
      input.setAttribute("aria-invalid", bad ? "true" : "false");
    }

    function buildRow(p) {
      var radio = h("input", {
        type: "radio", name: "hp-default-" + uid, value: p.id,
        checked: state.default.trim().toLowerCase() === p.name.trim().toLowerCase()
      });
      radio.addEventListener("change", function () {
        if (!radio.checked) return;
        state.default = p.name;
        state.providers.forEach(function (other) {
          var isDefault = state.default.trim().toLowerCase() === other.name.trim().toLowerCase();
          if (other.row && other.row.card) other.row.card.classList.toggle("hp-is-default", isDefault);
          if (other.row && other.row.radio) other.row.radio.checked = isDefault;
        });
        syncDirty();
      });

      var nameInput = h("input", {
        type: "text", value: p.name, placeholder: "endpoint name",
        autocomplete: "off", spellcheck: "false", "aria-label": "Provider name"
      });
      var urlInput = h("input", {
        type: "text", value: p.base_url, placeholder: "http://127.0.0.1:1234 or auto://studio",
        autocomplete: "off", spellcheck: "false", "aria-label": "Base URL"
      });
      var modelInput = h("input", {
        type: "text", value: p.model, placeholder: "model id (optional)",
        autocomplete: "off", spellcheck: "false", "aria-label": "Model"
      });
      var keyInput = h("input", {
        type: "password", value: p.keyMasked && !p.keyEdited ? MASK : p.api_key,
        placeholder: "no key", autocomplete: "off", spellcheck: "false",
        "aria-label": "API key"
      });
      var headersInput = h("textarea", {
        rows: "3", value: p.headersText, spellcheck: "false",
        placeholder: "X-Header: value\n(one per line, optional)",
        "aria-label": "Extra headers"
      });

      var keyHint = h("small", { class: "hp-hint" });
      var cardName = h("span", { class: "hp-card-name", text: p.name.trim() || "new provider" });
      var keyBadge = h("span", { class: "hp-badge" });

      var errors = h("div", { class: "hp-row-errors", role: "alert", hidden: true });

      function setKeyHint() {
        var parts = [];
        if (p.keyMasked && !p.keyEdited) {
          parts.push("Stored key unchanged — type a new key to replace it.");
        } else if (p.keyMasked && p.keyEdited) {
          parts.push("A new key will replace the stored one on save.");
        } else if (p.api_key) {
          parts.push("A new key will be stored on save.");
        } else {
          parts.push("No key — local endpoints usually need none.");
        }
        if (p.keyMasked && p.name.trim().toLowerCase() !== p.origName.toLowerCase()) {
          parts.push("Renaming clears the stored key on save (the server matches keys by name).");
        }
        keyHint.textContent = parts.join(" ");
        keyBadge.hidden = !p.keyMasked || p.keyEdited;
        keyBadge.textContent = "key stored";
      }

      nameInput.addEventListener("input", function () {
        var before = p.name;
        p.name = nameInput.value;
        cardName.textContent = p.name.trim() || "new provider";
        if (state.default === before) state.default = p.name;
        setKeyHint();
        markField(p, "name", false);
        syncDirty();
      });
      urlInput.addEventListener("input", function () {
        p.base_url = urlInput.value;
        markField(p, "base_url", false);
        syncDirty();
      });
      modelInput.addEventListener("input", function () {
        p.model = modelInput.value;
        syncDirty();
      });
      headersInput.addEventListener("input", function () {
        p.headersText = headersInput.value;
        markField(p, "headers", false);
        syncDirty();
      });
      keyInput.addEventListener("focus", function () {
        if (p.keyMasked && !p.keyEdited) keyInput.select();
      });
      keyInput.addEventListener("input", function () {
        var value = keyInput.value;
        if (p.keyMasked && value !== MASK) {
          // The user typed over the mask: treat what they typed as the key.
          if (value.slice(0, MASK.length) === MASK) value = value.slice(MASK.length);
          p.keyMasked = false;
          p.keyEdited = true;
          p.api_key = value;
          keyInput.value = value;
        } else {
          p.api_key = value;
          p.keyEdited = true;
        }
        setKeyHint();
        syncDirty();
      });

      var card = h("article", { class: "hp-card" });
      card.dataset.provider = p.id;
      card.classList.toggle(
        "hp-is-default",
        state.default.trim().toLowerCase() === p.name.trim().toLowerCase()
      );
      card.append(
        h("header", { class: "hp-card-head" },
          h("label", { class: "hp-default" }, radio, h("span", { text: "default" })),
          cardName,
          keyBadge,
          h("button", {
            class: "hp-btn hp-danger", type: "button", text: "Delete",
            "aria-label": "Delete provider " + (p.name || ""),
            onclick: function () { removeProvider(p); }
          })
        ),
        h("div", { class: "hp-grid" },
          label("Name", nameInput),
          label("Model", modelInput),
          label("Base URL", urlInput, "Server root, or auto://studio to follow Studio."),
          label("API key", keyInput),
          label("Headers", headersInput, 'One "Name: value" pair per line.')
        ),
        keyHint,
        errors
      );

      p.row = {
        fields: { name: nameInput, base_url: urlInput, model: modelInput, api_key: keyInput, headers: headersInput },
        errors: errors,
        card: card,
        radio: radio,
        nameInput: nameInput,
        keyInput: keyInput,
        keyHint: keyHint,
        setKeyHint: setKeyHint
      };

      setKeyHint();
      if (Object.keys(p.errors).length) rowErrors(p);
      return card;
    }

    function renderList() {
      clear(listEl);
      if (!state.providers.length) {
        listEl.append(h("div", { class: "hp-empty" },
          h("p", { text: state.status === "loading"
            ? "Loading providers…"
            : "No providers configured. Add one to point Hive at an OpenAI-compatible endpoint." }),
          !busy() ? h("button", {
            class: "hp-btn hp-add", type: "button", text: "+ Add provider",
            onclick: addProvider
          }) : null
        ));
        return;
      }
      state.providers.forEach(function (p) {
        listEl.append(buildRow(p));
      });
    }

    function render() {
      renderBanner();
      syncChrome();
      renderList();
    }

    /* -------------------------------------------------------- mutations */

    function addProvider() {
      if (busy()) return;
      var p = blankProvider(state.providers.length + 1);
      var names = {};
      state.providers.forEach(function (other) { names[other.name.trim().toLowerCase()] = true; });
      var n = state.providers.length + 1;
      while (names[p.name.toLowerCase()]) {
        n += 1;
        p.name = "provider-" + n;
      }
      state.providers.push(p);
      if (!state.default) state.default = p.name;
      syncDirty();
      renderList();
      var row = p.row;
      if (row) {
        row.nameInput.focus();
        row.nameInput.select();
      }
    }

    function removeProvider(p) {
      if (busy()) return;
      var wasDefault = state.default === p.name;
      state.providers = state.providers.filter(function (other) { return other !== p; });
      if (wasDefault) state.default = state.providers.length ? state.providers[0].name : "";
      syncDirty();
      renderList();
    }

    /* --------------------------------------------------------- validate */

    function parseHeaders(p) {
      try {
        return textToHeaders(p.headersText);
      } catch (err) {
        p.errors.headers = err.message;
        return null;
      }
    }

    function validate() {
      var seen = {};
      var bad = 0;
      state.providers.forEach(function (p) {
        p.errors = {};
        var name = p.name.trim();
        var lower = name.toLowerCase();
        if (!name) {
          p.errors.name = "A name is required.";
        } else if (seen[lower]) {
          p.errors.name = 'Another provider is already called "' + name + '".';
        } else {
          seen[lower] = true;
        }
        var url = p.base_url.trim();
        if (!url) {
          p.errors.base_url = "A base URL is required.";
        } else if (!/^(https?:\/\/|auto:\/\/)/i.test(url)) {
          p.errors.base_url = "Use http://, https:// or auto://.";
        }
        parseHeaders(p);
        if (Object.keys(p.errors).length) bad += 1;
      });
      if (state.providers.length && state.default && !seen[state.default.trim().toLowerCase()]) {
        // the default no longer exists (renamed/removed outside the panel)
        state.default = defaultName();
      }
      return bad;
    }

    /* ------------------------------------------------------------ fetch */

    function request(method, body) {
      var init = {
        method: method,
        cache: "no-store",
        headers: { accept: "application/json" }
      };
      if (body !== undefined) {
        init.headers["content-type"] = "application/json";
        init.body = JSON.stringify(body);
      }
      return fetch(endpoint, init).then(function (res) {
        return res.json().catch(function () { return null; }).then(function (data) {
          return { ok: res.ok, status: res.status, body: data };
        });
      });
    }

    function applyServerConfig(data) {
      data = data || {};
      state.revision = data.revision == null ? null : data.revision;
      state.default = typeof data.default === "string" ? data.default : "";
      state.file = typeof data.file === "string" ? data.file : (state.file || "");
      state.providers = (Array.isArray(data.providers) ? data.providers : []).map(normalize);
      // normalise the default's casing to the provider it names (the API
      // matches names case-insensitively, the radios do not)
      var wanted = state.default.trim().toLowerCase();
      var match = state.providers.filter(function (p) {
        return p.name.trim().toLowerCase() === wanted;
      })[0];
      if (match) state.default = match.name;
      else if (state.providers.length) state.default = state.providers[0].name;
      else state.default = "";
      state.baseline = signature();
      state.dirty = false;
      state.error = null;
      state.conflict = null;
    }

    function refresh(force) {
      if (state.dirty && !force && typeof global.confirm === "function") {
        if (!global.confirm("Discard unsaved provider changes and reload from the server?")) {
          return Promise.resolve(state);
        }
      }
      state.status = "loading";
      state.error = null;
      state.conflict = null;
      state.notice = null;
      render();
      return request("GET").then(function (res) {
        if (!res.ok) {
          throw new Error(detailOf(res.body) || ("GET " + endpoint + " returned " + res.status));
        }
        applyServerConfig(res.body);
        state.status = "ready";
      }).catch(function (err) {
        state.status = "error";
        state.error = (err && err.message) ? err.message : String(err);
      }).then(function () {
        render();
        return state;
      });
    }

    function save() {
      if (busy()) return Promise.resolve(state);
      var bad = validate();
      if (bad) {
        state.error = null;
        state.conflict = null;
        state.notice = null;
        render();
        renderBanner();
        return Promise.resolve(state);
      }
      var body;
      try {
        body = {
          providers: state.providers.map(toWire),
          default: state.default,
          persist: true,
          expectedRevision: state.revision
        };
      } catch (err) {
        state.error = (err && err.message) ? err.message : String(err);
        render();
        return Promise.resolve(state);
      }

      state.status = "saving";
      state.error = null;
      state.conflict = null;
      state.notice = null;
      syncChrome();
      renderBanner();

      return request("POST", body).then(function (res) {
        if (res.status === 409) {
          state.status = "ready";
          state.conflict = {
            message: detailOf(res.body) ||
              ("The server expected a different revision than " + state.revision + ".")
          };
          return;
        }
        if (!res.ok) {
          state.status = "ready";
          state.error = detailOf(res.body) || ("Save failed with HTTP " + res.status + ".");
          return;
        }
        applyServerConfig(res.body || {});
        state.status = "ready";
        state.notice = "Saved"
          + (res.body && res.body.persisted_to ? " to " + res.body.persisted_to : "")
          + (state.revision == null ? "" : " · revision " + state.revision);
      }).catch(function (err) {
        state.status = "error";
        state.error = "Could not reach the provider API: "
          + ((err && err.message) ? err.message : String(err));
      }).then(function () {
        render();
        return state;
      });
    }

    /* ------------------------------------------------------------ wire */

    saveBtn.addEventListener("click", function () { save(); });
    reloadBtn.addEventListener("click", function () { refresh(false); });
    addBtn.addEventListener("click", addProvider);

    render();
    var ready = refresh(true);

    return {
      refresh: function () { return refresh(false); },
      save: save,
      state: state,
      ready: ready,
      destroy: function () { clear(root); }
    };
  }

  global.HiveProvidersPanel = { mount: mount };
})(typeof window !== "undefined" ? window : globalThis);
