/* Suggestions inbox. Defines the panel at import time; network work starts only
 * when mount() is called. Suggestions are stored for human review and never run. */
(function (global) {
  "use strict";

  var ENDPOINT = "/api/hive_build/suggestions";

  function element(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function request(url, options) {
    return global.fetch(url, options).then(function (response) {
      return response.json().catch(function () { return {}; }).then(function (body) {
        if (!response.ok) throw new Error(body.error || ("Request failed (" + response.status + ")"));
        return body;
      });
    });
  }

  function mount(root) {
    if (!root || !root.appendChild) throw new TypeError("mount requires a DOM element");

    var state = { suggestions: [], summary: { total: 0, by_status: {} }, loading: false, error: "", unavailable: "" };
    var shell = element("section", "hive-suggestions");
    var header = element("header", "hs-head");
    var heading = element("div", "hs-heading");
    heading.appendChild(element("h2", "hs-title", "Suggestion inbox"));
    heading.appendChild(element("p", "hs-subtitle", "Planner ideas for human review — nothing is applied automatically."));
    var refreshButton = element("button", "hs-button", "Refresh");
    refreshButton.type = "button";
    header.appendChild(heading);
    header.appendChild(refreshButton);

    var message = element("p", "hs-message");
    message.setAttribute("role", "status");
    var list = element("div", "hs-list");
    var form = element("form", "hs-form");
    form.appendChild(element("h3", "hs-form-title", "Add a suggestion"));

    function addField(labelText, name, multiline, required) {
      var label = element("label", "hs-field");
      var caption = element("span", "hs-label", labelText);
      var input = document.createElement(multiline ? "textarea" : "input");
      input.name = name;
      input.required = Boolean(required);
      if (!multiline) input.type = "text";
      if (name === "source") input.placeholder = "manual";
      label.appendChild(caption);
      label.appendChild(input);
      form.appendChild(label);
      return input;
    }

    addField("Title", "title", false, true);
    addField("Problem", "problem", true, true);
    addField("Proposal", "proposal", true, true);
    var sourceInput = addField("Source (optional)", "source", false, false);
    var submit = element("button", "hs-button hs-button--primary", "Save suggestion");
    submit.type = "submit";
    form.appendChild(submit);

    shell.appendChild(header);
    shell.appendChild(message);
    shell.appendChild(form);
    shell.appendChild(list);
    root.replaceChildren(shell);

    function render() {
      list.replaceChildren();
      message.textContent = state.error || state.unavailable ||
        (state.loading ? "Loading suggestions…" : "");
      message.className = "hs-message" +
        (state.error ? " hs-message--error" : state.unavailable ? " hs-message--empty" : "");
      refreshButton.disabled = state.loading;
      submit.disabled = state.loading;

      if (!state.loading && !state.error && !state.unavailable && state.suggestions.length === 0) {
        message.textContent = "No suggestions yet. Add one above or refresh later.";
        message.className = "hs-message hs-message--empty";
      }

      state.suggestions.forEach(function (suggestion) {
        var card = element("article", "hs-card");
        var top = element("div", "hs-card-head");
        top.appendChild(element("h3", "hs-card-title", suggestion.title || "Untitled suggestion"));
        top.appendChild(element("span", "hs-status hs-status--" + (suggestion.status || "raw"), suggestion.status || "raw"));
        card.appendChild(top);
        card.appendChild(detail("Problem", suggestion.problem));
        card.appendChild(detail("Proposal", suggestion.proposal));
        var metadata = element("div", "hs-meta");
        metadata.appendChild(meta("Effort", suggestion.effort || "—"));
        metadata.appendChild(meta("Risk", suggestion.risk || "—"));
        metadata.appendChild(meta("Impact", suggestion.impact || "—"));
        card.appendChild(metadata);
        if (suggestion.source) card.appendChild(element("p", "hs-source", "Source: " + suggestion.source));

        if (suggestion.status === "raw") {
          var actions = element("div", "hs-actions");
          actions.appendChild(statusButton(suggestion, "accepted", "Accept"));
          actions.appendChild(statusButton(suggestion, "rejected", "Reject"));
          card.appendChild(actions);
        }
        list.appendChild(card);
      });
    }

    function detail(label, value) {
      var block = element("div", "hs-detail");
      block.appendChild(element("h4", "hs-detail-label", label));
      block.appendChild(element("p", "hs-detail-text", value || "—"));
      return block;
    }

    function meta(label, value) {
      var block = element("span", "hs-meta-item");
      block.appendChild(element("b", "", label + ": "));
      block.appendChild(document.createTextNode(String(value)));
      return block;
    }

    function statusButton(suggestion, status, label) {
      var button = element("button", "hs-button" + (status === "accepted" ? " hs-button--accept" : " hs-button--reject"), label);
      button.type = "button";
      button.addEventListener("click", function () {
        state.error = "";
        state.loading = true;
        render();
        request(ENDPOINT + "/" + encodeURIComponent(suggestion.id), {
          method: "PATCH",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ status: status })
        }).then(refresh).catch(fail);
      });
      return button;
    }

    function fail(error) {
      state.loading = false;
      state.error = error && error.message ? error.message : String(error || "Unknown error");
      render();
    }

    function refresh() {
      state.loading = true;
      state.error = "";
      render();
      return request(ENDPOINT + "?limit=50").then(function (body) {
        state.suggestions = Array.isArray(body.suggestions) ? body.suggestions : [];
        state.summary = body.summary || { total: state.suggestions.length, by_status: {} };
        state.unavailable = body.unavailable || "";
        state.loading = false;
        render();
        return state.suggestions;
      }).catch(function (error) {
        fail(error);
        throw error;
      });
    }

    refreshButton.addEventListener("click", function () { refresh().catch(function () {}); });
    form.addEventListener("submit", function (event) {
      event.preventDefault();
      if (!form.reportValidity()) return;
      var data = new FormData(form);
      var record = {
        title: data.get("title"),
        problem: data.get("problem"),
        proposal: data.get("proposal"),
        source: String(data.get("source") || "").trim() || "manual"
      };
      state.loading = true;
      state.error = "";
      render();
      request(ENDPOINT, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(record)
      }).then(function () {
        form.reset();
        sourceInput.value = "";
        return refresh();
      }).catch(fail);
    });

    render();
    refresh().catch(function () {});
    return {
      refresh: refresh,
      state: function () {
        return {
          suggestions: state.suggestions.map(function (item) { return Object.assign({}, item); }),
          summary: Object.assign({}, state.summary),
          loading: state.loading,
          error: state.error
        };
      }
    };
  }

  global.HiveSuggestionsPanel = { mount: mount };
}(window));
