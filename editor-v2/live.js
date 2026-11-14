// Ornis editor-v2 — live editor bridge to the editor-backend.
//
// Replaces the static mock data with the authoritative ECS snapshot served by
// `GET /api/scene` (with `GET /api/status` and the WS `/api/events` stream as
// the push channel, cursor polling as fallback). Hierarchy, inspector and the
// console render that snapshot; edits go back through `POST /api/command`
// (`create_entity`, `destroy_entity`, `set_component`, `select`,
// `save_scene`, `load_scene`) with the same ACK/`CommandCompleted`
// contract as `editor/`.
//
// The 3D viewport is `viewport.js` (WASM); this module only owns the DOM
// around it. Selection is authoritative: `editor.selected` from the
// snapshot drives the tree/inspector, tree clicks send `select`, and the
// WASM viewport sends the same command on its own picking.
(function () {
  "use strict";

  var POLL_MS = 1500;
  var RECENT_KEY = "ornis.recentProjects";
  var projectName = "demo";
  // Scene asset the live world was last loaded from/saved to. Recents store
  // it so the welcome sheet can reopen a project instead of only renaming it.
  var currentScenePath = "editor/scene.ron";

  var lastScene = null;
  var lastSceneSequence = 0;
  var lastStatusSequence = 0;
  var eventCursor = 0;
  var eventSocket = null;
  var eventSocketConnected = false;
  var nextRequestId = 1;
  var pendingCommands = new Map();
  var selectedKey = null;
  var lastInspectorKey = "";
  var lastEntityCount = -1;

  // Whether a keystroke targets a text field: viewport shortcuts must not
  // hijack the browser undo/redo inside inputs, textareas and rich-text
  // surfaces.
  function isEditingTarget(target) {
    if (!target) return false;
    var tag = target.tagName ? target.tagName.toLowerCase() : "";
    if (tag === "input" || tag === "textarea" || tag === "select") return true;
    return !!target.isContentEditable;
  }

  // ── DOM ────────────────────────────────────────────────────────────
  var tree = document.getElementById("tree");
  var treeFilter = document.getElementById("tree-filter");
  var inspector = document.getElementById("inspector");
  var nameInput = document.getElementById("inspector-name");
  var inspectorIcon = document.getElementById("inspector-icon");
  var previewIcon = document.getElementById("preview-icon");
  var consoleLog = document.getElementById("console-log");
  var tabStrip = document.getElementById("scene-tabs");
  var projectEl = document.querySelector(".center-top .project strong");

  // ── helpers ────────────────────────────────────────────────────────
  function icon(name) { return "icons/" + name + ".svg#icon"; }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function svgIcon(name) {
    var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    var use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", icon(name));
    svg.appendChild(use);
    return svg;
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function debounce(fn, ms) {
    var timer = null;
    function debounced() {
      clearTimeout(timer);
      timer = setTimeout(fn, ms);
    }
    debounced.cancel = function () {
      clearTimeout(timer);
      timer = null;
    };
    return debounced;
  }

  function entityKey(entity) { return entity.id + ":" + entity.generation; }

  function findEntity(scene, key) {
    var entities = (scene && scene.entities) || [];
    for (var i = 0; i < entities.length; i++) {
      if (entityKey(entities[i]) === key) return entities[i];
    }
    return null;
  }

  function componentsOf(entity) { return (entity && entity.components) || {}; }

  function entityNameOf(entity) {
    var name = componentsOf(entity).Name;
    return typeof name === "string" && name ? name : null;
  }

  function entityLabel(entity) {
    return entityNameOf(entity) || ("Entity " + entity.id + " (gen " + entity.generation + ")");
  }

  function entityIcon(entity) {
    var c = componentsOf(entity);
    if (c.Light) return "lightbulb";
    if (c.Camera) return "camera-wide";
    if (c.Mesh) return "cube-outline";
    if (c.SkelPlayer || c.Animator) return "sphere-toon";
    return "circle";
  }

  function rgbToHex(rgb) {
    return "#" + rgb.map(function (c) {
      return Math.round(clamp(c, 0, 1) * 255).toString(16).padStart(2, "0");
    }).join("");
  }

  function hexToRgb(hex) {
    return [1, 3, 5].map(function (i) { return parseInt(hex.slice(i, i + 2), 16) / 255; });
  }

  function formatTime() {
    var d = new Date();
    return d.toTimeString().slice(0, 8);
  }

  function log(message, isError) {
    if (!consoleLog) return;
    var row = el("div");
    var tag = el("span", "muted", "[" + (isError ? "error" : "live") + "]");
    row.appendChild(tag);
    row.appendChild(document.createTextNode(" " + message));
    consoleLog.appendChild(row);
    while (consoleLog.children.length > 200) consoleLog.removeChild(consoleLog.firstChild);
    consoleLog.scrollTop = consoleLog.scrollHeight;
  }

  // ── transport ──────────────────────────────────────────────────────
  function fetchJson(url, options) {
    return fetch(url, options).then(function (res) {
      if (!res.ok) throw new Error(url + ": HTTP " + res.status);
      return res.json();
    });
  }

  function sendCommand(type, data) {
    var requestId = nextRequestId++;
    return fetchJson("/api/command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type: type, request_id: requestId, data: data || {} })
    }).then(function (ack) {
      if (ack && ack.accepted === false) {
        throw new Error(ack.error || ("command rejected: " + type));
      }
      if (ack && ack.accepted === true && typeof ack.request_id === "number") {
        pendingCommands.set(ack.request_id, { type: type });
      }
      return ack;
    }).catch(function (err) {
      log(type + " failed: " + err.message, true);
      throw err;
    });
  }

  function setComponent(entity, component, value) {
    return sendCommand("set_component", {
      id: entity.id,
      generation: entity.generation,
      component: component,
      value: value
    }).then(afterMutation);
  }

  function afterMutation() {
    refreshScene();
    refreshStatus();
  }

  // ── hierarchy ──────────────────────────────────────────────────────
  function row(label, iconName, opts) {
    opts = opts || {};
    var b = el("button", "tree-row" + (opts.selected ? " selected" : ""));
    var caret = el("span", "caret" + (opts.collapsible ? " clickable" + (opts.collapsed ? "" : " open") : ""));
    if (opts.collapsible && opts.onToggle) {
      caret.addEventListener("click", function (ev) { ev.stopPropagation(); opts.onToggle(); });
    }
    b.appendChild(caret);
    b.appendChild(svgIcon(iconName));
    b.appendChild(el("span", null, label));
    return b;
  }

  var collapsed = {};

  function renderTree() {
    if (!tree) return;
    var q = treeFilter ? treeFilter.value.trim().toLowerCase() : "";
    tree.innerHTML = "";
    var entities = (lastScene && lastScene.entities) || [];
    if (!entities.length) {
      tree.appendChild(el("div", "muted", lastScene ? "Scene is empty — press + to create an entity" : "Waiting for the engine…"));
      return;
    }
    entities.forEach(function (entity) {
      // Editor chrome (gizmo handles) lives in the same snapshot but is not
      // scene content: keep it out of the hierarchy and inspector.
      if (entity.editor && entity.editor.editor_only) return;
      var key = entityKey(entity);
      var label = entityLabel(entity);
      if (q && label.toLowerCase().indexOf(q) < 0) return;
      var r = row(label, entityIcon(entity), {
        selected: key === selectedKey,
        collapsible: true,
        collapsed: !!collapsed[key],
        onToggle: function () { collapsed[key] = !collapsed[key]; renderTree(); }
      });
      r.dataset.id = key;
      r.title = "id " + entity.id + ", generation " + entity.generation;
      r.addEventListener("click", function () { selectEntity(key); });
      r.addEventListener("dblclick", function () { openEntityTab(key); });
      tree.appendChild(r);
      if (!collapsed[key]) {
        var kids = el("div", "tree-children");
        Object.keys(componentsOf(entity)).forEach(function (component) {
          var cr = row(component, componentIcon(component), {});
          cr.addEventListener("click", function () { selectEntity(key); });
          kids.appendChild(cr);
        });
        tree.appendChild(kids);
      }
    });
    if (!tree.children.length) tree.appendChild(el("div", "muted", "No entities match the filter."));
  }

  function componentIcon(component) {
    switch (component) {
      case "Transform": return "transform-gizmo";
      case "Material": return "material-toon";
      case "Mesh": return "cube-outline";
      case "Collider": return "weight";
      case "Skeleton":
      case "SkinnedMesh": return "sphere-toon";
      default: return "circle";
    }
  }

  function selectEntity(key) {
    selectedKey = key;
    lastInspectorKey = "";
    // Update the existing rows instead of re-rendering: rebuilding on click
    // would replace the node mid-gesture and swallow the second click of a
    // double-click (open-in-tab).
    updateTreeSelection();
    syncInspector();
    renderTabs();
    onSelectionChanged(key);
    // The authoritative world owns the selection; an empty payload clears.
    var entity = key && lastScene ? findEntity(lastScene, key) : null;
    var data = entity ? { id: entity.id, generation: entity.generation } : {};
    sendCommand("select", data).catch(function () {});
  }

  function clearSelection() {
    if (!selectedKey) return;
    selectedKey = null;
    lastInspectorKey = "";
    updateTreeSelection();
    syncInspector();
    renderTabs();
    onSelectionChanged(null);
    // Empty payload = authoritative clear (same as a viewport miss).
    sendCommand("select", {}).catch(function () {});
  }

  function updateTreeSelection() {
    if (!tree) return;
    tree.querySelectorAll(".tree-row.selected").forEach(function (row) {
      row.classList.remove("selected");
    });
    if (!selectedKey) return;
    var row = tree.querySelector('.tree-row[data-id="' + selectedKey + '"]');
    if (row) row.classList.add("selected");
  }

  // ── inspector ──────────────────────────────────────────────────────
  function inspectorHasFocus() {
    if (!inspector) return false;
    var active = document.activeElement;
    return !!active && active.tagName === "INPUT" && inspector.contains(active);
  }

  function syncInspector() {
    if (!lastScene) return;
    var entity = selectedKey ? findEntity(lastScene, selectedKey) : null;
    var key = entity
      ? JSON.stringify([entity.id, entity.generation, entity.components])
      : "none";
    if (key === lastInspectorKey) return;
    if (inspectorHasFocus()) return;
    lastInspectorKey = key;
    renderInspector(entity);
  }

  function section(title, open, iconName) {
    var d = el("details", "section");
    d.open = !!open;
    var s = el("summary");
    s.appendChild(svgIcon(iconName || "circle"));
    s.appendChild(document.createTextNode(title));
    d.appendChild(s);
    var body = el("div", "body");
    d.appendChild(body);
    return { root: d, body: body };
  }

  function field(label, control) {
    var f = el("div", "field");
    f.appendChild(el("span", null, label));
    f.appendChild(control);
    return f;
  }

  function vecRow(values, onChange) {
    var wrap = el("div", "vec");
    values.forEach(function (value, i) {
      var input = document.createElement("input");
      input.type = "number";
      input.step = "0.1";
      input.value = value;
      input.addEventListener("change", function () {
        onChange(i, parseFloat(input.value) || 0);
      });
      var cell = el("span", "vec-cell");
      cell.appendChild(input);
      wrap.appendChild(cell);
    });
    return wrap;
  }

  function sliderRow(label, value, min, max, step, fmt, onChange) {
    var r = el("div", "slider-row");
    r.appendChild(el("span", null, label));
    var input = document.createElement("input");
    input.type = "range";
    input.min = min;
    input.max = max;
    input.step = step;
    input.value = value;
    var out = el("span", null, fmt(value));
    input.addEventListener("input", function () {
      out.textContent = fmt(parseFloat(input.value));
      onChange(parseFloat(input.value));
    });
    r.appendChild(input);
    r.appendChild(out);
    return r;
  }

  function renderInspector(entity) {
    if (!inspector) return;
    inspector.innerHTML = "";
    if (nameInput) nameInput.value = entity ? (entityNameOf(entity) || "") : "";
    if (inspectorIcon) inspectorIcon.setAttribute("href", icon(entity ? entityIcon(entity) : "cube-outline"));
    if (previewIcon) previewIcon.setAttribute("href", icon(entity ? entityIcon(entity) : "cube-outline"));
    if (!entity) {
      inspector.appendChild(el("p", "muted", "Select an entity."));
      return;
    }

    var components = componentsOf(entity);

    if (components.Transform) {
      var t = section("Transform", true, "transform-gizmo");
      var translation = (components.Transform.translation || [0, 0, 0]).slice(0, 3);
      var scale = (components.Transform.scale || [1, 1, 1]).slice(0, 3);
      var sendTransform = debounce(function () {
        setComponent(entity, "Transform", {
          translation: translation.slice(),
          rotation: components.Transform.rotation || [0, 0, 0, 1],
          scale: scale.slice()
        }).catch(function () {});
      }, 250);
      t.body.appendChild(field("Position", vecRow(translation, function (i, v) {
        translation[i] = v;
        sendTransform();
      })));
      t.body.appendChild(field("Scale", vecRow(scale, function (i, v) {
        scale[i] = v;
        sendTransform();
      })));
      var rotation = components.Transform.rotation;
      if (Array.isArray(rotation)) {
        var ro = el("div", "muted", "Rotation (xyzw): " + rotation.join(", "));
        t.body.appendChild(ro);
      }
      inspector.appendChild(t.root);
    }

    if (components.Material) {
      var material = components.Material;
      var variantName = Object.keys(material)[0];
      var variant = variantName ? material[variantName] : null;
      if (variant) {
        var m = section("Material (" + variantName + ")", true, "material-toon");
        var baseColor = Array.isArray(variant.base_color) ? variant.base_color.slice(0, 3) : [1, 1, 1];
        var roughness = typeof variant.roughness === "number" ? variant.roughness : 0.5;
        var sendMaterial = debounce(function () {
          var fields = {};
          Object.keys(variant).forEach(function (key) { fields[key] = variant[key]; });
          fields.base_color = baseColor.slice();
          if ("roughness" in fields) fields.roughness = roughness;
          var value = {};
          value[variantName] = fields;
          setComponent(entity, "Material", value).catch(function () {});
        }, 250);

        var colorInput = document.createElement("input");
        colorInput.type = "color";
        colorInput.value = rgbToHex(baseColor);
        colorInput.addEventListener("input", function () {
          baseColor = hexToRgb(colorInput.value);
          sendMaterial();
        });
        m.body.appendChild(field("Base color", colorInput));
        m.body.appendChild(sliderRow("Roughness", roughness, 0, 1, 0.01,
          function (v) { return v.toFixed(2); },
          function (v) { roughness = v; sendMaterial(); }));
        inspector.appendChild(m.root);
      }
    }

    Object.keys(components).forEach(function (name) {
      if (name === "Transform" || name === "Material" || name === "Name") return;
      var s = section(name, false, componentIcon(name));
      var pre = el("pre", "muted");
      pre.textContent = JSON.stringify(components[name], null, 2);
      s.body.appendChild(pre);
      inspector.appendChild(s.root);
    });

    var del = el("button", "add-component", "Delete entity");
    del.addEventListener("click", deleteSelected);
    inspector.appendChild(del);
  }

  function deleteSelected() {
    var entity = selectedKey && lastScene ? findEntity(lastScene, selectedKey) : null;
    if (!entity) return;
    selectedKey = null;
    sendCommand("destroy_entity", { id: entity.id, generation: entity.generation })
      .then(afterMutation).catch(function () {});
  }

  // ── scene tabs ─────────────────────────────────────────────────────
  var openTabs = [{ id: "root", root: true }];

  function tabLabel(tab) {
    if (tab.root) return projectName;
    var entity = lastScene ? findEntity(lastScene, tab.id) : null;
    return entity ? entityLabel(entity) : tab.id;
  }

  function renderTabs() {
    if (!tabStrip) return;
    tabStrip.innerHTML = "";
    openTabs.forEach(function (tab) {
      var node = el("div", "scene-tab" + (tab.root ? " root" : "") +
        (tab.id === (selectedKey || "root") ? " active" : ""));
      node.dataset.id = tab.id;
      node.appendChild(svgIcon(tab.root ? "cube-outline" : entityIcon(findEntity(lastScene, tab.id) || {})));
      node.appendChild(el("span", "label", tabLabel(tab)));
      node.addEventListener("click", function () {
        if (tab.root) clearSelection();
        else selectEntity(tab.id);
      });
      if (!tab.root) {
        var close = el("button", "close", "×");
        close.title = "Close tab";
        close.addEventListener("click", function (ev) { ev.stopPropagation(); closeTab(tab.id); });
        node.appendChild(close);
      }
      tabStrip.appendChild(node);
    });
  }

  function openEntityTab(key) {
    if (!findEntity(lastScene, key)) return;
    if (!openTabs.some(function (t) { return t.id === key; })) openTabs.push({ id: key });
    selectEntity(key);
  }

  function closeTab(key) {
    openTabs = openTabs.filter(function (t) { return t.id !== key; });
    // Closing a tab does not deselect: selection belongs to the engine.
    renderTabs();
  }

  // ── scenes / projects ──────────────────────────────────────────────
  function setProjectName(name, path) {
    if (!name) return;
    if (path) currentScenePath = path;
    projectName = name;
    if (projectEl) projectEl.textContent = name;
    renderTabs();
    rememberProject(name, currentScenePath);
  }

  function rememberProject(name, path) {
    try {
      var recents = JSON.parse(localStorage.getItem(RECENT_KEY) || "[]");
      if (!Array.isArray(recents)) recents = [];
      recents = recents.filter(function (p) { return p && p.name !== name; });
      recents.unshift({ id: name, name: name, path: path || name, date: "just now" });
      localStorage.setItem(RECENT_KEY, JSON.stringify(recents.slice(0, 8)));
    } catch (e) { /* storage may be disabled; recents are best-effort */ }
  }

  function openScene(path, name) {
    if (!path) return Promise.resolve();
    return sendCommand("load_scene", { path: path }).then(function () {
      setProjectName(name || path.split("/").pop().replace(/\.(ron|glb|gltf)$/i, ""), path);
      afterMutation();
    }).catch(function () {});
  }

  function promptOpenScene() {
    var path = window.prompt("Scene asset path (.ron, .glb, .gltf):", "editor/scene.ron");
    if (path === null) return;
    path = path.trim();
    if (path) openScene(path);
  }

  function newEmptyProject() {
    return openScene("assets/templates/empty.ron", "untitled").then(function () {
      // A new project is not the template file: leave the path unset so the
      // first save asks where to write instead of overwriting the template.
      currentScenePath = null;
      rememberProject(projectName, null);
      log("new empty project — ⌘S to save");
    }).catch(function () {});
  }

  function saveScene() {
    var path = currentScenePath;
    if (!path) {
      path = window.prompt("Save scene as (under assets/ or editor/):", "assets/untitled.ron");
      if (path === null) return;
      path = path.trim();
      if (!path) return;
    }
    sendCommand("save_scene", { path: path }).then(function () {
      currentScenePath = path;
      rememberProject(projectName, path);
      log("scene saved to " + path + " (v" + ((lastScene && lastScene.version) || "?") + ")");
    }).catch(function () {});
  }

  // ── snapshots / events ─────────────────────────────────────────────
  function refreshScene() {
    return fetchJson("/api/scene").then(function (scene) {
      var sequence = scene && scene.sequence;
      if (typeof sequence === "number" && sequence < lastSceneSequence) return;
      if (typeof sequence === "number") lastSceneSequence = sequence;
      lastScene = scene;
      // `editor.selected` is the source of truth: the viewport replica
      // writes it through `select`, and tree clicks are confirmed here.
      var serverKey = null;
      (scene.entities || []).forEach(function (entity) {
        if (!serverKey && entity.editor && entity.editor.selected) serverKey = entityKey(entity);
      });
      if (serverKey !== selectedKey) {
        selectedKey = serverKey;
        lastInspectorKey = "";
      }
      renderTree();
      syncInspector();
      renderTabs();
      onSelectionChanged(selectedKey);
    }).catch(function () { /* engine offline — keep the last state */ });
  }

  // Hook for the viewport overlay: `key` mirrors `editor.selected` after
  // every snapshot (the replica/welcome code can override this).
  function onSelectionChanged(key) {
    if (window.OrnisSelection && typeof window.OrnisSelection.changed === "function") {
      window.OrnisSelection.changed(key);
    }
  }

  function refreshStatus() {
    return fetchJson("/api/status").then(function (status) {
      var sequence = status && status.sequence;
      if (typeof sequence === "number" && sequence < lastStatusSequence) return;
      if (typeof sequence === "number") lastStatusSequence = sequence;
      var count = status && typeof status.entity_count === "number" ? status.entity_count : 0;
      if (count !== lastEntityCount) {
        lastEntityCount = count;
        log("entities: " + count + " (v" + (status.version || "?") + ")");
      }
      if (status && typeof status.mode === "string") setEditorMode(status.mode);
    }).catch(function () { /* engine offline — keep the last state */ });
  }

  function handleEvents(events) {
    events.forEach(function (ev) {
      if (ev && typeof ev.sequence === "number" && ev.sequence > eventCursor) {
        eventCursor = ev.sequence;
      }
      var gap = ev && ev.EventGap;
      if (gap) {
        eventCursor = Math.max(0, (gap.oldest || 1) - 1);
        refreshScene();
        refreshStatus();
        return;
      }
      var completed = ev && ev.CommandCompleted;
      if (completed) {
        var pending = pendingCommands.get(completed.request_id);
        pendingCommands.delete(completed.request_id);
        if (!completed.success) {
          log((completed.command || (pending && pending.type) || "command") + ": " +
            (completed.error || "failed"), true);
        }
        return;
      }
      var custom = ev && ev.CustomEvent;
      if (!custom) return;
      var data = {};
      try { data = JSON.parse(custom.json_data); } catch (e) { return; }
      if (custom.cmd_type === "scene_saved") {
        log("saved " + (data.path || "scene") + " (v" + data.version + ")");
      } else if (custom.cmd_type === "scene_loaded") {
        log("loaded " + (data.path || "scene") + " — " + data.entity_count + " entities (v" + data.version + ")");
        refreshScene();
        refreshStatus();
      } else if (custom.cmd_type === "scene") {
        // A new snapshot is ready (selection, edits, spawns): fetch it now
        // instead of waiting for the poll; polling stays the fallback.
        refreshScene();
      } else if (custom.cmd_type === "mode_changed") {
        setEditorMode(data.mode);
        refreshScene();
      } else if (custom.cmd_type === "error") {
        log((data.command || "command") + ": " + (data.message || "error"), true);
      }
    });
  }

  function refreshEvents() {
    if (eventSocketConnected) return Promise.resolve();
    return fetchJson("/api/events?after=" + eventCursor)
      .then(handleEvents)
      .catch(function () { /* engine offline — keep the last state */ });
  }

  function connectEventStream() {
    if (!window.WebSocket) return;
    var scheme = location.protocol === "https:" ? "wss:" : "ws:";
    var socket = new WebSocket(scheme + "//" + location.host + "/api/events?after=" + eventCursor);
    eventSocket = socket;
    socket.onopen = function () {
      if (eventSocket !== socket) return;
      eventSocketConnected = true;
      log("event stream connected");
    };
    socket.onmessage = function (message) {
      try { handleEvents(JSON.parse(message.data)); }
      catch (e) { console.warn("[ornis-editor] invalid event payload", e); }
    };
    socket.onclose = function () {
      if (eventSocket !== socket) return;
      eventSocketConnected = false;
      eventSocket = null;
      setTimeout(connectEventStream, 2000);
    };
  }

  // ── UI wiring ──────────────────────────────────────────────────────
  if (treeFilter) treeFilter.addEventListener("input", renderTree);

  // Bottom panel tabs (Project / Console). The Project page is a placeholder:
  // the backend exposes no asset-listing endpoint, scenes are opened with ⌘O.
  document.querySelectorAll(".browser-panel .tab").forEach(function (tab) {
    tab.addEventListener("click", function () {
      document.querySelectorAll(".browser-panel .tab").forEach(function (other) {
        other.classList.toggle("active", other === tab);
      });
      var isAssets = tab.dataset.tab === "assets";
      var assetsPage = document.getElementById("tab-assets");
      var consolePage = document.getElementById("tab-console");
      if (assetsPage) assetsPage.classList.toggle("hidden", !isAssets);
      if (consolePage) consolePage.classList.toggle("hidden", isAssets);
    });
  });
  var crumb = document.getElementById("crumb");
  if (crumb) crumb.appendChild(el("span", "crumb-item", "assets"));
  var folderGrid = document.getElementById("folder-grid");
  if (folderGrid) {
    folderGrid.appendChild(el("p", "muted",
      "No asset-listing API yet — open a scene with ⌘O."));
  }

  if (nameInput) {
    nameInput.addEventListener("change", function () {
      var entity = selectedKey && lastScene ? findEntity(lastScene, selectedKey) : null;
      var name = nameInput.value.trim();
      if (!entity || !name || name === entityNameOf(entity)) return;
      setComponent(entity, "Name", name).catch(function () {});
    });
  }

  var addEntity = document.getElementById("add-entity");
  if (addEntity) {
    addEntity.addEventListener("click", function () {
      sendCommand("create_entity", { name: "Entity" }).then(afterMutation).catch(function () {});
    });
  }

  var addComponent = document.getElementById("add-component");
  if (addComponent) {
    addComponent.addEventListener("click", function () {
      log("component picker is not wired yet (follow-up after E2/E3)");
    });
  }

  document.querySelectorAll(".viewport-gizmos .gizmo[data-tool]").forEach(function (button) {
    button.addEventListener("click", function () {
      document.querySelectorAll(".viewport-gizmos .gizmo[data-tool]").forEach(function (other) {
        other.classList.toggle("active", other === button);
      });
      var api = window.OrnisViewport;
      if (api && api.setGizmoMode) api.setGizmoMode(button.dataset.tool);
      log("gizmo: " + button.dataset.tool);
    });
  });

  var spaceButton = document.getElementById("gizmo-space");
  if (spaceButton) {
    spaceButton.addEventListener("click", function () {
      var local = spaceButton.dataset.space !== "local";
      spaceButton.dataset.space = local ? "local" : "world";
      spaceButton.textContent = local ? "L" : "W";
      spaceButton.title = (local ? "Local" : "World") + " space (click to switch)";
      spaceButton.classList.toggle("active", local);
      var api = window.OrnisViewport;
      if (api && api.setGizmoSpace) api.setGizmoSpace(local ? "local" : "world");
      log("gizmo space: " + (local ? "local" : "world"));
    });
  }

  // ── play/edit transport (E4) ───────────────────────────────────────
  var btnPlay = document.getElementById("btn-play");
  var btnPause = document.getElementById("btn-pause");
  var playIcon = document.getElementById("play-icon");
  var modeIndicator = document.getElementById("mode-indicator");
  var editorMode = "edit";

  function renderMode() {
    var playing = editorMode !== "edit";
    if (btnPlay) {
      btnPlay.classList.toggle("playing", playing);
      btnPlay.title = playing ? "Stop (F5)" : "Play (F5)";
      if (playIcon) playIcon.setAttribute("href", icon(playing ? "stop" : "play"));
    }
    if (btnPause) {
      btnPause.disabled = !playing;
      btnPause.classList.toggle("active", editorMode === "paused");
      btnPause.title = editorMode === "paused" ? "Resume (Shift+F5)" : "Pause (Shift+F5)";
    }
    if (modeIndicator) {
      modeIndicator.textContent = editorMode.toUpperCase();
      modeIndicator.dataset.mode = editorMode;
    }
  }

  function setEditorMode(mode) {
    editorMode = mode === "play" || mode === "paused" ? mode : "edit";
    renderMode();
    // Relay to the replica so it gates picking/gizmo drag off in Play (E4).
    var api = window.OrnisViewport;
    if (api && api.setEditorMode) api.setEditorMode(editorMode);
  }

  function play() { sendCommand("play").then(afterMutation).catch(function () {}); }
  function pause() { sendCommand("pause").then(afterMutation).catch(function () {}); }
  function stop() { sendCommand("stop").then(afterMutation).catch(function () {}); }

  function togglePlay() {
    if (editorMode === "edit") play(); else stop();
  }
  function togglePause() {
    if (editorMode === "play") pause();
    else if (editorMode === "paused") play();
  }

  if (btnPlay) btnPlay.addEventListener("click", togglePlay);
  if (btnPause) btnPause.addEventListener("click", togglePause);
  renderMode();

  // ── snap controls (E4) ─────────────────────────────────────────────
  var snapToggle = document.getElementById("snap-toggle");
  var snapTranslate = document.getElementById("snap-translate");
  var snapRotate = document.getElementById("snap-rotate");
  var snapScale = document.getElementById("snap-scale");
  var snapEnabled = false;

  function pushSnap() {
    var api = window.OrnisViewport;
    if (api && api.setSnap) {
      api.setSnap({
        enabled: snapEnabled,
        translate: snapTranslate ? snapTranslate.value : 0.5,
        rotateDeg: snapRotate ? snapRotate.value : 15,
        scale: snapScale ? snapScale.value : 0.25
      });
    }
  }
  if (snapToggle) {
    snapToggle.addEventListener("click", function () {
      snapEnabled = !snapEnabled;
      snapToggle.classList.toggle("active", snapEnabled);
      snapToggle.setAttribute("aria-pressed", String(snapEnabled));
      pushSnap();
      log("snap: " + (snapEnabled ? "on" : "off"));
    });
  }
  [snapTranslate, snapRotate, snapScale].forEach(function (input) {
    if (input) input.addEventListener("change", pushSnap);
  });

  var commandPalette = document.getElementById("command-palette");
  if (commandPalette) {
    document.addEventListener("keydown", function (ev) {
      // The welcome surfaces own their own shortcuts while visible.
      if (document.querySelector(".wl-screen.wl-on")) return;
      if (ev.key === "F5") {
        // Ctrl/Cmd+F5 is the browser's hard refresh — leave it to the browser.
        if (ev.ctrlKey || ev.metaKey) return;
        ev.preventDefault();
        // Holding F5 auto-repeats keydown; toggle only on the first press
        // so a held key does not flap play/stop.
        if (ev.repeat) return;
        if (ev.shiftKey) togglePause(); else togglePlay();
        return;
      }
      if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "k") {
        ev.preventDefault();
        commandPalette.focus();
        commandPalette.select();
      } else if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "s") {
        ev.preventDefault();
        saveScene();
      } else if ((ev.metaKey || ev.ctrlKey) && !ev.shiftKey && ev.key.toLowerCase() === "z") {
        // The gizmo undo is a viewport shortcut: it must not hijack the
        // browser undo inside text fields (and Shift+Z is redo, not undo).
        if (isEditingTarget(ev.target)) return;
        ev.preventDefault();
        var api = window.OrnisViewport;
        if (api && api.undoGizmo) api.undoGizmo();
      } else if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "o") {
        ev.preventDefault();
        promptOpenScene();
      }
    });
    commandPalette.addEventListener("keydown", function (ev) {
      if (ev.key !== "Enter") return;
      var text = commandPalette.value.trim();
      commandPalette.value = "";
      commandPalette.blur();
      if (!text) return;
      if (text === "save") saveScene();
      else if (text === "reload") openScene("editor/scene.ron");
      else if (text.indexOf("load ") === 0) openScene(text.slice(5).trim());
      else if (text === "add") addEntity && addEntity.click();
      else if (text === "delete") deleteSelected();
      else log("unknown command: " + text, true);
    });
  }

  // Panel collapse toggles (same behaviour as the mock: the button docks in
  // the top bar while its panel is hidden).
  var layout = document.querySelector(".layout");
  var topRow = document.querySelector(".center-top .top-row");
  function wirePanelToggle(button, hideClass) {
    if (!button || !layout) return;
    var home = button.parentNode;
    button.addEventListener("click", function () {
      var hidden = layout.classList.toggle(hideClass);
      (hidden ? topRow : home).appendChild(button);
      button.classList.toggle("docked", hidden);
      button.title = hidden ? "Expand panel" : "Collapse panel";
    });
  }
  wirePanelToggle(document.getElementById("toggle-left"), "hide-left");
  wirePanelToggle(document.getElementById("toggle-right"), "hide-right");

  var settings = document.getElementById("btn-settings");
  if (settings) settings.addEventListener("click", function () { log("settings panel not wired yet"); });
  var previewSettings = document.getElementById("btn-preview-settings");
  if (previewSettings) previewSettings.addEventListener("click", function () { log("preview settings not wired yet"); });

  // ── init ───────────────────────────────────────────────────────────
  function init() {
    setProjectName(projectName);
    refreshScene();
    refreshStatus();
    setInterval(refreshScene, POLL_MS);
    setInterval(refreshStatus, POLL_MS);
    setInterval(refreshEvents, POLL_MS);
    connectEventStream();
    log("connecting to the engine…");
  }

  window.OrnisEditor = {
    refresh: refreshScene,
    refreshStatus: refreshStatus,
    sendCommand: sendCommand,
    select: selectEntity,
    openProject: function (path, name) {
      if (path && /\.(ron|glb|gltf)$/i.test(path) && path.charAt(0) !== "~") {
        return openScene(path, name);
      }
      setProjectName(name || projectName);
      return Promise.resolve();
    },
    openScene: openScene,
    promptOpenScene: promptOpenScene,
    newEmptyProject: newEmptyProject,
    saveScene: saveScene
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
