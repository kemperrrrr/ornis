// Ornis editor-v2 mockup logic (static demo data, no backend).
// Sample scene mirrors editor/scene.ron: five cubes + two lights + camera.
(function () {
  "use strict";

  var ICONS = "../editor/icons/";
  // editor-v2's own icons live in ./icons and take precedence over the shared set.
  var LOCAL_ICONS = { "material-toon": true, "camera-wide": true };
  function icon(name) { return (LOCAL_ICONS[name] ? "icons/" : ICONS) + name + ".svg#icon"; }

  var entities = [
    { id: "red", name: "Red Sphere", kind: "mesh", icon: "cube-outline",
      comps: ["transform", "material", "physics"],
      pos: [-5.6, 0, 0], color: "#cc3333", rough: 0.5, metal: false, mass: 12, body: "Dynamic", enabled: true },
    { id: "green", name: "Green Rough Sphere", kind: "mesh", icon: "cube-outline",
      comps: ["transform", "material", "physics"],
      pos: [-2.8, 0, 0], color: "#33aa33", rough: 0.7, metal: false, mass: 9, body: "Dynamic", enabled: true },
    { id: "blue", name: "Blue Smooth Sphere", kind: "mesh", icon: "cube-outline",
      comps: ["transform", "material", "physics"],
      pos: [0, 0, 0], color: "#3355cc", rough: 0.1, metal: false, mass: 9, body: "Static", enabled: true },
    { id: "gold", name: "Gold Sphere", kind: "mesh", icon: "cube-outline",
      comps: ["transform", "material", "physics"],
      pos: [2.8, 0, 0], color: "#e6b800", rough: 0.2, metal: true, mass: 20, body: "Dynamic", enabled: false },
    { id: "ceramic", name: "Ceramic Sphere", kind: "mesh", icon: "cube-outline",
      comps: ["transform", "material", "physics"],
      pos: [5.6, 0, 0], color: "#e8e8e8", rough: 0.35, metal: false, mass: 7, body: "Kinematic", enabled: true },
    { id: "key", name: "Key Light", kind: "light", icon: "lightbulb",
      comps: ["transform", "light"],
      pos: [3, 5, 2], color: "#fff4e0", intensity: 0.6, enabled: true },
    { id: "fill", name: "Fill Light", kind: "light", icon: "lightbulb",
      comps: ["transform", "light"],
      pos: [-4, 3, -2], color: "#ccd6ff", intensity: 0.3, enabled: true },
    { id: "cam", name: "Orbit Camera", kind: "camera", icon: "camera-wide",
      comps: ["transform", "camera"],
      pos: [0, 2.5, 9], fov: 60, enabled: true },
  ];
  var COMP_DEFS = {
    transform: { label: "Transform", icon: "transform-gizmo" },
    material: { label: "Material", icon: "material-toon" },
    physics: { label: "Physics Body", icon: "weight" },
    light: { label: "Light", icon: "lightbulb" },
    camera: { label: "Camera", icon: "camera-wide" },
  };
  var sceneTree = [
    { label: "demo", icon: "cube-outline",
      children: ["red", "green", "blue", "gold", "ceramic", "key", "fill", "cam"] },
  ];
  var FS = { name: "demo", children: [
    { name: "assets", children: [
      { name: "Scenes", children: [
        { name: "demo.ron", file: { icon: "file" } },
        { name: "Village", children: [
          { name: "exterior.ron", file: { icon: "file" } },
          { name: "interior.ron", file: { icon: "file" } },
        ]},
      ]},
      { name: "Materials", children: [
        { name: "Metals", children: [
          { name: "Gold material", file: { icon: "material-toon" } },
        ]},
        { name: "Ceramics", children: [
          { name: "Ceramic material", file: { icon: "material-toon" } },
        ]},
      ]},
      { name: "Prefabs", children: [
        { name: "Player spawn", file: { icon: "cube-outline" } },
      ]},
      { name: "Textures", children: [] },
      { name: "Sounds", children: [] },
    ]},
  ]};
  var selectedId = "blue";
  function byId(id) { return entities.find(function (e) { return e.id === id; }); }

  // ---- hierarchy tree (collapsible) ----
  var tree = document.getElementById("tree");
  var treeFilter = document.getElementById("tree-filter");
  var collapsed = {};

  function row(label, iconName, opts) {
    opts = opts || {};
    var b = document.createElement("button");
    b.className = "tree-row" + (opts.selected ? " selected" : "");
    var caret = document.createElement("span");
    caret.className = "caret" + (opts.collapsible ? " clickable" + (opts.collapsed ? "" : " open") : "");
    if (opts.collapsible && opts.onToggle) {
      caret.addEventListener("click", function (ev) { ev.stopPropagation(); opts.onToggle(); });
    }
    b.appendChild(caret);
    var svgNS = "http://www.w3.org/2000/svg";
    var svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    var use = document.createElementNS(svgNS, "use");
    use.setAttribute("href", icon(iconName));
    svg.appendChild(use);
    b.appendChild(svg);
    var span = document.createElement("span");
    span.textContent = label;
    b.appendChild(span);
    return b;
  }

  function renderTree() {
    var q = treeFilter.value.trim().toLowerCase();
    tree.innerHTML = "";
    sceneTree.forEach(function (group) {
      var gkey = "g:" + group.label;
      var wrap = document.createElement("div");
      var header = row(group.label, group.icon,
        { collapsible: true, collapsed: !!collapsed[gkey],
          onToggle: function () { collapsed[gkey] = !collapsed[gkey]; renderTree(); } });
      header.addEventListener("click", function () { collapsed[gkey] = !collapsed[gkey]; renderTree(); });
      wrap.appendChild(header);
      if (!collapsed[gkey]) {
        var kids = document.createElement("div");
        kids.className = "tree-children";
        group.children.forEach(function (child) {
          var id = typeof child === "string" ? child : null;
          var label = id ? byId(id).name : child.label;
          if (q && label.toLowerCase().indexOf(q) < 0) return;
          var ekey = "e:" + id;
          var r = row(label, id ? byId(id).icon : child.icon,
            { selected: id === selectedId, collapsible: !!id, collapsed: !!collapsed[ekey],
              onToggle: function () { collapsed[ekey] = !collapsed[ekey]; renderTree(); } });
          if (id) {
            r.dataset.id = id;
            r.title = "Double-click to open in a tab";
            r.addEventListener("click", function (ev) {
              select(id);
              if (ev.detail === 2) openEntityTab(id); // tree re-renders on click, so detect dblclick here too
            });
          }
          kids.appendChild(r);
          if (id && !collapsed[ekey]) {
            var sub = document.createElement("div");
            sub.className = "tree-children";
            (byId(id).comps || []).forEach(function (ckey) {
              var cdef = COMP_DEFS[ckey];
              var cr = row(cdef.label, cdef.icon, {});
              cr.addEventListener("click", function () { select(id); });
              sub.appendChild(cr);
            });
            kids.appendChild(sub);
          }
        });
        if (kids.children.length) wrap.appendChild(kids);
      }
      if (wrap.children.length > 1 || !q) tree.appendChild(wrap);
    });
  }

  // ---- viewport shapes: spheres for mesh entities ----
  var shapes = document.getElementById("shapes");
  function renderShapes() {
    shapes.innerHTML = "";
    entities.forEach(function (e) {
      if (e.kind !== "mesh") return;
      var d = document.createElement("div");
      d.className = "sphere" + (e.id === selectedId ? " selected" : "");
      d.title = e.name;
      d.style.background = "radial-gradient(circle at 35% 30%, #ffffff88, " + e.color + " 60%, #00000055)";
      d.addEventListener("click", function () { select(e.id); });
      shapes.appendChild(d);
    });
  }

  // ---- inspector ----
  var inspector = document.getElementById("inspector");
  var nameInput = document.getElementById("inspector-name");
  var inspectorIcon = document.getElementById("inspector-icon");
  var previewIcon = document.getElementById("preview-icon");

  function numVec(values, onChange) {
    var wrap = document.createElement("div");
    wrap.className = "vec";
    values.forEach(function (v, i) {
      var inp = document.createElement("input");
      inp.type = "number"; inp.step = "0.1"; inp.value = v;
      inp.addEventListener("change", function () { onChange(i, parseFloat(inp.value) || 0); });
      var cell = document.createElement("span"); // holds the inner axis-colour strip
      cell.className = "vec-cell";
      cell.appendChild(inp);
      wrap.appendChild(cell);
    });
    return wrap;
  }
  function sliderRow(label, value, min, max, step, fmt, onChange) {
    var r = document.createElement("div");
    r.className = "slider-row";
    var lab = document.createElement("span"); lab.textContent = label;
    var inp = document.createElement("input");
    inp.type = "range"; inp.min = min; inp.max = max; inp.step = step; inp.value = value;
    var out = document.createElement("span"); out.textContent = fmt(value);
    inp.addEventListener("input", function () {
      out.textContent = fmt(parseFloat(inp.value));
      onChange(parseFloat(inp.value));
    });
    r.appendChild(lab); r.appendChild(inp); r.appendChild(out);
    return r;
  }
  function section(title, open, iconName) {
    var d = document.createElement("details");
    d.className = "section";
    if (open) d.open = true;
    var s = document.createElement("summary");
    var svgNS = "http://www.w3.org/2000/svg";
    var svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("class", "section-icon");
    var use = document.createElementNS(svgNS, "use");
    use.setAttribute("href", icon(iconName || "cube-outline"));
    svg.appendChild(use);
    s.appendChild(svg);
    s.appendChild(document.createTextNode(title));
    d.appendChild(s);
    var body = document.createElement("div");
    body.className = "body";
    d.appendChild(body);
    return { root: d, body: body };
  }

  function renderInspector() {
    var e = byId(selectedId);
    inspector.innerHTML = "";
    if (!e) { inspector.innerHTML = '<p class="muted">Select an entity.</p>'; return; }
    nameInput.value = e.name;
    inspectorIcon.setAttribute("href", icon(e.icon));
    previewIcon.setAttribute("href", icon(e.icon));

    var t = section("Transform", true, "transform-gizmo");
    var f = document.createElement("div"); f.className = "field";
    var lab = document.createElement("span"); lab.textContent = "Position";
    f.appendChild(lab);
    f.appendChild(numVec(e.pos, function (i, v) { e.pos[i] = v; log("moved " + e.name + " → [" + e.pos.join(", ") + "]"); }));
    t.body.appendChild(f);
    inspector.appendChild(t.root);

    if (e.kind === "mesh") {
    var m = section("Material", true, "material-toon");
    var cf = document.createElement("div"); cf.className = "field";
    var cl = document.createElement("span"); cl.textContent = "Base color";
    var ci = document.createElement("input");
    ci.type = "color"; ci.value = e.color;
    ci.addEventListener("input", function () { e.color = ci.value; renderShapes(); });
    cf.appendChild(cl); cf.appendChild(ci);
    m.body.appendChild(cf);
    m.body.appendChild(sliderRow("Roughness", e.rough, 0, 1, 0.05,
      function (v) { return v.toFixed(2); }, function (v) { e.rough = v; }));
    var tog = document.createElement("label"); tog.className = "toggle-row";
    tog.textContent = "Metallic";
    var sw = document.createElement("span"); sw.className = "switch";
    var cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = e.metal;
    cb.addEventListener("change", function () { e.metal = cb.checked; });
    var tr = document.createElement("span"); tr.className = "track";
    sw.appendChild(cb); sw.appendChild(tr);
    tog.appendChild(sw);
    m.body.appendChild(tog);
    inspector.appendChild(m.root);

    var p = section("Physics Body", false, "weight");
    var mf = document.createElement("div"); mf.className = "field";
    var ml = document.createElement("span"); ml.textContent = "Mode";
    var sel = document.createElement("select");
    ["Dynamic", "Static", "Kinematic"].forEach(function (k) {
      var o = document.createElement("option"); o.textContent = k; o.selected = e.body === k;
      sel.appendChild(o);
    });
    sel.addEventListener("change", function () { e.body = sel.value; });
    mf.appendChild(ml); mf.appendChild(sel);
    p.body.appendChild(mf);
    p.body.appendChild(sliderRow("Mass", e.mass, 1, 40, 1,
      function (v) { return v.toFixed(0); }, function (v) { e.mass = v; }));
    var en = document.createElement("label"); en.className = "toggle-row";
    en.textContent = "Enabled";
    var sw2 = document.createElement("span"); sw2.className = "switch";
    var cb2 = document.createElement("input"); cb2.type = "checkbox"; cb2.checked = e.enabled;
    cb2.addEventListener("change", function () { e.enabled = cb2.checked; });
    var tr2 = document.createElement("span"); tr2.className = "track";
    sw2.appendChild(cb2); sw2.appendChild(tr2);
    en.appendChild(sw2);
    p.body.appendChild(en);
    inspector.appendChild(p.root);
    }

    if (e.kind === "light") {
      var l = section("Light", true, "lightbulb");
      var lf = document.createElement("div"); lf.className = "field";
      var ll = document.createElement("span"); ll.textContent = "Color";
      var li = document.createElement("input");
      li.type = "color"; li.value = e.color;
      li.addEventListener("input", function () { e.color = li.value; });
      lf.appendChild(ll); lf.appendChild(li);
      l.body.appendChild(lf);
      l.body.appendChild(sliderRow("Intensity", e.intensity, 0, 2, 0.05,
        function (v) { return v.toFixed(2); }, function (v) { e.intensity = v; }));
      var len = document.createElement("label"); len.className = "toggle-row";
      len.textContent = "Enabled";
      var lsw = document.createElement("span"); lsw.className = "switch";
      var lcb = document.createElement("input"); lcb.type = "checkbox"; lcb.checked = e.enabled;
      lcb.addEventListener("change", function () { e.enabled = lcb.checked; });
      var ltr = document.createElement("span"); ltr.className = "track";
      lsw.appendChild(lcb); lsw.appendChild(ltr);
      len.appendChild(lsw);
      l.body.appendChild(len);
      inspector.appendChild(l.root);
    }

    if (e.kind === "camera") {
      var cm = section("Camera", true, "camera-wide");
      cm.body.appendChild(sliderRow("Fov", e.fov, 20, 120, 1,
        function (v) { return v.toFixed(0) + "°"; }, function (v) { e.fov = v; }));
      inspector.appendChild(cm.root);
    }
  }

  function select(id) {
    selectedId = id;
    renderTree(); renderShapes(); renderInspector();
  }

  nameInput.addEventListener("change", function () {
    var e = byId(selectedId);
    if (e && nameInput.value.trim()) { e.name = nameInput.value.trim(); renderTree(); renderShapes(); renderTabLabels(); }
  });

  // ---- project browser: breadcrumb path + folder grid (folders only for now) ----
  var folderGrid = document.getElementById("folder-grid");
  var crumb = document.getElementById("crumb");
  var selectedFolder = ["assets"]; // path of folder names below the project root

  function fsNodeAt(path) {
    var node = FS;
    for (var i = 0; i < path.length; i++) {
      node = (node.children || []).filter(function (c) { return !c.file; })
        .find(function (c) { return c.name === path[i]; });
      if (!node) return FS;
    }
    return node;
  }

  function crumbItem(name, path) {
    var b = document.createElement("button");
    b.className = "crumb-item";
    var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    var use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", icon("folder"));
    svg.appendChild(use);
    b.appendChild(svg);
    b.appendChild(document.createTextNode(name));
    b.addEventListener("click", function () { selectedFolder = path; renderBrowser(); });
    return b;
  }

  function renderCrumb() {
    crumb.innerHTML = "";
    crumb.appendChild(crumbItem(FS.name, []));
    selectedFolder.forEach(function (seg, i) {
      var sep = document.createElement("span");
      sep.className = "crumb-sep";
      crumb.appendChild(sep);
      crumb.appendChild(crumbItem(seg, selectedFolder.slice(0, i + 1)));
    });
  }

  function renderGrid() {
    folderGrid.innerHTML = "";
    var node = fsNodeAt(selectedFolder);
    (node.children || []).forEach(function (child) {
      if (child.file) return; // files come in a later iteration
      var tile = document.createElement("button");
      tile.className = "folder-tile";
      tile.title = child.name;
      tile.innerHTML = '<svg viewBox="0 0 24 24"><use href="' + icon("folder") + '" /></svg>';
      var label = document.createElement("span");
      label.textContent = child.name;
      tile.appendChild(label);
      tile.addEventListener("dblclick", function () {
        selectedFolder = selectedFolder.concat([child.name]);
        renderBrowser();
      });
      tile.addEventListener("click", function () {
        folderGrid.querySelectorAll(".folder-tile.selected").forEach(function (t) { t.classList.remove("selected"); });
        tile.classList.add("selected");
      });
      folderGrid.appendChild(tile);
    });
    if (!folderGrid.children.length) {
      var empty = document.createElement("div");
      empty.className = "muted folder-empty";
      empty.textContent = "No folders here";
      folderGrid.appendChild(empty);
    }
  }

  function renderBrowser() { renderCrumb(); renderGrid(); }
  document.querySelectorAll(".tab").forEach(function (tab) {
    tab.addEventListener("click", function () {
      document.querySelectorAll(".tab").forEach(function (x) { x.classList.remove("active"); });
      tab.classList.add("active");
      var isAssets = tab.dataset.tab === "assets";
      document.getElementById("tab-assets").classList.toggle("hidden", !isAssets);
      document.getElementById("tab-console").classList.toggle("hidden", isAssets);
    });
  });

  var consoleLog = document.getElementById("console-log");
  function log(msg) {
    var div = document.createElement("div");
    var s = document.createElement("span"); s.className = "muted"; s.textContent = "[mock] ";
    div.appendChild(s);
    div.appendChild(document.createTextNode(msg));
    consoleLog.appendChild(div);
  }

  // ---- scene / entity tabs (second row of the top bar, Godot-style) ----
  var tabStrip = document.getElementById("scene-tabs");
  var TAB_ANIM_MS = 320; // keep in sync with --tab-anim in styles.css
  var openTabs = [{ id: "demo", root: true }]; // "demo" = root scene, always open
  var activeTab = "demo";
  var SVG_NS = "http://www.w3.org/2000/svg";

  function svgIcon(name) {
    var svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    var use = document.createElementNS(SVG_NS, "use");
    use.setAttribute("href", icon(name));
    svg.appendChild(use);
    return svg;
  }
  function tabEl(id) { return tabStrip.querySelector('.scene-tab[data-id="' + id + '"]:not(.leaving)'); }
  function tabLabel(t) { return t.root ? "demo" : (byId(t.id) || { name: t.id }).name; }
  function tabIcon(t) { return t.root ? sceneTree[0].icon : (byId(t.id) || { icon: "cube-outline" }).icon; }

  function buildTab(t) {
    var el = document.createElement("div");
    el.className = "scene-tab" + (t.root ? " root" : "");
    el.dataset.id = t.id;
    el.setAttribute("role", "tab");
    el.tabIndex = 0;
    el.title = tabLabel(t);
    el.appendChild(svgIcon(tabIcon(t)));
    var lab = document.createElement("span");
    lab.className = "label";
    lab.textContent = tabLabel(t);
    el.appendChild(lab);
    if (!t.root) {
      var x = document.createElement("button");
      x.className = "close";
      x.title = "Close tab";
      x.tabIndex = -1;
      x.appendChild(svgIcon("close"));
      x.addEventListener("click", function (ev) { ev.stopPropagation(); closeTab(t.id); });
      el.appendChild(x);
      el.addEventListener("auxclick", function (ev) { if (ev.button === 1) { ev.preventDefault(); closeTab(t.id); } });
    }
    el.addEventListener("click", function () { activateTab(t.id); });
    el.addEventListener("keydown", function (ev) {
      if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); activateTab(t.id); }
    });
    return el;
  }

  function syncActive() {
    tabStrip.querySelectorAll(".scene-tab").forEach(function (el) {
      var on = el.dataset.id === activeTab && !el.classList.contains("leaving");
      el.classList.toggle("active", on);
      el.setAttribute("aria-selected", on ? "true" : "false");
    });
  }

  function scrollTabIntoView(el) {
    var left = el.offsetLeft, right = left + el.offsetWidth;
    var view = tabStrip.scrollLeft, w = tabStrip.clientWidth, pad = 8;
    if (left - pad < view) tabStrip.scrollTo({ left: left - pad, behavior: "smooth" });
    else if (right + pad > view + w) tabStrip.scrollTo({ left: right + pad - w, behavior: "smooth" });
  }

  function activateTab(id) {
    if (!openTabs.some(function (t) { return t.id === id; })) return;
    var changed = activeTab !== id;
    activeTab = id;
    syncActive();
    var t = openTabs.find(function (t) { return t.id === id; });
    if (!t.root && byId(id)) select(id);
    var el = tabEl(id);
    if (el) scrollTabIntoView(el);
    if (changed) log("tab: switched to " + tabLabel(t) + " (mock)");
  }

  function openEntityTab(id) {
    if (!byId(id)) return;
    if (openTabs.some(function (t) { return t.id === id; })) { activateTab(id); return; }
    var t = { id: id };
    openTabs.push(t);
    var el = buildTab(t);
    tabStrip.appendChild(el);
    // Expand from 0 to the tab's natural width (measured first so the easing isn't cut short).
    var w = el.offsetWidth;
    freezeLabel(el);
    el.style.transition = "none";
    el.classList.add("entering");
    void el.offsetWidth; // commit the collapsed state
    el.style.transition = "";
    requestAnimationFrame(function () {
      el.style.maxWidth = w + "px";
      el.classList.remove("entering");
      setTimeout(function () { el.style.maxWidth = ""; unfreezeLabel(el); }, TAB_ANIM_MS + 40);
    });
    log("tab: opened " + tabLabel(t) + " (mock)");
    activeTab = id;
    syncActive();
    select(id);
    setTimeout(function () { if (activeTab === id) scrollTabIntoView(el); }, TAB_ANIM_MS); // after it has expanded
  }

  // Keep the label at its full width while the tab folds/unfolds, so the text
  // is clipped by the tab edge instead of re-ellipsizing every frame.
  function freezeLabel(el) {
    var lab = el.querySelector(".label");
    if (lab) lab.style.width = lab.offsetWidth + "px";
    el.classList.add("anim");
  }
  function unfreezeLabel(el) {
    var lab = el.querySelector(".label");
    if (lab) lab.style.width = "";
    el.classList.remove("anim");
  }

  function closeTab(id) {
    var i = openTabs.findIndex(function (t) { return t.id === id; });
    if (i < 0 || openTabs[i].root) return;
    var t = openTabs[i];
    openTabs.splice(i, 1);
    var el = tabEl(id);
    log("tab: closed " + tabLabel(t) + " (mock)");
    if (activeTab === id) {
      var next = openTabs[Math.min(i, openTabs.length - 1)];
      activeTab = null;
      activateTab(next.id);
    }
    if (el) {
      // Collapse from the real width (pin it without a transition first).
      var bg = getComputedStyle(el).backgroundColor;
      freezeLabel(el);
      el.style.transition = "none";
      el.style.maxWidth = el.offsetWidth + "px";
      void el.offsetWidth;
      el.style.transition = "";
      el.style.backgroundColor = bg; // keep its colour while it folds away
      el.classList.remove("active");
      el.classList.add("leaving");
      el.style.maxWidth = "";
      var done = false;
      var remove = function () { if (!done) { done = true; el.remove(); } };
      el.addEventListener("transitionend", function (ev) { if (ev.propertyName === "max-width") remove(); });
      setTimeout(remove, TAB_ANIM_MS + 80);
    }
  }

  function renderTabLabels() {
    openTabs.forEach(function (t) {
      var el = tabEl(t.id);
      if (!el) return;
      el.querySelector(".label").textContent = tabLabel(t);
      el.title = tabLabel(t);
    });
  }

  var TAB_SCROLL_SPEED = 0.35; // fraction of the raw wheel delta
  // Vertical wheel scrolls the strip horizontally (scrollbars are hidden globally).
  tabStrip.addEventListener("wheel", function (ev) {
    if (tabStrip.scrollWidth <= tabStrip.clientWidth) return;
    var d = Math.abs(ev.deltaX) > Math.abs(ev.deltaY) ? ev.deltaX : ev.deltaY;
    if (!d) return;
    ev.preventDefault();
    if (ev.deltaMode === 1) d *= 16;           // lines -> px
    tabStrip.scrollLeft += d * TAB_SCROLL_SPEED;
  }, { passive: false });
  // Double-click on an entity row in the hierarchy opens its tab.
  tree.addEventListener("dblclick", function (ev) {
    var r = ev.target.closest && ev.target.closest(".tree-row[data-id]");
    if (r) openEntityTab(r.dataset.id);
  });

  openTabs.forEach(function (t) { tabStrip.appendChild(buildTab(t)); });
  syncActive();

  // ---- toolbar wiring (fake): play/stop toggle + pause ----
  var playing = false;
  var paused = false;
  var playIcon = document.getElementById("play-icon");
  var btnPlay = document.getElementById("btn-play");
  var btnPause = document.getElementById("btn-pause");
  function setPaused(on) {
    paused = on;
    btnPause.classList.toggle("paused", on);
    btnPause.title = on ? "Resume" : "Pause";
  }
  function setPlaying(on) {
    playing = on;
    btnPlay.classList.toggle("playing", on);
    btnPlay.title = on ? "Stop" : "Play";
    playIcon.setAttribute("href", icon(on ? "stop" : "play"));
    btnPause.disabled = !on;
    setPaused(false);
    log(on ? "simulate started (mock)" : "simulate stopped (mock)");
  }
  btnPlay.addEventListener("click", function () { setPlaying(!playing); });
  btnPause.addEventListener("click", function () {
    if (!playing) return;
    setPaused(!paused);
    log(paused ? "simulate paused (mock)" : "simulate resumed (mock)");
  });
  document.querySelectorAll(".viewport-gizmos .gizmo").forEach(function (b) {
    b.addEventListener("click", function () {
      document.querySelectorAll(".viewport-gizmos .gizmo").forEach(function (x) { x.classList.remove("active"); });
      b.classList.add("active");
      log("gizmo: " + b.dataset.tool + " (mock)");
    });
  });
  // ---- side panel toggles ----
  var layout = document.querySelector(".layout");
  var toggleLeft = document.getElementById("toggle-left");
  var toggleRight = document.getElementById("toggle-right");
  var centerTop = document.querySelector(".center-top .top-row"); // docked toggles sit in the first row
  // Collapsing hides the whole panel and docks its button in the top bar;
  // expanding puts the button back into the panel's own bar.
  function wirePanelToggle(button, hideClass) {
    var home = button.parentNode;
    button.addEventListener("click", function () {
      var hidden = layout.classList.toggle(hideClass);
      (hidden ? centerTop : home).appendChild(button);
      button.classList.toggle("docked", hidden);
      button.title = hidden ? "Expand panel" : "Collapse panel";
    });
  }
  wirePanelToggle(toggleLeft, "hide-left");
  wirePanelToggle(toggleRight, "hide-right");
  document.getElementById("btn-settings").addEventListener("click", function () {
    log("Settings — panel comes in a later iteration (mock)");
  });
  document.getElementById("btn-preview-settings").addEventListener("click", function () {
    log("Preview settings — panel comes in a later iteration (mock)");
  });
  document.getElementById("add-component").addEventListener("click", function () {
    log("Add Component — picker comes in the next iteration (mock)");
  });
  document.getElementById("add-entity").addEventListener("click", function () {
    log("Add Entity — creation flow comes in the next iteration (mock)");
  });

  // ---- command palette (mock): Ctrl/Cmd+K focuses, Enter logs the query ----
  var palette = document.getElementById("command-palette");
  document.addEventListener("keydown", function (ev) {
    if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "k") {
      ev.preventDefault(); palette.focus(); palette.select();
    } else if (ev.key === "Escape" && document.activeElement === palette) {
      palette.value = ""; palette.blur();
    }
  });
  palette.addEventListener("keydown", function (ev) {
    if (ev.key === "Enter" && palette.value.trim()) {
      log("command: " + palette.value.trim() + " (mock)"); palette.value = "";
    }
  });
  treeFilter.addEventListener("input", renderTree);

  renderTree(); renderShapes(); renderInspector(); renderBrowser();
})();
