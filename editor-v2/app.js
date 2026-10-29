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

  // ---- 2D plane mode (mock of the K2 plane camera; no backend, no canvas) ----
  var viewportEl = document.getElementById("viewport");
  var gridBg = viewportEl.querySelector(".grid-bg");
  var viewMode = "perspective3d";
  // Same units as the Rust plane camera: world units on the sheet, y up.
  var view2d = { center: [0, 0], pixelsPerUnit: 28 };
  var BASE_PPU = 28; // mock grid baseline, matches the 28px CSS step

  function viewportSize() {
    return [viewportEl.clientWidth || 800, viewportEl.clientHeight || 600];
  }
  function halfHeight() {
    return viewportSize()[1] / (2 * view2d.pixelsPerUnit);
  }
  // Mirror of `PlaneCamera::screen_to_plane` (screen y down, sheet y up).
  function screenToWorld(px, py) {
    var size = viewportSize();
    return [
      view2d.center[0] + (px - size[0] / 2) / view2d.pixelsPerUnit,
      view2d.center[1] + (size[1] / 2 - py) / view2d.pixelsPerUnit,
    ];
  }
  // Mirror of the 1/2/5 ladder in `plane_grid` (mock resolution).
  function ladderStep(minPx) {
    var ppu = view2d.pixelsPerUnit;
    var mag = Math.floor(Math.log10(minPx / ppu)) - 1;
    for (;;) {
      var mantissas = [1, 2, 5];
      for (var i = 0; i < mantissas.length; i++) {
        var step = mantissas[i] * Math.pow(10, mag);
        if (step * ppu >= minPx) {
          return { step: step, px: step * ppu, mantissa: mantissas[i] };
        }
      }
      mag += 1;
    }
  }
  function paintMockGrid() {
    if (viewMode !== "plane2d") {
      gridBg.style.backgroundSize = "";
      gridBg.style.backgroundPosition = "";
      return;
    }
    var size = viewportSize();
    var ladder = ladderStep(48);
    var ox = size[0] / 2 - view2d.center[0] * view2d.pixelsPerUnit;
    var oy = size[1] / 2 + view2d.center[1] * view2d.pixelsPerUnit;
    gridBg.style.backgroundSize = ladder.px + "px " + ladder.px + "px";
    gridBg.style.backgroundPosition = ox + "px " + oy + "px";
  }
  // Same JSON shape as the WASM `viewport_state_json()` wire: drives the
  // grid overlay and the shape layer from one state object.
  function applyViewState(state) {
    if (typeof state === "string") {
      try { state = JSON.parse(state); } catch (e) { return; }
    }
    if (!state || state.mode !== "plane2d" || !state.grid) return;
    view2d.center = state.center.slice();
    view2d.pixelsPerUnit = state.pixels_per_unit;
    var grid = state.grid;
    gridBg.style.backgroundSize = grid.minor_step_px + "px " + grid.minor_step_px + "px";
    gridBg.style.backgroundPosition = grid.origin_px[0] + "px " + grid.origin_px[1] + "px";
    var scale = state.pixels_per_unit / BASE_PPU;
    shapes.style.transform =
      "translate(" + (-state.center[0] * state.pixels_per_unit) + "px," +
      (state.center[1] * state.pixels_per_unit) + "px) scale(" + scale + ")";
  }
  function mockWireState() {
    var size = viewportSize();
    var ladder = ladderStep(48);
    return {
      mode: "plane2d",
      center: view2d.center.slice(),
      half_height: halfHeight(),
      pixels_per_unit: view2d.pixelsPerUnit,
      grid: {
        minor_step: ladder.step,
        // Same cadence as the Rust `plane_grid`: 5 minor steps per major
        // line for the middle mantissa (2), 10 for the decade ones (1/5).
        major_every: ladder.mantissa === 2 ? 5 : 10,
        minor_step_px: ladder.px,
        origin_px: [
          size[0] / 2 - view2d.center[0] * view2d.pixelsPerUnit,
          size[1] / 2 + view2d.center[1] * view2d.pixelsPerUnit,
        ],
      },
      pointer_world: null,
    };
  }
  function setViewMode(mode) {
    viewMode = mode;
    var is2d = mode === "plane2d";
    viewportEl.classList.toggle("mode-2d", is2d);
    document.getElementById("mode-3d").classList.toggle("active", !is2d);
    document.getElementById("mode-2d").classList.toggle("active", is2d);
    document.getElementById("mode-3d").setAttribute("aria-pressed", String(!is2d));
    document.getElementById("mode-2d").setAttribute("aria-pressed", String(is2d));
    if (is2d) {
      applyViewState(mockWireState());
      log("viewport: plane2d (mock)");
    } else {
      shapes.style.transform = "";
      paintMockGrid();
      log("viewport: perspective3d (mock)");
    }
  }
  document.getElementById("mode-3d").addEventListener("click", function () { setViewMode("perspective3d"); });
  document.getElementById("mode-2d").addEventListener("click", function () { setViewMode("plane2d"); });
  // Wheel zooms around the cursor (same exp(delta*0.001) speed as the Rust
  // camera; hh clamp [0.01, 1000] m expressed as pixels-per-unit bounds).
  viewportEl.addEventListener("wheel", function (ev) {
    if (viewMode !== "plane2d") return;
    ev.preventDefault();
    var rect = viewportEl.getBoundingClientRect();
    var px = ev.clientX - rect.left, py = ev.clientY - rect.top;
    var before = screenToWorld(px, py);
    var size = viewportSize();
    var minPpu = size[1] / 2000, maxPpu = size[1] / 0.02;
    view2d.pixelsPerUnit = Math.min(maxPpu, Math.max(minPpu,
      view2d.pixelsPerUnit * Math.exp(-ev.deltaY * 0.001)));
    var after = screenToWorld(px, py);
    view2d.center[0] += before[0] - after[0];
    view2d.center[1] += before[1] - after[1];
    applyViewState(mockWireState());
  }, { passive: false });
  // Middle/right drag pans; the left button only selects, never pans.
  var panDrag = null;
  viewportEl.addEventListener("pointerdown", function (ev) {
    if (viewMode !== "plane2d") return;
    if (ev.button === 1 || ev.button === 2) {
      ev.preventDefault();
      panDrag = { x: ev.clientX, y: ev.clientY };
      viewportEl.setPointerCapture(ev.pointerId);
    }
  });
  viewportEl.addEventListener("pointermove", function (ev) {
    if (!panDrag) return;
    view2d.center[0] -= (ev.clientX - panDrag.x) / view2d.pixelsPerUnit;
    view2d.center[1] += (ev.clientY - panDrag.y) / view2d.pixelsPerUnit;
    panDrag = { x: ev.clientX, y: ev.clientY };
    applyViewState(mockWireState());
  });
  function endPan(ev) {
    if (panDrag && ev) {
      try { viewportEl.releasePointerCapture(ev.pointerId); } catch (e) { /* already released */ }
    }
    panDrag = null;
  }
  viewportEl.addEventListener("pointerup", endPan);
  viewportEl.addEventListener("pointercancel", endPan);
  viewportEl.addEventListener("contextmenu", function (ev) {
    if (viewMode === "plane2d") ev.preventDefault();
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

// ---- Welcome flow ---------------------------------------------------------------
// start (#wl-start, shown on load unless ?nowelcome) ─View all→ all projects (#wl-all, Esc/back → start)
// editor: click the project name in the top bar or ⇧⌘O → project sheet (#wl-sheet, Esc/Dismiss closes)
// sheet ─View all→ hub (#wl-hub, Esc/close → editor). Opening "demo" anywhere returns to the editor.
(function () {
  "use strict";
  var start = document.getElementById("wl-start");
  if (!start) return;

var PROJECTS = [
  { id: "demo", name: "demo", path: "~/AI-Projects/ornis/demo", date: "12 minutes ago", short: "12m ago", scenes: 1, pinned: true },
  { id: "terrain", name: "terrain_test", path: "~/AI-Projects/terrain_test", date: "Yesterday", short: "Yesterday", scenes: 3 },
  { id: "physics", name: "physics_sandbox", path: "~/AI-Projects/ornis/examples/physics_sandbox", date: "3 days ago", short: "3d ago", scenes: 5 },
  { id: "showcase", name: "ornis_showcase", path: "~/Projects/ornis_showcase", date: "Sep 18", short: "2w ago", scenes: 8 }
];
function _sphere(p, i, cx, cy, r, col) {
  var id = p + "s" + i;
  return '<defs><radialGradient id="' + id + '" cx="36%" cy="30%" r="78%">' +
    '<stop offset="0" stop-color="#fff" stop-opacity=".95"/><stop offset=".16" stop-color="' + col + '"/>' +
    '<stop offset=".7" stop-color="' + col + '"/><stop offset="1" stop-color="#0b0b12"/></radialGradient></defs>' +
    '<ellipse cx="' + cx + '" cy="' + (cy + r * 0.96) + '" rx="' + r * 0.95 + '" ry="' + r * 0.2 + '" fill="#000" opacity=".38"/>' +
    '<circle cx="' + cx + '" cy="' + cy + '" r="' + r + '" fill="url(#' + id + ')"/>';
}
function _cube(x, y, s, col) {           // isometric cube, (x, y) = bottom-front corner
  var h = s * 0.5, w = s * 0.866;
  var top = [[x, y - 2 * s + 0], [x + w, y - s - h], [x, y - s - 2 * h + 0], [x - w, y - s - h]];
  top = [[x, y - s], [x + w, y - s - h], [x, y - s - 2 * h], [x - w, y - s - h]];
  var L = [[x - w, y - s - h], [x, y - s], [x, y], [x - w, y - h]];
  var R = [[x, y - s], [x + w, y - s - h], [x + w, y - h], [x, y]];
  function pts(a) { return a.map(function (q) { return q[0].toFixed(1) + "," + q[1].toFixed(1); }).join(" "); }
  return '<ellipse cx="' + x + '" cy="' + (y - h * 0.4) + '" rx="' + w * 1.15 + '" ry="' + h * 0.75 + '" fill="#000" opacity=".3"/>' +
    '<polygon points="' + pts(L) + '" fill="' + col + '"/><polygon points="' + pts(L) + '" fill="#000" opacity=".28"/>' +
    '<polygon points="' + pts(R) + '" fill="' + col + '"/><polygon points="' + pts(R) + '" fill="#000" opacity=".5"/>' +
    '<polygon points="' + pts(top) + '" fill="' + col + '"/><polygon points="' + pts(top) + '" fill="#fff" opacity=".12"/>';
}
function _floor(p, hy, col, line) {
  var s = '<rect x="0" y="' + hy + '" width="320" height="' + (200 - hy) + '" fill="' + col + '"/><g stroke="' + line + '" stroke-width="1">';
  for (var i = -10; i <= 10; i++) s += '<line x1="' + (160 + i * 12) + '" y1="' + hy + '" x2="' + (160 + i * 64) + '" y2="200"/>';
  for (var k = 1; k <= 6; k++) { var y = hy + (200 - hy) * Math.pow(k / 6, 1.8); s += '<line x1="0" y1="' + y.toFixed(1) + '" x2="320" y2="' + y.toFixed(1) + '"/>'; }
  return s + '</g>';
}
function sceneSVG(id, prefix) {
  var p = (prefix || "t") + id, sky, body = "";
  if (id === "demo") {
    sky = ["#2f3240", "#1b1c22"];
    body = _floor(p, 128, "#24252c", "rgba(255,255,255,.06)");
    [["#d23b3b", 70], ["#3cae4a", 115], ["#3f6fe0", 160], ["#e0b021", 205], ["#d9d9de", 250]].forEach(function (c, i) {
      body += _sphere(p, i, c[1], 118, 18, c[0]); });
  } else if (id === "terrain") {
    sky = ["#e08a5c", "#5b3a6e"];
    body = '<circle cx="232" cy="92" r="20" fill="#ffd9a0" opacity=".9"/>' +
      '<polygon points="0,130 40,96 78,118 120,78 168,120 210,92 262,126 320,100 320,200 0,200" fill="#3d2a52"/>' +
      '<polygon points="0,150 50,124 96,142 150,112 204,146 252,128 320,146 320,200 0,200" fill="#2a1f3b"/>' +
      '<polygon points="0,172 70,154 140,170 214,150 320,168 320,200 0,200" fill="#1c1628"/>';
  } else if (id === "physics") {
    sky = ["#20403f", "#141d20"];
    body = _floor(p, 132, "#18272a", "rgba(120,255,230,.07)") +
      _cube(118, 182, 26, "#2fb3a0") + _cube(164, 182, 26, "#2fb3a0") + _cube(141, 141, 26, "#45d1bd") +
      _cube(214, 176, 18, "#e0a03a") + _sphere(p, 0, 236, 58, 13, "#ff6b5a") +
      '<path d="M236 76v10M231 82l5 6 5-6" stroke="#ff6b5a" stroke-width="1.6" fill="none" opacity=".5" stroke-linecap="round"/>';
  } else if (id === "empty") {
    sky = ["#2a2b33", "#17181d"];
    body = _floor(p, 120, "#202127", "rgba(255,255,255,.07)") +
      '<line x1="160" y1="150" x2="200" y2="150" stroke="#d24b4b" stroke-width="2"/><line x1="160" y1="150" x2="160" y2="112" stroke="#57c785" stroke-width="2"/><line x1="160" y1="150" x2="136" y2="166" stroke="#5796e8" stroke-width="2"/><circle cx="160" cy="150" r="3" fill="#fff"/>';
  } else if (id === "materials") {
    sky = ["#2b2b38", "#15151b"];
    body = _floor(p, 130, "#1e1e26", "rgba(255,255,255,.06)") +
      _sphere(p, 0, 92, 116, 24, "#c8ccd6") + _sphere(p, 1, 160, 116, 24, "#b0563a") + _sphere(p, 2, 228, 116, 24, "#8d97ff");
  } else if (id === "character") {
    sky = ["#3d5a80", "#1c2433"];
    body = _floor(p, 128, "#1d2633", "rgba(160,200,255,.08)") + _cube(96, 176, 22, "#6b7b93") + _cube(236, 168, 16, "#6b7b93") +
      '<ellipse cx="160" cy="166" rx="18" ry="5" fill="#000" opacity=".35"/><rect x="146" y="104" width="28" height="62" rx="14" fill="#e8b04a"/><rect x="146" y="104" width="28" height="62" rx="14" fill="url(#' + p + 'cap)"/>' +
      '<defs><linearGradient id="' + p + 'cap" x1="0" x2="1"><stop offset="0" stop-color="#fff" stop-opacity=".35"/><stop offset=".5" stop-color="#fff" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity=".35"/></linearGradient></defs>';
  } else if (id === "lighting") {
    sky = ["#1b1d2c", "#0d0e14"];
    body = _floor(p, 132, "#14151d", "rgba(255,255,255,.04)") +
      '<polygon points="150,0 170,0 214,170 106,170" fill="#ffd27a" opacity=".1"/><ellipse cx="160" cy="168" rx="56" ry="12" fill="#ffd27a" opacity=".14"/>' + _sphere(p, 0, 160, 136, 26, "#e9e4da");
  } else {
    sky = ["#3a2f6b", "#15131f"];
    body = '<ellipse cx="160" cy="112" rx="120" ry="70" fill="#7c8aff" opacity=".18"/>' +
      _floor(p, 136, "#1c1a2a", "rgba(124,138,255,.12)") +
      _sphere(p, 0, 128, 112, 32, "#8d97ff") + _cube(206, 160, 30, "#c25bd6");
  }
  return '<svg viewBox="0 0 320 200" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">' +
    '<defs><linearGradient id="' + p + 'sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="' + sky[0] + '"/><stop offset="1" stop-color="' + sky[1] + '"/></linearGradient></defs>' +
    '<rect width="320" height="200" fill="url(#' + p + 'sky)"/>' + body + '</svg>';
}
function fillThumbs(root) {
  (root || document).querySelectorAll("[data-thumb]").forEach(function (el, i) {
    el.insertAdjacentHTML("afterbegin", sceneSVG(el.getAttribute("data-thumb"), "wlt" + i + "_"));
  });
}

var LEARN_BG = { start: ["#7c8aff", "#3b2f9a"], editor: ["#3fb2a6", "#185259"], physics: ["#e08a5c", "#7d3047"],
  materials: ["#c25bd6", "#4f2380"], lighting: ["#e0b021", "#7d4c10"], scripting: ["#5796e8", "#1e3a73"] };
function fillLearn(root) {
  (root || document).querySelectorAll("[data-learn-bg]").forEach(function (el) {
    var c = LEARN_BG[el.getAttribute("data-learn-bg")] || ["#555", "#222"];
    el.style.background = "linear-gradient(135deg, " + c[0] + ", " + c[1] + ")";
    if (!el.style.position) el.style.position = "relative";
  });
}

  fillThumbs(); fillLearn();
  var S = { start: start, new: document.getElementById("wl-new"), all: document.getElementById("wl-all"), sheet: document.getElementById("wl-sheet"), hub: document.getElementById("wl-hub") };
  var ORDER = ["new", "hub", "all", "start", "sheet"];            // top-most first
  var layout = document.querySelector(".layout");
  var hubQ = document.getElementById("wl-hub-q");
  var params = new URLSearchParams(location.search);
  var lastFocus = null;

  function isOn(n) { return S[n].classList.contains("wl-on"); }
  function top() { for (var i = 0; i < ORDER.length; i++) if (isOn(ORDER[i])) return ORDER[i]; return null; }
  function sync() { if (layout) layout.inert = !!top(); }
  function show(n, instant) {
    var el = S[n];
    if (instant) { el.classList.add("wl-instant"); el.classList.add("wl-on"); void el.offsetWidth; el.classList.remove("wl-instant"); }
    else el.classList.add("wl-on");
    sync(); el.focus({ preventScroll: true });
  }
  function hide(n, instant) {
    var el = S[n];
    if (instant) { el.classList.add("wl-instant"); el.classList.remove("wl-on"); void el.offsetWidth; el.classList.remove("wl-instant"); }
    else el.classList.remove("wl-on");
    sync();
  }
  function toEditor() {
    ORDER.forEach(function (n) { if (isOn(n)) hide(n); });
    if (document.activeElement && document.activeElement !== document.body) document.activeElement.blur();
    if (lastFocus && document.contains(lastFocus) && !lastFocus.classList.contains("wl-project-btn")) { try { lastFocus.focus({ preventScroll: true }); } catch (e) {} }
    lastFocus = null;
  }
  function go(where) {
    if (where === "all") {
      setTab("all", "projects"); show("all");
      setTimeout(function () { if (isOn("all") && isOn("start")) hide("start", true); }, 330);
    } else if (where === "start") {
      show("start", true); hide("all"); S.start.focus({ preventScroll: true });
    } else if (where === "sheet") {
      if (!top()) lastFocus = document.activeElement;
      show("sheet");
    } else if (where === "hub") {
      setTab("hub", "projects"); show("hub"); hide("sheet");
    } else {
      toEditor();
    }
  }
  function openProject(id) {
    if (id === "demo") { console.log("[welcome] open project demo"); toEditor(); }
    else console.log("[welcome] open project " + id + " (mock)");
  }

  // Tabs inside the all-projects panel (v3) and the hub sidebar (v1).
  function setTab(screen, tab) {
    var p = screen === "all" ? "wl3-" : "wl1-";
    var root = S[screen];
    root.querySelectorAll("[data-wl-tab]").forEach(function (b) {
      if (b.classList.contains("wl1-new")) return;
      b.classList.toggle(screen === "all" ? "wl3-on" : "wl1-active", b.getAttribute("data-wl-tab") === tab);
      b.setAttribute("aria-selected", b.getAttribute("data-wl-tab") === tab ? "true" : "false");
    });
    root.querySelectorAll("[data-wl-pane]").forEach(function (pn) { pn.classList.toggle(p + "on", pn.getAttribute("data-wl-pane") === tab); });
    if (screen === "hub") {
      var meta = { projects: ["Projects", "4 projects", "Search projects"], templates: ["Templates", "6 templates", "Search templates"], learn: ["Learn", "6 guides", "Search guides"] }[tab];
      document.getElementById("wl-hub-title").textContent = meta[0];
      document.getElementById("wl-hub-sub").textContent = meta[1];
      hubQ.placeholder = meta[2]; hubQ.value = "";
      S.hub.querySelector(".wl1-select").style.display = tab === "projects" ? "" : "none";
    }
  }

  // One delegated click handler for every welcome surface.
  document.addEventListener("click", function (ev) {
    var t = ev.target.closest && ev.target.closest(".wl-root [data-wl-go], .wl-root [data-wl-project], .wl-root [data-wl-tab], .wl-root [data-wl-action], .wl-root [data-wl-template], .wl-root [data-wl-guide]");
    if (!t) return;
    ev.preventDefault();
    var screen = t.closest("#wl-all") ? "all" : t.closest("#wl-hub") ? "hub" : t.closest("#wl-sheet") ? "sheet" : "start";
    if (t.hasAttribute("data-wl-action")) { ev.stopPropagation(); runAction(t.getAttribute("data-wl-action")); return; }
    if (t.hasAttribute("data-wl-go")) go(t.getAttribute("data-wl-go"));
    else if (t.hasAttribute("data-wl-project")) openProject(t.getAttribute("data-wl-project"));
    else if (t.hasAttribute("data-wl-tab")) setTab(screen, t.getAttribute("data-wl-tab"));
    else if (t.hasAttribute("data-wl-template")) console.log("[welcome] new project from template " + t.getAttribute("data-wl-template") + " (stub)");
    else if (t.hasAttribute("data-wl-guide")) console.log("[welcome] open guide " + t.getAttribute("data-wl-guide") + " (stub)");
  });
  // Chips / grid-list toggles in the hub: purely visual.
  S.hub.addEventListener("click", function (ev) {
    var c = ev.target.closest(".wl1-chip, .wl1-seg button");
    if (!c) return;
    c.parentNode.querySelectorAll(c.tagName).forEach(function (x) { x.classList.toggle(c.classList.contains("wl1-chip") ? "wl1-on" : "wl1-on", x === c); });
  });

  // Start window (Xcode-style): recent projects come from localStorage ("ornis.recentProjects",
  // [{ id, name, path }]); until the editor writes that list we show the mock PROJECTS above.
  var RECENT_KEY = "ornis.recentProjects";
  function loadRecents() {
    try {
      var saved = JSON.parse(localStorage.getItem(RECENT_KEY) || "null");
      if (Array.isArray(saved)) return saved;
    } catch (e) { /* corrupted entry: fall back to mock data */ }
    return PROJECTS;
  }
  var FILE_ICO = '<svg class="wl-ico" viewBox="0 0 24 24"><path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M4 7.5l8 4.5 8-4.5M12 12v9"/></svg>';
  var list = document.getElementById("wl5-list"), listEmpty = document.getElementById("wl5-empty");
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function renderRecents() {
    var items = loadRecents();
    list.innerHTML = items.map(function (p, i) {
      return '<div class="wl5-item' + (i === 0 ? " wl5-sel" : "") + '" role="option" tabindex="0" data-wl5-recent="' + esc(p.id) + '" title="' + esc(p.path) + '">' +
        FILE_ICO + '<div class="wl5-tx"><div class="wl5-n">' + esc(p.name) + '</div><div class="wl5-p">' + esc(p.path) + '</div></div><span class="wl5-dt">' + esc(p.date || "") + "</span></div>";
    }).join("");
    listEmpty.hidden = items.length > 0;
  }
  renderRecents();
  list.addEventListener("click", function (ev) {
    var it = ev.target.closest("[data-wl5-recent]");
    if (!it) return;
    list.querySelectorAll(".wl5-sel").forEach(function (x) { x.classList.remove("wl5-sel"); });
    it.classList.add("wl5-sel");
  });
  list.addEventListener("dblclick", function (ev) {
    var it = ev.target.closest("[data-wl5-recent]");
    if (it) openProject(it.getAttribute("data-wl5-recent"));
  });
  list.addEventListener("keydown", function (ev) {
    var it = ev.target.closest("[data-wl5-recent]");
    if (it && ev.key === "Enter") { ev.preventDefault(); ev.stopPropagation(); openProject(it.getAttribute("data-wl5-recent")); }
  });

  // "Open Existing Project": no native open flow in the mockup yet, so pick a folder via the browser.
  var openInput = document.getElementById("wl5-open-input");
  openInput.addEventListener("change", function () {
    var f = openInput.files && openInput.files[0];
    var name = f ? (f.webkitRelativePath || f.name).split("/")[0] : null;
    openInput.value = "";
    if (name) { console.log("[welcome] open existing project folder " + name); toEditor(); }
  });

  // Template picker ("New Project"). Icons are stroke-only; parts with class wl5-a use the accent.
  var TEMPLATES = [
    { id: "empty", name: "Empty Project", desc: "Blank scene with a camera and light", sec: "games", cat: ["3d", "2d"], ico: '<rect x="8" y="8" width="32" height="32" rx="4"/><path class="wl5-a" d="M24 18v12M18 24h12"/>' },
    { id: "first-person", name: "First-Person", desc: "FPS controller, mouse look, crosshair", sec: "games", cat: ["3d"], ico: '<circle cx="24" cy="24" r="13"/><path class="wl5-a" d="M24 7v8M24 33v8M7 24h8M33 24h8"/><circle class="wl5-a" cx="24" cy="24" r="2"/>' },
    { id: "third-person", name: "Third-Person", desc: "Orbit camera and character movement", sec: "games", cat: ["3d"], ico: '<circle cx="24" cy="13" r="5"/><path d="M15 38v-9a9 9 0 0 1 18 0v9"/><path class="wl5-a" d="M8 42h32"/><path class="wl5-a" d="M6 8l5 3M42 8l-5 3"/>' },
    { id: "top-down", name: "Top-Down", desc: "Isometric camera with click-to-move", sec: "games", cat: ["3d", "2d"], ico: '<rect x="7" y="7" width="34" height="34" rx="3"/><path d="M7 19h34M7 30h34M19 7v34M30 7v34"/><circle class="wl5-a" cx="24.5" cy="24.5" r="3.5"/>' },
    { id: "platformer-2d", name: "Platformer 2D", desc: "Side-scroller with tight jump physics", sec: "games", cat: ["2d"], ico: '<path d="M5 40h14M27 30h16M12 22h12"/><rect class="wl5-a" x="15" y="11" width="6" height="8" rx="1.5"/><path class="wl5-a" d="M26 20q5-6 9 4"/>' },
    { id: "racing", name: "Racing", desc: "Vehicle physics and a lap track", sec: "games", cat: ["3d"], ico: '<path d="M8 30l3-9a4 4 0 0 1 4-3h18a4 4 0 0 1 4 3l3 9v6H8z"/><circle cx="15" cy="36" r="3"/><circle cx="33" cy="36" r="3"/><path class="wl5-a" d="M4 12h10M2 17h8"/>' },
    { id: "open-world", name: "Open World", desc: "Streaming procedural terrain", sec: "games", cat: ["3d"], ico: '<path d="M4 38l11-16 7 9 6-8 16 15z"/><path class="wl5-a" d="M4 43c6-3 10 0 14-2s8-3 12-1 9 1 14-1"/><circle cx="36" cy="11" r="4"/>' },
    { id: "puzzle", name: "Puzzle", desc: "Grid logic and level progression", sec: "games", cat: ["2d", "3d"], ico: '<path d="M10 10h10a4 4 0 1 1 8 0h10v10a4 4 0 1 0 0 8v10H28a4 4 0 1 0-8 0H10V28a4 4 0 1 1 0-8z"/><circle class="wl5-a" cx="24" cy="24" r="3"/>' },
    { id: "multiplayer-arena", name: "Multiplayer Arena", desc: "Server-authoritative arena match", sec: "games", cat: ["multiplayer", "3d"], ico: '<ellipse cx="24" cy="30" rx="18" ry="8"/><circle class="wl5-a" cx="15" cy="15" r="4"/><circle class="wl5-a" cx="33" cy="15" r="4"/><path d="M19 15h10"/><path class="wl5-a" d="M24 11v8"/>' },
    { id: "coop-survival", name: "Co-op Survival", desc: "Shared world for 2–4 players", sec: "games", cat: ["multiplayer", "3d"], ico: '<path d="M24 6l14 5v11c0 9-6 16-14 20-8-4-14-11-14-20V11z"/><path class="wl5-a" d="M17 24l5 5 9-10"/>' },
    { id: "physics-playground", name: "Physics Playground", desc: "Rigid bodies, joints and forces", sec: "tools", cat: ["3d", "other"], ico: '<rect x="8" y="28" width="12" height="12" rx="1.5"/><rect x="20" y="28" width="12" height="12" rx="1.5"/><rect x="14" y="16" width="12" height="12" rx="1.5"/><circle class="wl5-a" cx="36" cy="12" r="5"/><path class="wl5-a" d="M36 20v5M33 23l3 3 3-3"/>' },
    { id: "mesh-editing", name: "Mesh Editing Sample", desc: "Edit vertices and faces in-scene", sec: "tools", cat: ["3d", "other"], ico: '<path d="M24 5l17 10v18L24 43 7 33V15z"/><path d="M7 15l17 10 17-10M24 25v18"/><circle class="wl5-a" cx="24" cy="5" r="2.5"/><circle class="wl5-a" cx="41" cy="15" r="2.5"/><circle class="wl5-a" cx="24" cy="25" r="2.5"/>' },
    { id: "terrain-sample", name: "Procedural Terrain", desc: "Heightmap noise and erosion", sec: "tools", cat: ["3d", "other"], ico: '<path d="M4 34l8-6 6 4 8-10 8 6 10-8"/><path d="M4 42l8-6 6 4 8-10 8 6 10-8" opacity=".5"/><path class="wl5-a" d="M24 6v8M20 10h8"/>' },
    { id: "ui-sample", name: "UI & Menus", desc: "Menus, HUD and settings screens", sec: "tools", cat: ["2d", "other"], ico: '<rect x="7" y="9" width="34" height="30" rx="3"/><path d="M7 16h34"/><path class="wl5-a" d="M13 24h14M13 30h22"/>' },
    { id: "netcode-sample", name: "Netcode Sample", desc: "Replication and client prediction", sec: "tools", cat: ["multiplayer", "other"], ico: '<rect x="6" y="8" width="14" height="10" rx="2"/><rect x="28" y="30" width="14" height="10" rx="2"/><path class="wl5-a" d="M13 18v8h22v4"/><path class="wl5-a" d="M31 27l4 3 4-3"/>' }
  ];
  var SECTIONS = [["games", "Games"], ["tools", "Tools & Samples"]];
  var pick = { cat: "all", q: "", sel: "empty" };
  var body = document.getElementById("wl5-body"), tq = document.getElementById("wl5-q");
  function renderTemplates() {
    var q = pick.q.trim().toLowerCase(), html = "";
    SECTIONS.forEach(function (s) {
      var ts = TEMPLATES.filter(function (t) {
        return t.sec === s[0] && (pick.cat === "all" || t.cat.indexOf(pick.cat) >= 0) && (!q || t.name.toLowerCase().indexOf(q) >= 0);
      });
      if (!ts.length) return;
      html += '<section class="wl5-sec"><h3>' + s[1] + '</h3><div class="wl5-grid">' + ts.map(function (t) {
        return '<button class="wl5-tpl' + (t.id === pick.sel ? " wl5-sel" : "") + '" data-wl5-tpl="' + t.id + '"><svg class="wl-ico" viewBox="0 0 48 48">' +
          t.ico + '</svg><span><b>' + t.name + "</b><small>" + t.desc + "</small></span></button>";
      }).join("") + "</div></section>";
    });
    var cur = TEMPLATES.filter(function (t) { return t.id === pick.sel; })[0];
    document.getElementById("wl5-chosen").textContent = cur ? "Selected: " + cur.name : "";
    body.innerHTML = html || '<div class="wl5-none">No templates match the filter.</div>';
  }
  function openPicker() {
    pick.q = ""; tq.value = "";
    renderTemplates(); show("new");
  }
  S.new.addEventListener("click", function (ev) {
    var c = ev.target.closest("[data-wl5-cat]"), t = ev.target.closest("[data-wl5-tpl]"), b = ev.target.closest("[data-wl5]");
    if (c) {
      pick.cat = c.getAttribute("data-wl5-cat");
      S.new.querySelectorAll("[data-wl5-cat]").forEach(function (x) { x.classList.toggle("wl5-on", x === c); });
      renderTemplates();
    } else if (t) {
      pick.sel = t.getAttribute("data-wl5-tpl"); renderTemplates();
    } else if (b && b.getAttribute("data-wl5") === "cancel") {
      hide("new"); S.start.focus({ preventScroll: true });
    } else if (b && b.getAttribute("data-wl5") === "next") {
      S.new.dataset.template = pick.sel;
      console.log("[welcome] new project from template " + pick.sel);
      toEditor();
    }
  });
  body.addEventListener("dblclick", function (ev) { if (ev.target.closest("[data-wl5-tpl]")) S.new.querySelector('[data-wl5="next"]').click(); });
  tq.addEventListener("input", function () { pick.q = tq.value; renderTemplates(); });
  function runAction(a) {
    if (a === "new") openPicker();
    else if (a === "open") openInput.click();
    else console.log("[welcome] " + a + " (stub)");
  }

  // Keyboard: Enter/Space activates focused cards & rows; Esc steps back; ⌘K focuses the overlay search; ⇧⌘O toggles the sheet.
  window.addEventListener("keydown", function (ev) {
    var k = ev.key, mod = ev.metaKey || ev.ctrlKey, t = top();
    var isO = mod && ev.shiftKey && k.toLowerCase() === "o";
    if (!t) {
      if (isO) { ev.preventDefault(); ev.stopPropagation(); go("sheet"); }
      return;
    }
    var a = document.activeElement;
    if (k === "Escape") {
      ev.preventDefault(); ev.stopPropagation();
      if (a && a.tagName === "INPUT" && a.value) { a.value = ""; a.dispatchEvent(new Event("input")); return; }
      if (t === "new") { hide("new"); S.start.focus({ preventScroll: true }); } else if (t === "all") go("start"); else go("editor");
    } else if (mod && k.toLowerCase() === "k") {
      ev.preventDefault(); ev.stopPropagation();
      if (t === "new") document.getElementById("wl5-q").focus(); else if (t === "hub") hubQ.focus();
    } else if (isO && t === "sheet") {
      ev.preventDefault(); ev.stopPropagation(); go("editor");
    } else if ((k === "Enter" || k === " ") && a && a.closest && a.closest(".wl-root") && a.matches("[data-wl5-recent], [data-wl-project], [data-wl-template], [data-wl-guide], [data-wl-tab], a[data-wl-go]")) {
      ev.preventDefault(); ev.stopPropagation(); a.click();
    }
  }, true);

  // The project name in the top bar opens the sheet.
  var proj = document.querySelector(".center-top .project");
  if (proj) {
    proj.classList.add("wl-project-btn");
    proj.title = "Projects (⇧⌘O)";
    proj.setAttribute("role", "button"); proj.tabIndex = 0;
    proj.insertAdjacentHTML("beforeend", '<svg class="wl-ico" viewBox="0 0 24 24"><path d="M7 10l5 5 5-5"/></svg>');
    proj.addEventListener("click", function () { go("sheet"); });
    proj.addEventListener("keydown", function (ev) { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); go("sheet"); } });
  }

  if (params.has("nowelcome") && !params.has("welcome")) hide("start", true);
  else { sync(); start.focus({ preventScroll: true }); }
})();
