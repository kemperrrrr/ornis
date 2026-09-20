// Ornis editor-v2 mockup logic (static demo data, no backend).
// Sample scene mirrors editor/scene.ron: five cubes + two lights + camera.
(function () {
  "use strict";

  var ICONS = "../editor/icons/";
  function icon(name) { return ICONS + name + ".svg#icon"; }

  var entities = [
    { id: "red", name: "Red Sphere", kind: "mesh", icon: "cube",
      comps: ["transform", "material", "physics"],
      pos: [-5.6, 0, 0], color: "#cc3333", rough: 0.5, metal: false, mass: 12, body: "Dynamic", enabled: true },
    { id: "green", name: "Green Rough Sphere", kind: "mesh", icon: "cube",
      comps: ["transform", "material", "physics"],
      pos: [-2.8, 0, 0], color: "#33aa33", rough: 0.7, metal: false, mass: 9, body: "Dynamic", enabled: true },
    { id: "blue", name: "Blue Smooth Sphere", kind: "mesh", icon: "cube",
      comps: ["transform", "material", "physics"],
      pos: [0, 0, 0], color: "#3355cc", rough: 0.1, metal: false, mass: 9, body: "Static", enabled: true },
    { id: "gold", name: "Gold Sphere", kind: "mesh", icon: "cube",
      comps: ["transform", "material", "physics"],
      pos: [2.8, 0, 0], color: "#e6b800", rough: 0.2, metal: true, mass: 20, body: "Dynamic", enabled: false },
    { id: "ceramic", name: "Ceramic Sphere", kind: "mesh", icon: "cube",
      comps: ["transform", "material", "physics"],
      pos: [5.6, 0, 0], color: "#e8e8e8", rough: 0.35, metal: false, mass: 7, body: "Kinematic", enabled: true },
    { id: "key", name: "Key Light", kind: "light", icon: "lightbulb",
      comps: ["transform", "light"],
      pos: [3, 5, 2], color: "#fff4e0", intensity: 0.6, enabled: true },
    { id: "fill", name: "Fill Light", kind: "light", icon: "lightbulb",
      comps: ["transform", "light"],
      pos: [-4, 3, -2], color: "#ccd6ff", intensity: 0.3, enabled: true },
    { id: "cam", name: "Orbit Camera", kind: "camera", icon: "camera",
      comps: ["transform", "camera"],
      pos: [0, 2.5, 9], fov: 60, enabled: true },
  ];
  var COMP_DEFS = {
    transform: { label: "Transform", icon: "transform-gizmo" },
    material: { label: "Material", icon: "select-color" },
    physics: { label: "Physics Body", icon: "weight" },
    light: { label: "Light", icon: "lightbulb" },
    camera: { label: "Camera", icon: "camera" },
  };
  var sceneTree = [
    { label: "demo", icon: "cube",
      children: ["red", "green", "blue", "gold", "ceramic", "key", "fill", "cam"] },
  ];
  var FS = { name: "Project", children: [
    { name: "Scenes", children: [
      { name: "demo.ron", file: { icon: "file" } },
      { name: "Village", children: [
        { name: "exterior.ron", file: { icon: "file" } },
        { name: "interior.ron", file: { icon: "file" } },
      ]},
    ]},
    { name: "Materials", children: [
      { name: "Metals", children: [
        { name: "Gold material", file: { icon: "select-color" } },
      ]},
      { name: "Ceramics", children: [
        { name: "Ceramic material", file: { icon: "select-color" } },
      ]},
    ]},
    { name: "Prefabs", children: [
      { name: "Player spawn", file: { icon: "cube" } },
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
    caret.className = "caret" + (opts.collapsible ? " clickable" : "");
    caret.textContent = opts.collapsible ? (opts.collapsed ? "▸" : "▾") : "";
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
          if (id) r.addEventListener("click", function () { select(id); });
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

  function numVec(values, onChange) {
    var wrap = document.createElement("div");
    wrap.className = "vec";
    values.forEach(function (v, i) {
      var inp = document.createElement("input");
      inp.type = "number"; inp.step = "0.1"; inp.value = v;
      inp.addEventListener("change", function () { onChange(i, parseFloat(inp.value) || 0); });
      wrap.appendChild(inp);
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
    use.setAttribute("href", icon(iconName || "cube"));
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

    var t = section("Transform", true, "transform-gizmo");
    var f = document.createElement("div"); f.className = "field";
    var lab = document.createElement("span"); lab.textContent = "Position";
    f.appendChild(lab);
    f.appendChild(numVec(e.pos, function (i, v) { e.pos[i] = v; log("moved " + e.name + " → [" + e.pos.join(", ") + "]"); }));
    t.body.appendChild(f);
    inspector.appendChild(t.root);

    if (e.kind === "mesh") {
    var m = section("Material", true, "select-color");
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
      var cm = section("Camera", true, "camera");
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
    if (e && nameInput.value.trim()) { e.name = nameInput.value.trim(); renderTree(); renderShapes(); }
  });

  // ---- project browser: folder tree (left) + contents (right) ----
  var folderTree = document.getElementById("folder-tree");
  var contentList = document.getElementById("content-list");
  var assetFilter = document.getElementById("asset-filter");
  var crumb = document.getElementById("crumb");
  var selectedFolder = []; // path of folder names from project root

  function fsNodeAt(path) {
    var node = FS;
    for (var i = 0; i < path.length; i++) {
      node = (node.children || []).filter(function (c) { return !c.file; })
        .find(function (c) { return c.name === path[i]; });
      if (!node) return FS;
    }
    return node;
  }

  function appendFolderNode(node, parent, path) {
    var key = "fp:" + path.join("/");
    var isCollapsed = !!collapsed[key];
    var isSel = path.length === selectedFolder.length &&
      path.every(function (v, i) { return v === selectedFolder[i]; });
    var r = row(node.name, isCollapsed ? "folder" : "folder-open",
      { selected: isSel, collapsible: true, collapsed: isCollapsed,
        onToggle: function () { collapsed[key] = !collapsed[key]; renderBrowser(); } });
    r.addEventListener("click", function () { selectedFolder = path; renderBrowser(); });
    parent.appendChild(r);
    if (!isCollapsed) {
      var sub = document.createElement("div");
      sub.className = "tree-children";
      (node.children || []).forEach(function (c) {
        if (!c.file) appendFolderNode(c, sub, path.concat([c.name]));
      });
      if (sub.children.length) parent.appendChild(sub);
    }
  }

  function renderFolderTree() {
    folderTree.innerHTML = "";
    (FS.children || []).forEach(function (node) {
      if (!node.file) appendFolderNode(node, folderTree, [node.name]);
    });
  }

  function renderContents() {
    crumb.innerHTML = "";
    var root = document.createElement("button");
    root.textContent = "Project";
    root.addEventListener("click", function () { selectedFolder = []; renderBrowser(); });
    crumb.appendChild(root);
    selectedFolder.forEach(function (seg, i) {
      crumb.appendChild(document.createTextNode(" › "));
      var b = document.createElement("button");
      b.textContent = seg;
      b.addEventListener("click", function () {
        selectedFolder = selectedFolder.slice(0, i + 1);
        renderBrowser();
      });
      crumb.appendChild(b);
    });
    contentList.innerHTML = "";
    var q = assetFilter.value.trim().toLowerCase();
    var node = fsNodeAt(selectedFolder);
    (node.children || []).forEach(function (child) {
      if (q && child.name.toLowerCase().indexOf(q) < 0) return;
      var r = row(child.name, child.file ? child.file.icon : "folder", {});
      if (!child.file) {
        r.addEventListener("click", function () {
          selectedFolder = selectedFolder.concat([child.name]);
          renderBrowser();
        });
      }
      contentList.appendChild(r);
    });
  }

  function renderBrowser() { renderFolderTree(); renderContents(); }
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

  // ---- toolbar wiring (fake): single play/stop toggle ----
  var pill = document.getElementById("status-pill");
  var playing = false;
  var playIcon = document.getElementById("play-icon");
  function setPlaying(on) {
    playing = on;
    document.getElementById("btn-play").classList.toggle("playing", on);
    document.getElementById("btn-play").title = on ? "Stop" : "Play";
    playIcon.setAttribute("href", icon(on ? "stop" : "play"));
    pill.textContent = on ? "Simulating…" : "Ready";
    log(on ? "simulate started (mock)" : "simulate stopped (mock)");
  }
  document.getElementById("btn-play").addEventListener("click", function () { setPlaying(!playing); });
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
  toggleLeft.addEventListener("click", function () {
    layout.classList.toggle("hide-left");
  });
  toggleRight.addEventListener("click", function () {
    layout.classList.toggle("hide-right");
  });
  document.getElementById("add-component").addEventListener("click", function () {
    log("Add Component — picker comes in the next iteration (mock)");
  });
  document.getElementById("add-entity").addEventListener("click", function () {
    log("Add Entity — creation flow comes in the next iteration (mock)");
  });

  treeFilter.addEventListener("input", renderTree);
  assetFilter.addEventListener("input", renderContents);

  renderTree(); renderShapes(); renderInspector(); renderBrowser();
})();
