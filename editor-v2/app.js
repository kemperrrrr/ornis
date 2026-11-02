
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
    var rec = (typeof loadRecents === "function" ? loadRecents() : PROJECTS) || PROJECTS;
    var p = rec.filter(function (x) { return x && x.id === id; })[0] ||
      PROJECTS.filter(function (x) { return x.id === id; })[0];
    if (window.OrnisEditor) {
      if (id === "demo") window.OrnisEditor.openProject(null, "demo");
      else if (p && /\.(ron|glb|gltf)$/i.test(p.path || "")) {
        window.OrnisEditor.openScene(p.path, p.name);
      } else {
        window.OrnisEditor.openProject(null, (p && p.name) || id);
        console.log("[welcome] project " + id + ": pass a scene path or use the native shell to switch projects");
      }
    }
    toEditor();
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
    else if (t.hasAttribute("data-wl-template")) {
      var tpl = t.getAttribute("data-wl-template");
      if (tpl === "empty" && window.OrnisEditor) { window.OrnisEditor.newEmptyProject(); toEditor(); }
      else window.alert("Template \"" + tpl + "\" is not implemented yet — only Empty Project creates a live scene.");
    }
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
  var FILE_ICO = '<svg class="wl-ico" viewBox="0 0 24 24"><path d="M6.5 3.5h7l4 4v13h-11z"/><path d="M13.5 3.5v4h4M9 12.5h6M9 15.5h6M9 18h3.5"/></svg>';
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

  // "Open Existing Project": the browser cannot reveal a folder's absolute
  // path, so ask for the scene asset path the server can resolve.
  var openInput = document.getElementById("wl5-open-input");
  openInput.addEventListener("change", function () {
    openInput.value = "";
    if (window.OrnisEditor) window.OrnisEditor.promptOpenScene();
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
      if (pick.sel === "empty" && window.OrnisEditor) {
        window.OrnisEditor.newEmptyProject();
        toEditor();
      } else {
        window.alert("Template \"" + pick.sel + "\" is not implemented yet — only Empty Project creates a live scene.");
      }
    }
  });
  body.addEventListener("dblclick", function (ev) { if (ev.target.closest("[data-wl5-tpl]")) S.new.querySelector('[data-wl5="next"]').click(); });
  tq.addEventListener("input", function () { pick.q = tq.value; renderTemplates(); });
  function runAction(a) {
    if (a === "new") openPicker();
    else if (a === "open") {
      if (window.OrnisEditor) window.OrnisEditor.promptOpenScene();
      else openInput.click();
    } else console.log("[welcome] " + a + " (stub)");
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
