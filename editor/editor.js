// Ornis Editor — REST bridge to the engine (remote.rs on :3420).
// Polls GET /api/scene + GET /api/status and renders the live ECS state
// into the Hierarchy panel, the Inspector and the footer. Commands
// (create/rename/destroy entity, set transform/material) go through
// POST /api/command. The WASM viewport (viewport.js) is separate
// and renders its own static scene.ron.

// ES module: top-level scope is private to this file; loaded as
// <script type="module"> from index.html.

var POLL_MS = 1500;

// ── Live state ──────────────────────────────────────────────────────

var lastScene = null;
var hasLiveScene = false;   // false until the first successful fetch
var selectedKey = null;     // "id:generation" of the selected entity
var lastInspectorKey = '';  // fingerprint of the rendered inspector
var activeDrag = null;      // slider drag in progress (see makeSlider)

function fetchJson(url, options) {
    return fetch(url, options).then(function(res) {
        if (!res.ok) throw new Error(url + ': HTTP ' + res.status);
        return res.json();
    });
}

function sendCommand(type, data) {
    return fetchJson('/api/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type: type, data: data || {} })
    });
}

// ── Helpers ─────────────────────────────────────────────────────────

function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
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
    debounced.cancel = function() {
        clearTimeout(timer);
        timer = null;
    };
    return debounced;
}

function entityKey(entity) {
    return entity.id + ':' + entity.generation;
}

function findEntity(scene, key) {
    var entities = (scene && scene.entities) || [];
    for (var i = 0; i < entities.length; i++) {
        if (entityKey(entities[i]) === key) return entities[i];
    }
    return null;
}

function iconSvg(href) {
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">' +
        '<use href="' + href + '" /></svg>';
}

function rgbToHex(rgb) {
    return '#' + rgb.map(function(c) {
        return Math.round(clamp(c, 0, 1) * 255).toString(16).padStart(2, '0');
    }).join('');
}

function hexToRgb(hex) {
    return [1, 3, 5].map(function(i) {
        return parseInt(hex.slice(i, i + 2), 16) / 255;
    });
}

// ── Hierarchy panel ─────────────────────────────────────────────────

function hierarchyPanel() {
    return document.getElementById('hierarchy-panel');
}

// Canonical scene shape: entity.components is a map keyed by the
// registry component name (serde-canonical payloads).
function entityNameOf(entity) {
    var components = entity.components || {};
    return components.Name || null;
}

function entityLabel(entity) {
    var name = entityNameOf(entity);
    if (name) return name;
    return 'Entity ' + entity.id + ' (gen ' + entity.generation + ')';
}

function buildEntityNode(entity) {
    var details = document.createElement('details');
    details.open = true;
    details.dataset.key = entityKey(entity);

    var summary = document.createElement('summary');
    var left = document.createElement('div');
    left.className = 'left';

    var arrow = document.createElement('div');
    arrow.className = 'arrow';
    arrow.style.opacity = '0';

    var icon = document.createElement('div');
    icon.className = 'icon custom';
    icon.innerHTML = iconSvg('icons/circle.svg#icon');

    var label = document.createElement('span');
    label.textContent = entityLabel(entity);
    label.title = 'id ' + entity.id + ', generation ' + entity.generation +
        ' · components: ' + Object.keys(entity.components || {}).join(', ');

    left.appendChild(arrow);
    left.appendChild(icon);
    left.appendChild(label);
    summary.appendChild(left);
    details.appendChild(summary);

    if (details.dataset.key === selectedKey) {
        details.classList.add('active');
    }

    summary.addEventListener('click', function() {
        selectEntity(details.dataset.key);
    });

    return details;
}

function renderHierarchy(scene) {
    var panel = hierarchyPanel();
    if (!panel) return;
    panel.textContent = '';
    var entities = (scene && scene.entities) || [];
    if (entities.length === 0) {
        var empty = document.createElement('div');
        empty.className = 'hierarchy-empty';
        empty.textContent = 'No entities — press + to create one';
        panel.appendChild(empty);
        return;
    }
    entities.forEach(function(entity) {
        panel.appendChild(buildEntityNode(entity));
    });
}

function selectEntity(key) {
    selectedKey = key;
    lastInspectorKey = ''; // force inspector re-render
    var panel = hierarchyPanel();
    if (panel) {
        panel.querySelectorAll('details.active').forEach(function(node) {
            node.classList.remove('active');
        });
        var node = panel.querySelector('details[data-key="' + key + '"]');
        if (node) node.classList.add('active');
    }
    syncInspector();
}

function refreshScene() {
    return fetchJson('/api/scene')
        .then(function(scene) {
            hasLiveScene = true;
            lastScene = scene;
            if (selectedKey && !findEntity(scene, selectedKey)) {
                selectedKey = null; // selected entity is gone
            }
            renderHierarchy(scene);
            syncInspector();
        })
        .catch(function() { /* engine offline — keep last state */ });
}

// ── Inspector panel ─────────────────────────────────────────────────

function inspectorPanel() {
    return document.getElementById('inspector');
}

function inspectorHasFocus() {
    var panel = inspectorPanel();
    if (!panel) return false;
    var active = document.activeElement;
    return active && active.tagName === 'INPUT' && panel.contains(active);
}

// Re-renders the inspector when the selected entity changed, unless the
// user is editing it right now (dragging a slider or typing in a field).
function syncInspector() {
    if (!hasLiveScene) return; // old server — keep the static mockup
    var entity = selectedKey ? findEntity(lastScene, selectedKey) : null;
    var key = entity
        ? JSON.stringify([entity.id, entity.generation, entity.components])
        : 'none';
    if (key === lastInspectorKey) return;
    if (activeDrag || inspectorHasFocus()) return;
    lastInspectorKey = key;
    buildInspector(entity);
}

function buildDetails(title, iconHref, color) {
    var details = el('details');
    details.open = true;

    var summary = el('summary');
    var left = el('div', 'left');
    left.appendChild(el('div', 'arrow'));
    var icon = el('div', 'icon');
    icon.style.fill = color;
    icon.innerHTML = iconSvg(iconHref);
    left.appendChild(icon);
    left.appendChild(document.createTextNode(title));

    var right = el('div', 'right');
    var dots = el('div', 'icon');
    dots.innerHTML = iconSvg('icons/dots-vertical.svg#icon');
    right.appendChild(dots);

    summary.appendChild(left);
    summary.appendChild(right);
    details.appendChild(summary);

    var content = el('div', 'content');
    details.appendChild(content);
    return details;
}

// Drag-slider in the style of sliders.js (pointer-lock drag), but bound
// dynamically — inspector DOM is rebuilt on every selection change.
// opts.onChange(value, final) fires on every step; final=true on release.
function makeSlider(opts) {
    var min = opts.min;
    var max = opts.max;
    var step = opts.step;
    var decimals = (typeof opts.decimals === 'number') ? opts.decimals : 1;
    var value = clamp(parseFloat(opts.value) || 0, min, max);

    var slider = el('div', 'slider');
    var fill = el('div', 'fill');
    var valueEl = el('span', 'value');
    slider.appendChild(fill);
    slider.appendChild(valueEl);

    function render() {
        var percent = (value - min) / (max - min) || 0;
        fill.style.width = (percent * 100) + '%';
        valueEl.textContent = value.toFixed(decimals);
    }

    function setValue(v) {
        value = parseFloat(clamp(Math.round(v / step) * step, min, max).toFixed(6));
        render();
    }

    slider.addEventListener('mousedown', function(e) {
        if (e.button !== 0) return;
        activeDrag = {
            onMove: function(dx) {
                setValue(value + dx * step);
                opts.onChange(value, false);
            },
            onEnd: function() {
                opts.onChange(value, true);
            }
        };
        if (slider.requestPointerLock) slider.requestPointerLock();
        e.preventDefault();
    });

    setValue(value);
    return slider;
}

document.addEventListener('mousemove', function(e) {
    if (activeDrag) activeDrag.onMove(e.movementX || 0);
});
document.addEventListener('mouseup', function() {
    if (!activeDrag) return;
    var drag = activeDrag;
    activeDrag = null;
    if (document.exitPointerLock) document.exitPointerLock();
    drag.onEnd();
});

function buildNameRow(panel, entity) {
    var row = el('div', 'name');
    var icon = el('div', 'icon');
    icon.innerHTML = iconSvg('icons/circle.svg#icon');
    var input = document.createElement('input');
    input.type = 'text';
    input.placeholder = 'Name';
    input.value = entityNameOf(entity) || '';
    input.addEventListener('change', function() {
        var name = input.value.trim();
        if (!name || name === entityNameOf(entity)) return;
        // Name is a newtype component: its canonical payload is a
        // bare JSON string.
        sendCommand('set_component', {
            id: entity.id,
            generation: entity.generation,
            component: 'Name',
            value: name
        }).then(refreshScene).catch(function() {});
    });
    row.appendChild(icon);
    row.appendChild(input);
    panel.appendChild(row);
}

function buildTransform(panel, entity) {
    var transform = (entity.components || {}).Transform;
    if (!transform || !Array.isArray(transform.translation)) return;

    var translation = transform.translation.slice(0, 3);
    var details = buildDetails('Transform', 'icons/transform-gizmo.svg#icon', '#5796e8');
    var content = details.querySelector('.content');

    function sendNow() {
        // set_component replaces the WHOLE component: resend the
        // other fields from the last scene snapshot.
        sendCommand('set_component', {
            id: entity.id,
            generation: entity.generation,
            component: 'Transform',
            value: {
                translation: translation.slice(),
                rotation: transform.rotation,
                scale: transform.scale
            }
        }).then(refreshScene).then(refreshStatus).catch(function() {});
    }
    var sendDebounced = debounce(sendNow, 300);

    ['X', 'Y', 'Z'].forEach(function(axis, i) {
        var item = el('div', 'item');
        item.appendChild(el('span', null, 'Translation ' + axis));
        item.appendChild(makeSlider({
            min: -50,
            max: 50,
            step: 0.1,
            decimals: 1,
            value: translation[i],
            onChange: function(v, final) {
                translation[i] = v;
                if (final) {
                    sendDebounced.cancel();
                    sendNow();
                } else {
                    sendDebounced();
                }
            }
        }));
        content.appendChild(item);
    });

    panel.appendChild(details);
}

function buildMaterial(panel, entity) {
    var material = (entity.components || {}).Material;
    if (!material) return;

    // Canonical serde shape: {"Dielectric": {...}}, {"Metal": {...}}, …
    var variantName = Object.keys(material)[0];
    if (!variantName) return;
    var variant = material[variantName];

    var baseColor = Array.isArray(variant.base_color)
        ? variant.base_color.slice(0, 3) : [1, 1, 1];
    var roughness = (typeof variant.roughness === 'number')
        ? variant.roughness : 0.5;

    var details = buildDetails('Material', 'icons/shapes.svg#icon', '#a156d6');
    var content = details.querySelector('.content');

    function sendNow() {
        // Resend the whole variant: untouched fields are preserved,
        // only base_color/roughness get the edited values.
        var fields = {};
        Object.keys(variant).forEach(function(key) { fields[key] = variant[key]; });
        fields.base_color = baseColor.slice();
        if ('roughness' in fields) fields.roughness = roughness;
        var value = {};
        value[variantName] = fields;
        sendCommand('set_component', {
            id: entity.id,
            generation: entity.generation,
            component: 'Material',
            value: value
        }).then(refreshScene).then(refreshStatus).catch(function() {});
    }
    var sendDebounced = debounce(sendNow, 300);

    var colorItem = el('div', 'item');
    colorItem.appendChild(el('span', null, 'Base Color'));
    var colorInput = document.createElement('input');
    colorInput.type = 'color';
    colorInput.className = 'material-color';
    colorInput.value = rgbToHex(baseColor);
    colorInput.addEventListener('input', function() {
        baseColor = hexToRgb(colorInput.value);
        sendDebounced();
    });
    colorItem.appendChild(colorInput);
    content.appendChild(colorItem);

    var roughnessItem = el('div', 'item');
    roughnessItem.appendChild(el('span', null, 'Roughness'));
    roughnessItem.appendChild(makeSlider({
        min: 0,
        max: 1,
        step: 0.01,
        decimals: 2,
        value: roughness,
        onChange: function(v, final) {
            roughness = v;
            if (final) {
                sendDebounced.cancel();
                sendNow();
            } else {
                sendDebounced();
            }
        }
    }));
    content.appendChild(roughnessItem);

    panel.appendChild(details);
}

function buildInspector(entity) {
    var panel = inspectorPanel();
    if (!panel) return;

    // Drop the static mockup blocks and previous live content,
    // keep only the tab list.
    Array.prototype.slice.call(panel.children).forEach(function(child) {
        if (!child.classList.contains('tab-list')) panel.removeChild(child);
    });

    if (!entity) {
        panel.appendChild(el('div', 'inspector-empty', 'Select an entity'));
        return;
    }

    buildNameRow(panel, entity);
    buildTransform(panel, entity);
    buildMaterial(panel, entity);

    var deleteBtn = el('button', 'bottom-button', 'Delete Entity');
    deleteBtn.addEventListener('click', deleteSelected);
    panel.appendChild(deleteBtn);
}

function deleteSelected() {
    if (!selectedKey || !lastScene) return;
    var entity = findEntity(lastScene, selectedKey);
    if (!entity) return;
    selectedKey = null;
    sendCommand('destroy_entity', {
        id: entity.id,
        generation: entity.generation
    }).then(refreshScene).then(refreshStatus).catch(function() {});
}

function initHierarchyKeys() {
    var panel = hierarchyPanel();
    if (!panel) return;
    panel.addEventListener('keydown', function(e) {
        if (e.key === 'Delete' || e.key === 'Backspace') {
            deleteSelected();
            e.preventDefault();
        }
    });
}

// ── Status (footer) ─────────────────────────────────────────────────

function renderStatus(status) {
    var statusEl = document.getElementById('status-entities');
    if (!statusEl) return;
    var count = (status && typeof status.entity_count === 'number')
        ? status.entity_count : 0;
    statusEl.textContent = 'Entities: ' + count;
}

function refreshStatus() {
    return fetchJson('/api/status')
        .then(renderStatus)
        .catch(function() { /* engine offline — keep last state */ });
}

// ── Commands ────────────────────────────────────────────────────────

function initCreateEntity() {
    // Reuse the "+" icon in the Hierarchy tab list.
    var panel = document.querySelector('.panel.left-upper');
    if (!panel) return;
    var btn = panel.querySelector('.tab-list .new-tab');
    if (!btn) return;
    btn.style.cursor = 'pointer';
    btn.addEventListener('click', function() {
        sendCommand('create_entity', {})
            .then(refreshScene)
            .then(refreshStatus)
            .catch(function() {});
    });
}

// ── File menu (save/reload scene) ─────────────────────────────────

// File-op results arrive as CustomEvents on /api/events
// (scene_saved/scene_loaded/error); the latest one is shown in the footer.
function showFileStatus(text) {
    var node = document.getElementById('status-file');
    if (node) node.textContent = text;
}

function initFileMenu() {
    var menu = document.getElementById('file-menu');
    if (!menu) return;
    // Hover opens the dropdown via CSS; click toggles it too (touch).
    menu.querySelector('span').addEventListener('click', function() {
        menu.classList.toggle('open');
    });
    document.addEventListener('click', function(e) {
        if (!menu.contains(e.target)) menu.classList.remove('open');
    });

    var save = document.getElementById('file-save');
    var reload = document.getElementById('file-reload');
    if (save) save.addEventListener('click', function() {
        menu.classList.remove('open');
        showFileStatus('Saving…');
        sendCommand('save_scene', {})
            .then(refreshEvents)
            .catch(function() { showFileStatus('Save failed: engine offline'); });
    });
    if (reload) reload.addEventListener('click', function() {
        menu.classList.remove('open');
        showFileStatus('Reloading…');
        sendCommand('load_scene', {})
            .then(refreshEvents)
            .then(refreshScene)
            .then(refreshStatus)
            .catch(function() { showFileStatus('Reload failed: engine offline'); });
    });
}

// Poll /api/events and surface scene save/load results in the footer.
// The endpoint drains its buffer per request — this is its only consumer.
function refreshEvents() {
    return fetchJson('/api/events')
        .then(function(events) {
            events.forEach(function(ev) {
                var custom = ev && ev.CustomEvent;
                if (!custom) return;
                var data = {};
                try { data = JSON.parse(custom.json_data); } catch (e) { return; }
                if (custom.cmd_type === 'scene_saved') {
                    showFileStatus('Saved ' + (data.path || 'scene') + ' (v' + data.version + ')');
                } else if (custom.cmd_type === 'scene_loaded') {
                    showFileStatus('Loaded ' + (data.path || 'scene') + ' — ' +
                        data.entity_count + ' entities (v' + data.version + ')');
                    refreshScene();
                    refreshStatus();
                } else if (custom.cmd_type === 'error') {
                    showFileStatus('Error (' + data.command + '): ' + data.message);
                }
            });
        })
        .catch(function() { /* engine offline — keep last state */ });
}

// ── Init ────────────────────────────────────────────────────────────

function init() {
    initCreateEntity();
    initHierarchyKeys();
    initFileMenu();
    refreshScene();
    refreshStatus();
    setInterval(refreshScene, POLL_MS);
    setInterval(refreshStatus, POLL_MS);
    setInterval(refreshEvents, POLL_MS);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

window.OrnisEditor = {
    refreshScene: refreshScene,
    refreshStatus: refreshStatus,
    refreshEvents: refreshEvents,
    sendCommand: sendCommand,
    selectEntity: selectEntity
};

