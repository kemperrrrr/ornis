// Ornis Editor — REST bridge to the engine (remote.rs on :3420).
// Polls GET /api/scene + GET /api/status and renders the live ECS state
// into the Hierarchy panel, the Inspector and the footer. Commands
// (create/rename/destroy entity, set transform/material) go through
// POST /api/command. The WASM viewport (viewport.js) is separate
// and renders its own static scene.ron.

(function() {
    'use strict';

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

    function entityLabel(entity) {
        if (entity.name) return entity.name;
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
            ' · components: ' + (entity.components || []).join(', ');

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
            ? JSON.stringify([entity.id, entity.generation, entity.name,
                entity.transform, entity.material])
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
        input.value = entity.name || '';
        input.addEventListener('change', function() {
            var name = input.value.trim();
            if (!name || name === entity.name) return;
            sendCommand('rename_entity', {
                id: entity.id,
                generation: entity.generation,
                name: name
            }).then(refreshScene).catch(function() {});
        });
        row.appendChild(icon);
        row.appendChild(input);
        panel.appendChild(row);
    }

    function buildTransform(panel, entity) {
        var transform = entity.transform;
        if (!transform || !Array.isArray(transform.translation)) return;

        var translation = transform.translation.slice(0, 3);
        var details = buildDetails('Transform', 'icons/transform-gizmo.svg#icon', '#5796e8');
        var content = details.querySelector('.content');

        function sendNow() {
            sendCommand('set_transform', {
                id: entity.id,
                generation: entity.generation,
                translation: translation.slice()
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
        var material = entity.material;
        if (!material) return;

        var baseColor = Array.isArray(material.base_color)
            ? material.base_color.slice(0, 3) : [1, 1, 1];
        var roughness = (typeof material.roughness === 'number')
            ? material.roughness : 0.5;

        var details = buildDetails('Material', 'icons/shapes.svg#icon', '#a156d6');
        var content = details.querySelector('.content');

        function sendNow() {
            sendCommand('set_material', {
                id: entity.id,
                generation: entity.generation,
                material: {
                    kind: material.kind || 'dielectric',
                    base_color: baseColor.slice(),
                    roughness: roughness
                }
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

    // ── Init ────────────────────────────────────────────────────────────

    function init() {
        initCreateEntity();
        initHierarchyKeys();
        refreshScene();
        refreshStatus();
        setInterval(refreshScene, POLL_MS);
        setInterval(refreshStatus, POLL_MS);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    window.OrnisEditor = {
        refreshScene: refreshScene,
        refreshStatus: refreshStatus,
        sendCommand: sendCommand,
        selectEntity: selectEntity
    };

})();
