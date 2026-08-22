// Ornis Editor — REST bridge to the engine (remote.rs on :3420).
// Polls GET /api/scene + GET /api/status and renders the live ECS state
// into the Hierarchy panel and the footer. Commands (create entity) go
// through POST /api/command. The WASM viewport (viewport.js) is separate
// and renders its own static scene.ron.

(function() {
    'use strict';

    var POLL_MS = 1500;

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

        var summary = document.createElement('summary');
        var left = document.createElement('div');
        left.className = 'left';

        var arrow = document.createElement('div');
        arrow.className = 'arrow';
        arrow.style.opacity = '0';

        var icon = document.createElement('div');
        icon.className = 'icon custom';
        icon.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">' +
            '<use href="icons/circle.svg#icon" /></svg>';

        var label = document.createElement('span');
        label.textContent = entityLabel(entity);
        label.title = 'id ' + entity.id + ', generation ' + entity.generation +
            ' · components: ' + (entity.components || []).join(', ');

        left.appendChild(arrow);
        left.appendChild(icon);
        left.appendChild(label);
        summary.appendChild(left);
        details.appendChild(summary);

        summary.addEventListener('click', function() {
            var panel = hierarchyPanel();
            if (!panel) return;
            panel.querySelectorAll('details.active').forEach(function(el) {
                el.classList.remove('active');
            });
            details.classList.add('active');
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

    function refreshScene() {
        return fetchJson('/api/scene')
            .then(renderHierarchy)
            .catch(function() { /* engine offline — keep last state */ });
    }

    // ── Status (footer) ─────────────────────────────────────────────────

    function renderStatus(status) {
        var el = document.getElementById('status-entities');
        if (!el) return;
        var count = (status && typeof status.entity_count === 'number')
            ? status.entity_count : 0;
        el.textContent = 'Entities: ' + count;
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
        sendCommand: sendCommand
    };

})();
