        // Ornis Engine — WebGPU WASM viewport
        // Build: wasm-pack build crates/wasm --target web --out-dir editor/pkg
        import init, { start_renderer, set_viewport_mode, viewport_state_json } from './pkg/ornis_wasm.js';

        const canvasEl = document.getElementById('bevy');
        const viewportEl = document.querySelector('.viewport');
        const gridEl = document.getElementById('plane-grid');
        const modeBtn = document.getElementById('viewport-mode');

        // Pointer events (orbit camera, handled inside the WASM module)
        // need touch-action:none so the browser doesn't hijack drags.
        if (canvasEl) canvasEl.style.touchAction = 'none';

        // The WASM module owns the backing store (it resizes every frame
        // at devicePixelRatio); this boot-time resize only prevents a
        // wrong-size flash before the first frame.
        const resizeCanvas = () => {
            if (!viewportEl || !canvasEl) return;
            const dpr = window.devicePixelRatio || 1;
            canvasEl.width = Math.max(1, Math.round(viewportEl.clientWidth * dpr));
            canvasEl.height = Math.max(1, Math.round(viewportEl.clientHeight * dpr));
        };
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        // 2D/3D toggle next to canvas#bevy. The WASM module owns the mode;
        // the button only queues the switch and mirrors the applied mode
        // from `viewport_state_json().mode` — the loop applies the request
        // on the next frame, so flipping eagerly would desync on a rejected
        // or still-pending request.
        if (modeBtn) {
            modeBtn.addEventListener('click', () => {
                if (typeof set_viewport_mode !== 'function') return;
                set_viewport_mode(currentMode() === 'plane2d' ? 'perspective3d' : 'plane2d');
            });
        }

        // The applied viewport mode from the wire state ('perspective3d'
        // until the loop reports otherwise).
        const currentMode = () => {
            try {
                const state = JSON.parse(viewport_state_json());
                return state && state.mode === 'plane2d' ? 'plane2d' : 'perspective3d';
            } catch (e) {
                return 'perspective3d';
            }
        };

        // Overlay: pointer-events:none div over the canvas, driven by
        // viewport_state_json(). Minor/major lines via layered gradients;
        // the mode button mirrors the applied mode in the same pass.
        const paintOverlay = () => {
            if (typeof viewport_state_json !== 'function') return;
            let state = null;
            try { state = JSON.parse(viewport_state_json()); } catch (e) { /* keep overlay hidden */ }
            const is2d = !!(state && state.mode === 'plane2d');
            if (modeBtn) {
                modeBtn.textContent = is2d ? '2D' : '3D';
                modeBtn.setAttribute('aria-pressed', String(is2d));
            }
            const grid = is2d ? state.grid : null;
            if (!gridEl || !grid) {
                if (gridEl) gridEl.hidden = true;
                return;
            }
            gridEl.hidden = false;
            // Wire grid values are backing-store (device) pixels; the
            // overlay paints in CSS pixels.
            const dpr = window.devicePixelRatio || 1;
            const minor = grid.minor_step_px / dpr;
            const major = minor * grid.major_every;
            const ox = grid.origin_px[0] / dpr;
            const oy = grid.origin_px[1] / dpr;
            gridEl.style.backgroundImage =
                'linear-gradient(rgba(255,255,255,.10) 1px, transparent 1px),' +
                'linear-gradient(90deg, rgba(255,255,255,.10) 1px, transparent 1px),' +
                'linear-gradient(rgba(255,255,255,.22) 1px, transparent 1px),' +
                'linear-gradient(90deg, rgba(255,255,255,.22) 1px, transparent 1px)';
            gridEl.style.backgroundSize =
                `${minor}px ${minor}px, ${minor}px ${minor}px, ${major}px ${major}px, ${major}px ${major}px`;
            gridEl.style.backgroundPosition =
                `${ox}px ${oy}px, ${ox}px ${oy}px, ${ox}px ${oy}px, ${ox}px ${oy}px`;
        };
        const overlayLoop = () => { paintOverlay(); requestAnimationFrame(overlayLoop); };

        async function boot() {
            console.log('[ornis-editor] loading WASM...');
            await init();
            console.log('[ornis-editor] WASM loaded, starting renderer...');
            await start_renderer('bevy');
            console.log('[ornis-editor] renderer started');
            requestAnimationFrame(overlayLoop);
        }
        boot().catch(err => {
            console.error('[ornis-editor] failed to start:', err);
            canvasEl.style.background = '#300';
            canvasEl.title = 'WebGPU init failed: ' + err.message;
        });
