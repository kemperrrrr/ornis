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

        const resizeCanvas = () => {
            if (!viewportEl || !canvasEl) return;
            canvasEl.width = viewportEl.clientWidth;
            canvasEl.height = viewportEl.clientHeight;
        };
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        // 2D/3D toggle next to canvas#bevy. The WASM module owns the mode;
        // the overlay below only reads it back for the grid.
        let mode2d = false;
        if (modeBtn) {
            modeBtn.addEventListener('click', () => {
                const next = mode2d ? 'perspective3d' : 'plane2d';
                if (typeof set_viewport_mode === 'function' && set_viewport_mode(next)) {
                    mode2d = !mode2d;
                    modeBtn.textContent = mode2d ? '2D' : '3D';
                    modeBtn.setAttribute('aria-pressed', String(mode2d));
                }
            });
        }

        // Grid overlay: pointer-events:none div over the canvas, driven by
        // viewport_state_json(). Minor/major lines via layered gradients.
        const paintGrid = () => {
            if (!gridEl || typeof viewport_state_json !== 'function') return;
            let state = null;
            try { state = JSON.parse(viewport_state_json()); } catch (e) { /* keep overlay hidden */ }
            const grid = state && state.mode === 'plane2d' ? state.grid : null;
            if (!grid) {
                gridEl.hidden = true;
                return;
            }
            gridEl.hidden = false;
            const minor = grid.minor_step_px;
            const major = minor * grid.major_every;
            const [ox, oy] = grid.origin_px;
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
        const overlayLoop = () => { paintGrid(); requestAnimationFrame(overlayLoop); };

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
