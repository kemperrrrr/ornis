        // Ornis Engine — WebGPU WASM viewport
        // Build: wasm-pack build crates/wasm --target web --out-dir editor/pkg
        import init, { start_renderer } from './pkg/ornis_wasm.js';

        const canvasEl = document.getElementById('bevy');
        const viewportEl = document.querySelector('.viewport');

        const resizeCanvas = () => {
            if (!viewportEl || !canvasEl) return;
            canvasEl.width = viewportEl.clientWidth;
            canvasEl.height = viewportEl.clientHeight;
        };
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        async function boot() {
            console.log('[ornis-editor] loading WASM...');
            await init();
            console.log('[ornis-editor] WASM loaded, starting renderer...');
            await start_renderer('bevy');
            console.log('[ornis-editor] renderer started');
        }
        boot().catch(err => {
            console.error('[ornis-editor] failed to start:', err);
            canvasEl.style.background = '#300';
            canvasEl.title = 'WebGPU init failed: ' + err.message;
        });
