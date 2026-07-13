// Unified Editor - Slider and Interactive Components
// This replaces the Bevy mockup's sliders.js with our own implementation
// In the future, these will be handled by Rust via our ECS bridge

(function() {
    'use strict';

    // Initialize all sliders on the page
    function initSliders() {
        const sliders = document.querySelectorAll('.slider[data-min][data-max][data-step][data-value]');
        sliders.forEach(initSlider);
    }

    function initSlider(slider) {
        const min = parseFloat(slider.dataset.min) || 0;
        const max = parseFloat(slider.dataset.max) || 100;
        const step = parseFloat(slider.dataset.step) || 1;
        let value = parseFloat(slider.dataset.value) || min;

        const fill = slider.querySelector('.fill');
        const valueEl = slider.querySelector('.value');

        // Clamp value
        value = Math.max(min, Math.min(max, value));
        slider.dataset.value = value;

        // Update visual
        updateSlider(slider, value);

        // Mouse events
        let isDragging = false;

        function updateFromMouse(e) {
            const rect = slider.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const percent = Math.max(0, Math.min(1, x / rect.width));
            const newValue = min + (max - min) * percent;
            const steppedValue = Math.round((newValue - min) / step) * step + min;
            const clampedValue = Math.max(min, Math.min(max, steppedValue));
            
            if (clampedValue !== value) {
                value = clampedValue;
                slider.dataset.value = value;
                updateSlider(slider, value);
            }
        }

        function onMouseDown(e) {
            isDragging = true;
            updateFromMouse(e);
            e.preventDefault();
        }

        function onMouseMove(e) {
            if (isDragging) {
                updateFromMouse(e);
            }
        }

        function onMouseUp() {
            isDragging = false;
        }

        slider.addEventListener('mousedown', onMouseDown);
        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);

        // Touch support
        slider.addEventListener('touchstart', (e) => {
            isDragging = true;
            updateFromMouse(e.touches[0]);
            e.preventDefault();
        }, { passive: false });

        document.addEventListener('touchmove', (e) => {
            if (isDragging) {
                updateFromMouse(e.touches[0]);
                e.preventDefault();
            }
        }, { passive: false });

        document.addEventListener('touchend', () => {
            isDragging = false;
        });

        // Keyboard support
        slider.addEventListener('keydown', (e) => {
            let newValue = value;
            switch (e.key) {
                case 'ArrowRight':
                case 'ArrowUp':
                    newValue = Math.min(max, value + step);
                    break;
                case 'ArrowLeft':
                case 'ArrowDown':
                    newValue = Math.max(min, value - step);
                    break;
                case 'Home':
                    newValue = min;
                    break;
                case 'End':
                    newValue = max;
                    break;
                default:
                    return;
            }
            if (newValue !== value) {
                value = newValue;
                slider.dataset.value = value;
                updateSlider(slider, value);
                e.preventDefault();
            }
        });

        // Make slider focusable
        slider.setAttribute('tabindex', '0');
        slider.setAttribute('role', 'slider');
        slider.setAttribute('aria-valuemin', min);
        slider.setAttribute('aria-valuemax', max);
        slider.setAttribute('aria-valuenow', value);
        slider.setAttribute('aria-valuetext', value + (slider.dataset.suffix || ''));

        // Cleanup on remove
        const observer = new MutationObserver((mutations) => {
            if (!document.body.contains(slider)) {
                document.removeEventListener('mousemove', onMouseMove);
                document.removeEventListener('mouseup', onMouseUp);
                document.removeEventListener('touchmove', onMouseMove);
                document.removeEventListener('touchend', onMouseUp);
                observer.disconnect();
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });
    }

    function updateSlider(slider, value) {
        const min = parseFloat(slider.dataset.min) || 0;
        const max = parseFloat(slider.dataset.max) || 100;
        const percent = (value - min) / (max - min) || 0;
        
        const fill = slider.querySelector('.fill');
        const valueEl = slider.querySelector('.value');
        
        if (fill) {
            fill.style.width = (percent * 100) + '%';
        }
        
        if (valueEl) {
            const suffix = slider.dataset.suffix || '';
            valueEl.textContent = value + (slider.dataset.suffix || '');
        }
        
        slider.dataset.value = value;
        slider.setAttribute('aria-valuenow', value);
        slider.setAttribute('aria-valuetext', value + (slider.dataset.suffix || ''));
    }

    // Color picker
    function initColorPickers() {
        const pickers = document.querySelectorAll('.color-picker');
        pickers.forEach(initColorPicker);
    }

    function initColorPicker(picker) {
        const dot = picker.querySelector('.color-picker-dot');
        if (!dot) return;

        function updateFromMouse(e) {
            const rect = picker.getBoundingClientRect();
            const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            const y = Math.max(0, Math.min(1, 1 - (e.clientY - rect.top) / rect.height));
            
            // Simple HSV color picker
            const h = x * 360;
            const s = y;
            const v = 1;
            
            const rgb = hsvToRgb(h, s, v);
            const hex = rgbToHex(rgb);
            
            picker.style.backgroundColor = '#' + hex;
            if (dot) {
                dot.style.left = (x * 100) + '%';
                dot.style.top = ((1 - y) * 100) + '%';
            }
            
            picker.dataset.value = '#' + hex;
        }

        let isDragging = false;

        function onMouseDown(e) {
            updateFromMouse(e);
            picker.dataset.dragging = 'true';
            e.preventDefault();
        }

        function onMouseMove(e) {
            if (picker.dataset.dragging === 'true') {
                updateFromMouse(e);
            }
        }

        function onMouseUp() {
            picker.dataset.dragging = 'false';
        }

        picker.addEventListener('mousedown', onMouseDown);
        document.addEventListener('mousemove', (e) => {
            if (document.activeElement === picker || picker.matches(':hover')) {
                updateFromMouse(e);
            }
        });
        document.addEventListener('mouseup', () => {});

        picker.addEventListener('touchstart', (e) => {
            updateFromMouse(e.touches[0]);
            e.preventDefault();
        }, { passive: false });

        document.addEventListener('touchmove', (e) => {
            if (e.target === picker || picker.contains(e.target)) {
                updateFromMouse(e.touches[0]);
                e.preventDefault();
            }
        }, { passive: false });
    }

    // HSV to RGB conversion
    function hsvToRgb(h, s, v) {
        const i = Math.floor(h / 60);
        const f = h / 60 - i;
        const p = v * (1 - s);
        const q = v * (1 - f * s);
        const t = v * (1 - (1 - f) * s);

        switch (i % 6) {
            case 0: return [v, t, p];
            case 1: return [q, v, p];
            case 2: return [p, v, t];
            case 3: return [p, q, v];
            case 4: return [t, p, v];
            case 5: return [v, p, q];
        }
        return [0, 0, 0];
    }

    function rgbToHex(rgb) {
        const r = Math.round(rgb[0] * 255).toString(16).padStart(2, '0');
        const g = Math.round(rgb[1] * 255).toString(16).padStart(2, '0');
        const b = Math.round(rgb[2] * 255).toString(16).padStart(2, '0');
        return r + g + b;
    }

    // Vector3 inputs
    function initVec3Inputs() {
        const inputs = document.querySelectorAll('.vec3 .transform-input');
        inputs.forEach(input => {
            input.addEventListener('change', (e) => {
                // Value changed - will be picked up by Rust via ECS bridge
                input.dataset.dirty = 'true';
            });
        });
    }

    // Entity name input
    function initEntityNameInput() {
        const input = document.getElementById('entity-name');
        if (input) {
            input.addEventListener('change', (e) => {
                input.dataset.dirty = 'true';
            });
        }
    }

    // Transform inputs
    function initTransformInputs() {
        const inputs = document.querySelectorAll('.transform-input');
        inputs.forEach(input => {
            input.addEventListener('change', (e) => {
                input.dataset.dirty = 'true';
            });
        });
    }

    // Add component button
    function initAddComponent() {
        const btn = document.getElementById('add-component-btn');
        if (btn) {
            btn.addEventListener('click', () => {
                // Dispatch event to Rust
                window.dispatchEvent(new CustomEvent('ornis:add-component', {
                    detail: {}
                }));
            });
        }
    }

    // Hierarchy search
    function initHierarchySearch() {
        const input = document.getElementById('hierarchy-search');
        if (input) {
            input.addEventListener('input', (e) => {
                const filter = e.target.value.toLowerCase();
                const items = document.querySelectorAll('#hierarchy-panel details, #hierarchy-panel summary');
                items.forEach(item => {
                    const text = item.textContent.toLowerCase();
                    item.style.display = text.includes(filter) ? '' : 'none';
                });
            });
        }
    }

    // Color picker copy to clipboard
    function initColorCopy() {
        const colors = document.querySelectorAll('.color');
        colors.forEach(color => {
            color.addEventListener('click', () => {
                const value = color.querySelector('.color-value');
                if (value) {
                    navigator.clipboard.writeText(value.textContent);
                    const span = color.querySelector('.popup span');
                    if (span) {
                        const original = span.textContent;
                        span.textContent = 'Copied!';
                        setTimeout(() => {
                            span.textContent = original;
                        }, 1000);
                    }
                }
            });
        });
    }

    // Font size controls
    function initFontSize() {
        const fontSizeValue = document.getElementById('font-size-value');
        const fontSizeMinus = document.getElementById('font-size-minus');
        const fontSizePlus = document.getElementById('font-size-plus');
        const fontSizeReset = document.getElementById('font-size-reset');

        let fontSize = 12;

        function updateFontSize() {
            if (fontSizeValue) fontSizeValue.textContent = fontSize + 'px';
            document.documentElement.style.fontSize = fontSize + 'px';
        }

        if (fontSizeMinus) {
            fontSizeMinus.addEventListener('click', () => {
                fontSize = Math.max(8, fontSize - 1);
                updateFontSize();
            });
        }

        if (fontSizePlus) {
            fontSizePlus.addEventListener('click', () => {
                fontSize = Math.min(24, fontSize + 1);
                updateFontSize();
            });
        }

        if (fontSizeReset) {
            fontSizeReset.addEventListener('click', () => {
                fontSize = 12;
                updateFontSize();
            });
        }
    }

    // Initialize everything when DOM is ready
    function init() {
        initSliders();
        initColorPickers();
        initVec3Inputs();
        initEntityNameInput();
        initTransformInputs();
        initAddComponent();
        initHierarchySearch();
        initColorCopy();
        initFontSize();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    // Expose for Rust to call
    window.OrnisEditor = {
        initSliders,
        initColorPickers,
        updateSlider: (slider, value) => {
            const el = document.querySelector(slider);
            if (el) {
                el.dataset.value = value;
                // Update visual
                const fill = el.querySelector('.fill');
                const valueEl = el.querySelector('.value');
                const min = parseFloat(el.dataset.min) || 0;
                const max = parseFloat(el.dataset.max) || 100;
                const percent = (value - min) / (max - min) || 0;
                if (fill) fill.style.width = (percent * 100) + '%';
                if (valueEl) valueEl.textContent = value + (el.dataset.suffix || '');
            }
        },
        toggle: () => {
            window.dispatchEvent(new CustomEvent('ornis:toggle'));
        },
        close: () => {
            window.dispatchEvent(new CustomEvent('ornis:close'));
        }
    };

})();