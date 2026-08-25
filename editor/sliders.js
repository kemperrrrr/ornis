// Slider widget: drag to change value, click to open a numeric edit box.
const sliders = document.querySelectorAll(".slider");

function clampValue(v, min, max) {
    return Math.max(min, Math.min(max, v));
}

function renderSlider(slider) {
    slider.fill.style.width = (slider.value / slider.max) * 100 + "%";
    slider.valueElement.innerText = slider.value.toFixed(1) + slider.suffix;
}

// Replace the static value text with an inline <input> until blur/Enter.
function enterEditMode(slider) {
    slider.editMode = true;
    const input = document.createElement("input");
    input.type = "text";
    input.value = slider.value.toFixed(2);
    input.className = "edit";

    function removeInput() {
        slider.editMode = false;
        input.remove();
        document.removeEventListener("mousedown", onDocMouseDown);
        document.removeEventListener("keydown", onDocKeyDown);
    }

    // Ignore the mousedown that lands inside the input itself.
    function onDocMouseDown(e) {
        if (e.target === input) {
            return;
        }
        removeInput();
    }

    function onDocKeyDown(e) {
        if (e.key === "Enter" || e.key === "Escape") {
            removeInput();
        }
    }

    document.addEventListener("mousedown", onDocMouseDown);
    document.addEventListener("keydown", onDocKeyDown);

    input.addEventListener("blur", () => {
        removeInput();
    });

    input.addEventListener("change", () => {
        slider.value = clampValue(parseFloat(input.value), slider.min, slider.max);
        renderSlider(slider);
    });

    slider.appendChild(input);
    input.focus();
    input.select();
}

sliders.forEach((sliderEl) => {
    const slider = {
        el: sliderEl,
        suffix: sliderEl.dataset.suffix || "",
        min: parseFloat(sliderEl.dataset.min) || 0,
        max: parseFloat(sliderEl.dataset.max) || 100,
        step: parseFloat(sliderEl.dataset.step) || 1,
        dragging: false,
        editMode: false,
        pointerStartPosX: 0,
        pointerStartPosY: 0,
        pointerEndPosX: 0,
        pointerEndPosY: 0,
        value: parseFloat(sliderEl.dataset.value) || 64,
        fill: sliderEl.querySelector(".fill"),
        valueElement: sliderEl.querySelector(".value"),
    };

    renderSlider(slider);

    sliderEl.addEventListener("mousedown", (e) => {
        // Clicks on the edit box belong to edit mode, not dragging.
        if (e.target.classList.contains("edit")) {
            return;
        }
        slider.dragging = true;
        slider.pointerStartPosX = e.pageX;
        slider.pointerStartPosY = e.pageY;
        slider.pointerEndPosX = e.pageX;
        slider.pointerEndPosY = e.pageY;
        // Cursor lock API:
        // https://developer.mozilla.org/en-US/docs/Web/API/Pointer_Lock_API
        sliderEl.requestPointerLock();
    });

    document.addEventListener("mouseup", () => {
        if (!slider.dragging) {
            return;
        }
        slider.dragging = false;
        document.exitPointerLock();
        // A drag that never moved was a click -> open the edit box.
        const moved =
            slider.pointerStartPosX !== slider.pointerEndPosX ||
            slider.pointerStartPosY !== slider.pointerEndPosY;
        if (!moved && !slider.editMode) {
            enterEditMode(slider);
        }
    });

    document.addEventListener("mousemove", (e) => {
        if (!slider.dragging) {
            return;
        }
        slider.value = clampValue(slider.value + e.movementX * slider.step, slider.min, slider.max);
        slider.pointerEndPosX += e.movementX;
        slider.pointerEndPosY += e.movementY;
        renderSlider(slider);
    });
});
