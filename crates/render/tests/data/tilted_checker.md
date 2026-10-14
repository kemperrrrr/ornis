# Analytic tilted-checker golden

`tilted_checker.rgba`: 128×128, RGBA8 sRGB, top row first; alpha always 255.
The background is black. A projected plane spans NDC corners
(-0.875, -0.875), (0.875, -0.875), (-0.25, 0.75), (0.25, 0.75).
Covered pixel centers are `[188,188,188,255]`: the sRGB encoding of the
linear-light average of equal-area black/white checks (rounded to RGBA8).

This is an analytical reference, **not** a capture blessed from the renderer.
For each pixel center `(x+0.5, y+0.5)`, transform to NDC, linearly interpolate
half-width between 0.875 at y=-0.875 and 0.25 at y=0.75, and test coverage.
The test allows only a rasterization-edge budget; its base-mip-only control
must visibly differ. The actual plane has perspective-correct UVs with far
clip w=8 and near w=1, repeating a 256² checker 32 times.
