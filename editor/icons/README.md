# editor/icons

Single home for all editor SVG assets. UI icons are black-base
(no hardcoded fill); color comes only from `.tint-*` modifiers
in `index.css`. All icons share `viewBox 0 0 24 24` with the glyph
fitted into a 20x20 box. Logo (`Owl_bored.svg`) and
`outdated_inkscape_workspace.svg` are exempt (not recolorable icons).

## Provenance

16 icons vendored from Material Design Icons
(https://github.com/Templarian/MaterialDesign, Apache-2.0):

| file          | MDI source   | file           | MDI source    |
|---------------|--------------|----------------|---------------|
| stop.svg      | stop         | image.svg      | file-image    |
| undo.svg      | undo         | video.svg      | file-video    |
| redo.svg      | redo         | music.svg      | file-music    |
| save.svg      | content-save | font.svg       | format-text   |
| cube.svg      | cube         | trash.svg      | delete        |
| audio.svg     | volume-high  | eye.svg        | eye           |
| script.svg    | code-braces  | settings.svg   | cog           |
| particles.svg | creation     | duplicate.svg  | see custom row below |
| sphere.svg    | sphere       |                |               |
| camera.svg    | camera (replaced legacy camcorder) | |              |
| cube-outline.svg | cube-outline (wireframe pair to cube) | |           |
| dock-left.svg | custom outline (square, MDI dock-left was solid) | dock-right.svg | custom outline (mirror) |
| dock-bottom.svg | custom outline | duplicate.svg | custom: two outline portrait sheets, tight overlap |

Each was re-saved in the repo's icon format (`id="icon"`,
black fill, 20px-fitted) — geometry unchanged.
