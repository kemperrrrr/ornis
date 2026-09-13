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
| redo.svg      | redo         | music.svg      | music (was file-music) |
| save.svg      | content-save | font.svg       | format-text   |
| cube.svg      | cube         | trash.svg      | delete        |
| audio.svg     | volume-high  | eye.svg        | eye           |
| script.svg    | script (was code-braces) | settings.svg   | cog           |
| particles.svg | creation     | duplicate.svg  | see custom row below |
| sphere.svg    | sphere       |                |               |
| camera.svg    | camera (replaced legacy camcorder) | |              |
| cube-outline.svg | cube-outline (wireframe pair to cube) | |           |
| dock-left.svg | custom outline (square, MDI dock-left was solid) | dock-right.svg | custom outline (mirror) |
| dock-bottom.svg | custom outline | duplicate.svg | custom: two outline portrait sheets, tight overlap |

Each was re-saved in the repo's icon format (`id="icon"`,
black fill, 20px-fitted) — geometry unchanged.

## Provenance (batch 2, uncommitted)

141 icons vendored from Material Design Icons (Apache-2.0),
file name == MDI source name (e.g. `magnify.svg` ← `mdi-magnify`):
chart-bubble, cursor-default(-outline), cursor-move, cylinder,
delete, download(-outline), file-multiple(-outline),
filter(-outline), floppy, folder-open(-outline), folder-outline,
help, information, language-css3/html5/javascript,
language-markdown(-outline), language-python, language-rust,
layers(-outline), layers-triple(-outline), leaf,
lightbulb-on(-outline), link, lock(-outline),
lock-open-variant(-outline), magnify, memory, menu,
microphone(-outline/-off/-plus), music-note-eighth, package-variant-closed,
paperclip, pencil(-outline), pentagon(-outline),
pine-tree-variant(-outline), pound, puzzle(-outline), replay,
rotate-orbit, script-outline, search-web, select(-color/-search),
selection(-drag/-search), shape-square-plus,
shape-square-rounded-plus, share-variant(-outline), source-branch,
square(-outline), star(-outline), sticker-text(-outline),
subdirectory-arrow-left/right, tag(-outline), tag-text(-outline),
text(-box-outline/-long), timer(-outline),
timer-sand(-complete/-empty), toggle-switch(-off/-outline),
toolbox(-outline), tools, toy-brick(-outline), tray-arrow-down/up,
tree(-outline), tune(-variant/-vertical/-vertical-variant), usb,
variable, vector-bezier/curve/polygon,
vector-polyline(-edit/-minus/-plus/-remove), volume-high/low/medium,
volume-minus/mute/off/plus/source/variant-off, walk, water(-outline),
water-plus(-outline), water-remove(-outline), waves,
weather-lightning-rainy, weather-pouring, web, weight,
white-balance-sunny, wifi.
Same repo format (`id="icon"`, black fill, 20px-fitted);
glyph bbox measured by rasterizing the original (alpha channel).

4 icons vendored from SVG Repo (www.svgrepo.com, per-icon license
— see upstream page before distribution):
`blocks-group-svgrepo-com.svg`, `cursor-hand-grab-svgrepo-com.svg`,
`cursor-hand-svgrepo-com.svg`, `particle-svgrepo-com.svg`.
Stripped root `fill`, dropped bg rect/title, re-gridded 36→24
and 20px-fitted like the rest.

Custom (already in repo format, untouched):
`music_file.svg` (= old file-music glyph), `script_bracketes.svg`
(= old code-braces glyph) — preserve the glyphs replaced in
`music.svg` / `script.svg`.

Not a tintable icon, kept as exempt illustration (like `Owl_bored.svg`,
render via `<img>`, not `<use>`): `Camera-photo.svg` — "Photo Camera"
by Jakub Steiner (jimmac.musichall.cz) via The Tango! Desktop Project.
Upstream (Wikimedia Commons) lists it as public domain — the Tango
project dedicated it to PD worldwide. NOTE: the file's embedded RDF
still carries a stale `cc-by-sa-2.0` tag from the original 2005 upload;
upstream relicensed to PD since. Unlike the MDI batch above, it is not
Apache-2.0. Keep the provenance row if the file stays.

Confirmed duplicates (byte-identical geometry — verified by diff,
not just eyeballed; owner to resolve, do not keep both):
`audio.svg` = `volume-high.svg` (neither referenced in code),
`trash.svg` = `delete.svg` (neither referenced),
`lightbulb.svg` = `lightbulb-on.svg` (`lightbulb.svg` is live in
editor code — drop the new `lightbulb-on.svg`).
