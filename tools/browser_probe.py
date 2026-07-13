#!/usr/bin/env python3
"""Headless browser probe for the Ornis UI layout engine.

Loads the real editor HTML in Chromium, walks the DOM in pre-order (skipping
the same tags the Rust engine skips), and dumps a `browser_layout.json` in the
exact format produced by `LayoutTree::to_json` (see
`crates/ui/examples/serialize_layout.rs`). `diff.py` then compares the two.

Usage:
    python3 tools/browser_probe.py [html_path] [width] [height]
    # defaults: crates/ui/assets/editor/index.html  1280 800
    # -> writes browser_layout.json next to CWD

Requires:  pip install playwright && playwright install chromium
"""
import json
import sys
from pathlib import Path

SKIP_TAGS = {
    "head", "style", "script", "title", "meta", "link", "noscript", "template",
    "path",  # our engine folds <path> into its parent <svg> node (b1 fix)
}

# Styles we copy from getComputedStyle so the output mirrors the Rust engine's
# `styles` map (subset that matters for layout/visual diffing).
STYLE_KEYS = [
    "box-sizing", "color", "font-family", "font-size", "font-weight",
    "background-color", "width", "height", "position", "display",
    "margin", "margin-right", "padding", "object-fit", "fill", "visibility",
]


def label_for(el):
    dom_id = el.get("id")
    if dom_id:
        return f"#{dom_id}"
    cls = el.get("class")
    if cls:
        return f".{cls}"
    return el.tag_name.lower()


def collect(root, out):
    """Pre-order walk, mirroring the Rust engine's node construction order."""
    from playwright.sync_api import sync_playwright

    def walk(node, page):
        # element or text node
        if node["nodeType"] == 3:  # TEXT_NODE
            info = page.evaluate(
                """(n) => {
                    const r = n.getBoundingClientRect();
                    return {tag: '#text', rect: {x:r.x,y:r.y,width:r.width,height:r.height}};
                }""",
                node,
            )
            out.append({
                "dom_class": None,
                "dom_id": None,
                "has_background_image": False,
                "image_intrinsic": None,
                "label": "#text",
                "rect": info["rect"],
                "styles": {},
                "svg": {"has_path": False},
                "tag": "#text",
            })
            return
        tag = node["tagName"].lower()
        if tag in SKIP_TAGS:
            return
        info = page.evaluate(
            """(el) => {
                const cs = getComputedStyle(el);
                const r = el.getBoundingClientRect();
                const styles = {};
                const keys = %r;
                for (const k of keys) styles[k] = cs.getPropertyValue(k);
                let vb = el.getAttribute('viewBox');
                let has_path = false;
                if (el.tagName.toLowerCase() === 'svg') {
                    has_path = !!el.querySelector('path');
                } else if (el.tagName.toLowerCase() === 'path') {
                    has_path = true;
                }
                const ii = (el.tagName.toLowerCase() === 'img')
                    ? [el.naturalWidth, el.naturalHeight] : null;
                return {
                    tag: el.tagName.toLowerCase(),
                    id: el.id || null,
                    cls: el.getAttribute('class') || null,
                    rect: {x:r.x, y:r.y, width:r.width, height:r.height},
                    styles, viewBox: vb, has_path, image_intrinsic: ii,
                };
            }""" % (STYLE_KEYS,),
            node,
        )
        svg = {"has_path": bool(info["has_path"])}
        if info["viewBox"]:
            svg["view_box"] = [float(v) for v in info["viewBox"].split()[:4]]
        out.append({
            "dom_class": info["cls"],
            "dom_id": info["id"],
            "has_background_image": False,
            "image_intrinsic": info["image_intrinsic"],
            "label": label_for(info),
            "rect": {k: float(v) for k, v in info["rect"].items()},
            "styles": {k: v.strip() for k, v in info["styles"].items()},
            "svg": svg,
            "tag": info["tag"],
        })
        for child in node["children"]:
            walk(child, page)

    walk(root, None)  # placeholder; real walk done in main()


def main():
    html_path = sys.argv[1] if len(sys.argv) > 1 else "crates/ui/assets/editor/index.html"
    width = int(sys.argv[2]) if len(sys.argv) > 2 else 1280
    height = int(sys.argv[3]) if len(sys.argv) > 3 else 800

    html_path = Path(html_path)
    if not html_path.exists():
        print(f"ERROR: html not found: {html_path}", file=sys.stderr)
        sys.exit(1)
    # In this sandbox `file://` navigation hangs, so inline the linked CSS and
    # load via set_content (geometry/layout is identical, no network needed).
    html = html_path.read_text()
    css_path = html_path.parent / "index.css"
    if css_path.exists():
        css = css_path.read_text()
        html = html.replace('<link rel="stylesheet" href="../index.css">',
                             f"<style>{css}</style>")
        # drop favicon links that would trigger network probes
        html = html.replace('<link rel="icon" href="../favicon.png" type="image/png" />', "")
        html = html.replace('<link rel="shortcut icon" href="../favicon.png" type="image/png" />', "")
    # Strip <script> blocks: the editor HTML imports a WebGL/wasm module from
    # the network; we only need static layout/geometry, so scripts are removed.
    import re
    html = re.sub(r"<script\b[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    # Drop any remaining external src/href that would force a network wait.
    html = re.sub(r'src="https?://[^"]*"', 'src=""', html)
    html = re.sub(r'href="https?://[^"]*"', 'href=""', html)

    from playwright.sync_api import sync_playwright

    out = []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": width, "height": height})
        page.set_content(html, wait_until="domcontentloaded", timeout=15000)
        page.wait_for_timeout(400)  # let layout settle
        # Build a serializable tree of element+text nodes in pre-order.
        tree = page.evaluate(
            """() => {
                const SKIP = new Set(%r);
                function ser(n) {
                    if (n.nodeType === 3) return {nodeType:3, children:[]};
                    const tag = (n.tagName||'').toLowerCase();
                    if (SKIP.has(tag)) return null;
                    const kids = [];
                    for (const c of n.childNodes) {
                        const s = ser(c);
                        if (s) kids.push(s);
                    }
                    return {nodeType:1, tagName:n.tagName, children:kids};
                }
                return ser(document.documentElement);
            }""" % (list(SKIP_TAGS),)
        )

        # Walk in Python, pulling rect/styles per node via a JS helper.
        # We re-evaluate per node using a path index is overkill; instead
        # collect everything in one JS pass that returns the full array.
        nodes = page.evaluate(
            """() => {
                const SKIP = new Set(%r);
                const KEYS = %r;
                const res = [];
                function walk(n, parents) {
                    if (!n || n.nodeType === 8 || n.nodeType === 10) return; // comment/doctype
                    if (n.nodeType === 3) {
                        try {
                            const r = n.getBoundingClientRect();
                            res.push({nodeType:3,
                                rect:{x:r.x,y:r.y,width:r.width,height:r.height}});
                        } catch (e) {}
                        return;
                    }
                    const tag = (n.tagName||'').toLowerCase();
                    if (SKIP.has(tag)) return;
                    const chain = parents.slice(-4).map(p => {
                        const pcs = getComputedStyle(p);
                        return (p.tagName||'').toLowerCase() + (p.getAttribute('class') ? '.' + p.getAttribute('class').split(' ')[0] : '') +
                               ' [w=' + pcs.width + ',fs=' + pcs.fontSize + ']';
                    });
                    try {
                        const cs = getComputedStyle(n);
                        const r = n.getBoundingClientRect();
                        const styles = {};
                        for (const k of KEYS) styles[k] = cs.getPropertyValue(k);
                        const vb = n.getAttribute('viewBox');
                        const width_attr = (tag === 'svg') ? n.getAttribute('width') : null;
                        const height_attr = (tag === 'svg') ? n.getAttribute('height') : null;
                        let has_path = false;
                        if (tag === 'svg') has_path = !!n.querySelector('path');
                        else if (tag === 'path') has_path = true;
                        const ii = (tag === 'img') ? [n.naturalWidth, n.naturalHeight] : null;
                        res.push({nodeType:1, tag, id:n.id||null,
                            cls:n.getAttribute('class')||null,
                            rect:{x:r.x,y:r.y,width:r.width,height:r.height},
                            styles, viewBox:vb, has_path, image_intrinsic:ii,
                            width_attr, height_attr,
                            parent_chain:chain});
                    } catch (e) {}
                    if (n.childNodes) for (const c of n.childNodes) walk(c, parents.concat([n]));
                }
                walk(document.documentElement, []);
                return res;
            }""" % (list(SKIP_TAGS), STYLE_KEYS)
        )

        for n in nodes:
            if n["nodeType"] == 3:
                out.append({
                    "dom_class": None, "dom_id": None,
                    "has_background_image": False, "image_intrinsic": None,
                    "label": "#text",
                    "rect": {k: float(v) for k, v in n["rect"].items()},
                    "styles": {}, "svg": {"has_path": False}, "tag": "#text",
                })
                continue
            svg = {"has_path": bool(n["has_path"])}
            if n["viewBox"]:
                svg["view_box"] = [float(v) for v in n["viewBox"].split()[:4]]
            if n.get("width_attr"):
                svg["width_attr"] = float(n["width_attr"])
            if n.get("height_attr"):
                svg["height_attr"] = float(n["height_attr"])
            label = f"#{n['id']}" if n["id"] else (f".{n['cls']}" if n["cls"] else n["tag"])
            out.append({
                "dom_class": n["cls"], "dom_id": n["id"],
                "has_background_image": False, "image_intrinsic": n["image_intrinsic"],
                "label": label,
                "rect": {k: float(v) for k, v in n["rect"].items()},
                "styles": {k: v.strip() for k, v in n["styles"].items()},
                "svg": svg, "tag": n["tag"],
                "parent_chain": n.get("parent_chain"),
            })
        browser.close()

    doc = {"node_count": len(out), "viewport": {"width": width, "height": height}, "nodes": out}
    Path("browser_layout.json").write_text(json.dumps(doc, indent=2))
    print(f"wrote browser_layout.json  (viewport {width}x{height}, {len(out)} nodes)")


if __name__ == "__main__":
    main()
