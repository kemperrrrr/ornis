#!/usr/bin/env python3
"""Extract all elements' bounding rects + computed styles from a local HTML file
using Playwright + Chromium, then save as JSON for diffing against our engine."""

import json
import sys
import os
from playwright.sync_api import sync_playwright

def main():
    viewport_w = int(sys.argv[1]) if len(sys.argv) > 1 else 1920
    viewport_h = int(sys.argv[2]) if len(sys.argv) > 2 else 1080
    html_path = sys.argv[3] if len(sys.argv) > 3 else "editor_test.html"
    output_path = sys.argv[4] if len(sys.argv) > 4 else "chromium_layout.json"

    # Resolve absolute path
    html_abs = os.path.abspath(html_path)
    if not os.path.exists(html_abs):
        print(f"ERROR: {html_abs} not found")
        sys.exit(1)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": viewport_w, "height": viewport_h})

        # The page embeds a Bevy demo that fetches WASM/JS from an external
        # URL; when the network stalls, DOMContentLoaded never fires.
        # Layout comparison doesn't need it — cut all network requests.
        page.route("**/*", lambda r: r.abort()
                   if r.request.url.startswith(("http://", "https://"))
                   else r.continue_())

        # domcontentloaded: the file references resources that may 404 from
        # this location (../index.css, fonts), and "load" waits for them all.
        page.goto("file://" + html_abs, wait_until="domcontentloaded")
        page.wait_for_timeout(300)

        # Inject :root font-size to match our engine's rem_base
        page.evaluate("""
            const s = document.createElement('style');
            s.textContent = ':root { font-size: 12px !important; }';
            document.head.appendChild(s);
        """)
        page.wait_for_timeout(200)

        data = page.evaluate(f"""
            (() => {{
                const skipTags = new Set(['html','head','script','style','meta','link','title']);
                const all = document.querySelectorAll('*');
                const nodes = [];

                function clsOf(el) {{
                    // SVG elements expose className as SVGAnimatedString, not a
                    // string — use the attribute instead.
                    return (el.getAttribute('class') || '').trim();
                }}

                function parentChain(el) {{
                    const c = [];
                    let cur = el.parentElement;
                    while (cur) {{
                        const t = cur.tagName.toLowerCase();
                        const cls = clsOf(cur);
                        c.push(t + (cls ? '.' + cls.split(/\s+/).join('.') : ''));
                        cur = cur.parentElement;
                    }}
                    return c;
                }}

                const keepProps = [
                    'display','position','width','height','min-width','min-height',
                    'max-width','max-height',
                    'margin','margin-left','margin-right','margin-top','margin-bottom',
                    'padding','padding-left','padding-right','padding-top','padding-bottom',
                    'font-size','font-family','font-weight','line-height',
                    'text-align','color','background-color','background-image',
                    'border','border-radius','box-sizing',
                    'overflow','overflow-x','overflow-y',
                    'flex-direction','flex-wrap','flex-grow','flex-shrink','flex-basis',
                    'align-items','align-content','justify-content','gap',
                    'left','top','right','bottom',
                    'opacity','visibility','pointer-events','cursor',
                    'white-space','text-overflow','letter-spacing','word-spacing',
                    'user-select','object-fit','transform','transform-origin',
                    'fill','stroke','stroke-width',
                ];

                for (const el of all) {{
                    const tag = el.tagName.toLowerCase();
                    if (skipTags.has(tag)) continue;

                    const r = el.getBoundingClientRect();
                    const s = window.getComputedStyle(el);
                    const styles = {{}};
                    for (const p of keepProps) {{
                        const v = s.getPropertyValue(p);
                        if (v && v !== 'none' && v !== 'normal') {{
                            styles[p] = v;
                        }}
                    }}

                    const cls = clsOf(el);
                    nodes.push({{
                        tag,
                        dom_class: cls || null,
                        dom_id: el.id || null,
                        label: tag + (cls ? '.' + cls.split(/\\s+/).join('.') : ''),
                        rect: {{
                            x: Math.round(r.x * 100) / 100,
                            y: Math.round(r.y * 100) / 100,
                            width: Math.round(r.width * 100) / 100,
                            height: Math.round(r.height * 100) / 100,
                        }},
                        styles,
                        parent_chain: parentChain(el),
                    }});
                }}
                return {{
                    viewport: {{ width: {viewport_w}, height: {viewport_h} }},
                    node_count: nodes.length,
                    nodes,
                }};
            }})()
        """)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"wrote {output_path}  (viewport {viewport_w}x{viewport_h}, {data['node_count']} nodes)")
        browser.close()

if __name__ == '__main__':
    main()
