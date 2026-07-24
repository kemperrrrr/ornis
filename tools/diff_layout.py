#!/usr/bin/env python3
"""Diff our engine's layout JSON against Chromium's JSON.

Usage:
    python3 diff_layout.py ui_layout.json chromium_layout.json [--tolerance 1.0] [--output diff_report.json]
"""

import json
import sys
import math

# CSS property defaults (what the browser resolves when no explicit value is set)
CSS_DEFAULTS = {
    'position': 'static',
    'display': 'inline',
    'visibility': 'visible',
    'opacity': '1',
    'overflow': 'visible',
    'overflow-x': 'visible',
    'overflow-y': 'visible',
    'flex-grow': '0',
    'flex-shrink': '1',
    'flex-basis': 'auto',
    'align-items': 'stretch',
    'align-content': 'stretch',
    'justify-content': 'flex-start',
    'flex-direction': 'row',
    'flex-wrap': 'nowrap',
    'gap': 'normal',
    'margin': '0px',
    'margin-left': '0px',
    'margin-right': '0px',
    'margin-top': '0px',
    'margin-bottom': '0px',
    'padding': '0px',
    'padding-left': '0px',
    'padding-right': '0px',
    'padding-top': '0px',
    'padding-bottom': '0px',
    'min-width': 'auto',
    'min-height': 'auto',
    'max-width': 'none',
    'max-height': 'none',
    'background-color': 'rgba(0, 0, 0, 0)',
    'border': '0px none rgb(0, 0, 0)',
    'border-radius': '0px',
    'box-sizing': 'content-box',
    'text-align': 'start',
    'font-weight': '400',
    'line-height': 'normal',
    'white-space': 'normal',
    'letter-spacing': 'normal',
    'word-spacing': '0px',
    'left': 'auto',
    'right': 'auto',
    'top': 'auto',
    'bottom': 'auto',
    'pointer-events': 'auto',
    'cursor': 'auto',
    'user-select': 'auto',
    'object-fit': 'fill',
    'fill': 'rgb(0, 0, 0)',
    'stroke': 'none',
    'stroke-width': '1px',
}

def fill_defaults(styles):
    """Add default CSS values for properties not explicitly set."""
    result = dict(styles)
    for prop, default in CSS_DEFAULTS.items():
        if prop not in result:
            result[prop] = default
    return result

def load_json(path):
    with open(path) as f:
        return json.load(f)

def _norm_entry(e):
    """Normalize one parent_chain entry: `Tag.ClassA ClassB` -> `tag.classa.classb`."""
    e = e.strip().lower()
    if not e:
        return e
    if ' ' in e:
        tag, _, classes = e.partition(' ')
        e = tag + '.' + '.'.join(classes.split())
    return e

def build_index(nodes, parent_first):
    """Map path-key -> list of nodes (document order) so duplicate labels
    (several `div.tab` siblings) pair up by occurrence order.

    `parent_first` describes the source's parent_chain order: Chromium's
    probe walks up from the immediate parent (parent-first), our engine's
    to_json stores it root-first. The `html` entry is dropped on both sides
    because one dump may include the <html> node and the other not.
    """
    idx = {}
    for n in nodes:
        label = n.get('label') or n.get('tag') or 'unknown'
        entries = [_norm_entry(e) for e in (n.get('parent_chain') or [])]
        entries = [e for e in entries if e and e != 'html']
        if parent_first:
            entries.reverse()
        key = '/'.join(entries + [label])
        idx.setdefault(key, []).append(n)
    return idx

def _hex_to_rgb(v):
    v = v.lstrip('#')
    if len(v) == 3:
        v = ''.join(c * 2 for c in v)
    if len(v) == 6:
        r, g, b = (int(v[i:i+2], 16) for i in (0, 2, 4))
        return f'rgb({r}, {g}, {b})'
    if len(v) == 8:
        r, g, b, a = (int(v[i:i+2], 16) for i in (0, 2, 4, 6))
        alpha = round(a / 255, 3)
        return f'rgba({r}, {g}, {b}, {alpha})'
    return None

def normalize_value(v, viewport=None):
    """Normalize a CSS value so our serialized form compares equal to
    Chromium's resolved form: hex->rgb, `0`->`0px`, vw/vh/rem->px."""
    if v is None:
        return None
    v = str(v).strip().lower()
    if not v:
        return v
    if v.startswith('#'):
        return _hex_to_rgb(v) or v
    if v == '0':
        return '0px'
    if viewport:
        try:
            if v.endswith('vw'):
                return f"{float(v[:-2]) * viewport.get('width', 0) / 100}px"
            if v.endswith('vh'):
                return f"{float(v[:-2]) * viewport.get('height', 0) / 100}px"
            if v.endswith('rem'):
                return f"{float(v[:-3]) * 12}px"  # probe injects :root{font-size:12px}
        except ValueError:
            pass
    return v

def values_equal(ov, cv, viewport=None, tolerance=0.5):
    """Compare two CSS values after normalization; px numbers get tolerance."""
    ov = normalize_value(ov, viewport)
    cv = normalize_value(cv, viewport)
    if ov == cv:
        return True
    def px(v):
        if v and v.endswith('px'):
            try:
                return float(v[:-2])
            except ValueError:
                return None
        return None
    op, cp = px(ov), px(cv)
    if op is not None and cp is not None:
        return abs(op - cp) <= tolerance
    return False

# Known structural differences between the two dumps, not bugs:
# - our engine folds <path> geometry into the parent <svg> node instead of
#   creating layout nodes for SVG children
# - our dump includes the <html> root node, the Chromium probe skips it
KNOWN_STRUCTURAL = lambda key: key == 'html' or key.endswith('/svg/path')

# CSS properties we intentionally don't emit yet (tracked in the report as
# known gaps, kept out of the noise of real style diffs).
KNOWN_GAP_PROPS = {'text-overflow', 'transform-origin'}

def diff_layout(ours, chromium, tolerance=0.5):
    """Compare two layout JSONs and return structured diff."""
    ours_nodes = ours.get('nodes', [])
    chr_nodes = chromium.get('nodes', [])
    viewport = chromium.get('viewport') or ours.get('viewport') or {}

    # Match nodes by full root-first path (tag.class chain) so duplicate
    # labels (several `div.tab` siblings) pair up by document order.
    # Both dumps store parent_chain parent-first (verified against actual
    # output); text nodes exist only in our dump, so they are skipped.
    ours_idx = build_index([n for n in ours_nodes
                            if not (n.get('label') or n.get('tag') or '').startswith('#')],
                           parent_first=True)
    chr_idx = build_index(chr_nodes, parent_first=True)

    missing_in_ours = []
    missing_in_chromium = []
    pairs = []
    for key, chr_list in chr_idx.items():
        our_list = ours_idx.get(key, [])
        for i, chr_node in enumerate(chr_list):
            if i < len(our_list):
                pairs.append((key, our_list[i], chr_node))
            elif not KNOWN_STRUCTURAL(key):
                missing_in_ours.append(key)
    for key, our_list in ours_idx.items():
        chr_list = chr_idx.get(key, [])
        for i in range(len(chr_list), len(our_list)):
            if not KNOWN_STRUCTURAL(key):
                missing_in_chromium.append(key)

    rect_diffs = []
    style_diffs = []
    known_gap_diffs = []
    matched = 0

    for lbl, our_node, chr_node in pairs:
        matched += 1

        # Compare rect
        cr = chr_node['rect']
        or_ = our_node['rect']
        dx = abs(cr['x'] - or_['x'])
        dy = abs(cr['y'] - or_['y'])
        dw = abs(cr['width'] - or_['width'])
        dh = abs(cr['height'] - or_['height'])
        if dx > tolerance or dy > tolerance or dw > tolerance or dh > tolerance:
            rect_diffs.append({
                'label': lbl,
                'ours': or_,
                'chromium': cr,
                'diff': {'dx': round(dx,1), 'dy': round(dy,1), 'dw': round(dw,1), 'dh': round(dh,1)},
            })

        # Compare styles (only keys present in both)
        cs = fill_defaults(chr_node.get('styles', {}))
        os_ = fill_defaults(our_node.get('styles', {}))
        all_keys = set(cs.keys()) | set(os_.keys())

        # Skip font-family (Chromium resolves to actual fonts, we often differ)
        interesting_keys = [k for k in all_keys if k != 'font-family']

        for key in interesting_keys:
            cv = cs.get(key)
            ov = os_.get(key)
            # Properties we intentionally don't emit yet: count separately.
            if key in KNOWN_GAP_PROPS:
                if cv != ov:
                    known_gap_diffs.append({'label': lbl, 'property': key,
                                            'ours': ov, 'chromium': cv})
                continue
            # A `0px none <color>` border is invisible; the color slot just
            # reflects different defaults (black vs currentColor).
            if key == 'border' and str(ov).startswith('0px none') \
                    and str(cv).startswith('0px none'):
                continue
            if cv is None and ov is not None:
                style_diffs.append({
                    'label': lbl,
                    'property': key,
                    'issue': 'missing_in_chromium',
                    'ours': ov,
                    'chromium': None,
                })
            elif ov is None and cv is not None:
                style_diffs.append({
                    'label': lbl,
                    'property': key,
                    'issue': 'missing_in_ours',
                    'chromium': cv,
                    'ours': None,
                })
            elif not values_equal(ov, cv, viewport, tolerance):
                style_diffs.append({
                    'label': lbl,
                    'property': key,
                    'issue': 'value_mismatch',
                    'ours': ov,
                    'chromium': cv,
                })

    report = {
        'viewport_ours': ours.get('viewport', {}),
        'viewport_chromium': chromium.get('viewport', {}),
        'summary': {
            'ours_nodes': len(ours_nodes),
            'chromium_nodes': len(chr_nodes),
            'matched': matched,
            'missing_in_ours': len(missing_in_ours),
            'missing_in_chromium': len(missing_in_chromium),
            'rect_diffs': len(rect_diffs),
            'style_diffs': len(style_diffs),
            'known_gap_diffs': len(known_gap_diffs),
        },
        'missing_in_ours': missing_in_ours,
        'missing_in_chromium': missing_in_chromium,
        'rect_diffs': rect_diffs,
        'style_diffs': style_diffs,
        'known_gap_diffs': known_gap_diffs,
    }
    return report

def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)

    ours_path = args[0]
    chr_path = args[1]
    tolerance = float(args[args.index('--tolerance') + 1]) if '--tolerance' in args else 0.5
    output = args[args.index('--output') + 1] if '--output' in args else 'diff_report.json'

    ours = load_json(ours_path)
    chromium = load_json(chr_path)
    report = diff_layout(ours, chromium, tolerance)

    with open(output, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    s = report['summary']
    print(f"=== Layout Diff Report ===")
    print(f"Ours: {s['ours_nodes']} nodes, Chromium: {s['chromium_nodes']} nodes")
    print(f"Matched: {s['matched']}")
    print(f"Missing in ours: {s['missing_in_ours']}")
    print(f"Missing in Chromium: {s['missing_in_chromium']}")
    print(f"Rect diffs: {s['rect_diffs']}  (tolerance={tolerance}px)")
    print(f"Style diffs: {s['style_diffs']}")
    if s['rect_diffs'] > 0:
        print(f"\nTop rect diffs:")
        for d in report['rect_diffs'][:10]:
            print(f"  {d['label']}: dx={d['diff']['dx']} dy={d['diff']['dy']} dw={d['diff']['dw']} dh={d['diff']['dh']}")
            print(f"    ours:     {d['ours']}")
            print(f"    chromium: {d['chromium']}")
    if s['style_diffs'] > 0:
        print(f"\nTop style diffs:")
        for d in report['style_diffs'][:10]:
            print(f"  {d['label']}.{d['property']}: {d['issue']}")
            print(f"    ours:     {d.get('ours', 'N/A')}")
            print(f"    chromium: {d.get('chromium', 'N/A')}")
    print(f"\nFull report: {output}")

if __name__ == '__main__':
    main()
