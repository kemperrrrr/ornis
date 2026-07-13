#!/usr/bin/env python3
"""Compare SVG sizes between the Rust layout engine and a headless browser.

Matches SVGs by a *suffix* of their DOM ancestor chain (closest 4 ancestors,
normalized to leaf->root, first class only). The two engines can differ in how
many wrapper elements they emit (our engine adds `div.hierarchy`, `div.content`,
`div.tab` that the browser DOM does not have), so exact full-chain matching
fails; a suffix match is robust to those extra deep ancestors.

Usage:
    python3 tools/compare_svg.py [our.json] [browser.json]
    # defaults: ui_layout.json  browser_layout.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

OUR = sys.argv[1] if len(sys.argv) > 1 else "ui_layout.json"
BROWSER = sys.argv[2] if len(sys.argv) > 2 else "browser_layout.json"

TOL = 0.6          # px tolerance for "match"
TAIL = 4           # number of closest ancestors to use as the match key


def segs(chain, leaf_first):
    """Normalize a parent_chain to 'tag.class' tokens (first class only),
    strip browser '[w=..,fs=..]' annotations and synthetic body/html, and
    orient to leaf->root."""
    if not chain:
        return []
    out = []
    for c in chain:
        s = str(c)
        if " [" in s:
            s = s.split(" [")[0]
        parts = s.split(".", 1)
        if len(parts) == 2:
            head, rest = parts
            s = f"{head}.{rest.split('.')[0]}"
        out.append(s)
    out = [t for t in out if t not in ("body", "html")]
    if not leaf_first:
        out = list(reversed(out))
    return out


def suffix_key(chain, leaf_first):
    """Match key = closest TAIL ancestors (leaf->root). Our engine reports
    leaf->root already; the browser probe reports root->leaf, so flip it."""
    return " > ".join(segs(chain, leaf_first)[-TAIL:])


def collect(doc, leaf_first):
    svgs = [n for n in doc["nodes"] if n.get("tag") == "svg"]
    by_chain = defaultdict(list)
    for n in svgs:
        by_chain[suffix_key(n.get("parent_chain"), leaf_first)].append(n)
    return svgs, by_chain


def main():
    ours = json.load(open(OUR))
    br = json.load(open(BROWSER))
    our_svgs, our_by = collect(ours, leaf_first=True)
    br_svgs, br_by = collect(br, leaf_first=False)

    print(f"our svgs: {len(our_svgs)}  browser svgs: {len(br_svgs)}")
    print(f"{'chain':58} {'ours':>10} {'brwsr':>10}  flag")
    print("-" * 82)

    matched = mismatched = 0
    keys = sorted(set(our_by) | set(br_by))
    for k in keys:
        ou = our_by.get(k, [])
        b = br_by.get(k, [])
        n = max(len(ou), len(b))
        for i in range(n):
            o = ou[i] if i < len(ou) else None
            bb = b[i] if i < len(b) else None
            ow = oh = bw = bh = None
            if o:
                ow, oh = o["rect"]["width"], o["rect"]["height"]
            if bb:
                bw, bh = bb["rect"]["width"], bb["rect"]["height"]
            if o and bb:
                ok = abs(ow - bw) <= TOL and abs(oh - bh) <= TOL
                matched += 1 if ok else 0
                mismatched += 0 if ok else 1
                flag = "OK " if ok else "MISMATCH"
            elif o:
                flag = "ONLY-OURS"
            else:
                flag = "ONLY-BROWSER"
            os_ = f"{ow:.1f}x{oh:.1f}" if o else "-"
            bs_ = f"{bw:.1f}x{bh:.1f}" if bb else "-"
            print(f"{k[-56:]:58} {os_:>10} {bs_:>10}  {flag}")

    print("-" * 82)
    print(f"matched: {matched}   mismatched: {mismatched}   "
          f"only_ours: {len(our_svgs) - matched - mismatched}   "
          f"only_browser: {len(br_svgs) - matched - mismatched}")


if __name__ == "__main__":
    main()
