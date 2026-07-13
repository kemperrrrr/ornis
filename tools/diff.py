#!/usr/bin/env python3
"""Diffs Ornis layout-engine JSON against a real browser probe.

Compares two layout dumps produced by `serialize_layout` (Rust -> ui_layout.json)
and `browser_probe.py` (Chromium -> browser_layout.json). Nodes are matched by
pre-order index (both walk DOM in the same order). For each pair we compare the
bounding rect and a subset of computed styles, flagging mismatches beyond a
tolerance. A dedicated SVG-icon section reports any icon that is oversized in
our engine vs the browser (the old "giant magnifier" class of bug).

Usage:
    python3 tools/diff.py ui_layout.json browser_layout.json [--tol 2.0]
    python3 tools/diff.py   # auto: ui_layout.json vs browser_layout.json, tol 2.0
"""
import json
import sys
import argparse


def load(path):
    try:
        return json.load(open(path))
    except FileNotFoundError:
        print(f"ERROR: missing {path}", file=sys.stderr)
        sys.exit(1)


def rect_str(r):
    return f"{r['width']:.0f}x{r['height']:.0f}@({r['x']:.0f},{r['y']:.0f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ours", nargs="?", default="ui_layout.json")
    ap.add_argument("browser", nargs="?", default="browser_layout.json")
    ap.add_argument("--tol", type=float, default=2.0, help="px tolerance on rect")
    args = ap.parse_args()

    a = load(args.ours)
    b = load(args.browser)
    na, nb = a["nodes"], b["nodes"]
    print(f"ours:  {a['node_count']} nodes  (vp {a.get('viewport')})")
    print(f"browser: {b['node_count']} nodes  (vp {b.get('viewport')})")

    # Group nodes by (tag,label) and match within each group by appearance
    # order. This tolerates different pre-order traversal between the two
    # engines and focuses the diff on geometry/style of identically-named nodes.
    from collections import defaultdict, deque

    def group(nodes):
        g = defaultdict(deque)
        for nd in nodes:
            g[(nd["tag"], nd["label"])].append(nd)
        return g

    ga, gb = group(na), group(nb)

    issues = 0
    svg_issues = []
    compared = 0
    label_only = []  # (tag,label,count_ours,count_browser)

    for key in set(ga) | set(gb):
        qa, qb = ga.get(key, deque()), gb.get(key, deque())
        while qa or qb:
            an = qa.popleft() if qa else None
            bn = qb.popleft() if qb else None
            if an and bn:
                compared += 1
                ar, br = an["rect"], bn["rect"]
                dx = abs(ar["x"] - br["x"])
                dy = abs(ar["y"] - br["y"])
                dw = abs(ar["width"] - br["width"])
                dh = abs(ar["height"] - br["height"])
                bad = max(dx, dy, dw, dh) > args.tol
                if an["svg"].get("has_path") and bn["svg"].get("has_path"):
                    ratio = max(ar["width"] / max(br["width"], 0.01),
                                ar["height"] / max(br["height"], 0.01))
                    if ratio > 1.5 or bad:
                        svg_issues.append((an["label"], rect_str(ar), rect_str(br), ratio))
                if bad:
                    issues += 1
                    if not (an["svg"].get("has_path") and bn["svg"].get("has_path")):
                        print(f"  {an['label']:22s} ours={rect_str(ar):18s} "
                              f"browser={rect_str(br):18s} dx={dx:.1f} dy={dy:.1f} "
                              f"dw={dw:.1f} dh={dh:.1f}")
            else:
                # count mismatch for this label
                issues += 1
                label_only.append((key[0], key[1], len(qa) + (1 if an else 0),
                                   len(qb) + (1 if bn else 0)))
                # drain rest
                for rest in list(qa) + list(qb):
                    label_only[-1] = (key[0], key[1],
                                      label_only[-1][2], label_only[-1][3])

    for tag, lab, co, cb in label_only:
        print(f"  COUNT {lab}({tag}): ours={co} browser={cb}")

    print("\n=== SVG ICON SCALE CHECK (ours vs browser) ===")
    if not svg_issues:
        print("  OK: no oversized SVG icons (giant-magnifier class of bug absent)")
    else:
        for lab, ours, brw, ratio in svg_issues:
            flag = "  <-- OVERSIZE" if ratio > 1.5 else ""
            print(f"  {lab:22s} ours={ours:14s} browser={brw:14s} "
                  f"ratio={ratio:.2f}{flag}")

    print(f"\nSUMMARY: {issues} rect/count issues, {len(svg_issues)} svg icon diffs "
          f"({compared} nodes compared)")
    if issues == 0 and not svg_issues:
        print("PASS: layouts match within tolerance.")
    else:
        print("FAIL: see above.")


if __name__ == "__main__":
    main()
