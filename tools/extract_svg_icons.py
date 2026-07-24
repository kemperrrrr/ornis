#!/usr/bin/env python3
"""Extract inline SVG icons from editor index.html into icons/sprite.svg
and replace them with <use href="icons/sprite.svg#id"/> references."""
import re, hashlib, os

HTML = 'crates/ui/assets/editor/index.html'
html = open(HTML).read()
svgs = list(re.finditer(r'<svg\b.*?</svg>', html, re.S))

def norm(s):
    vb = re.search(r'viewBox="([^"]*)"', s)
    ds = [re.sub(r'\s+', '', d) for d in re.findall(r'\bd="([^"]*)"', s)]
    others = re.findall(r'<(circle|rect|line|polyline|polygon|ellipse)\b([^>]*)>', s)
    o = [t + re.sub(r'\s+', '', a) for t, a in others]
    return (vb.group(1) if vb else '') + '|' + '|'.join(sorted(ds)) + '|' + '|'.join(sorted(o))

groups, order = {}, []
for i, m in enumerate(svgs):
    h = hashlib.md5(norm(m.group(0)).encode()).hexdigest()[:8]
    if h not in groups:
        groups[h] = []
        order.append(h)
    groups[h].append(i)

NAMES = {
    '5107518b': 'play',
    '50cf6054': 'pause',
    '2ad38577': 'skip-next',
    'cd95dc81': 'error-octagon',
    '9077cb84': 'warning-triangle',
    '49194196': 'close',
    '26d7d96c': 'search',
    'a13cc5cf': 'person',
    '87942018': 'camera',
    'ea2fa497': 'circle',
    'd4a267dd': 'transform-gizmo',
    '1fce0894': 'dots-vertical',
    'ff46b553': 'shapes',
    'ffd03c06': 'lightbulb',
    '87f17ff6': 'file-document',
    '0ffd5a69': 'plus',
    'ceb19aad': 'check',
}
missing = [h for h in order if h not in NAMES]
assert not missing, f"unnamed groups: {missing}"

symbols = []
for h in order:
    s = svgs[groups[h][0]].group(0)
    vb = re.search(r'viewBox="([^"]*)"', s).group(1)
    inner = re.sub(r'^<svg\b[^>]*>|</svg>$', '', s, flags=re.S).strip()
    inner = re.sub(r'\s*\n\s*', ' ', inner)
    symbols.append(f'  <symbol id="{NAMES[h]}" viewBox="{vb}">{inner}</symbol>')

sprite = ('<svg xmlns="http://www.w3.org/2000/svg">\n'
          '  <!-- Ornis editor icon sprite. Usage: <svg ...><use href="icons/sprite.svg#ID"/></svg> -->\n'
          + '\n'.join(symbols) + '\n</svg>\n')
os.makedirs('crates/ui/assets/editor/icons', exist_ok=True)
open('crates/ui/assets/editor/icons/sprite.svg', 'w').write(sprite)

out, last = [], 0
for i, m in enumerate(svgs):
    out.append(html[last:m.start()])
    h = hashlib.md5(norm(m.group(0)).encode()).hexdigest()[:8]
    open_tag = re.match(r'<svg\b[^>]*>', m.group(0)).group(0)
    # The <symbol> carries the real viewBox. On the outer <svg> we keep a
    # zero-origin viewBox with the same aspect: (a) an outer viewBox with a
    # non-zero origin ("0 -960 960 960") shifts the <use> viewport out of the
    # visible region and clips the icon entirely; (b) with no outer viewBox
    # the <svg> loses its intrinsic aspect ratio and falls back to the 300x150
    # replaced-element default instead of filling its .icon container.
    vb = re.search(r'viewBox="([^"]*)"', m.group(0))
    if vb:
        parts = vb.group(1).replace(',', ' ').split()
        zero_vb = f'0 0 {parts[2]} {parts[3]}'
        open_tag = re.sub(r'viewBox="[^"]*"', f'viewBox="{zero_vb}"', open_tag)
    out.append(f'{open_tag}<use href="icons/sprite.svg#{NAMES[h]}" /></svg>')
    last = m.end()
out.append(html[last:])
open(HTML, 'w').write(''.join(out))

print(f"replaced {len(svgs)} inline SVGs with {len(order)} symbols")
for h in order:
    print(f"  {NAMES[h]:16s} x{len(groups[h])}")
