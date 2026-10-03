#!/usr/bin/env python3
"""Analytic SVG generator for a shaded rounded-edge cube icon (PBR or toon).

Sibling of gen_sphere_svg.py: same lights, material, tone mapping and CLI
style; the colour / Shader / sphere-region code is imported from there.

Geometry: a rounded box = inner box (half-size h = s - r) Minkowski-summed
with a sphere of radius r (r = bevel * s), seen orthographically along -Z
(screen coords y up, z toward the viewer), after a yaw (about Y) and a pitch
(about X).  With directional lights and an orthographic camera the colour of
a point depends on its normal N only, and the surface splits into
  * faces   - N constant  -> one flat colour each (no gradient is physically
                             justified: flat face, parallel light, ortho view);
  * edges   - quarter cylinders, N = cos(t) a + sin(t) b -> iso-colour lines
              are straight lines parallel to the edge, so an edge is a stack of
              parallelograms;
  * corners - sphere octants.  A sphere of radius r has exactly the shading of
              the material ball, so the ball body of gen_sphere_svg (radius r)
              is defined once and <use>d at each visible corner, masked to the
              projected octant.

Composition (no gradients, filters, raster or blend modes):
  1. one mask = the exact silhouette (inner hull offset by r), so the outer
     antialiased edge is formed once;
  2. edge strips: per edge, nested opaque layers [u_k, end] (each layer is
     painted over the previous one, so a band boundary always lies on an opaque
     layer below - no conflation seams).  All layers share the far end, which
     is pushed past the silhouette (masked) or into the next face's interior.
     The strip ends at a vertex are exact elliptic arcs (projected quarter
     circles); every layer runs a little past them onto the corner sphere
     (along a latitude circle, so it never leaves the octant);
  3. faces, exact parallelograms, painted over the strip overshoot;
  4. corners on top: <use> of the ball inside a <mask> of the exact projected
     octant (a mask, not clip-path, so its edge is antialiased once, over the
     strip run-on below).
Toon: the same layout with the colour function replaced by flat bands
(N.L thresholds), a crisp highlight {N.H >= cos(size)} (on the bevels it is a
straight strip along the edge, on a corner an ellipse spot), an optional rim
and an outline stroke around the silhouette.

Flags not listed here (e.g. --diffuse-step, --spec-step, --fill-*) are
passed through to gen_sphere_svg's parser and affect the corner ball.
"""
import argparse
import math
import os
import re
import sys

sys.dont_write_bytecode = True  # importing the sibling must not leave __pycache__ in tools/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_sphere_svg as gs  # noqa: E402
from gen_sphere_svg import Shader, Geo, bands, hex_to_rgb, rgb_to_hex, normalize  # noqa: E402


def dot(p, q):
    return sum(x * y for x, y in zip(p, q))


def add(*vs):
    return tuple(sum(c) for c in zip(*vs))


def mul(v, k):
    return tuple(x * k for x in v)


def matvec(M, v):
    return tuple(dot(row, v) for row in M)


def rotation(yaw, pitch):
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    Ry = ((cy, 0, sy), (0, 1, 0), (-sy, 0, cy))
    Rx = ((1, 0, 0), (0, cp, -sp), (0, sp, cp))
    return tuple(tuple(sum(Rx[i][k] * Ry[k][j] for k in range(3)) for j in range(3)) for i in range(3))


AX = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


# --------------------------------------------------------------- geometry --

class Cube:
    def __init__(self, a):
        self.M = rotation(a.yaw, a.pitch)
        # bevel < 0.005 snaps to a sharp cube (r = 0): below that the bevel is
        # thinner than ~0.03 viewBox units and only adds hairline layers
        b = 0.0 if a.bevel < 0.005 else min(a.bevel, 0.95)
        self.sharp = b == 0.0
        sig = [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        vx = max(abs(matvec(self.M, mul(s, 1 - b))[0]) for s in sig) + b
        vy = max(abs(matvec(self.M, mul(s, 1 - b))[1]) for s in sig) + b
        self.s = 0.5 * a.size / max(vx, vy)
        self.r = b * self.s
        self.h = self.s - self.r
        self.cx, self.cy = a.cx, a.cy
        self.o = a.overshoot
        self.geo = Geo(a.cx, a.cy, self.r, a.precision, a.overshoot)

    def view(self, p):
        return matvec(self.M, p)

    def svg(self, p_obj):
        q = self.view(p_obj)
        return (self.cx + q[0], self.cy - q[1])

    def P(self, q):
        return self.geo.P(q)

    def poly(self, pts):
        return 'M' + ' L'.join(self.P(q) for q in pts) + ' Z'

    # silhouette: convex hull of the projected inner vertices offset by rad
    def silhouette_d(self, rad):
        h = self.h
        pts = [self.svg((sx * h, sy * h, sz * h)) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        pts = sorted(set((round(x, 9), round(y, 9)) for x, y in pts))

        def cross(o, p, q):
            return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
        lo, hi = [], []
        for p in pts:
            while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 1e-9:
                lo.pop()
            lo.append(p)
        for p in reversed(pts):
            while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 1e-9:
                hi.pop()
            hi.append(p)
        hull = lo[:-1] + hi[:-1]
        area = sum(hull[i][0] * hull[(i + 1) % len(hull)][1] - hull[(i + 1) % len(hull)][0] * hull[i][1]
                   for i in range(len(hull)))
        if area < 0:
            hull.reverse()  # visually clockwise in svg (y-down) coords -> sweep 1
        n = len(hull)
        nrm = []
        for i in range(n):
            p, q = hull[i], hull[(i + 1) % n]
            dx, dy = q[0] - p[0], q[1] - p[1]
            l = math.hypot(dx, dy)
            nrm.append((dy / l, -dx / l))
        f = self.geo.f
        if rad <= 1e-9:
            # sharp corners: offset (inset for rad < 0) polygon, mitred joins
            pts = []
            for i in range(n):
                np_, nc = nrm[i - 1], nrm[i]
                k = rad / (1 + np_[0] * nc[0] + np_[1] * nc[1])
                pts.append((hull[i][0] + k * (np_[0] + nc[0]), hull[i][1] + k * (np_[1] + nc[1])))
            return self.poly(pts)
        d = []
        for i in range(n):
            p, q = hull[i], hull[(i + 1) % n]
            nn, nx = nrm[i], nrm[(i + 1) % n]
            a0 = (p[0] + rad * nn[0], p[1] + rad * nn[1])
            a1 = (q[0] + rad * nn[0], q[1] + rad * nn[1])
            b1 = (q[0] + rad * nx[0], q[1] + rad * nx[1])
            if i == 0:
                d.append('M' + self.P(a0))
            d.append('L' + self.P(a1))
            d.append('A%s %s 0 0 1 %s' % (f(rad), f(rad), self.P(b1)))
        return ' '.join(d) + ' Z'

    def faces(self):
        h, r = self.h, self.r
        out = []
        for i in range(3):
            j, k = [x for x in range(3) if x != i]
            for s in (-1, 1):
                n = mul(AX[i], s)
                N = self.view(n)
                if N[2] <= 1e-6:
                    continue
                pts = []
                for sj, sk in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
                    p = add(mul(AX[i], s * h + s * r), mul(AX[j], sj * h), mul(AX[k], sk * h))
                    pts.append(self.svg(p))
                out.append((N, pts))
        return out

    def sharp_faces(self):
        """r = 0: visible faces as (N, quad svg pts, [view normal of the
        face across each quad edge])"""
        h = self.h
        out = []
        loop = ((1, 1), (1, -1), (-1, -1), (-1, 1))
        for i in range(3):
            j, k = [x for x in range(3) if x != i]
            for s in (-1, 1):
                N = self.view(mul(AX[i], s))
                if N[2] <= 1e-6:
                    continue
                pts = [self.svg(add(mul(AX[i], s * h), mul(AX[j], sj * h), mul(AX[k], sk * h))) for sj, sk in loop]
                nb = []
                for m in range(4):
                    (sj0, sk0), (sj1, sk1) = loop[m], loop[(m + 1) % 4]
                    nb.append(self.view(mul(AX[j], sj0)) if sj0 == sj1 else self.view(mul(AX[k], sk0)))
                out.append((N, pts, nb))
        return out

    def edges(self):
        """yield dict(e, c, a, b, t0, t1, end) with theta running from the
        visible-face side (t0) to t1; end = 'face' or 'horizon'."""
        h = self.h
        out = []
        for i in range(3):
            j, k = [x for x in range(3) if x != i]
            for sj in (-1, 1):
                for sk in (-1, 1):
                    a, b = mul(AX[j], sj), mul(AX[k], sk)
                    A, B = self.view(a)[2], self.view(b)[2]
                    c = add(mul(a, h), mul(b, h))
                    if A <= 1e-9 and B <= 1e-9:
                        continue
                    if A > 1e-9 and B > 1e-9:
                        out.append(dict(e=AX[i], c=c, a=a, b=b, t0=0.0, t1=math.pi / 2, end='face'))
                        continue
                    # n_z(t) = A cos t + B sin t; zero at th
                    th = math.atan2(A, -B) % math.pi  # A cos + B sin = 0
                    if A > 1e-9:  # visible at t=0, horizon at th
                        out.append(dict(e=AX[i], c=c, a=a, b=b, t0=0.0, t1=th, end='horizon'))
                    else:
                        out.append(dict(e=AX[i], c=c, a=a, b=b, t0=math.pi / 2, t1=th, end='horizon'))
        return out

    def corners(self, nseg=48):
        """yield (vertex svg centre, region path d) for visible corners."""
        h, r = self.h, self.r
        res = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    sg = (sx, sy, sz)
                    av = [self.view(mul(AX[i], sg[i])) for i in range(3)]
                    pts = []
                    for i in range(3):
                        p, q = av[i], av[(i + 1) % 3]
                        for m in range(nseg):
                            t = 0.5 * math.pi * m / nseg
                            pts.append(add(mul(p, math.cos(t)), mul(q, math.sin(t))))
                    if all(p[2] < 0 for p in pts):
                        continue
                    seq = []
                    L = len(pts)
                    for m in range(L):
                        p, q = pts[m], pts[(m + 1) % L]
                        if p[2] >= 0:
                            seq.append(('s', p))
                        if (p[2] >= 0) != (q[2] >= 0):
                            f = p[2] / (p[2] - q[2])
                            x = add(mul(p, 1 - f), mul(q, f))
                            seq.append(('h', normalize((x[0], x[1], 0.0)), p[2] >= 0))
                    # insert horizon arcs between an exit and the next entry
                    out = []
                    n = len(seq)
                    for m in range(n):
                        it = seq[m]
                        out.append(it)
                        if it[0] == 'h' and it[2]:  # exit
                            nxt = next(seq[(m + d) % n] for d in range(1, n + 1) if seq[(m + d) % n][0] == 'h')
                            a0 = math.atan2(it[1][1], it[1][0])
                            a1 = math.atan2(nxt[1][1], nxt[1][0])
                            da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
                            steps = max(2, int(abs(da) / math.radians(4)))
                            for kk in range(1, steps):
                                aa = a0 + da * kk / steps
                                out.append(('h', (math.cos(aa), math.sin(aa), 0.0), None))
                    vc = self.svg(mul(sg, h))
                    poly = []
                    for it in out:
                        N = it[1]
                        rad = r if it[0] == 's' else r + self.o
                        poly.append((vc[0] + rad * N[0], vc[1] - rad * N[1]))
                    res.append((vc, self.poly(poly)))
        return res


# ---------------------------------------------------------------- strips --

def strip_layers(cube, ed, colour_at, cuts):
    """cuts: increasing u in [0,1) starting with 0 (band starts); colour_at(i)
    -> hex.  Returns svg paths (nested opaque layers)."""
    r, h = cube.r, cube.h
    m = 0.35 * r       # overshoot into the neighbouring face
    t0, t1 = ed['t0'], ed['t1']
    sgn = 1.0 if t1 >= t0 else -1.0
    a, b, e, c = ed['a'], ed['b'], ed['e'], ed['c']

    def N(t):
        return add(mul(a, math.cos(t)), mul(b, math.sin(t)))

    def T(t):  # unit tangent toward increasing u
        return mul(add(mul(a, -math.sin(t)), mul(b, math.cos(t))), sgn)

    def pt(t, along, off=0.0, rad=None):
        rr = r if rad is None else rad
        return cube.svg(add(c, mul(N(t), rr), mul(T(t), off), mul(e, along)))
    H = h
    # the strip end at a vertex is the projection of the quarter circle
    # v + r N(t): an ellipse arc (centre v, semi-axes r and r|e_z|)
    ev = cube.view(e)
    phi = math.atan2(ev[0], ev[1])  # svg direction perpendicular to e_xy
    ry = max(r * abs(ev[2]), 1e-4 * r)
    geo = cube.geo

    phi_x = 0.3  # run past the vertex: latitude phi_x on the corner sphere
    dt = 0.15

    def W(t, sgn_e):
        """point on the corner sphere just past the vertex (inside the octant)"""
        v = add(c, mul(e, sgn_e * H))
        return cube.svg(add(v, mul(e, sgn_e * r * math.sin(phi_x)), mul(N(t), r * math.cos(phi_x))))

    def lat_arc(sgn_e, ta, tb):
        C = cube.svg(add(c, mul(e, sgn_e * (H + r * math.sin(phi_x)))))
        k = math.cos(phi_x)
        return geo.arc_through(C, r * k, ry * k, phi, W(ta, sgn_e), W(0.5 * (ta + tb), sgn_e), W(tb, sgn_e))

    ro = r + max(cube.o, 0.15 * r)

    def merid_arc(sgn_e, t, back=False):
        """vertex arc point -> W along the meridian (exact; for the first
        layer, whose start lies on the corner octant's border)"""
        v = add(c, mul(e, sgn_e * H))
        n = cube.view(N(t))
        nv = (n[1] * ev[2] - n[2] * ev[1], n[2] * ev[0] - n[0] * ev[2], n[0] * ev[1] - n[1] * ev[0])
        pm = cube.svg(add(v, mul(N(t), r * math.cos(phi_x / 2)), mul(e, sgn_e * r * math.sin(phi_x / 2))))
        p0, p1 = pt(t, sgn_e * H), W(t, sgn_e)
        if back:
            p0, p1 = p1, p0
        return geo.arc_through(cube.svg(v), r, max(r * abs(nv[2]), 1e-4 * r), math.atan2(nv[0], nv[1]), p0, pm, p1)

    def out_pt(sgn_e):
        C = cube.svg(add(c, mul(e, sgn_e * H)))
        w = W(t1, sgn_e)
        dx, dy = w[0] - C[0], w[1] - C[1]
        l = math.hypot(dx, dy)
        return (C[0] + ro * dx / l, C[1] + ro * dy / l)

    def layer(ta, first):
        d = []
        if first:
            d.append('M' + cube.P(pt(ta, H, -m)))
            d.append('L' + cube.P(pt(ta, H)))
        else:
            d.append('M' + cube.P(pt(ta, H)))
        # past the vertex: along latitude phi_x up to t1 - dt, then straight
        # back to the vertex arc at t1.  The shared far end of all layers thus
        # lies inside the corner octant (opaque corner on top), never on its
        # border.
        te = t1 - sgn * dt
        far = (te - ta) * sgn > 1e-6
        d.append(merid_arc(1, ta) if first else 'L' + cube.P(W(ta, 1)))
        if far:
            d.append(lat_arc(1, ta, te))
        if ed['end'] == 'face':
            # into the next face via its interior (the face is painted on top)
            d.append('L' + cube.P(pt(t1, H)))
            d.append('L' + cube.P(pt(t1, H - m, m)))
            d.append('L' + cube.P(pt(t1, -H + m, m)))
            d.append('L' + cube.P(pt(t1, -H)))
        else:
            # out past the silhouette, radially from the corner centre
            d.append('L' + cube.P(out_pt(1)))
            d.append('L' + cube.P(out_pt(-1)))
        if far:
            d.append('L' + cube.P(W(te, -1)))
            d.append(lat_arc(-1, te, ta))
        else:
            d.append('L' + cube.P(W(ta, -1)))
        d.append(merid_arc(-1, ta, True) if first else 'L' + cube.P(pt(ta, -H)))
        if first:
            d.append('L' + cube.P(pt(ta, -H, -m)))
        return ' '.join(d) + ' Z'
    out = []
    prev = None
    for i, u in enumerate(cuts):
        col = colour_at(i)
        if col == prev:
            continue
        prev = col
        t = t0 + (t1 - t0) * u
        out.append('<path d="%s" fill="%s"/>' % (layer(t, i == 0), col))
    return out


def strip_normal(ed, u):
    t = ed['t0'] + (ed['t1'] - ed['t0']) * u
    return add(mul(ed['a'], math.cos(t)), mul(ed['b'], math.sin(t)))


# ------------------------------------------------------------------ ball --

BALL_RE = re.compile(r'^<svg[^>]*><defs><mask id="([^"]+)">.*?</mask>(.*)</defs>'
                     r'<g mask="url\(#\1\)">\n(.*)\n</g>(.*)</svg>\n$', re.S)


def ball(a, cube, toon):
    b = argparse.Namespace(**vars(a))
    b.cx = b.cy = 0.0
    b.r = cube.r
    b.mask_id = a.mask_id + 'B'
    b.shading = 'toon' if toon else 'pbr'
    b.outline = 0.0
    svg = gs.build_svg(b)[0]
    m = BALL_RE.match(svg)
    if not m or m.group(4).strip():
        raise SystemExit('unexpected gen_sphere_svg output format')
    return m.group(2), m.group(3)


# ------------------------------------------------------------------ toon --

def toon_colour_fn(a, sh):
    """same band colours / highlight / rim rules as gen_sphere_svg.build_toon"""
    n = max(2, a.toon_bands)
    if a.toon_thresholds:
        th = sorted(float(x) for x in a.toon_thresholds.split(','))
        n = len(th) + 1
    elif n == 2:
        th = [0.1]
    else:
        th = [0.1 + (0.62 - 0.1) * i / (n - 2) for i in range(n - 1)]
    lift = tuple(0.5 * c * b * (1 - sh.m) for c, b in zip(sh.Lc2, sh.base)) if sh.on2 else (0.0, 0.0, 0.0)

    def col_at(t):
        return sh.out(add(sh.key_lin(t), lift))
    edges = [None] + th + [1.0]
    cols = [col_at(0.0)] + [col_at(0.5 * (edges[i] + edges[i + 1])) for i in range(1, n)]
    mix = lambda p, q, f: tuple(x + (y - x) * f for x, y in zip(p, q))
    hc = mix(cols[-1], hex_to_rgb(a.light_color), 0.8)
    cosS = math.cos(math.radians(a.toon_highlight_size))
    rim = None
    if a.toon_rim > 0:
        src = sh.L2 if sh.on2 else (-sh.L[0], -sh.L[1], 0.0)
        A = normalize((src[0], src[1], 0.0)) if math.hypot(src[0], src[1]) > 1e-6 else (-0.6, -0.8, 0.0)
        rimc = mix(cols[min(1, n - 1)], hex_to_rgb(a.light2_color if sh.on2 else a.light_color), 0.25)
        rim = (A, rgb_to_hex(rimc))
    hexcols = [rgb_to_hex(c) for c in cols]

    def fn(N):
        if a.toon_highlight and dot(N, sh.H) >= cosS:
            return rgb_to_hex(hc)
        if rim and math.hypot(N[0], N[1]) >= 1 - a.toon_rim and dot(N, rim[0]) >= a.toon_rim_spread:
            return rim[1]
        t = dot(N, sh.L)
        return hexcols[sum(1 for x in th if t >= x)]
    return fn, cols


def piecewise_cuts(fn, n=2000):
    """breakpoints of a piecewise-constant fn on [0,1] (bisection refined)"""
    cuts = [0.0]
    cur = fn(0.0)
    prev_u = 0.0
    for i in range(1, n + 1):
        u = i / n
        v = fn(u)
        if v != cur:
            lo, hi = prev_u, u
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if fn(mid) == cur:
                    lo = mid
                else:
                    hi = mid
            cuts.append(hi)
            cur = v
        prev_u = u
    return cuts


# ----------------------------------------------------------------- build --

def outline_colour(a, cols0):
    return tuple(c * 0.45 for c in cols0) if a.outline_color == 'auto' else hex_to_rgb(a.outline_color)


def build_sharp(a, sh, cube, colour, tcols):
    """Sharp cube (bevel 0): every visible face is one exact polygon with its
    flat colour.  Seam-free composition under the single silhouette mask:
    faces are sorted by projected area (smallest first); the first one is
    painted as the whole silhouette (overshot), each later face is exact on
    its edges with earlier faces (which lie under it) and pushed out on its
    other edges - into faces painted later or past the silhouette - so no
    antialiased edge ever sits on an uncovered pixel."""
    toon = tcols is not None
    pid = a.mask_id
    f = cube.geo.f
    faces = cube.sharp_faces()

    def area(p):
        return abs(sum(p[i][0] * p[(i + 1) % 4][1] - p[(i + 1) % 4][0] * p[i][1] for i in range(4))) / 2
    faces.sort(key=lambda fc: area(fc[1]))
    widths = [area(fc[1]) / max(max(math.dist(fc[1][m], fc[1][(m + 1) % 4]) for m in range(4)), 1e-9)
              for fc in faces]
    key = lambda N: tuple(round(x, 6) for x in N)
    order = {key(fc[0]): i for i, fc in enumerate(faces)}
    body = ['<path d="%s" fill="%s"/>' % (cube.silhouette_d(cube.o), colour(faces[0][0]))]
    for idx, (N, pts, nb) in enumerate(faces[1:], 1):
        # outward offset per edge
        cen = (sum(p[0] for p in pts) / 4, sum(p[1] for p in pts) / 4)
        lines = []
        for m in range(4):
            p, q = pts[m], pts[(m + 1) % 4]
            dx, dy = q[0] - p[0], q[1] - p[1]
            l = math.hypot(dx, dy)
            nx, ny = dy / l, -dx / l
            if (p[0] - cen[0]) * nx + (p[1] - cen[1]) * ny < 0:
                nx, ny = -nx, -ny
            j = order.get(key(nb[m]))
            if j is None:
                off = cube.o                      # silhouette edge (masked)
            elif j < idx:
                off = 0.0                         # earlier face is underneath
            else:
                off = min(0.4, 0.45 * widths[j])  # under a later face
            lines.append(((p[0] + off * nx, p[1] + off * ny), (dx, dy)))
        poly = []
        for m in range(4):
            (p1, d1), (p2, d2) = lines[m - 1], lines[m]
            den = d1[0] * d2[1] - d1[1] * d2[0]
            t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
            poly.append((p1[0] + t * d1[0], p1[1] + t * d1[1]))
        body.append('<path d="%s" fill="%s"/>' % (cube.poly(poly), colour(N)))
    # 'auto' line colour: darkened shadow band (toon) / darkest face (pbr)
    darkest = min((fc[0] for fc in faces), key=lambda n: dot(n, sh.L))
    oc = outline_colour(a, tcols[0] if toon else sh.shade(darkest))
    if a.edge_lines > 0:
        # inner edges = edges shared by two visible faces
        seen = set()
        for N, pts, nb in faces:
            for m in range(4):
                if key(nb[m]) in order:
                    seg = tuple(sorted((tuple(round(c, 6) for c in pts[m]), tuple(round(c, 6) for c in pts[(m + 1) % 4]))))
                    if seg in seen:
                        continue
                    seen.add(seg)
                    body.append('<path d="M%s L%s" fill="none" stroke="%s" stroke-width="%s" stroke-linecap="round"/>' % (
                        cube.P(seg[0]), cube.P(seg[1]), rgb_to_hex(oc), f(a.edge_lines)))
    out = []
    if toon and a.outline > 0:
        out.append('<path d="%s" fill="none" stroke="%s" stroke-width="%s" stroke-linejoin="miter"/>' % (
            cube.silhouette_d(-a.outline * a.outline_inset), rgb_to_hex(oc), f(a.outline)))
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" id="icon" viewBox="0 0 {vb} {vb}">'
           '<defs><mask id="{id}"><path d="{sil}" fill="#fff"/></mask></defs>'
           '<g mask="url(#{id})">\n{body}\n</g>{out}</svg>\n').format(
        vb=f(a.viewbox), id=pid, sil=cube.silhouette_d(0.0),
        body='\n'.join(body), out=''.join('\n' + x for x in out))
    return svg, len(faces), 0, cube


def build(a):
    toon = a.shading == 'toon'
    sh = Shader(a)
    cube = Cube(a)
    view = cube.view
    if toon:
        tfn, tcols = toon_colour_fn(a, sh)
        colour = tfn
    else:
        colour = lambda N: rgb_to_hex(sh.shade(N))
    pid = a.mask_id
    defs = []
    body = []
    # base under everything (never visible unless something leaks)
    base_col = colour(view(normalize((-1.0, -1.0, 1.0))))
    body.append('<path d="%s" fill="%s"/>' % (cube.silhouette_d(cube.r + cube.o), base_col))
    if cube.sharp:
        return build_sharp(a, sh, cube, colour, tcols if toon else None)
    nstrip = 0
    step = a.strip_step / 255.0
    for ed in cube.edges():
        Nf = lambda u, ed=ed: view(strip_normal(ed, u))
        if toon:
            cuts = piecewise_cuts(lambda u: colour(Nf(u)))
            ends = cuts[1:] + [1.0]
            cols = [colour(Nf(0.5 * (p + q))) for p, q in zip(cuts, ends)]
        else:
            cfn = lambda u: sh.shade(Nf(u))
            e_ = bands(cfn, 0.0, 1.0, step, n=4000)
            cuts = e_[:-1]
            cols = [rgb_to_hex(cfn(0.5 * (p + q))) for p, q in zip(e_[:-1], e_[1:])]
        layers = strip_layers(cube, ed, lambda i: cols[i], cuts)
        nstrip += len(layers)
        body += layers
    for N, pts in cube.faces():
        body.append('<path d="%s" fill="%s"/>' % (cube.poly(pts), colour(N)))
    bdefs, bbody = ball(a, cube, toon)
    defs.append(bdefs)
    defs.append('<g id="%s-ball">\n%s\n</g>' % (pid, bbody))
    corners = cube.corners()
    for i, (vc, d) in enumerate(corners):
        mid = '%s-c%d' % (pid, i)
        defs.append('<mask id="%s"><path d="%s" fill="#fff"/></mask>' % (mid, d))
        body.append('<g mask="url(#%s)"><use href="#%s-ball" x="%s" y="%s"/></g>' % (
            mid, pid, cube.geo.f(vc[0]), cube.geo.f(vc[1])))
    out = []
    if toon and a.outline > 0:
        oc = outline_colour(a, tcols[0])
        out.append('<path d="%s" fill="none" stroke="%s" stroke-width="%s" stroke-linejoin="round"/>' % (
            cube.silhouette_d(cube.r - a.outline * a.outline_inset), rgb_to_hex(oc), cube.geo.f(a.outline)))
    f = cube.geo.f
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" id="icon" viewBox="0 0 {vb} {vb}">'
           '<defs><mask id="{id}"><path d="{sil}" fill="#fff"/></mask>{defs}</defs>'
           '<g mask="url(#{id})">\n{body}\n</g>{out}</svg>\n').format(
        vb=f(a.viewbox), id=pid, sil=cube.silhouette_d(cube.r),
        defs=''.join('\n' + x for x in defs), body='\n'.join(body), out=''.join('\n' + x for x in out))
    return svg, nstrip, len(corners), cube


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-o', '--output', default='cube-pbr.svg')
    ap.add_argument('--shading', choices=('pbr', 'toon'), default='pbr')
    ap.add_argument('--base', default='#9a9aa0', help='base colour (sRGB hex)')
    ap.add_argument('--roughness', type=float, default=0.45, help='(0.45: wider, softer bevel highlights than the sphere default 0.3)')
    ap.add_argument('--metallic', type=float, default=0.0)
    ap.add_argument('--light', type=float, nargs=3, default=(0.5, 0.6, 0.62), metavar=('X', 'Y', 'Z'),
                    help='light direction, screen coords, y up, z toward viewer (normalized)')
    ap.add_argument('--light-color', default='#ffffff')
    ap.add_argument('--intensity', type=float, default=1.6, help='light intensity (Lambert, no 1/pi)')
    ap.add_argument('--light2', type=float, nargs=3, default=(-0.5, -0.35, 0.8), metavar=('X', 'Y', 'Z'),
                    help='fill light direction (same coords as --light)')
    ap.add_argument('--light2-color', default='#ffffff')
    ap.add_argument('--intensity2', type=float, default=0.3, help='fill light intensity, 0 disables')
    ap.add_argument('--spec2', type=float, default=0.0, help='fill light specular multiplier (0 = off)')
    ap.add_argument('--ambient', type=float, default=0.06, help='uniform ambient, fraction of albedo')
    ap.add_argument('--exposure', type=float, default=1.0)
    ap.add_argument('--tonemap', choices=('soft', 'clamp', 'reinhard', 'aces'), default='soft')
    ap.add_argument('--knee', type=float, default=0.8, help="shoulder start for 'soft' tonemap")
    # cube
    ap.add_argument('--size', type=float, default=20, help='silhouette fits a size x size box (viewBox units)')
    ap.add_argument('--bevel', type=float, default=0.22,
                    help='edge radius as a fraction of the half-size (up to 0.95); 0 (or < 0.005) = sharp cube')
    ap.add_argument('--yaw', type=float, default=45.0, help='rotation about the vertical axis, degrees')
    ap.add_argument('--pitch', type=float, default=35.264, help='tilt toward the viewer (top face visible), degrees; '
                    '35.264 = true isometric')
    ap.add_argument('--strip-step', type=float, default=1.5, help='pbr: max sRGB step (0-255) per edge strip band')
    # toon
    ap.add_argument('--toon-bands', type=int, default=3, help='toon: number of flat N.L bands')
    ap.add_argument('--toon-thresholds', default='', help='toon: comma-separated N.L cutoffs (overrides --toon-bands)')
    ap.add_argument('--toon-highlight', type=int, default=1, help='toon: crisp specular highlight (1/0)')
    ap.add_argument('--toon-highlight-size', type=float, default=24.0,
                    help='toon: highlight = normals within this angle of H, degrees (spot on a corner, '
                         'strip along a bevel)')
    ap.add_argument('--toon-rim', type=float, default=0.0,
                    help='toon: rim on normals with |N_xy| >= 1 - width, on the fill-light side (0 = off)')
    ap.add_argument('--toon-rim-spread', type=float, default=0.35)
    ap.add_argument('--outline', type=float, default=0.6, help='toon: silhouette stroke width, viewBox units (0 = off)')
    ap.add_argument('--outline-color', default='auto', help="toon: hex colour or 'auto' (darkened shadow colour)")
    ap.add_argument('--outline-inset', type=float, default=0.0,
                    help='toon: move the stroke inward by this fraction of its width (0 = centred on silhouette)')
    ap.add_argument('--edge-lines', type=float, default=0.0,
                    help='sharp cube only: stroke width (viewBox units) of the inner edges between visible faces, '
                         'outline colour (0 = off)')
    ap.add_argument('--viewbox', type=float, default=24)
    ap.add_argument('--cx', type=float, default=12)
    ap.add_argument('--cy', type=float, default=12)
    ap.add_argument('--overshoot', type=float, default=0.3, help='inner shapes extend past the silhouette')
    ap.add_argument('--precision', type=int, default=3)
    ap.add_argument('--mask-id', default='cube')
    ns, rest = ap.parse_known_args(argv)
    a = gs.parse(rest)  # sphere defaults + pass-through flags (validated there)
    for k, v in vars(ns).items():
        setattr(a, k, v)
    return a


def main(argv=None):
    a = parse(argv)
    svg, ns, nc, cube = build(a)
    with open(a.output, 'w') as fh:
        fh.write(svg)
    what = 'faces (sharp)' if cube.sharp else 'strip layers'
    sys.stderr.write('wrote %s: %d bytes, %d %s, %d corners, s=%.3f r=%.3f\n' % (
        a.output, len(svg.encode()), ns, what, nc, cube.s, cube.r))


if __name__ == '__main__':
    main()
