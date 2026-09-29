#!/usr/bin/env python3
"""Analytic SVG generator for a shaded PBR "material ball" icon.

Pure Python stdlib. Orthographic view along -Z, V = (0,0,1), screen coords y-up.

Key fact: for a fixed unit vector A, the set {N on sphere : N.A = c} is a circle
on the sphere whose orthographic projection is an ellipse (centre c*A_xy,
semi-axes rho=sqrt(1-c^2) along perp(A_xy) and rho*|A_z| along A_xy).  The
visible region {N.A >= c} is bounded by the *front* arc of that ellipse plus
an arc of the silhouette circle, so every iso-brightness region is an exact
path made of SVG elliptical/circular arcs.

Shading (all in linear light, then tone map + sRGB encode):
  diffuse(N.L)  = ambient*albedo_amb + Lc*(1-metallic)*base*max(N.L,0)
  specular(N.H) = Lc*F(V.H)*D_GGX(N.H)/(4*(V.H)^2) * max(N.H,0)
    F = Schlick with F0 = mix(0.04, base, metallic); V.H is constant because
    V and L are fixed.  The Kelemen visibility 1/(4 (L.H)^2) = 1/(4 (V.H)^2)
    is constant too, and the cosine factor is approximated by N.H, so the
    specular term depends on N.H only -> its contours are H-ellipses.

Fill light (--light2/--intensity2): diffuse becomes a function of N.L1 and
N.L2, composed as (bottom to top):
  1. key-shadow stack: opaque L2-ellipse regions, colour out(ambient +
     fill(N.L2)) (exact where N.L1 < 0), pre-compensated for layer 3's zone 0;
  2. key stack: opaque L1-ellipse regions {N.L1 >= t}, colour out(ambient +
     key(N.L1) + a constant part of the fill);
  3. lit fill: the lit side is split into ~13 key zones (equal steps of key
     colour); each zone is one <g opacity> masked to {tau_j <= N.L1 <
     tau_j+1} containing opaque <use> copies of the L2 regions.  Group
     opacity and cell colours are least-squares fitted against the exact
     two-light result sampled along each L2 contour inside the zone
     (source-over is affine in the colour underneath).
  4. key specular (and optional fill specular) stacks, as below.
SVG composition (specular):
  * specular regions {N.H >= t_k}, composited with plain source-over.
    They are opaque shapes inside a few <g opacity=...> groups (so each
    pixel goes through only ~4 translucent composites: Chrome/Skia truncates
    on every 8-bit blend, and 40 stacked fill-opacity layers lose ~10 levels).
    Each shape's colour (and each group's opacity) is solved so that, at a
    representative point on its contour (mean normal t*H), the composited
    sRGB equals the exact tone-mapped sRGB of diffuse+specular.
  * everything sits in one group masked once by the sphere circle.  A <mask>
    is used rather than clip-path: Chrome applies a group clip-path per child
    draw, compounding antialiasing at the silhouette over ~140 layers.
    Inner shapes overshoot the disk slightly so only the mask forms the edge.
No raster images, gradients, filters or blend modes.
"""
import argparse
import bisect
import math
import sys

# ----------------------------------------------------------------- colour --

def srgb_to_lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lin_to_srgb(c):
    c = min(max(c, 0.0), 1.0)
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def hex_to_rgb(h):
    h = h.lstrip('#')
    if len(h) == 3:
        h = ''.join(ch * 2 for ch in h)
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(int(round(min(max(v, 0.0), 1.0) * 255)) for v in rgb)


def tonemap(x, mode, knee):
    x = max(x, 0.0)
    if mode == 'clamp':
        return min(x, 1.0)
    if mode == 'reinhard':
        return x / (1.0 + x)
    if mode == 'aces':  # Narkowicz fit
        return min(max((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0), 1.0)
    # 'soft': identity below knee, exponential shoulder to 1 above (C1)
    if x <= knee:
        return x
    w = 1.0 - knee
    return knee + w * (1.0 - math.exp(-(x - knee) / w))


def normalize(v):
    n = math.sqrt(sum(a * a for a in v))
    return tuple(a / n for a in v)


# ---------------------------------------------------------------- shading --

class Shader:
    def __init__(self, a):
        self.base = tuple(srgb_to_lin(c) for c in hex_to_rgb(a.base))
        self.m = min(max(a.metallic, 0.0), 1.0)
        self.rough = min(max(a.roughness, 0.02), 1.0)
        self.alpha = self.rough * self.rough
        self.L = normalize(a.light)
        self.V = (0.0, 0.0, 1.0)
        self.H = normalize(tuple(l + v for l, v in zip(self.L, self.V)))
        self.Lc = tuple(srgb_to_lin(c) * a.intensity for c in hex_to_rgb(a.light_color))
        self.ambient = a.ambient
        self.exposure = a.exposure
        self.tm = a.tonemap
        self.knee = a.knee
        m = self.m
        self.F0 = tuple(0.04 * (1 - m) + b * m for b in self.base)
        self.VH = self.H[2]
        self.F = tuple(f0 + (1 - f0) * (1 - self.VH) ** 5 for f0 in self.F0)
        self.HL = sum(h * l for h, l in zip(self.H, self.L))
        amb_alb = tuple((1 - m) * b + m * f for b, f in zip(self.base, self.F0))
        self.amb = tuple(self.ambient * al * lc / max(a.intensity, 1e-9) * 1.0 for al, lc in zip(amb_alb, self.Lc)) \
            if a.ambient_tinted else tuple(self.ambient * al for al in amb_alb)
        # fill light
        self.L2 = normalize(a.light2)
        self.Lc2 = tuple(srgb_to_lin(c) * max(a.intensity2, 0.0) for c in hex_to_rgb(a.light2_color))
        self.on2 = a.intensity2 > 0
        self.H2 = normalize(tuple(l + v for l, v in zip(self.L2, self.V)))
        self.VH2 = self.H2[2]
        self.F2 = tuple(f0 + (1 - f0) * (1 - self.VH2) ** 5 for f0 in self.F0)
        self.spec2 = max(a.spec2, 0.0) if self.on2 else 0.0
        self.bake = a.fill_bake
        self.bake_mode = a.fill_bake_mode
        dot = lambda p, q: sum(x * y for x, y in zip(p, q))
        self.HL2 = dot(self.H, self.L2)      # for rep points on H-circles
        self.H2L = dot(self.H2, self.L)
        self.H2L2 = dot(self.H2, self.L2)

    def key_lin(self, ndl):
        """ambient + key-light diffuse (linear)"""
        k = (1 - self.m) * max(ndl, 0.0)
        return tuple(am + lc * b * k for am, lc, b in zip(self.amb, self.Lc, self.base))

    def fill_lin(self, ndl2):
        """fill-light diffuse (linear), without ambient"""
        k = (1 - self.m) * max(ndl2, 0.0)
        return tuple(lc * b * k for lc, b in zip(self.Lc2, self.base))

    def t2star(self, t1):
        """fill N.L2 baked into the opaque key stack at key value t1: middle
        of the (clamped) N.L2 range over the visible circle N.L1 = t1"""
        if not self.on2:
            return 0.0
        if self.bake_mode == 'const':
            if not hasattr(self, '_t2c'):
                v = []
                for iy in range(80):
                    for ix in range(80):
                        x, y = (ix + 0.5) / 40 - 1, (iy + 0.5) / 40 - 1
                        if x * x + y * y < 1:
                            n = (x, y, math.sqrt(1 - x * x - y * y))
                            if sum(p * q for p, q in zip(n, self.L)) >= 0:
                                v.append(max(sum(p * q for p, q in zip(n, self.L2)), 0.0))
                v.sort()
                self._t2c = 0.5 * (v[0] + v[-1]) * self.bake if v else 0.0
            return self._t2c
        key = round(t1, 5)
        cache = self.__dict__.setdefault('_t2s', {})
        if key not in cache:
            pts = circle_samples(self.L, t1, 72)
            if not pts:
                cache[key] = 0.0
            else:
                v = [max(sum(p * q for p, q in zip(n, self.L2)), 0.0) for n in pts]
                cache[key] = 0.5 * (min(v) + max(v)) * self.bake
        return cache[key]

    def base_lin(self, t1):
        """colour of the opaque key stack (key + ambient + baked part of fill)"""
        return self.diffuse_lin(t1, self.t2star(t1))

    def diffuse_lin(self, ndl, ndl2=None):
        d = self.key_lin(ndl)
        if ndl2 is None or not self.on2:
            return d
        return tuple(x + y for x, y in zip(d, self.fill_lin(ndl2)))

    def _ggx(self, t, VH):
        t = max(t, 0.0)
        a2 = self.alpha * self.alpha
        d = a2 / (math.pi * (t * t * (a2 - 1) + 1) ** 2)
        return d / (4 * VH * VH) * t

    def spec_lin(self, ndh):
        k = self._ggx(ndh, self.VH)
        return tuple(lc * f * k for lc, f in zip(self.Lc, self.F))

    def spec2_lin(self, ndh2):
        k = self._ggx(ndh2, self.VH2) * self.spec2
        return tuple(lc * f * k for lc, f in zip(self.Lc2, self.F2))

    def out(self, lin):
        return tuple(lin_to_srgb(tonemap(self.exposure * c, self.tm, self.knee)) for c in lin)

    def shade(self, N):
        dot = lambda p, q: sum(x * y for x, y in zip(p, q))
        tot = self.diffuse_lin(dot(N, self.L), dot(N, self.L2))
        tot = tuple(x + y for x, y in zip(tot, self.spec_lin(dot(N, self.H))))
        if self.spec2 > 0:
            tot = tuple(x + y for x, y in zip(tot, self.spec2_lin(dot(N, self.H2))))
        return self.out(tot)


# --------------------------------------------------------------- geometry --

class Geo:
    def __init__(self, cx, cy, r, prec, overshoot):
        self.cx, self.cy, self.r, self.p = cx, cy, r, prec
        self.rext = r + overshoot

    def f(self, v):
        s = ('%.' + str(self.p) + 'f') % v
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
        if s in ('-0', ''):
            s = '0'
        if s.startswith('0.'):
            s = s[1:]
        elif s.startswith('-0.'):
            s = '-' + s[2:]
        return s

    def pt(self, x, y):  # unit-sphere math coords -> svg
        return (self.cx + self.r * x, self.cy - self.r * y)

    def P(self, q):
        return self.f(q[0]) + ' ' + self.f(q[1])

    def circle_d(self, rad):
        c = (self.cx, self.cy)
        return ('M%s A%s %s 0 1 1 %s A%s %s 0 1 1 %s Z' % (
            self.P((c[0] + rad, c[1])), self.f(rad), self.f(rad), self.P((c[0] - rad, c[1])),
            self.f(rad), self.f(rad), self.P((c[0] + rad, c[1]))))

    def arc_through(self, C, rx, ry, phi, P0, Pm, P1):
        """Elliptic arc (svg coords) from P0 via Pm to P1, as 'A' commands."""
        cp, sp = math.cos(phi), math.sin(phi)

        def ang(Q):
            dx, dy = Q[0] - C[0], Q[1] - C[1]
            return math.atan2((-sp * dx + cp * dy) / ry, (cp * dx + sp * dy) / rx)
        t0, tm, t1 = ang(P0), ang(Pm), ang(P1)
        tau = 2 * math.pi
        fwd = (t1 - t0) % tau
        fm = (tm - t0) % tau
        if fm <= fwd:
            spans = (fm, fwd - fm)
        else:
            spans = (-((t0 - tm) % tau), -((tm - t1) % tau))
        out = []
        ends = (Pm, P1)
        pd = math.degrees(phi)
        for sp_, E in zip(spans, ends):
            out.append('A%s %s %s %d %d %s' % (self.f(rx), self.f(ry), self.f(pd),
                                                1 if abs(sp_) > math.pi else 0,
                                                1 if sp_ > 0 else 0, self.P(E)))
        return ' '.join(out)

    def region(self, A, c):
        """Return (kind, d, fill_rule) for visible region {N.A >= c}."""
        Ax, Ay, Az = A
        s = math.hypot(Ax, Ay)
        if c >= 1.0:
            return ('empty', None, None)
        if c <= -1.0:
            return ('full', self.circle_d(self.rext), None)
        rho = math.sqrt(1 - c * c)
        if s > 1e-9:
            e1 = (-Ay / s, Ax / s)
            e2 = (Ax / s, Ay / s)
        else:
            e1, e2 = (1.0, 0.0), (0.0, 1.0)
        zc, zr = c * Az, rho * s
        R = self.r
        Csvg = self.pt(c * Ax, c * Ay)
        rx, ry = R * rho, R * rho * abs(Az)
        phi = math.atan2(-e1[1], e1[0])
        if zc - zr >= 0 or zc + zr <= 0:
            front = zc - zr >= 0
            if not front:
                return ('full', self.circle_d(self.rext), None) if Az - c >= 0 else ('empty', None, None)
            if ry < 1e-4:
                return ('empty', None, None) if Az > 0 else ('full', self.circle_d(self.rext), None)
            u = (math.cos(phi), math.sin(phi))
            p0 = (Csvg[0] + rx * u[0], Csvg[1] + rx * u[1])
            p1 = (Csvg[0] - rx * u[0], Csvg[1] - rx * u[1])
            pd = self.f(math.degrees(phi))
            ed = 'M%s A%s %s %s 0 1 %s A%s %s %s 0 1 %s Z' % (
                self.P(p0), self.f(rx), self.f(ry), pd, self.P(p1),
                self.f(rx), self.f(ry), pd, self.P(p0))
            if Az > 0:
                return ('shape', ed, None)
            return ('shape', self.circle_d(self.rext) + ' ' + ed, 'evenodd')
        # crossing: tangent points on silhouette
        k1 = c / s
        k2 = math.sqrt(max(1 - k1 * k1, 0.0))
        Tp = (k1 * e2[0] + k2 * e1[0], k1 * e2[1] + k2 * e1[1])
        Tm = (k1 * e2[0] - k2 * e1[0], k1 * e2[1] - k2 * e1[1])
        u3 = normalize((-Az * Ax, -Az * Ay, 1 - Az * Az))
        M = (c * Ax + rho * u3[0], c * Ay + rho * u3[1])
        S = e2
        g = self.rext / R
        TpS, TmS, MS = self.pt(*Tp), self.pt(*Tm), self.pt(*M)
        TpE, TmE, SE = self.pt(Tp[0] * g, Tp[1] * g), self.pt(Tm[0] * g, Tm[1] * g), self.pt(S[0] * g, S[1] * g)
        parts = ['M' + self.P(TpS)]
        if ry < 1e-4:
            parts.append('L' + self.P(TmS))
        else:
            parts.append(self.arc_through(Csvg, rx, ry, phi, TpS, MS, TmS))
        parts.append('L' + self.P(TmE))
        parts.append(self.arc_through((self.cx, self.cy), self.rext, self.rext, 0.0, TmE, SE, TpE))
        parts.append('Z')
        return ('shape', ' '.join(parts), None)


# ---------------------------------------------------------------- levels --

def bands(colour_fn, t0, t1, step, n=6000, min_width=0.0):
    """Split [t0,t1] into bands where colour changes by <= step (0..1 units)."""
    edges = [t0]
    ref = colour_fn(t0)
    for i in range(1, n + 1):
        t = t0 + (t1 - t0) * i / n
        c = colour_fn(t)
        if max(abs(a - b) for a, b in zip(c, ref)) > step and t - edges[-1] >= min_width:
            edges.append(t)
            ref = c
    if edges[-1] < t1:
        edges.append(t1)
    return edges


def circle_samples(A, c, n=96):
    """Visible (z >= 0) points of the sphere circle {N.A = c}."""
    if abs(c) >= 1:
        return []
    rho = math.sqrt(1 - c * c)
    ref = (0.0, 0.0, 1.0) if abs(A[2]) < 0.9 else (1.0, 0.0, 0.0)
    d = sum(x * y for x, y in zip(ref, A))
    u = normalize(tuple(r - d * x for r, x in zip(ref, A)))
    v = (A[1] * u[2] - A[2] * u[1], A[2] * u[0] - A[0] * u[2], A[0] * u[1] - A[1] * u[0])
    pts = []
    for i in range(n):
        th = 2 * math.pi * i / n
        p = tuple(c * x + rho * (math.cos(th) * uu + math.sin(th) * vv) for x, uu, vv in zip(A, u, v))
        if p[2] >= 0:
            pts.append(p)
    return pts


def need_alpha(P, T):
    """smallest source-over alpha that can move colour P to T (per channel, colour in [0,1])"""
    need = 0.0
    for p, t_ in zip(P, T):
        if t_ > p + 1e-9:
            need = max(need, (t_ - p) / max(1 - p, 1e-9))
        elif t_ < p - 1e-9:
            need = max(need, (p - t_) / max(p, 1e-9))
    return need


def fmt_op(al):
    return ('%.3f' % al).rstrip('0').rstrip('.').lstrip('0') or '0'


def emit_groups(groups):
    out = []
    for al, items, paths in groups:
        op = '' if al >= 1.0 else ' opacity="%s"' % fmt_op(al)
        out.append('<g%s>\n%s\n</g>' % (op, '\n'.join(paths)))
    return out


def spec_stack(a, sh, geo, H, spec_fn, base_lin_fn, step_s, tag):
    """Specular regions {N.H >= t}: opaque shapes inside few opacity groups.
    base_lin_fn(N) = exact linear diffuse at normal N (evaluated at the
    contour's mean normal t*H)."""
    cache = {}

    def base_samples(t):
        """exact diffuse (linear) at visible points of the contour N.H = t"""
        k = round(t, 6)
        if k not in cache:
            # the contour mean was tested and differs < 0.3/255 from the value at
            # the mean normal t*H, so use the (much cheaper) single point
            cache[k] = [base_lin_fn(tuple(t * h for h in H))]
        return cache[k]

    def mean_out(lins, add):
        # mean sRGB over the contour: composites are affine in P, so fitting the
        # mean is the least-squares fit over the contour
        acc = [0.0, 0.0, 0.0]
        for dl in lins:
            o = sh.out(tuple(x + y for x, y in zip(dl, add)))
            for ch in range(3):
                acc[ch] += o[ch]
        return tuple(v / len(lins) for v in acc)
    zero = (0.0, 0.0, 0.0)

    def spec_delta(t):
        lins = base_samples(t)
        base = mean_out(lins, zero)
        tot = mean_out(lins, spec_fn(t))
        return tuple(y - x for x, y in zip(base, tot))
    thr = a.spec_threshold / 255.0
    tlo = 1.0
    N = 4000
    for i in range(N + 1):
        t = i / N
        if max(abs(v) for v in spec_delta(t)) >= thr:
            tlo = t
            break
    groups = []           # [alpha, [(t, colour)], [svg paths]]
    layers = []
    if tlo >= 1.0:
        return groups, layers
    edges = bands(spec_delta, tlo, 1.0, step_s)

    def below(t, upto):
        P = mean_out(base_samples(t), zero)
        for al, items, _ in groups[:upto]:
            c = items[-1][1]  # topmost opaque shape in that group covers t
            P = tuple(p * (1 - al) + q * al for p, q in zip(P, c))
        return P

    blist = []
    for ta, tb in zip(edges[:-1], edges[1:]):
        lins = base_samples(ta)
        T = mean_out(lins, spec_fn(0.5 * (ta + tb)))
        blist.append((ta, T, need_alpha(mean_out(lins, zero), T)))
    acum, m_ = [], 0.0
    for _, _, n_ in blist:
        m_ = max(m_, n_)
        acum.append(m_)
    for bi, (ta, T, _) in enumerate(blist):
        kind, d, rule = geo.region(H, ta)
        if kind == 'empty':
            continue
        if groups:
            P = below(ta, len(groups) - 1)
            if need_alpha(P, T) > groups[-1][0] + 1e-9:
                groups.append(None)
        else:
            groups.append(None)
        if groups[-1] is None:
            P = below(ta, len(groups) - 1)
            n = need_alpha(P, T)
            if n < 2e-4 and len(groups) == 1:
                groups.pop()
                continue
            j = bi
            while j + 1 < len(blist) and acum[j + 1] <= a.spec_group_ratio * max(acum[bi], 1e-3):
                j += 1
            al = n
            for jj in range(bi, j + 1):
                al = max(al, need_alpha(below(blist[jj][0], len(groups) - 1), blist[jj][1]))
            al = min(1.0, math.ceil(al * 1000) / 1000.0)
            groups[-1] = [al, [], []]
        al = groups[-1][0]
        col = tuple(min(max(p + (t_ - p) / al, 0.0), 1.0) for p, t_ in zip(P, T))
        colq = tuple(round(c * 255) / 255.0 for c in col)
        groups[-1][1].append((ta, colq))
        groups[-1][2].append('<path d="%s" fill="%s"%s/>' % (d, rgb_to_hex(colq),
                                                            ' fill-rule="evenodd"' if rule else ''))
        layers.append((ta, al, colq))
    return groups, layers


def fill_stack(a, sh, geo, step, defs):
    """Fill-light diffuse.  Returns (shadow_elements, lit_groups).

    * Key shadow (N.L1 < 0): the colour there is out(ambient + fill(N.L2)),
      a function of N.L2 only -> an ordinary opaque stack of L2-ellipse
      regions, drawn *under* the opaque key stack (which covers N.L1 >= 0).
    * Key-lit side: the fill is added with translucent source-over layers,
      result = P*(1-A) + K, i.e. an sRGB increment affine in the key colour P.
      The lit side is cut into key zones by colour (tau_1=0 < tau_2 < ...),
      and every zone gets one <g opacity> group masked by {0 <= N.L1 <
      tau_j+1}.  Masks are nested, so darker zones pass through more groups
      and get a larger cumulative A: the true increment (gamma + tone map) is
      steeper in the dark.  Per fill band the group colour is least-squares
      fitted over the visible points of that band's contour inside the zone.
      Nested (not complementary) masks avoid the source-over conflation seam
      that two complementary translucent edges would produce."""
    L2 = sh.L2
    s2 = math.hypot(L2[0], L2[1])
    tmin_vis = max(0.0, -s2 if L2[2] >= 0 else -1.0)
    fcol = lambda t: sh.out(sh.diffuse_lin(-1.0, t))  # ambient + fill (key shadow)
    edges = bands(fcol, tmin_vis, 1.0, step)
    pid = a.mask_id
    tks = []
    fids = []
    shadow_raw = [(None, sh.out(sh.diffuse_lin(-1.0, -1.0)))]  # (use id, exact colour); None = disk
    zone0 = None
    for ta, tb in zip(edges[:-1], edges[1:]):
        kind, d, rule = geo.region(L2, ta)
        if kind == 'empty':
            continue
        fid = '%s-f%d' % (pid, len(tks))
        defs.append('<path id="%s" d="%s"%s/>' % (fid, d, ' fill-rule="evenodd"' if rule else ''))
        tks.append((ta, tb))
        fids.append(fid)
        shadow_raw.append((fid, fcol(0.5 * (ta + tb))))
    # lit zones
    Z = max(2, a.fill_zones)
    taus = [0.0]
    P0, P1 = sh.out(sh.base_lin(0.0))[1], sh.out(sh.base_lin(1.0))[1]
    for i in range(1, Z - 1):
        target = P0 + (P1 - P0) * i / (Z - 1)
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if sh.out(sh.base_lin(mid))[1] < target:
                lo = mid
            else:
                hi = mid
        taus.append(0.5 * (lo + hi))
    if a.fill_split_ends and len(taus) > 1:
        # the zones at the terminator and at the key highlight carry the largest
        # fit error (steep gamma / wide area): halve them
        extra = [0.5 * (taus[-1] + 1.0)]
        if a.fill_split_ends > 1:
            extra.append(0.5 * taus[1])
        taus = sorted(taus + extra)
    nz = len(taus)
    taus_hi = taus[1:] + [1.0]
    sh.fill_taus = taus

    def zone_of(t1):
        if t1 < 0:
            return -1
        return max(j for j in range(nz) if t1 >= taus[j])

    def sample(t1, tm):
        kl = sh.key_lin(t1)
        P = sh.out(sh.base_lin(t1))
        T = sh.out(tuple(x + y for x, y in zip(kl, sh.fill_lin(tm))))
        return (P, tuple(y - x for x, y in zip(P, T)))
    zs_all = []
    for ta, tb in tks:
        tm = 0.5 * (ta + tb)
        zs = [[] for _ in range(nz)]
        for p in circle_samples(L2, ta, 180):
            t1 = sum(x * y for x, y in zip(p, sh.L))
            j = zone_of(t1)
            if j >= 0:
                zs[j].append(sample(t1, tm))
        for j in range(nz):
            if not zs[j]:  # contour misses this zone: sample the zone's middle key value
                zs[j].append(sample(0.5 * (taus[j] + taus_hi[j]), tm))
        zs_all.append(zs)
    # grid of visible normals: occupancy of (zone, band) and samples for the
    # band below the fill terminator (N.L2 < 0, where the baked fill must be removed)
    neg = [[] for _ in range(nz)]
    occ = set()
    starts = [ta for ta, _ in tks]
    NG = 360  # fine grid: thin zone/band cells near the terminator must not be missed
    for iy in range(NG):
        y = (iy + 0.5) / (NG / 2) - 1
        for ix in range(NG):
            x = (ix + 0.5) / (NG / 2) - 1
            r2 = x * x + y * y
            if r2 >= 1:
                continue
            zz = math.sqrt(1 - r2)
            t1 = x * sh.L[0] + y * sh.L[1] + zz * sh.L[2]
            if t1 < 0:
                continue
            j = zone_of(t1)
            t2 = x * L2[0] + y * L2[1] + zz * L2[2]
            k = bisect.bisect_right(starts, t2)
            occ.add((j, k))
            if k == 0 and ix % 6 == 0 and iy % 6 == 0:
                neg[j].append(sample(t1, -1.0))
    for j in range(nz):
        if not neg[j]:
            neg[j].append(sample(0.5 * (taus[j] + taus_hi[j]), -1.0))
    zs_all = [neg] + zs_all
    tks = [(-2.0, -2.0)] + tks
    fid = '%s-f' % pid
    defs.append('<path id="%s" d="%s"/>' % (fid, geo.circle_d(geo.rext)))
    fids = [fid] + fids
    # region paths for the masks
    kind, d0, rule0 = geo.region(sh.L, 0.0)
    if kind == 'empty' or not tks:
        sh.fill_shadow = [(-2.0, shadow_raw[0][1])] + [(tks[i + 1][0], c) for i, (_, c) in enumerate(shadow_raw[1:])]
        return [('<path d="%s" fill="%s"/>' % (geo.circle_d(geo.rext), rgb_to_hex(c)) if f is None else
                 '<use href="#%s" fill="%s"/>' % (f, rgb_to_hex(c))) for f, c in shadow_raw], []
    regs = []
    for j in range(1, nz):
        kind, d, rule = geo.region(sh.L, taus[j])
        regs.append(None if kind == 'empty' else d)
    nb = len(tks)
    Acum = [0.0] * nz
    Kst = [[[0.0, 0.0, 0.0] for _ in range(nb)] for _ in range(nz)]
    groups = []
    for j in range(nz - 1, -1, -1):
        # slope of the increment vs P inside zone j (pooled over bands)
        cov = var = 0.0
        for zs in zs_all:
            z = zs[j]
            n = len(z)
            mP = [sum(P[ch] for P, _ in z) / n for ch in range(3)]
            mD = [sum(D[ch] for _, D in z) / n for ch in range(3)]
            cov += sum((P[ch] - mP[ch]) * (D[ch] - mD[ch]) for P, D in z for ch in range(3))
            var += sum((P[ch] - mP[ch]) ** 2 for P, _ in z for ch in range(3))
        Afit = min(max(-cov / var if var > 1e-9 else 0.0, 0.0), 0.95)
        al = max(1e-3, 1 - (1 - Afit) / (1 - Acum[j]))

        def cols(al):
            An = 1 - (1 - Acum[j]) * (1 - al)
            out = []
            for k, zs in enumerate(zs_all):
                z = zs[j]
                n = len(z)
                out.append(tuple((sum(D[ch] + An * P[ch] for P, D in z) / n - Kst[j][k][ch] * (1 - al)) / al
                                 for ch in range(3)))
            return out
        def score(al):
            # worst error over all samples of this zone, with colours clamped to [0,1]
            An = 1 - (1 - Acum[j]) * (1 - al)
            worst = 0.0
            for cc, zs, kst in zip(cols(al), zs_all, Kst[j]):
                c = [min(max(x, 0.0), 1.0) for x in cc]
                for P, D in zs[j]:
                    for ch in range(3):
                        r = P[ch] * (1 - An) + kst[ch] * (1 - al) + al * c[ch]
                        worst = max(worst, abs(r - P[ch] - D[ch]))
            return worst
        cands = [al] + [0.01 * i for i in range(2, 101)]
        al = min(cands, key=lambda x: (round(score(x) * 255 * 4), x))
        al = min(1.0, math.ceil(al * 1000) / 1000.0)
        cs = [tuple(round(min(max(c, 0.0), 1.0) * 255) / 255.0 for c in cc) for cc in cols(al)]
        if a.debug:
            sys.stderr.write('zone %d tau %.3f Afit %.3f al %.3f Acum %.3f maxerr %.2f\n' % (j, taus[j], Afit, al, Acum[j], score(al) * 255))
        paths, prev = [], None
        for k, cq in enumerate(cs):
            if cq == prev or not any((jj, k) in occ for jj in ([j] if a.fill_exclusive else range(j + 1))):
                continue
            paths.append('<use href="#%s" fill="%s"/>' % (fids[k], rgb_to_hex(cq)))
            prev = cq
        if j == 0 and paths:
            # zone 0 also covers the key shadow (so its mask edge does not coincide
            # with the key-stack terminator edge, which would make an AA seam);
            # the opaque shadow colours are pre-compensated for this layer.
            applied, cur = [], None
            emitted = set(p.split('"')[1][1:] for p in paths)
            for k, cq in enumerate(cs):
                if fids[k] in emitted and (cur is None or cq != cur):
                    cur = cq
                applied.append(cur)
            zone0 = (al, applied)
        # mask {tau_j <= N.L1 < tau_j+1}  (zone 0: N.L1 < tau_1, shadow included)
        mid = '%s-m%d' % (pid, j)
        lo_d = geo.circle_d(geo.rext) if j == 0 else regs[j - 1]
        if lo_d is None:
            continue
        if j == nz - 1 or regs[j] is None:
            md = '<path d="%s" fill="#fff"/>' % lo_d
        else:
            md = '<path d="%s %s" fill="#fff" fill-rule="evenodd"/>' % (lo_d, regs[j])
        if paths:
            defs.append('<mask id="%s">%s</mask>' % (mid, md))
            op = '' if al >= 1.0 else ' opacity="%s"' % fmt_op(al)
            groups.append(('<g%s mask="url(#%s)">\n%s\n</g>' % (op, mid, '\n'.join(paths)),
                           al, [(tks[k][0], cs[k]) for k in range(nb)], j))
        for J in ([j] if a.fill_exclusive else range(j + 1)):
            Acum[J] = 1 - (1 - Acum[J]) * (1 - al)
            for k in range(nb):
                Kst[J][k] = [Kst[J][k][ch] * (1 - al) + al * cs[k][ch] for ch in range(3)]
    # opaque key-shadow stack, compensated for the zone-0 layer drawn over it
    shadow = []
    comp = []
    for i, (fid_, S) in enumerate(shadow_raw):
        if zone0 is not None and zone0[1][i] is not None:
            al0, c0 = zone0[0], zone0[1][i]
            S = tuple(min(max((x - al0 * y) / (1 - al0), 0.0), 1.0) if al0 < 1 else x for x, y in zip(S, c0))
        S = tuple(round(x * 255) / 255.0 for x in S)
        comp.append(S)
        if fid_ is None:
            shadow.append('<path d="%s" fill="%s"/>' % (geo.circle_d(geo.rext), rgb_to_hex(S)))
        else:
            shadow.append('<use href="#%s" fill="%s"/>' % (fid_, rgb_to_hex(S)))
    sh.fill_shadow = [(-2.0, comp[0])] + [(tks[i + 1][0], comp[i + 1]) for i in range(len(comp) - 1)]
    return shadow, groups


def build_toon(a):
    """Cel shading: a few flat bands with crisp (antialiased) edges.

    Every band is the exact region {N.L1 >= t} (ellipse front arc + silhouette
    arc), drawn as nested opaque shapes: each band is a full region painted
    over the previous one, so a lower band always extends under the next and
    no two translucent/AA edges meet edge to edge (no seams).
      * band colours: out(ambient + fill_lift + key * rep_i), rep_i = the
        band's middle N.L (shadow band: 0).  The fill light (--light2) does
        not get its own bands; it contributes a uniform lift of
        0.5*intensity2 (roughly its average over the visible hemisphere) and
        drives the direction of the optional rim.
      * highlight: region {N.H >= cos(size)}, flat light-coloured spot.
      * rim: {|N_xy| >= 1 - width} (concentric circles, N.V = const) clipped to
        the fill-light side {N.R >= spread}, R = screen-plane direction of the
        fill light (mirrored key if the fill is off); flat, lightened mid colour.
      * outline: circle stroke centred on the silhouette.
    Everything but the outline is masked once by the sphere circle."""
    sh = Shader(a)
    geo = Geo(a.cx, a.cy, a.r, a.precision, a.overshoot)
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
        k = sh.key_lin(t)
        return sh.out(tuple(x + y for x, y in zip(k, lift)))
    edges = [None] + th + [1.0]
    cols = []
    for i in range(n):
        if i == 0:
            cols.append(col_at(0.0))
        else:
            cols.append(col_at(0.5 * (edges[i] + edges[i + 1])))
    pid = a.mask_id
    els = ['<path d="%s" fill="%s"/>' % (geo.circle_d(geo.rext), rgb_to_hex(cols[0]))]
    for i in range(1, n):
        kind, d, rule = geo.region(sh.L, edges[i])
        if kind == 'empty':
            continue
        els.append('<path d="%s" fill="%s"%s/>' % (d, rgb_to_hex(cols[i]), ' fill-rule="evenodd"' if rule else ''))
    defs = []
    mix = lambda p, q, f: tuple(x + (y - x) * f for x, y in zip(p, q))
    lc1 = hex_to_rgb(a.light_color)
    if a.toon_rim > 0:
        src = sh.L2 if sh.on2 else (-sh.L[0], -sh.L[1], 0.0)
        A = normalize((src[0], src[1], 0.0)) if math.hypot(src[0], src[1]) > 1e-6 else (-0.6, -0.8, 0.0)
        kind, d, rule = geo.region(A, a.toon_rim_spread)
        if kind != 'empty':
            rin = a.r * (1 - a.toon_rim)
            ring = geo.circle_d(geo.rext) + ' ' + geo.circle_d(rin)
            rimc = mix(cols[min(1, n - 1)], hex_to_rgb(a.light2_color if sh.on2 else a.light_color), 0.25)
            clip = ''
            if kind != 'full':
                defs.append('<clipPath id="%s-r"><path d="%s"%s/></clipPath>' % (
                    pid, d, ' clip-rule="evenodd"' if rule else ''))
                clip = ' clip-path="url(#%s-r)"' % pid
            els.append('<path d="%s" fill="%s" fill-rule="evenodd"%s/>' % (ring, rgb_to_hex(rimc), clip))
    if a.toon_highlight:
        kind, d, rule = geo.region(sh.H, math.cos(math.radians(a.toon_highlight_size)))
        if kind != 'empty':
            hc = mix(cols[-1], lc1, 0.8)
            els.append('<path d="%s" fill="%s"%s/>' % (d, rgb_to_hex(hc), ' fill-rule="evenodd"' if rule else ''))
    out = []
    if a.outline > 0:
        if a.outline_color == 'auto':
            oc = tuple(c * 0.45 for c in cols[0])
        else:
            oc = hex_to_rgb(a.outline_color)
        out.append('<circle cx="%s" cy="%s" r="%s" fill="none" stroke="%s" stroke-width="%s"/>' % (
            geo.f(a.cx), geo.f(a.cy), geo.f(a.r - a.outline * a.outline_inset), rgb_to_hex(oc), geo.f(a.outline)))
    vb = a.viewbox
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" id="icon" viewBox="0 0 {vb} {vb}">'
           '<defs><mask id="{id}"><circle cx="{cx}" cy="{cy}" r="{r}" fill="#fff"/></mask>{defs}</defs>'
           '<g mask="url(#{id})">\n{body}\n</g>{out}</svg>\n').format(
        vb=geo.f(vb), id=pid, cx=geo.f(a.cx), cy=geo.f(a.cy), r=geo.f(a.r),
        defs=''.join(defs), body='\n'.join(els), out=''.join('\n' + x for x in out))
    return svg, n, 1 if a.toon_highlight else 0, sh


def build_svg(a):
    if getattr(a, 'shading', 'pbr') == 'toon':
        return build_toon(a)
    sh = Shader(a)
    geo = Geo(a.cx, a.cy, a.r, a.precision, a.overshoot)
    els = []
    defs = []
    step_d = a.diffuse_step / 255.0
    step_s = a.spec_step / 255.0

    # --- key-light diffuse stack (opaque)
    L = sh.L
    sL = math.hypot(L[0], L[1])
    tmin_vis = -sL if L[2] >= 0 else -1.0
    dcol = lambda t: sh.out(sh.base_lin(t))
    fill = None
    if sh.on2:
        fill = fill_stack(a, sh, geo, a.fill_step / 255.0, defs)
        els += fill[0]  # key-shadow colours (ambient + fill), opaque
    else:
        els.append('<path d="%s" fill="%s"/>' % (geo.circle_d(geo.rext), rgb_to_hex(dcol(-1))))
    ndiff = 0
    if tmin_vis < 1:
        tstart = max(0.0, tmin_vis)
        edges = bands(dcol, tstart, 1.0, step_d)
        for ta, tb in zip(edges[:-1], edges[1:]):
            kind, d, rule = geo.region(L, ta)
            if kind == 'empty':
                continue
            col = rgb_to_hex(dcol(0.5 * (ta + tb)))
            els.append('<path d="%s" fill="%s"%s/>' % (d, col, ' fill-rule="evenodd"' if rule else ''))
            ndiff += 1
    sh.fill_groups = []
    if fill:
        for g_ in fill[1]:
            els.append(g_[0])
        sh.fill_groups = [g_[1:] for g_ in fill[1]]
        ndiff += len(fill[0])

    # --- key specular; base = exact two-light diffuse at the rep point
    # (mean of N over the circle N.H = t is t*H)
    dot = lambda p, q: sum(x * y for x, y in zip(p, q))
    g1, l1 = spec_stack(a, sh, geo, sh.H, sh.spec_lin,
                        lambda N: sh.diffuse_lin(dot(N, sh.L), dot(N, sh.L2)), step_s, 's')
    sh.spec_groups = [(g[0], g[1]) for g in g1]
    sh.spec_layers = l1
    els += emit_groups(g1)
    nspec = len(l1)
    # --- optional fill specular
    sh.spec2_groups = []
    if sh.spec2 > 0:
        g2, l2 = spec_stack(a, sh, geo, sh.H2, sh.spec2_lin,
                            lambda N: sh.diffuse_lin(dot(N, sh.L), dot(N, sh.L2)), step_s, 'f')
        sh.spec2_groups = [(g[0], g[1]) for g in g2]
        els += emit_groups(g2)
        nspec += len(l2)

    vb = a.viewbox
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" id="icon" viewBox="0 0 {vb} {vb}">'
           '<defs><mask id="{id}"><circle cx="{cx}" cy="{cy}" r="{r}" fill="#fff"/></mask>{defs}</defs>'
           '<g mask="url(#{id})">\n{body}\n</g></svg>\n').format(
        vb=geo.f(vb), id=a.mask_id, cx=geo.f(a.cx), cy=geo.f(a.cy), r=geo.f(a.r), defs=''.join('\n' + x for x in defs), body='\n'.join(els))
    return svg, ndiff, nspec, sh


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-o', '--output', default='material-pbr.svg')
    ap.add_argument('--base', default='#9a9aa0', help='base colour (sRGB hex)')
    ap.add_argument('--roughness', type=float, default=0.3)
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
    ap.add_argument('--fill-step', type=float, default=2.0, help='max sRGB step (0-255) per fill band')
    ap.add_argument('--fill-groups', type=int, default=3, help='opacity groups for the fill stack')
    ap.add_argument('--fill-bake', type=float, default=0.6,
                    help='fraction of the mid fill level baked into the opaque key stack')
    ap.add_argument('--fill-exclusive', type=int, default=1, help='1: disjoint zone masks, 0: nested')
    ap.add_argument('--fill-bake-mode', choices=('circle', 'const'), default='const')
    ap.add_argument('--fill-split-ends', type=int, default=2, help='1: halve the brightest key zone, 2: also the terminator zone')
    ap.add_argument('--fill-zones', type=int, default=12, help='key-light zones the fill bands are split into')
    ap.add_argument('--ambient', type=float, default=0.06, help='uniform ambient, fraction of albedo')
    ap.add_argument('--ambient-tinted', action='store_true', help='tint ambient by light colour')
    ap.add_argument('--exposure', type=float, default=1.0)
    ap.add_argument('--tonemap', choices=('soft', 'clamp', 'reinhard', 'aces'), default='soft')
    ap.add_argument('--knee', type=float, default=0.8, help="shoulder start for 'soft' tonemap")
    ap.add_argument('--diffuse-step', type=float, default=2.0, help='max sRGB step (0-255) per diffuse band')
    ap.add_argument('--spec-step', type=float, default=2.0, help='max sRGB step (0-255) per specular band')
    ap.add_argument('--spec-threshold', type=float, default=0.5, help='ignore specular below this (0-255)')
    ap.add_argument('--spec-group-ratio', type=float, default=3.0,
                    help='specular shapes are grouped; each group is one translucent composite, sized to last '
                         'until the cumulative specular alpha grows by this factor')
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--shading', choices=('pbr', 'toon'), default='pbr')
    ap.add_argument('--toon-bands', type=int, default=3, help='toon: number of flat N.L bands')
    ap.add_argument('--toon-thresholds', default='', help='toon: comma-separated N.L cutoffs (overrides --toon-bands)')
    ap.add_argument('--toon-highlight', type=int, default=1, help='toon: crisp specular spot (1/0)')
    ap.add_argument('--toon-highlight-size', type=float, default=6.5, help='toon: spot angular radius, degrees')
    ap.add_argument('--toon-rim', type=float, default=0.0,
                    help='toon: rim band width as fraction of the radius, on the fill-light side (0 = off)')
    ap.add_argument('--toon-rim-spread', type=float, default=0.35,
                    help='toon: rim covers silhouette points with N.R >= this (R = fill direction in screen plane)')
    ap.add_argument('--outline', type=float, default=0.6, help='toon: silhouette stroke width, viewBox units (0 = off)')
    ap.add_argument('--outline-color', default='auto', help="toon: hex colour or 'auto' (darkened shadow colour)")
    ap.add_argument('--outline-inset', type=float, default=0.0,
                    help='toon: move the stroke inward by this fraction of its width (0 = centred on silhouette)')
    ap.add_argument('--viewbox', type=float, default=24)
    ap.add_argument('--cx', type=float, default=12)
    ap.add_argument('--cy', type=float, default=12)
    ap.add_argument('--r', type=float, default=10)
    ap.add_argument('--overshoot', type=float, default=0.3, help='inner shapes extend past silhouette')
    ap.add_argument('--precision', type=int, default=3)
    ap.add_argument('--mask-id', default='pbrS')
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    svg, nd, ns, sh = build_svg(a)
    with open(a.output, 'w') as fh:
        fh.write(svg)
    sys.stderr.write('wrote %s: %d bytes, %d diffuse + %d specular layers, L=%s H=%s\n' % (
        a.output, len(svg.encode()), nd, ns,
        tuple(round(v, 4) for v in sh.L), tuple(round(v, 4) for v in sh.H)))


if __name__ == '__main__':
    main()
