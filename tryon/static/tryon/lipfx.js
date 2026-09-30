/*
 * LipFX - pure-JS port of the finishes in tryon/services/lip_engine.py.
 * Works on a small ROI (the lips + margin), so it is cheap enough for live video.
 * No DOM access: runs in the browser (window.LipFX) and in Node (tests).
 */
(function (root) {
  'use strict';
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  const smoothstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };

  // ---- Tunables (keep in sync with tryon/services/lip_engine.py) ----
  // Paint colour = LIP_MIX * lipstick + (1 - LIP_MIX) * the user's own average lip colour.
  // A real colour mix (not just the alpha compositing in render()), so the lipstick's hue is
  // already warmed/cooled by the person's own pigment before it touches the video frame.
  const LIP_MIX = 0.7;
  // How strongly the LIP'S OWN TEXTURE (not the room's lighting - see shade() below) lightens a
  // naturally-raised/highlighted spot, and darkens a naturally-recessed one, for each finish.
  const TONE = { matte: [0.35, 0.35], satin: [0.50, 0.30], glossy: [0.65, 0.25], velvet: [0.25, 0.45] };
  // Glossy only: extra whitening on the very brightest texture ridges - the "wet" specular look.
  const GLOSSY_SPECULAR = 0.35;

  // ---------- Gaussian blur (3 box passes, edge-replicated) ----------
  function boxesForGauss(sigma, n) {
    const wi = Math.sqrt((12 * sigma * sigma) / n + 1);
    let wl = Math.floor(wi); if (wl % 2 === 0) wl--;
    const wu = wl + 2;
    const m = Math.round((12 * sigma * sigma - n * wl * wl - 4 * n * wl - 3 * n) / (-4 * wl - 4));
    return Array.from({ length: n }, (_, i) => (i < m ? wl : wu));
  }
  function boxH(s, d, w, h, r) {
    const k = 1 / (2 * r + 1);
    for (let y = 0; y < h; y++) {
      const b = y * w; let acc = 0;
      for (let i = -r; i <= r; i++) acc += s[b + clamp(i, 0, w - 1)];
      d[b] = acc * k;
      for (let x = 1; x < w; x++) {
        acc += s[b + Math.min(x + r, w - 1)] - s[b + Math.max(x - r - 1, 0)];
        d[b + x] = acc * k;
      }
    }
  }
  function boxV(s, d, w, h, r) {
    const k = 1 / (2 * r + 1);
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) acc += s[clamp(i, 0, h - 1) * w + x];
      d[x] = acc * k;
      for (let y = 1; y < h; y++) {
        acc += s[Math.min(y + r, h - 1) * w + x] - s[Math.max(y - r - 1, 0) * w + x];
        d[y * w + x] = acc * k;
      }
    }
  }
  function gauss(src, w, h, sigma) {
    let a = Float32Array.from(src);
    if (sigma < 0.4) return a;
    let b = new Float32Array(a.length);
    for (const bw of boxesForGauss(sigma, 3)) {
      const r = (bw - 1) >> 1;
      boxH(a, b, w, h, r); boxV(b, a, w, h, r);
    }
    return a;
  }

  // ---------- Helpers ----------
  // For pixels with flag=1: distance to the nearest flag=0 pixel (3-4 chamfer). oob = value used outside the ROI.
  function chamfer(flag, w, h, oob) {
    const INF = 1e9, d = new Float32Array(w * h);
    for (let i = 0; i < d.length; i++) d[i] = flag[i] ? INF : 0;
    const at = (x, y) => (x < 0 || y < 0 || x >= w || y >= h ? oob : d[y * w + x]);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x; if (d[i] === 0) continue;
      d[i] = Math.min(d[i], at(x - 1, y) + 1, at(x, y - 1) + 1, at(x - 1, y - 1) + 1.4142, at(x + 1, y - 1) + 1.4142);
    }
    for (let y = h - 1; y >= 0; y--) for (let x = w - 1; x >= 0; x--) {
      const i = y * w + x; if (d[i] === 0) continue;
      d[i] = Math.min(d[i], at(x + 1, y) + 1, at(x, y + 1) + 1, at(x + 1, y + 1) + 1.4142, at(x - 1, y + 1) + 1.4142);
    }
    return d;
  }
  function distanceMap(mask, w, h) {              // 0 at lip edge -> 1 at centre
    const d = chamfer(Uint8Array.from(mask, v => (v > 0.5 ? 1 : 0)), w, h, 0);
    let mx = 0; for (let i = 0; i < d.length; i++) if (d[i] > mx) mx = d[i];
    if (mx > 0) for (let i = 0; i < d.length; i++) d[i] /= mx;
    return d;
  }
  function percentiles(vals, mask, ps) {           // over mask>0.5, histogram based (fast)
    const BINS = 512, hist = new Uint32Array(BINS); let n = 0;
    for (let i = 0; i < vals.length; i++) if (mask[i] > 0.5) { hist[clamp((vals[i] * (BINS - 1)) | 0, 0, BINS - 1)]++; n++; }
    return ps.map(p => { let c = 0; const t = (p / 100) * n;
      for (let b = 0; b < BINS; b++) { c += hist[b]; if (c >= t) return b / (BINS - 1); } return 1; });
  }
  function makeNoise(seed = 42, size = 256) {      // pre-blurred grain tile, std ~ 7 levels
    let s = seed >>> 0;
    const rnd = () => { s = (s + 0x6D2B79F5) >>> 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t ^= t + Math.imul(t ^ (t >>> 7), 61 | t); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    const g = new Float32Array(size * size);
    for (let i = 0; i < g.length; i++) g[i] = Math.sqrt(-2 * Math.log(rnd() + 1e-9)) * Math.cos(2 * Math.PI * rnd());
    const b = gauss(g, size, size, 0.7); for (let i = 0; i < b.length; i++) b[i] *= 7;
    return { tile: b, size };
  }


  // ---------- Landmark filter: One Euro (steady when still, no lag when moving) ----------
  class OneEuro {
    constructor(minCutoff = 1.2, beta = 0.1, dCutoff = 1.0) { this.mc = minCutoff; this.b = beta; this.dc = dCutoff; this.x = null; this.dx = 0; this.t = 0; }
    static a(cutoff, dt) { return 1 / (1 + 1 / (2 * Math.PI * cutoff) / dt); }
    filter(v, t) {
      if (this.x === null) { this.x = v; this.t = t; this.dx = 0; return v; }
      const dt = Math.max(t - this.t, 1e-3); this.t = t;
      this.dx += OneEuro.a(this.dc, dt) * ((v - this.x) / dt - this.dx);
      this.x += OneEuro.a(this.mc + this.b * Math.abs(this.dx), dt) * (v - this.x);
      return this.x;
    }
  }

  // ---------- Smooth lip outline (Catmull-Rom; the two mouth corners stay sharp) ----------
  function curveChain(pts, seg = 6) {
    const n = pts.length, P = i => pts[Math.max(0, Math.min(n - 1, i))], out = [];
    for (let i = 0; i < n - 1; i++) {
      const p0 = P(i - 1), p1 = P(i), p2 = P(i + 1), p3 = P(i + 2);
      for (let s = 0; s < seg; s++) {
        const t = s / seg, t2 = t * t, t3 = t2 * t, f = k => 0.5 * (2 * p1[k] + (p2[k] - p0[k]) * t +
          (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2 + (3 * p1[k] - p0[k] - 3 * p2[k] + p3[k]) * t3);
        out.push([f(0), f(1)]);
      }
    }
    out.push([pts[n - 1][0], pts[n - 1][1]]);
    return out;
  }
  // 20 ordered contour points (corner, upper lip ..., corner, lower lip ...) -> dense closed polyline
  function lipLoop(P, seg = 6) {
    const upper = curveChain(P.slice(0, 11), seg), lower = curveChain(P.slice(10).concat([P[0]]), seg);
    return upper.concat(lower.slice(1));
  }

  // ---------- 3D depth of the lip region: quadratic surface fitted to the landmarks' z ----------
  function solve(A, b) {
    const n = b.length;
    for (let i = 0; i < n; i++) {
      let p = i; for (let r = i + 1; r < n; r++) if (Math.abs(A[r][i]) > Math.abs(A[p][i])) p = r;
      [A[i], A[p]] = [A[p], A[i]]; [b[i], b[p]] = [b[p], b[i]];
      for (let r = i + 1; r < n; r++) { const f = A[r][i] / A[i][i]; for (let c = i; c < n; c++) A[r][c] -= f * A[i][c]; b[r] -= f * b[i]; }
    }
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) { let t = b[i]; for (let c = i + 1; c < n; c++) t -= A[i][c] * x[c]; x[i] = t / A[i][i]; }
    return x;
  }
  // pts: [[x, y, z]] in pixels (z in the same scale as x). depth d(u,v) = a0 + a1u + a2v + a3u² + a4uv + a5v²
  function fitSurface(pts, lipW) {
    const n = pts.length; let cx = 0, cy = 0; for (const p of pts) { cx += p[0]; cy += p[1]; } cx /= n; cy /= n;
    const A = Array.from({ length: 6 }, () => new Array(6).fill(0)), b = new Array(6).fill(0);
    for (const p of pts) {
      const u = (p[0] - cx) / lipW, v = (p[1] - cy) / lipW, d = p[2] / lipW, f = [1, u, v, u * u, u * v, v * v];
      for (let i = 0; i < 6; i++) { b[i] += f[i] * d; for (let j = 0; j < 6; j++) A[i][j] += f[i] * f[j]; }
    }
    for (let i = 0; i < 6; i++) A[i][i] += 1e-4 * n;
    return { cx, cy, sc: lipW, a: solve(A, b) };
  }
  // cos(angle between the lip surface normal and the camera) at pixel (x, y): 1 = facing us, 0 = edge-on
  function facingAt(fit, x, y) {
    const u = (x - fit.cx) / fit.sc, v = (y - fit.cy) / fit.sc, a = fit.a;
    const gx = a[1] + 2 * a[3] * u + a[4] * v, gy = a[2] + a[4] * u + 2 * a[5] * v;
    return 1 / Math.sqrt(1 + gx * gx + gy * gy);
  }

  // ---------- Final lip mask: snap to the real lip border, feather, fade at grazing angles ----------
  /**
   * mOuter / mInner: anti-aliased fills of the outer lip outline and of the mouth opening (Float32, 0..1).
   * o: { lipW, refine (default true), band (max edge move, fraction of lip width, default 0.05), fit, ox, oy }  ->  Float32Array mask
   */
  function finishMask(mOuter, mInner, rgba, w, h, o) {
    const n = w * h, lipW = o.lipW;
    let outer = mOuter;
    if (o.refine !== false) {
      const hardO = new Uint8Array(n), inv = new Uint8Array(n), hardG = new Uint8Array(n);
      for (let i = 0; i < n; i++) { const ho = mOuter[i] > 0.5 ? 1 : 0; hardO[i] = ho; inv[i] = 1 - ho; hardG[i] = ho && mInner[i] < 0.5 ? 1 : 0; }
      const dIn = chamfer(hardO, w, h, 0), dOut = chamfer(inv, w, h, 1e9), dGeo = chamfer(hardG, w, h, 0);
      const delta = Math.max(2, (o.band ?? 0.05) * lipW), coreD = Math.max(2, 0.05 * lipW), ring = 0.15 * lipW;
      const f0 = new Float32Array(n);                                      // red chroma (Cr): lips > skin
      for (let i = 0; i < n; i++) f0[i] = 0.5 * rgba[4 * i] - 0.4187 * rgba[4 * i + 1] - 0.0813 * rgba[4 * i + 2];
      const f = gauss(f0, w, h, 1.0);
      let s1 = 0, q1 = 0, n1 = 0, s2 = 0, q2 = 0, n2 = 0;
      for (let i = 0; i < n; i++) {
        if (dGeo[i] > coreD && dIn[i] > delta) { s1 += f[i]; q1 += f[i] * f[i]; n1++; }                       // sure lip
        else if (!hardO[i] && dOut[i] > delta && dOut[i] < delta + ring) { s2 += f[i]; q2 += f[i] * f[i]; n2++; } // skin ring
      }
      if (n1 > 30 && n2 > 30) {
        const mu1 = s1 / n1, mu2 = s2 / n2, sd = Math.sqrt(Math.max(((q1 / n1 - mu1 * mu1) + (q2 / n2 - mu2 * mu2)) / 2, 1e-3));
        const dl = mu1 - mu2;
        // confidence: how well red chroma separates lip from skin in THIS frame (0 = don't touch the geometry)
        const conf = dl < 4 ? 0 : clamp((dl / sd - 0.8) / 1.2, 0, 1);
        if (conf > 0) {
          outer = new Float32Array(n);
          for (let i = 0; i < n; i++) {
            const grow = !hardO[i];                                        // turning a SKIN pixel into lip
            const a = grow ? dOut[i] : dIn[i];                              // distance to the geometric border
            if (a >= delta) { outer[i] = mOuter[i]; continue; }
            // Asymmetric on purpose: bleeding lipstick onto skin (growing) is far more visible than
            // trimming a pixel of lip back to skin (shrinking), so growing needs much stronger evidence.
            const p = grow ? smoothstep(0.55, 0.9, (f[i] - mu2) / dl) : smoothstep(0.15, 0.5, (f[i] - mu2) / dl);
            const k = (grow ? conf * 0.6 : conf) * (1 - smoothstep(0.55 * delta, delta, a));   // fades to 0 at the band edge
            outer[i] = mOuter[i] * (1 - k) + p * k;
          }
        }
      }
    }
    const m0 = new Float32Array(n);
    for (let i = 0; i < n; i++) m0[i] = outer[i] * (1 - mInner[i]);
    const m = gauss(m0, w, h, Math.max(1, lipW * 0.02));
    if (o.fit) for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {   // depth-aware fade at grazing angles
      const i = y * w + x; if (m[i] > 0) m[i] *= smoothstep(0.25, 0.5, facingAt(o.fit, x + o.ox, y + o.oy));
    }
    return m;
  }

  // ---------- Main entry ----------
  /**
   * rgba: Uint8ClampedArray (ROI, modified in place) | w,h: ROI size | mask: Float32Array 0..1
   * o: { color:[r,g,b], alpha, finish, lipW, noise, ox, oy, state }
   *    state: object kept between frames (smooths the light map so it does not flicker)
   */
  function render(rgba, w, h, mask, o) {
    const n = w * h, A = o.alpha, lipW = o.lipW;
    const R = new Float32Array(n), G = new Float32Array(n), B = new Float32Array(n), L = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      R[i] = rgba[4 * i]; G[i] = rgba[4 * i + 1]; B[i] = rgba[4 * i + 2];
      L[i] = (0.299 * R[i] + 0.587 * G[i] + 0.114 * B[i]) / 255;
    }
    // The user's average lip colour (raw pixels, before any tint), smoothed over frames so it doesn't flicker.
    let mr = 0, mg = 0, mb = 0, mc = 0;
    for (let i = 0; i < n; i++) if (mask[i] > 0.5) { mr += R[i]; mg += G[i]; mb += B[i]; mc++; }
    let mean = mc ? [mr / mc, mg / mc, mb / mc] : o.color.slice();
    if (o.state) { const p = o.state.mean; if (p) mean = mean.map((v, k) => p[k] + 0.2 * (v - p[k])); o.state.mean = mean; }
    const mix = o.mix ?? LIP_MIX;
    const cr = mix * o.color[0] + (1 - mix) * mean[0], cg = mix * o.color[1] + (1 - mix) * mean[1], cb = mix * o.color[2] + (1 - mix) * mean[2];
    const planes = (r, g, b) => ({ r, g, b });
    const soften = (p, amount) => {
      const s = Math.max(1, lipW * 0.006);
      const br = gauss(p.r, w, h, s), bg = gauss(p.g, w, h, s), bb = gauss(p.b, w, h, s);
      for (let i = 0; i < n; i++) { const m = mask[i] * amount;
        p.r[i] = p.r[i] * (1 - m) + br[i] * m; p.g[i] = p.g[i] * (1 - m) + bg[i] * m; p.b[i] = p.b[i] * (1 - m) + bb[i] * m; }
      return p;
    };
    const addWhite = (p, map) => { for (let i = 0; i < n; i++) { const v = 255 * map[i]; p.r[i] += v; p.g[i] += v; p.b[i] += v; } return p; };
    let depth = null; const getDepth = () => depth || (depth = distanceMap(mask, w, h));
    const band = (c, wd) => { const d = getDepth(), out = new Float32Array(n);
      for (let i = 0; i < n; i++) { const t = (d[i] - c) / wd; out[i] = Math.exp(-t * t) * mask[i]; } return out; };

    // Local texture lightness (~0.6 recess .. 1.0 neutral .. 1.4 highlight ridge): a small blur of the
    // lips' own luma divided by a WIDER blur of the same pixels.  A slow, room-wide lighting gradient
    // shows up almost identically in both blurs and cancels out of the ratio; what survives is texture
    // that changes over a shorter distance than the wide blur - the vermillion border, the cupid's bow,
    // the seam, the corners.  That is "the brightness of the lip's own texture", not the room's light.
    const s1 = 0.8, s2 = Math.max(2, lipW * 0.06), Lm = new Float32Array(n);
    for (let i = 0; i < n; i++) Lm[i] = L[i] * mask[i];
    const a1 = gauss(Lm, w, h, s1), b1 = gauss(mask, w, h, s1), a2 = gauss(Lm, w, h, s2), b2 = gauss(mask, w, h, s2);
    const tHi = new Float32Array(n), tLo = new Float32Array(n);         // highlight / recess amounts, 0..1
    for (let i = 0; i < n; i++) {
      const fine = a1[i] / Math.max(b1[i], 1e-3), mid = a2[i] / Math.max(b2[i], 1e-3);
      let sh = clamp(fine / Math.max(mid, 0.05), 0.6, 1.4);
      const conf = smoothstep(0.05, 0.3, b2[i]);                        // fade to neutral at the mask edge
      sh = 1 + (sh - 1) * conf;
      tHi[i] = clamp((sh - 1) / 0.4, 0, 1); tLo[i] = clamp((1 - sh) / 0.4, 0, 1);
    }

    // Paint the lips at a constant opacity everywhere, but let the texture above brighten/darken the
    // paint itself first (highlight -> towards white, recess -> towards black) before it's laid down.
    const tinted = (lightGain, darkGain, specGain) => {
      const out = planes(new Float32Array(n), new Float32Array(n), new Float32Array(n));
      for (let i = 0; i < n; i++) {
        const wt = mask[i] * A;
        let tr = cr + tHi[i] * lightGain * (255 - cr), tg = cg + tHi[i] * lightGain * (255 - cg), tb = cb + tHi[i] * lightGain * (255 - cb);
        if (specGain) { const s = tHi[i] * tHi[i] * specGain; tr += s * (255 - tr); tg += s * (255 - tg); tb += s * (255 - tb); }
        const k = 1 - tLo[i] * darkGain; tr *= k; tg *= k; tb *= k;
        out.r[i] = R[i] * (1 - wt) + tr * wt; out.g[i] = G[i] * (1 - wt) + tg * wt; out.b[i] = B[i] * (1 - wt) + tb * wt;
      }
      return out;
    };
    let out;
    switch (o.finish) {
      case 'glossy':
        out = soften(tinted(...TONE.glossy, GLOSSY_SPECULAR), 0.35); break;
      case 'satin':
        out = soften(tinted(...TONE.satin), 0.2); break;
      case 'velvet': {
        out = soften(tinted(...TONE.velvet), 0.5);
        const nz = o.noise, d = getDepth();
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const i = y * w + x;
          const g = nz ? nz.tile[(((y + o.oy) % nz.size + nz.size) % nz.size) * nz.size + (((x + o.ox) % nz.size + nz.size) % nz.size)] * mask[i] : 0;
          const e = (1 - d[i]) * (1 - d[i]), k = 1 - 0.18 * e * mask[i];
          out.r[i] = (out.r[i] + g) * k; out.g[i] = (out.g[i] + g) * k; out.b[i] = (out.b[i] + g) * k; }
        const bd = band(0.6, 0.3); for (let i = 0; i < n; i++) bd[i] *= 0.05;
        out = addWhite(out, bd); break;
      }
      default: out = tinted(...TONE.matte);
    }
    for (let i = 0; i < n; i++) if (mask[i] > 0) {         // only touch lip pixels
      rgba[4 * i] = out.r[i]; rgba[4 * i + 1] = out.g[i]; rgba[4 * i + 2] = out.b[i];
    }
  }

  const api = { render, gauss, makeNoise, OneEuro, curveChain, lipLoop, fitSurface, facingAt, finishMask };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.LipFX = api;
})(typeof self !== 'undefined' ? self : this);