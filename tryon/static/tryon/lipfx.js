/*
 * LipFX - pure-JS port of the finishes in tryon/services/lip_engine.py.
 * Works on a small ROI (the lips + margin), so it is cheap enough for live video.
 * No DOM access: runs in the browser (window.LipFX) and in Node (tests).
 */
(function (root) {
  'use strict';
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  const LIGHT = { matte: 0.5, glossy: 1.0, velvet: 0.6 };

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
  function distanceMap(mask, w, h) {              // 0 at lip edge -> 1 at centre (chamfer 3-4)
    const INF = 1e9, d = new Float32Array(w * h);
    for (let i = 0; i < d.length; i++) d[i] = mask[i] > 0.5 ? INF : 0;
    const at = (x, y) => (x < 0 || y < 0 || x >= w || y >= h ? 0 : d[y * w + x]);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x; if (d[i] === 0) continue;
      d[i] = Math.min(d[i], at(x - 1, y) + 1, at(x, y - 1) + 1, at(x - 1, y - 1) + 1.4142, at(x + 1, y - 1) + 1.4142);
    }
    let mx = 0;
    for (let y = h - 1; y >= 0; y--) for (let x = w - 1; x >= 0; x--) {
      const i = y * w + x; if (d[i] === 0) continue;
      d[i] = Math.min(d[i], at(x + 1, y) + 1, at(x, y + 1) + 1, at(x + 1, y + 1) + 1.4142, at(x - 1, y + 1) + 1.4142);
      if (d[i] > mx) mx = d[i];
    }
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

  // ---------- Main entry ----------
  /**
   * rgba: Uint8ClampedArray (ROI, modified in place) | w,h: ROI size | mask: Float32Array 0..1
   * o: { color:[r,g,b], alpha, finish, lipW, noise, ox, oy, state }
   *    state: object kept between frames (smooths the light map so it does not flicker)
   */
  function render(rgba, w, h, mask, o) {
    const n = w * h, A = o.alpha, [cr, cg, cb] = o.color, lipW = o.lipW;
    const R = new Float32Array(n), G = new Float32Array(n), B = new Float32Array(n), L = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      R[i] = rgba[4 * i]; G[i] = rgba[4 * i + 1]; B[i] = rgba[4 * i + 2];
      L[i] = (0.299 * R[i] + 0.587 * G[i] + 0.114 * B[i]) / 255;
    }
    const planes = (r, g, b) => ({ r, g, b });
    const copy = () => planes(Float32Array.from(R), Float32Array.from(G), Float32Array.from(B));

    const composite = (src, wfn) => {                     // out = img*(1-w) + colour*w
      const out = planes(new Float32Array(n), new Float32Array(n), new Float32Array(n));
      for (let i = 0; i < n; i++) { const wt = wfn ? wfn(i) : mask[i] * A;
        out.r[i] = src.r[i] * (1 - wt) + cr * wt; out.g[i] = src.g[i] * (1 - wt) + cg * wt; out.b[i] = src.b[i] * (1 - wt) + cb * wt; }
      return out;
    };
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
    const highlights = strength => {
      let sum = 0, sum2 = 0, cnt = 0;
      for (let i = 0; i < n; i++) if (mask[i] > 0.5) { sum += L[i]; sum2 += L[i] * L[i]; cnt++; }
      const mu = sum / Math.max(cnt, 1), sd = Math.sqrt(Math.max(sum2 / Math.max(cnt, 1) - mu * mu, 0)) + 1e-3;
      const s = new Float32Array(n);
      for (let i = 0; i < n; i++) s[i] = Math.pow(clamp((L[i] - (mu + 0.4 * sd)) / (2 * sd), 0, 1), 1.5) * mask[i];
      const b = gauss(s, w, h, lipW * 0.008); for (let i = 0; i < n; i++) b[i] *= strength; return b;
    };
    const lightMap = () => {                               // 0 shadow .. 1 brightest spot of the lips
      const lum = gauss(L, w, h, Math.max(1.5, lipW * 0.03));
      let [lo, hi] = percentiles(lum, mask, [10, 99.7]);
      const st = o.state;
      if (st) { if (st.lo !== undefined) { lo = st.lo + 0.15 * (lo - st.lo); hi = st.hi + 0.15 * (hi - st.hi); } st.lo = lo; st.hi = hi; }
      const out = new Float32Array(n), span = Math.max(hi - lo, 1e-3);
      for (let i = 0; i < n; i++) { let t = clamp((lum[i] - lo) / span, 0, 1); t = t * t * (3 - 2 * t); out[i] = Math.pow(t, 2.2); }
      return out;
    };

        // Total tint weight stays constant everywhere; only its split between colour and white
    // changes with the light (0 = all colour, 1 = all white).
    const tintByLight = strength => {
      const s = lightMap(), out = planes(new Float32Array(n), new Float32Array(n), new Float32Array(n));
      for (let i = 0; i < n; i++) { const wt = mask[i] * A, k = s[i] * strength;
        out.r[i] = R[i] * (1 - wt) + ((1 - k) * cr + k * 255) * wt;
        out.g[i] = G[i] * (1 - wt) + ((1 - k) * cg + k * 255) * wt;
        out.b[i] = B[i] * (1 - wt) + ((1 - k) * cb + k * 255) * wt; }
      return out;
    };
    let out;
    switch (o.finish) {
      case 'glossy':
        out = soften(tintByLight(LIGHT.glossy), 0.35); break;
      case 'satin': {
        out = soften(composite(copy()), 0.2);
        const hl = highlights(0.35), bd = band(0.55, 0.35);
        const mix = new Float32Array(n); for (let i = 0; i < n; i++) mix[i] = hl[i] + bd[i] * 0.10;
        out = addWhite(out, mix); break;
      }
      case 'velvet': {
        out = soften(tintByLight(LIGHT.velvet), 0.5);
        const nz = o.noise, d = getDepth();
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const i = y * w + x;
          const g = nz ? nz.tile[(((y + o.oy) % nz.size + nz.size) % nz.size) * nz.size + (((x + o.ox) % nz.size + nz.size) % nz.size)] * mask[i] : 0;
          const e = (1 - d[i]) * (1 - d[i]), k = 1 - 0.18 * e * mask[i];
          out.r[i] = (out.r[i] + g) * k; out.g[i] = (out.g[i] + g) * k; out.b[i] = (out.b[i] + g) * k; }
        const bd = band(0.6, 0.3); for (let i = 0; i < n; i++) bd[i] *= 0.05;
        out = addWhite(out, bd); break;
      }
      default: out = tintByLight(LIGHT.matte)
    }
    for (let i = 0; i < n; i++) if (mask[i] > 0) {         // only touch lip pixels
      rgba[4 * i] = out.r[i]; rgba[4 * i + 1] = out.g[i]; rgba[4 * i + 2] = out.b[i];
    }
  }

  const api = { render, gauss, makeNoise };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.LipFX = api;
})(typeof self !== 'undefined' ? self : this);
