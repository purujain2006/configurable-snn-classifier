/* gesture.js: a simulated 128 x 128 event camera watching simple arm gestures.
   Moving forearms and hands change the brightness under them; a pixel emits
   an event when its log brightness has moved by a fixed step since its last
   event, as a real dynamic vision sensor does. */
(function () {
  "use strict";
  const RES = 128, DT = 0.004, DUR = 1.4, STEP = 0.2, BASE = 0.25;

  function rwave(u) {
    const E = [42, 82], a = -Math.PI / 2 + 0.62 * Math.sin(2 * Math.PI * u / 0.7);
    const H = [E[0] + 28 * Math.cos(a), E[1] + 28 * Math.sin(a)];
    return [{ a: E, b: H, w: 3.6, amp: 0.65 }, { a: H, b: H, w: 6, amp: 0.85 }];
  }
  const KINDS = {
    rwave,
    lwave: u => rwave(u).map(p => ({ ...p, a: [127 - p.a[0], p.a[1]], b: [127 - p.b[0], p.b[1]] })),
    clap(u) {
      const d = 5 + 20 * (0.5 + 0.5 * Math.cos(2 * Math.PI * u / 0.6));
      const HL = [64 - d, 74], HR = [64 + d, 74];
      return [{ a: [36, 98], b: HL, w: 3.6, amp: 0.65 }, { a: [92, 98], b: HR, w: 3.6, amp: 0.65 },
        { a: HL, b: HL, w: 6, amp: 0.85 }, { a: HR, b: HR, w: 6, amp: 0.85 }];
    },
    roll(u) {
      const th = 2 * Math.PI * u / 0.55, c = Math.cos(th), s = Math.sin(th);
      const H1 = [64 + 13 * c, 80 + 7 * s], H2 = [64 - 13 * c, 80 - 7 * s];
      return [{ a: [38, 102], b: H1, w: 3.6, amp: 0.65 }, { a: [90, 102], b: H2, w: 3.6, amp: 0.65 },
        { a: H1, b: H1, w: 5.5, amp: 0.85 }, { a: H2, b: H2, w: 5.5, amp: 0.85 }];
    }
  };
  const NAMES = { rwave: "right hand wave", lwave: "left hand wave", clap: "hand clap", roll: "arm roll" };

  function dist2(px, py, p) {
    const [ax, ay] = p.a, [bx, by] = p.b, vx = bx - ax, vy = by - ay;
    const L2 = vx * vx + vy * vy;
    let t = L2 ? ((px - ax) * vx + (py - ay) * vy) / L2 : 0;
    t = Math.max(0, Math.min(1, t));
    const dx = px - (ax + t * vx), dy = py - (ay + t * vy);
    return dx * dx + dy * dy;
  }
  function intensity(px, py, parts) {
    let v = BASE;
    for (const p of parts) v += p.amp * Math.exp(-dist2(px, py, p) / (p.w * p.w));
    return v;
  }
  function bbox(parts) {
    let x0 = RES, y0 = RES, x1 = 0, y1 = 0;
    for (const p of parts) {
      const m = p.w * 2.6 + 2;
      x0 = Math.min(x0, p.a[0] - m, p.b[0] - m); x1 = Math.max(x1, p.a[0] + m, p.b[0] + m);
      y0 = Math.min(y0, p.a[1] - m, p.b[1] - m); y1 = Math.max(y1, p.a[1] + m, p.b[1] + m);
    }
    return [Math.max(0, Math.floor(x0)), Math.max(0, Math.floor(y0)), Math.min(RES - 1, Math.ceil(x1)), Math.min(RES - 1, Math.ceil(y1))];
  }

  const cache = {};
  function makeClip(kind) {
    if (cache[kind]) return cache[kind];
    const fn = KINDS[kind], r = Learn.rng(kind.length * 97 + 5);
    const ref = new Float32Array(RES * RES).fill(Math.log(BASE));
    let parts = fn(0);
    let [bx0, by0, bx1, by1] = bbox(parts);
    for (let y = by0; y <= by1; y++) for (let x = bx0; x <= bx1; x++) ref[y * RES + x] = Math.log(intensity(x, y, parts));
    const ts = [], xs = [], ys = [], ps = [];
    const steps = Math.round(DUR / DT);
    for (let k = 1; k <= steps; k++) {
      const u = k * DT;
      parts = fn(u);
      const nb = bbox(parts);
      const x0 = Math.min(bx0, nb[0]), y0 = Math.min(by0, nb[1]), x1 = Math.max(bx1, nb[2]), y1 = Math.max(by1, nb[3]);
      for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
        const i = y * RES + x, d = Math.log(intensity(x, y, parts)) - ref[i];
        if (Math.abs(d) > STEP) {
          const p = d > 0 ? 1 : -1;
          ref[i] += p * STEP;
          ts.push(u - DT * r()); xs.push(x); ys.push(y); ps.push(p);
        }
      }
      [bx0, by0, bx1, by1] = nb;
      if (r() < 0.5) { ts.push(u - DT * r()); xs.push(Math.floor(r() * RES)); ys.push(Math.floor(r() * RES)); ps.push(r() < 0.5 ? 1 : -1); }
    }
    const order = ts.map((_, i) => i).sort((a, b) => ts[a] - ts[b]);
    const clip = {
      kind, name: NAMES[kind], dur: DUR, n: order.length,
      t: Float32Array.from(order, i => ts[i]), x: Uint8Array.from(order, i => xs[i]),
      y: Uint8Array.from(order, i => ys[i]), p: Int8Array.from(order, i => ps[i])
    };
    cache[kind] = clip;
    return clip;
  }

  /* counts of on and off events for events [i0, i1) */
  function frame(clip, i0, i1) {
    const on = new Uint16Array(RES * RES), off = new Uint16Array(RES * RES);
    for (let i = i0; i < i1; i++) (clip.p[i] > 0 ? on : off)[clip.y[i] * RES + clip.x[i]]++;
    return { on, off, t0: clip.t[i0] || 0, t1: clip.t[Math.max(i0, i1 - 1)] || 0 };
  }
  /* T slices with equal numbers of events */
  function slices(clip, T) {
    const out = [];
    for (let k = 0; k < T; k++) out.push(frame(clip, Math.round(k * clip.n / T), Math.round((k + 1) * clip.n / T)));
    return out;
  }
  /* sum a RES x RES count map into N x N blocks; crop = [x0, y0, side] limits it to a square */
  function resample(counts, N, crop) {
    const [x0, y0, side] = crop || [0, 0, RES];
    const out = new Float32Array(N * N);
    for (let y = y0; y < y0 + side; y++) {
      const ty = Math.floor((y - y0) * N / side);
      for (let x = x0; x < x0 + side; x++) {
        const c = counts[y * RES + x];
        if (c) out[ty * N + Math.floor((x - x0) * N / side)] += c;
      }
    }
    return out;
  }
  /* draw an on/off pair (any typed arrays of size N*N) into a canvas; vmax scales intensity */
  function paint(canvas, on, off, N, vmax) {
    canvas.width = N; canvas.height = N;
    const ctx = canvas.getContext("2d"), img = ctx.createImageData(N, N);
    vmax = vmax || 1;
    for (let i = 0; i < N * N; i++) {
      const a = Math.min(1, on[i] / vmax), b = Math.min(1, off[i] / vmax), k = i * 4;
      img.data[k] = Math.min(255, 8 + 216 * a + 184 * b);
      img.data[k + 1] = Math.min(255, 8 + 161 * a + 114 * b);
      img.data[k + 2] = Math.min(255, 8 + 71 * a + 84 * b);
      img.data[k + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  window.Gesture = { RES, makeClip, frame, slices, resample, paint, NAMES };
})();
