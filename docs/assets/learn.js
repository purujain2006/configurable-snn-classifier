/* learn.js: shared helpers for the walkthrough pages (A to E).
   SVG building, a seeded random generator, and an animation loop that only
   runs while its visual is on screen and starts paused for reduced motion. */
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";

  function S(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    if (attrs) for (const k in attrs) {
      if (attrs[k] !== undefined && attrs[k] !== null) e.setAttribute(k, attrs[k]);
    }
    if (parent) parent.appendChild(e);
    return e;
  }
  function T(parent, x, y, text, cls, attrs) {
    const t = S("text", Object.assign({ x, y, class: cls || "lt" }, attrs || {}), parent);
    t.textContent = text;
    return t;
  }
  function svg(container, w, h, label) {
    container.innerHTML = "";
    const s = S("svg", { viewBox: `0 0 ${w} ${h}`, role: "img", "aria-label": label || "" }, container);
    return s;
  }
  function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); }

  const C = {
    text: "#f6f4f1", dim: "#cfcbc4", muted: "#a7a29a", faint: "#7d7871",
    accent: "#e0a94f", accentDim: "#b3853d", rust: "#c07a5c", green: "#9bb17a",
    red: "#cf6b58", s1: "#ece7de", s4: "#a7a29a",
    line: "rgba(255,255,255,0.13)", lineSoft: "rgba(255,255,255,0.075)",
    lineStrong: "rgba(255,255,255,0.24)", bg: "#050505", cell: "#121110", panel: "#0a0a0a"
  };

  function rng(seed) {
    let a = (seed >>> 0) || 1;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) {
    const u = Math.max(1e-9, r()), v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  const reduced = !!(window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches);

  /* loop(el, step, opts): calls step(dt) every frame while playing and visible.
     step returns false to stop. opts.button toggles play and pause. */
  function loop(el, step, opts) {
    opts = opts || {};
    let playing = !reduced && opts.autoplay !== false;
    let visible = false, raf = 0, last = 0;
    const btn = opts.button;
    function label() {
      if (btn) btn.textContent = playing ? (opts.pauseLabel || "Pause") : (opts.playLabel || "Play");
    }
    function frame(ts) {
      raf = 0;
      if (!playing || !visible) return;
      const dt = last ? Math.min(0.1, (ts - last) / 1000) : 0;
      last = ts;
      if (step(dt) === false) { playing = false; label(); return; }
      raf = requestAnimationFrame(frame);
    }
    function kick() { if (playing && visible && !raf) { last = 0; raf = requestAnimationFrame(frame); } }
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(es => { visible = es[es.length - 1].isIntersecting; kick(); },
        { threshold: 0.02 }).observe(el);
    } else { visible = true; }
    if (btn) btn.addEventListener("click", () => { playing = !playing; label(); kick(); });
    label();
    return {
      play() { playing = true; label(); kick(); },
      pause() { playing = false; label(); },
      get playing() { return playing; }
    };
  }

  /* ticker(rate): turns frame times into whole ticks at `rate` per second */
  function ticker(rate) {
    let acc = 0;
    return {
      get rate() { return rate; },
      set rate(v) { rate = v; },
      ticks(dt) { acc += dt * rate; const n = Math.floor(acc); acc -= n; return n; }
    };
  }

  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const lerp = (a, b, t) => a + (b - a) * t;
  function fmt(n, d) { return Number(n).toLocaleString("en-US", { minimumFractionDigits: d || 0, maximumFractionDigits: d || 0 }); }

  /* colour for a signed value in [-1, 1]: amber positive, rust negative */
  function signedFill(v, base) {
    const a = clamp(Math.abs(v), 0, 1);
    if (a < 0.02) return base || C.cell;
    return v > 0 ? `rgba(224,169,79,${0.15 + 0.85 * a})` : `rgba(192,122,92,${0.15 + 0.85 * a})`;
  }

  /* seg(el, onChange): wires a .seg of buttons with data-v, returns setter */
  function seg(el, onChange) {
    const btns = [...el.querySelectorAll("button")];
    function set(v) {
      btns.forEach(b => b.classList.toggle("on", b.dataset.v === String(v)));
      onChange(v);
    }
    btns.forEach(b => b.addEventListener("click", () => set(b.dataset.v)));
    const first = btns.find(b => b.classList.contains("on")) || btns[0];
    if (first) set(first.dataset.v);
    // the shared "Reset controls" button resets inputs; put segments back too
    const demo = el.closest(".demo");
    if (demo && first) demo.addEventListener("demo:reset", () => set(first.dataset.v));
    return set;
  }

  /* bind a range input to an <output> and a callback; map turns the raw value */
  function range(id, onInput, map) {
    const el = document.getElementById(id);
    const out = document.getElementById(id + "-out");
    if (!el) return null;
    function upd() {
      const v = map ? map(+el.value) : +el.value;
      if (out) out.textContent = typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(2)) : v;
      onInput(v);
    }
    el.addEventListener("input", upd);
    upd();
    return el;
  }

  function onReady(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else fn();
  }

  window.Learn = { S, T, svg, clear, C, rng, gauss, loop, ticker, reduced, clamp, lerp, fmt, signedFill, seg, range, onReady };
})();
