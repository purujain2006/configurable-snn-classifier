/* gloss.js: hover definitions. The first time a glossary term appears in a
   page's prose it gets a dotted underline; pointing at it, focusing it or
   tapping it shows the definition from glossary-data.js and a link to where
   the site explains it. A term explained on an earlier page is underlined
   again once three sections have passed since its last underline, since a
   reader meeting it pages later has often forgotten it. */
(function () {
  "use strict";
  const G = window.GLOSSARY;
  if (!G || document.body.dataset.gloss === "off") return;

  const SKIP = "a, code, pre, kbd, h1, h2, h3, h4, button, label, select, option, output, svg, script, style, " +
    "summary, .nav, .toc, .foot-nav, footer, .site-foot, .aka-head, .aka-sub, .seg, .stat, .kicker, .c-tags, .c-num, " +
    ".gl, .demo-tag, .demo-title, .verdict-tag, .check-list, .mm-kind, .readout, .formula, .tbl th, .sc-range, .eb-row, .legend, .refresh";
  const byAlias = new Map();
  const sens = [], insens = [];
  G.forEach(g => g.a.forEach(a => {
    const caseSensitive = (a.match(/[A-Z]/g) || []).length >= 2;
    byAlias.set(caseSensitive ? a : a.toLowerCase(), g);
    (caseSensitive ? sens : insens).push(a);
  }));
  const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const alt = list => list.sort((x, y) => y.length - x.length).map(esc).join("|");
  const B0 = "(?<![A-Za-z0-9_\\-])", B1 = "(?![A-Za-z0-9_])";
  const reS = new RegExp(B0 + "(" + alt(sens) + ")" + B1, "g");
  const reI = new RegExp(B0 + "(" + alt(insens) + ")" + B1, "gi");

  function lookup(word) { return byAlias.get(word) || byAlias.get(word.toLowerCase()); }

  const ORDER = ["index.html", "why.html", "basics.html", "data.html", "pipeline.html", "settings.html",
    "bits.html", "neurons.html", "hardware.html", "results.html"];
  const here = location.pathname.split("/").pop() || "index.html";
  const hereAt = ORDER.indexOf(here);
  const fromEarlierPage = g => { const i = ORDER.indexOf(g.see.split("#")[0]); return hereAt > 0 && i >= 0 && i < hereAt; };

  function wrap(root) {
    const done = new Set();
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(n) {
        if (!n.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
        const p = n.parentElement;
        if (!p || p.closest(SKIP)) return NodeFilter.FILTER_REJECT;
        // inside a demo, only its description and notes are prose; the rest is controls and output
        if (p.closest(".demo") && !p.closest(".demo-sub, .demo-note")) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      }
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    let section = null, k = 0;
    const lastAt = new Map();          // term id -> section number of its last underline
    nodes.forEach(node => {
      const sec = node.parentElement.closest("section");
      if (sec !== section) {
        section = sec; k++;
        G.forEach(g => { if (fromEarlierPage(g) && k - (lastAt.has(g.id) ? lastAt.get(g.id) : -9) >= 3) done.delete(g.id); });
      }
      const text = node.nodeValue;
      const hits = [];
      for (const re of [reS, reI]) {
        re.lastIndex = 0;
        let m;
        while ((m = re.exec(text))) {
          const g = lookup(m[1]);
          if (g && !done.has(g.id)) hits.push({ i: m.index, len: m[1].length, g });
        }
      }
      if (!hits.length) return;
      hits.sort((x, y) => x.i - y.i || y.len - x.len);
      const keep = [];
      let end = -1;
      hits.forEach(h => {
        if (h.i >= end && !done.has(h.g.id)) { keep.push(h); done.add(h.g.id); lastAt.set(h.g.id, k); end = h.i + h.len; }
      });
      if (!keep.length) return;
      const frag = document.createDocumentFragment();
      let pos = 0;
      keep.forEach(h => {
        if (h.i > pos) frag.appendChild(document.createTextNode(text.slice(pos, h.i)));
        const span = document.createElement("span");
        span.className = "gl";
        span.tabIndex = 0;
        span.setAttribute("role", "button");
        span.setAttribute("aria-label", `${text.substr(h.i, h.len)}: show definition`);
        span.dataset.gl = h.g.id;
        span.textContent = text.substr(h.i, h.len);
        frag.appendChild(span);
        pos = h.i + h.len;
      });
      if (pos < text.length) frag.appendChild(document.createTextNode(text.slice(pos)));
      node.parentNode.replaceChild(frag, node);
    });
  }

  /* one popup, moved to whichever term is active */
  let pop = null, current = null, hideT = 0;
  const byId = new Map(G.map(g => [g.id, g]));
  function show(span) {
    const g = byId.get(span.dataset.gl);
    if (!g) return;
    clearTimeout(hideT);
    if (!pop) {
      pop = document.createElement("div");
      pop.className = "gl-pop";
      pop.setAttribute("role", "tooltip");
      pop.addEventListener("mouseenter", () => clearTimeout(hideT));
      pop.addEventListener("mouseleave", () => hide(250));
      document.body.appendChild(pop);
    }
    const here = location.pathname.split("/").pop() || "index.html";
    const seeHere = g.see.split("#")[0] === here;
    pop.innerHTML = `<div class="gl-t">${g.t}</div><div class="gl-d">${g.d}</div>` +
      (g.w ? `<div class="gl-w"><span class="gl-wk">Why it exists</span>${g.w}</div>` : "") +
      `<div class="gl-links">${seeHere ? "" : `<a href="${g.see}">Where it is explained</a>`}<a href="glossary.html#${g.id}">Glossary</a></div>`;
    current = span;
    pop.classList.add("on");
    place();
  }
  function place() {
    if (!pop || !current) return;
    const r = current.getBoundingClientRect(), vw = document.documentElement.clientWidth;
    if (vw < 560) { pop.style.left = "8px"; pop.style.right = "8px"; pop.style.top = "auto"; pop.style.bottom = "8px"; pop.style.position = "fixed"; return; }
    pop.style.position = "absolute"; pop.style.right = "auto"; pop.style.bottom = "auto";
    const w = Math.min(340, vw - 24);
    pop.style.width = w + "px";
    let x = r.left + window.scrollX + r.width / 2 - w / 2;
    x = Math.max(window.scrollX + 12, Math.min(x, window.scrollX + vw - w - 12));
    pop.style.left = x + "px";
    const below = r.bottom + window.scrollY + 8;
    pop.style.top = below + "px";
  }
  function hide(delay) {
    clearTimeout(hideT);
    hideT = setTimeout(() => { if (pop) pop.classList.remove("on"); current = null; }, delay || 0);
  }

  function boot() {
    const main = document.querySelector("main") || document.body;
    try { wrap(main); } catch (e) { console.error("gloss", e); }
    const hdr = document.querySelector(".hero .lede");
    if (hdr) try { wrap(hdr); } catch (e) { /* lede is optional */ }
    document.addEventListener("mouseover", e => { const s = e.target.closest && e.target.closest(".gl"); if (s) show(s); });
    document.addEventListener("mouseout", e => { const s = e.target.closest && e.target.closest(".gl"); if (s) hide(250); });
    document.addEventListener("focusin", e => { const s = e.target.closest && e.target.closest(".gl"); if (s) show(s); });
    document.addEventListener("focusout", e => { if (e.target.closest && e.target.closest(".gl")) hide(150); });
    document.addEventListener("click", e => {
      const s = e.target.closest && e.target.closest(".gl");
      if (s) { if (current === s && pop && pop.classList.contains("on")) hide(); else show(s); }
      else if (!(e.target.closest && e.target.closest(".gl-pop"))) hide();
    });
    document.addEventListener("keydown", e => {
      if (e.key === "Escape") hide();
      const s = e.target.closest && e.target.closest(".gl");
      if (s && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); show(s); }
    });
    window.addEventListener("resize", place);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
