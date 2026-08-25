/* fileview.js — scrollable source excerpts with line numbers and highlights.

   Markup:
     <div class="file-view" data-file="tools/x.py" data-start="108" data-hl="3,7-9"
          data-note="what to look at">
       <pre class="fv-src">...raw source, HTML-escaped...</pre>
     </div>

   data-start is the line number the excerpt begins at in the real file, so the
   gutter matches what you would see in an editor. data-hl is relative to the
   excerpt, one-based, and accepts single lines and ranges.

   Lines are rendered individually rather than handed to Prism, because a
   tokenizer that emits elements spanning newlines makes per-line numbering and
   highlighting unreliable. Colouring here is a small per-line pass, so it can
   never straddle a line boundary.
*/
(function () {
  "use strict";

  const KW = new Set(("def class return if elif else for while in not and or is None True False " +
    "import from as with try except raise lambda yield pass break continue global assert").split(" "));
  const BI = new Set(("self int float str bool list dict tuple set len range print round abs min " +
    "max sum sorted enumerate zip getattr setattr isinstance").split(" "));

  const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  // ONE pass over the raw text, escaping each token as it is emitted. Running
  // several .replace() calls in sequence would let a later pattern match inside
  // the markup an earlier one produced -- the word "class" in `<i class="fv-s">`
  // being the case that caught this out.
  const TOKEN = /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(\b\d+\.?\d*(?:[eE][-+]?\d+)?\b)|([A-Za-z_]\w*)/g;

  function colour(line) {
    // a comment runs to end of line, so split it off before anything else
    const hash = findComment(line);
    const code = hash >= 0 ? line.slice(0, hash) : line;
    const tail = hash >= 0 ? line.slice(hash) : "";

    let out = "", last = 0, m;
    TOKEN.lastIndex = 0;
    while ((m = TOKEN.exec(code))) {
      out += esc(code.slice(last, m.index));
      const t = m[0];
      if (m[1]) out += `<i class="fv-s">${esc(t)}</i>`;
      else if (m[2]) out += `<i class="fv-n">${esc(t)}</i>`;
      else if (KW.has(t)) out += `<i class="fv-k">${esc(t)}</i>`;
      else if (BI.has(t)) out += `<i class="fv-b">${esc(t)}</i>`;
      else out += esc(t);
      last = m.index + t.length;
    }
    out += esc(code.slice(last));
    if (tail) out += `<i class="fv-c">${esc(tail)}</i>`;
    return out;
  }

  // '#' inside a string literal is not a comment
  function findComment(line) {
    let q = null;
    for (let i = 0; i < line.length; i++) {
      const c = line[i];
      if (q) { if (c === q && line[i - 1] !== "\\") q = null; }
      else if (c === '"' || c === "'") q = c;
      else if (c === "#") return i;
    }
    return -1;
  }

  function parseRanges(spec, count) {
    const hl = new Set();
    (spec || "").split(",").forEach(part => {
      part = part.trim();
      if (!part) return;
      const m = part.match(/^(\d+)\s*-\s*(\d+)$/);
      if (m) { for (let i = +m[1]; i <= +m[2]; i++) hl.add(i); }
      else if (/^\d+$/.test(part)) hl.add(+part);
    });
    return hl;
  }

  function render(view) {
    const src = view.querySelector(".fv-src");
    if (!src) return;
    const raw = src.textContent.replace(/^\n/, "").replace(/\s+$/, "");
    const lines = raw.split("\n");
    const start = parseInt(view.dataset.start || "1", 10);
    const hl = parseRanges(view.dataset.hl, lines.length);
    const plain = view.dataset.plain === "true";

    const body = document.createElement("div");
    body.className = "fv-body";
    body.innerHTML = lines.map((ln, i) => {
      const n = start + i;
      const on = hl.has(i + 1) ? " fv-hl" : "";
      return `<div class="fv-line${on}"><span class="fv-num">${n}</span>` +
             `<span class="fv-code">${plain ? esc(ln) || "&nbsp;" : colour(ln) || "&nbsp;"}</span></div>`;
    }).join("");

    const head = document.createElement("div");
    head.className = "fv-head";
    const shown = hl.size ? `${hl.size} line${hl.size > 1 ? "s" : ""} highlighted` : `${lines.length} lines`;
    head.innerHTML = `<span class="fv-name">${esc(view.dataset.file || "source")}</span>` +
                     `<span class="fv-meta">lines ${start}&#8211;${start + lines.length - 1} &#183; ${shown}</span>`;

    src.remove();
    view.prepend(body);
    view.prepend(head);

    if (view.dataset.note) {
      const note = document.createElement("div");
      note.className = "fv-note";
      note.textContent = view.dataset.note;
      view.append(note);
    }
    // jump the scroll to the first highlighted line so it is visible on load
    const first = body.querySelector(".fv-hl");
    if (first) {
      const target = first.offsetTop - body.clientHeight / 3;
      if (target > 0) body.scrollTop = target;
    }
  }

  function init() {
    document.querySelectorAll(".file-view").forEach(render);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
