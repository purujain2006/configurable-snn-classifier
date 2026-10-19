/* guide.js: filter the question map and open every answer. */
(function () {
  "use strict";
  const box = document.getElementById("mm-filter");
  const all = document.getElementById("mm-open");
  const empty = document.getElementById("mm-empty");
  const qs = [...document.querySelectorAll(".mm-q")];
  if (!box) return;
  function filter() {
    const words = box.value.toLowerCase().split(/\s+/).filter(Boolean);
    let shown = 0;
    qs.forEach(q => {
      const text = q.textContent.toLowerCase();
      const hit = words.every(w => text.includes(w));
      q.hidden = !hit;
      if (hit) shown++;
      if (words.length && hit) q.open = true;
    });
    document.querySelectorAll(".mm-branch").forEach(b => { b.hidden = !b.querySelector(".mm-q:not([hidden])"); });
    empty.hidden = shown > 0;
  }
  box.addEventListener("input", filter);
  all.addEventListener("click", () => {
    const open = qs.some(q => !q.open);
    qs.forEach(q => { q.open = open; });
    all.textContent = open ? "Close all" : "Open all";
  });
})();
