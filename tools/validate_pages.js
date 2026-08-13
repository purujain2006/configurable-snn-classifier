// Structural validation: every element the demo scripts reach for must exist on
// the page that loads them, tags must balance, and cross-refs must resolve.
const fs = require('fs'), path = require('path');
const DOCS = path.join(__dirname, '..', 'docs');
let fail = 0;
const bad = m => { console.log('  FAIL ' + m); fail++; };

const pages = fs.readdirSync(DOCS).filter(f => f.endsWith('.html'));

// ---- 1. ids referenced by each page's scripts exist on that page ----
for (const p of pages) {
  const html = fs.readFileSync(path.join(DOCS, p), 'utf8');
  const ids = new Set([...html.matchAll(/\sid="([^"]+)"/g)].map(m => m[1]));
  const scripts = [...html.matchAll(/<script src="(assets\/[^"]+)"/g)].map(m => m[1]);
  let checked = 0;
  for (const s of scripts) {
    const sp = path.join(DOCS, s);
    if (!fs.existsSync(sp)) { bad(`${p}: missing script ${s}`); continue; }
    const js = fs.readFileSync(sp, 'utf8');
    const usesDollar = /\$\s*=\s*id\s*=>\s*document\.getElementById/.test(js);
    if (usesDollar) {
      for (const m of js.matchAll(/(?<![\w.])\$\(\s*["'`]([^"'`]+)["'`]\s*\)/g)) {
        checked++;
        if (!ids.has(m[1])) bad(`${p} -> ${s}: $("${m[1]}") has no matching id`);
      }
    }
    for (const m of js.matchAll(/getElementById\(\s*["'`]([^"'`]+)["'`]\s*\)/g)) {
      checked++;
      if (!ids.has(m[1])) bad(`${p} -> ${s}: getElementById("${m[1]}") has no matching id`);
    }
    for (const m of js.matchAll(/querySelector(?:All)?\(\s*["'`]#([A-Za-z][\w-]*)["'`]/g)) {
      checked++;
      if (!ids.has(m[1])) bad(`${p} -> ${s}: querySelector("#${m[1]}") has no matching id`);
    }
  }
  console.log(`${p}: ${ids.size} ids, ${scripts.length} scripts, ${checked} element lookups checked`);
}

// ---- 2. tag balance for structural elements ----
const PAIRED = ['section','div','p','table','thead','tbody','tr','td','th','ul','ol','li','dl','dt','dd','details','figure','main','header','footer','nav','aside','pre','h2','h3','h4'];
const VOID = new Set(['br','hr','img','input','meta','link','path','rect','circle','line','use','source','col']);
for (const p of pages) {
  const html = fs.readFileSync(path.join(DOCS, p), 'utf8')
    .replace(/<!--[\s\S]*?-->/g, '')
    .replace(/<pre[\s\S]*?<\/pre>/g, '<pre></pre>')
    .replace(/<svg[\s\S]*?<\/svg>/g, '');
  const stack = [];
  for (const m of html.matchAll(/<(\/?)([a-zA-Z][\w-]*)([^>]*?)(\/?)>/g)) {
    const [, close, tag, attrs, selfClose] = m;
    const t = tag.toLowerCase();
    if (!PAIRED.includes(t)) continue;
    if (selfClose || VOID.has(t)) continue;
    if (close) {
      const top = stack.pop();
      if (top !== t) bad(`${p}: </${t}> closes <${top || 'nothing'}>`);
    } else stack.push(t);
  }
  if (stack.length) bad(`${p}: unclosed ${stack.join(', ')}`);
}

// ---- 3. section cross-references resolve to a real heading ----
const headings = new Set();
for (const p of pages) {
  const html = fs.readFileSync(path.join(DOCS, p), 'utf8');
  for (const m of html.matchAll(/<span class="sec-num">([\d.]+)<\/span>/g)) headings.add(m[1]);
}
for (const p of pages) {
  const html = fs.readFileSync(path.join(DOCS, p), 'utf8');
  for (const m of html.matchAll(/[Ss]ections?\s+(\d+\.\d+)(?:\s+and\s+(\d+\.\d+))?/g)) {
    for (const ref of [m[1], m[2]]) {
      if (ref && !headings.has(ref)) bad(`${p}: reference to section ${ref}, which does not exist`);
    }
  }
}
console.log(`\n${headings.size} section headings found across the site`);

// ---- 4. internal links resolve ----
for (const p of pages) {
  const html = fs.readFileSync(path.join(DOCS, p), 'utf8');
  for (const m of html.matchAll(/href="([^"#:]+\.html)(#([^"]+))?"/g)) {
    if (!fs.existsSync(path.join(DOCS, m[1]))) bad(`${p}: link to missing page ${m[1]}`);
    else if (m[3]) {
      const target = fs.readFileSync(path.join(DOCS, m[1]), 'utf8');
      if (!target.includes(`id="${m[3]}"`)) bad(`${p}: link ${m[1]}#${m[3]} has no target`);
    }
  }
}

// ---- 5. no stray double spaces or empty paragraphs in prose ----
for (const p of pages) {
  const html = fs.readFileSync(path.join(DOCS, p), 'utf8');
  for (const m of html.matchAll(/<p([^>]*)>([\s\S]{0,400}?)<\/p>/g)) {
    const isRenderTarget = /\sid=/.test(m[1]);
    const t = m[2].replace(/<[^>]+>/g,'').replace(/&\w+;|&#\d+;/g,'x');
    if (/[a-z]\s{2,}[a-z]/i.test(t.replace(/\n\s+/g, ' '))) bad(`${p}: double space inside a paragraph: "${t.trim().slice(0,70)}"`);
    if (!t.trim() && !isRenderTarget) bad(`${p}: empty paragraph with no id to fill it`);
  }
}

console.log(fail === 0 ? '\nALL STRUCTURAL CHECKS PASSED' : `\n${fail} FAILURES`);
process.exit(fail ? 1 : 0);
