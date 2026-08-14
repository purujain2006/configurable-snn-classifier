// Fails if a technical term is used in prose before the site defines it.
// Reading order is the page order in the nav, then position within the page.
const fs = require('fs'), path = require('path');
const DOCS = path.join(__dirname, '..', 'docs');
const ORDER = ['index.html', 'neurons.html', 'architecture.html', 'hardware.html',
               'training.html', 'search.html', 'reference.html'];

// term -> the exact markup that constitutes its definition
const TERMS = [
  ['synapse',            'index.html',        '<dt>synapse</dt>'],
  ['axon',               'index.html',        '<dt>axon</dt>'],
  ['fan-in',             'index.html',        '<dt>fan-in</dt>'],
  ['fan-out',            'index.html',        '<dt>fan-out</dt>'],
  ['connection limit',   'index.html',        'connection/routing limits'],
  ['routing',            'index.html',        "counts rows in the platform's routing table"],
  ['converter',          'index.html',        'class="term">converter<'],
  ['truncat',            'index.html',        'class="term">Truncation<'],
  ['membrane potential', 'index.html',        'class="term">membrane potential<'],
  ['w_alpha',            'index.html',        '<code>w_alpha</code> and fixes it at 1'],
  ['batch normalization','index.html',        'a layer that rescales each channel using statistics'],
  ['leaky integrate',    'index.html',        '<strong>leaky integrate-and-fire</strong>'],
  ['tdbn',               'architecture.html', '<code>tdbn</code> is a'],
  ['dilation',           'architecture.html', "<code>dilation</code> spaces a kernel's taps"],
  ['global average pool','architecture.html', 'Global average pooling reduces each channel to its mean'],
  ['rasteriz',           'training.html',     '<dt>rasterization</dt>'],
  ['surrogate',          'neurons.html',      'Substituting the derivative of a smooth function'],
  ['straight-through',   'hardware.html',     'Straight-through estimator: forward uses'],
  ['flush',              'index.html',        'class="term">flush<'],
];

// Words that must never appear in prose at all.
const BANNED = [
  ['synaptic',     'use "connection"'],
  ['receptive field', 'say what it means instead'],
  ['pre-neuron',   'say "input to the neuron"'],
  ['logit',        'say "class score"'],
];
// The one place the hardware term is named on purpose, matched on block text.
const BANNED_EXEMPT = [/^\s*Hardware documentation calls the table the routing table/];

// Blank out regions with equal-length spaces so every index stays valid
// against the ORIGINAL file. Replacing with shorter text would shift them.
function blank(html, re) {
  return html.replace(re, m => ' '.repeat(m.length));
}
function prose(html) {
  let h = html;
  for (const re of [/<script[\s\S]*?<\/script>/gi, /<style[\s\S]*?<\/style>/gi,
                    /<pre[\s\S]*?<\/pre>/gi, /<svg[\s\S]*?<\/svg>/gi,
                    /<div class="c-tags">[\s\S]*?<\/div>/gi, /<code[\s\S]*?<\/code>/gi]) {
    h = blank(h, re);
  }
  const out = [];
  const re = /<(p|li|dd|dt|h1|h2|h3|td|span class="demo-sub")\b[^>]*>([\s\S]*?)<\/(?:p|li|dd|dt|h1|h2|h3|td|span)>/gi;
  let m;
  while ((m = re.exec(h))) out.push({ text: m[2].replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' '), at: m.index, end: m.index + m[0].length });
  return out;
}
const lineOf = (s, i) => s.slice(0, i).split('\n').length;

let fail = 0;
const raw = {}, order = {};
ORDER.forEach((f, i) => {
  const p = path.join(DOCS, f);
  if (fs.existsSync(p)) { raw[f] = fs.readFileSync(p, 'utf8'); order[f] = i; }
});

// ---- 1. every term is defined before it is used ----
for (const [term, defFile, defMark] of TERMS) {
  const defIdx = (raw[defFile] || '').indexOf(defMark);
  if (defIdx < 0) { console.log(`  FAIL definition for "${term}" not found in ${defFile}`); fail++; continue; }
  const defPos = order[defFile] * 1e9 + defIdx;
  const rx = new RegExp(term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i');
  let firstPos = Infinity, firstEnd = Infinity, where = '';
  for (const f of ORDER) {
    if (!raw[f]) continue;
    for (const b of prose(raw[f])) {
      if (rx.test(b.text)) {
        const pos = order[f] * 1e9 + b.at;
        if (pos < firstPos) { firstPos = pos; firstEnd = order[f] * 1e9 + b.end; where = `${f}:${lineOf(raw[f], b.at)}`; }
        break;
      }
    }
    if (firstPos < Infinity) break;
  }
  if (firstPos === Infinity) continue;
  // A definition inside the very block that first uses the term counts as defined.
  if (defPos > firstEnd) {
    console.log(`  FAIL "${term}" first used at ${where}, but defined later in ${defFile}`);
    fail++;
  } else {
    console.log(`  ok   ${term.padEnd(20)} defined ${defFile.padEnd(18)} first used ${where}`);
  }
}

// ---- 2. banned words absent from prose ----
console.log('');
for (const f of ORDER) {
  if (!raw[f]) continue;
  for (const b of prose(raw[f])) {
    if (BANNED_EXEMPT.some(rx => rx.test(b.text))) continue;
    for (const [word, advice] of BANNED) {
      if (new RegExp(`\\b${word}`, 'i').test(b.text)) {
        console.log(`  FAIL ${f}:${lineOf(raw[f], b.at)} uses banned "${word}" — ${advice}`);
        console.log(`       "${b.text.trim().slice(0, 110)}"`);
        fail++;
      }
    }
  }
}

console.log(fail === 0 ? '\nJARGON CHECK PASSED' : `\n${fail} JARGON FAILURES`);
process.exit(fail ? 1 : 0);
