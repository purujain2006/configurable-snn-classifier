// Fails if a technical term is used in prose before the site defines it.
// Reading order is the page order in the nav, then position within the page.
const fs = require('fs'), path = require('path');
const DOCS = path.join(__dirname, '..', '..', 'docs');
const ORDER = ['index.html', 'neurons.html', 'architecture.html', 'hardware.html',
               'training.html', 'search.html', 'results.html', 'reference.html'];

// term -> the exact markup that constitutes its definition
const TERMS = [
  ['neuromorphic',       'index.html',        'neuromorphic computer, meaning one built'],
  ['FPGA',               'index.html',        'FPGAs, chips whose circuits can be reprogrammed'],
  [' spike',             'index.html',        'A spike is a single event'],
  ['membrane potential', 'index.html',        'class="term">membrane potential<'],
  ['leaky integrate',    'index.html',        '<strong>leaky integrate-and-fire</strong>'],
  ['synapse',            'index.html',        '<dt>synapse</dt>'],
  ['axon',               'index.html',        '<dt>axon</dt>'],
  ['fan-in',             'index.html',        '<dt>fan-in</dt>'],
  ['fan-out',            'index.html',        '<dt>fan-out</dt>'],
  ['INT16',              'index.html',        '<dt>INT16</dt>'],
  ['batch normalization','index.html',        'the layer that rescales each channel during training'],
  ['folding',            'index.html',        'a step called folding'],
  ['bias neuron',        'index.html',        '<em>bias neuron</em>'],
  ['validation',         'index.html',        'A validation set, held back from training'],
  ['convolutional',      'index.html',        'networks that slide small learned filters'],
  ['epoch',              'neurons.html',      '(passes through the training data)'],
  ['surrogate',          'neurons.html',      'class="term">surrogate gradient<'],
  ['straight-through',   'neurons.html',      '<em>straight-through estimator</em>'],
  ['global average pool','architecture.html', '<dt>flatten, GAP</dt>'],
  ['dropout',            'architecture.html', 'dropout (silencing random units during training'],
  ['overfitting',        'architecture.html', 'overfitting: learning details of the training clips'],
  ['SynOp',              'architecture.html', 'SynOps (synaptic operations)'],
  ['w_alpha',            'hardware.html',     'The scale <code>w_alpha</code> says'],
  ['flush',              'training.html',     'called flush steps'],
  ['hyperparameter',     'search.html',       'class="term">hyperparameter<'],
];

// Words that must never appear in prose at all.
const BANNED = [
  ['synaptic',     'use "connection"'],
  ['receptive field', 'say what it means instead'],
  ['pre-neuron',   'say "input to the neuron"'],
  ['logit',        'say "class score"'],
];

//031749
// The one place the hardware term is named on purpose, matched on block text.
const BANNED_EXEMPT = [
  /^\s*Hardware documentation calls the table the routing table/,
  // SynOps stands for synaptic operations. That is the term the literature
  // uses and the reason the metric is comparable across platforms, so the
  // house preference for "connection" does not apply to the name itself.
  /synaptic operation/i,
];

// Blank out regions with equal-length spaces so every index stays valid
// against the ORIGINAL file. Replacing with shorter text would shift them.
function blank(html, re) {
  return html.replace(re, m => ' '.repeat(m.length));
}
// `skipLede` drops the page's opening summary. A lede naming the topics its own
// page goes on to define is doing its job, not using a term prematurely.
function prose(html, skipLede) {
  let h = html;
  const strip = [/<script[\s\S]*?<\/script>/gi, /<style[\s\S]*?<\/style>/gi,
                 /<pre[\s\S]*?<\/pre>/gi, /<svg[\s\S]*?<\/svg>/gi,
                 /<div class="c-tags">[\s\S]*?<\/div>/gi, /<code[\s\S]*?<\/code>/gi];
  if (skipLede) strip.push(/<p class="lede">[\s\S]*?<\/p>/gi);
  for (const re of strip) h = blank(h, re);
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
    for (const b of prose(raw[f], true)) {
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
