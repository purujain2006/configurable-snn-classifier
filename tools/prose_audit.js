// Scans docs/*.html prose for AI-slop writing patterns.
const fs = require('fs'), path = require('path');
const DOCS = path.join(__dirname, '..', 'docs');
const ORDER = ['index.html','neurons.html','architecture.html','hardware.html','training.html','search.html','results.html','reference.html'];

function textBlocks(html) {
  // strip code/pre/script/style/svg entirely
  let h = html
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<pre[\s\S]*?<\/pre>/gi, ' ')
    .replace(/<svg[\s\S]*?<\/svg>/gi, ' ')
    .replace(/<code[\s\S]*?<\/code>/gi, ' CODE ');
  const blocks = [];
  const re = /<(p|li|dd|h1|h2|h3|td|div class="lede")\b[^>]*>([\s\S]*?)<\/\1>/gi;
  let m;
  while ((m = re.exec(h))) {
    let t = m[2].replace(/<[^>]+>/g, ' ')
      .replace(/&#8722;/g,'-').replace(/&#215;/g,'x').replace(/&#183;/g,'.')
      .replace(/&#(\d+);/g, (_,d)=>String.fromCharCode(+d))
      .replace(/&[a-z]+;/gi,' ')
      .replace(/\s+/g,' ').trim();
    if (t.length > 25) blocks.push({tag:m[1], text:t, idx:m.index});
  }
  return blocks;
}
function lineOf(html, idx){ return html.slice(0, idx).split('\n').length; }

const RULES = [
  { id:'notXbutY', re:/\bnot\s+(?:just\s+|merely\s+|simply\s+|only\s+)?(?:a|an|the|about|that)?\b[^.,;]{2,45}?[,]?\s+(?:but|it(?:'s| is)|rather)\b/gi,
    label:'"not X but/it is Y" construction' },
  { id:'triad', re:/\b\w+ly\b[^.,;]{0,25},\s+\b\w+ly\b[^.,;]{0,25},\s+and\s+\b\w+ly\b/gi, label:'adverbial triad' },
  { id:'listTriad', re:/(?:^|[.;:]\s)(?:[A-Z]?[a-z]+\s+){0,3}\b\w+s\b,\s+\b\w+s\b,\s+and\s+\b\w+s\b\./gm, label:'noun triad ending a sentence' },
  { id:'emdash', re:/—|&mdash;| -- /g, label:'em dash' },
  { id:'itsNotAbout', re:/\bit(?:'s| is)\s+(?:not|worth noting|important to note|the case that)\b/gi, label:'filler / hedging opener' },
  { id:'metaphor', re:/\b(?:journey|landscape|tapestry|symphony|orchestrat\w+|dance|marriage of|at its heart|at its core|under the hood|the secret sauce|magic|elegant(?:ly)?|beautiful(?:ly)?|powerful(?:ly)?|seamless(?:ly)?|robust|cutting[- ]edge|game[- ]chang\w+|unlock\w*|leverag\w+|delve|realm|navigat\w+ the|paint\w* a picture|bird'?s[- ]eye)\b/gi,
    label:'stock metaphor / marketing adjective' },
  { id:'thisIsWhere', re:/\bthis is (?:where|why|what|how)\b|\benter\s+the\b|\bthat(?:'s| is) where\b/gi, label:'"this is where/why" transition' },
  { id:'inOtherWords', re:/\bin other words\b|\bput (?:simply|another way)\b|\bthink of (?:it|this) as\b|\bimagine\b/gi, label:'restatement filler' },
  { id:'fwdRef', re:/\b(?:as we(?:'ll| will)|we(?:'ll| will) (?:see|cover|get to|explore)|later (?:on|we)|coming up|in the next section|below,? we)\b/gi, label:'forward reference' },
  { id:'weRoyal', re:/\b(?:we|our|us)\b|\blet's\b/gi, label:'first person' },
  { id:'crucial', re:/\b(?:crucial(?:ly)?|essential(?:ly)?|vital|key to|critical(?:ly)?|fundamental(?:ly)?|significantly|notably|importantly)\b/gi, label:'intensifier' },
  { id:'ensure', re:/\b(?:ensur\w+|facilitat\w+|utiliz\w+|myriad|plethora|comprehensive|holistic|streamlin\w+|optimiz\w+ for|in order to)\b/gi, label:'inflated diction' },
];

let total = 0;
const perPage = {};
for (const f of ORDER) {
  const p = path.join(DOCS, f);
  if (!fs.existsSync(p)) continue;
  const html = fs.readFileSync(p, 'utf8');
  const blocks = textBlocks(html);
  const hits = [];
  for (const b of blocks) {
    for (const r of RULES) {
      r.re.lastIndex = 0;
      const found = b.text.match(r.re);
      if (found) hits.push({rule:r.id, label:r.label, line:lineOf(html,b.idx), sample:found.slice(0,3).join(' | '), ctx:b.text.slice(0,150)});
    }
  }
  // cadence: only compare sentences INSIDE one block, so adjacent paragraphs
  // starting with "The" are not mistaken for deliberate anaphora.
  const w = x => x.trim().split(/\s+/).length;
  const first = x => (x.trim().split(/\s+/)[0]||'').toLowerCase().replace(/[^a-z]/g,'');
  for (const b of blocks) {
    if (b.tag !== 'p' && b.tag !== 'dd' && b.tag !== 'li') continue;
    const ss = b.text.split(/(?<=[.?!])\s+/).filter(x=>x.trim());
    const line = lineOf(html, b.idx);
    for (let i=0;i+2<ss.length;i++){
      const a=ss[i], b2=ss[i+1], c=ss[i+2];
      if (first(a)&&first(a)===first(b2)&&first(b2)===first(c))
        hits.push({rule:'anaphora', label:'three sentences in one block with the same opening word', line, sample:first(a), ctx:a.slice(0,90)});
      if (w(a)>7 && Math.abs(w(a)-w(b2))<=2 && Math.abs(w(b2)-w(c))<=2)
        hits.push({rule:'isoLength', label:'three consecutive sentences of near-identical length', line, sample:`${w(a)}/${w(b2)}/${w(c)} words`, ctx:(a+' '+b2).slice(0,130)});
    }
    // repeated sentence *shape*: same opening 2-word bigram twice in a block
    for (let i=0;i+1<ss.length;i++){
      const g = x => x.trim().split(/\s+/).slice(0,2).join(' ').toLowerCase().replace(/[^a-z ]/g,'');
      if (g(ss[i]) && g(ss[i])===g(ss[i+1]) && w(ss[i])>5)
        hits.push({rule:'bigram', label:'two consecutive sentences opening with the same two words', line, sample:g(ss[i]), ctx:(ss[i]+' | '+ss[i+1]).slice(0,130)});
    }
  }
  perPage[f] = hits; total += hits.length;
  const byRule = {};
  for (const h of hits) (byRule[h.rule] ||= []).push(h);
  console.log(`\n=== ${f}  (${blocks.length} prose blocks, ${hits.length} flags) ===`);
  for (const k of Object.keys(byRule).sort()) {
    console.log(`  [${k}] ${byRule[k][0].label}  x${byRule[k].length}`);
    for (const h of byRule[k].slice(0,6)) console.log(`     L${h.line}: ${h.sample}   << ${h.ctx}`);
  }
}
console.log(`\nTOTAL FLAGS: ${total}`);
