"""Emit .file-view blocks for results.html straight from the real source files.

Hand-copying source into HTML drifts the moment the file changes, and escaping
it by hand invites mistakes. This reads the actual lines, escapes them, and
writes markup the fileview.js component renders with a gutter and highlights.

Usage:  python tools/make_fileviews.py > /tmp/fileviews.html
"""
import html
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (id, path, first line, last line, highlighted lines relative to the excerpt, note)
EXCERPTS = [
    ("claims", "Practice2.py", 2428, 2435, "2-5",
     "The narrowing and its evidence sit on the same screen, so the reason a range "
     "exists is readable from the file."),
    ("epoch-report", "Practice2.py", 2812, 2822, "6-7",
     "Reported every epoch during both phases. val_accuracy is a placeholder here, "
     "holding the same float number as float_val_accuracy."),
    ("deploy-report", "Practice2.py", 2827, 2841, "5,8-10",
     "Reported once, after training. This is where val_accuracy stops being a "
     "placeholder and becomes the converted accuracy, and an undeployable "
     "configuration is scored zero."),
    ("analysis-load", "tools/trial_analysis.py", 108, 131, "9-18",
     "Leaderboards give the configuration and score. trials.jsonl is the only place "
     "the deployment fields appear, so the two have to be read together."),
    ("analysis-perm", "tools/trial_analysis.py", 86, 101, "10-15",
     "Kruskal-Wallis with a permutation p-value. The statistic is computed on the "
     "real grouping, then on thousands of shuffles, and the p-value is the share of "
     "shuffles that did as well by chance."),
]


def emit(eid, path, first, last, hl, note):
    full = os.path.join(ROOT, path)
    with open(full, encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    body_lines = lines[first - 1:last]
    # fileview.js strips a leading blank line, which would slide the whole
    # gutter down by one. Drop them here and move `first` to compensate, so the
    # numbers in the margin always name the real lines.
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
        first += 1
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()
    body = "\n".join(body_lines)
    print(f'  <div class="file-view" id="fv-{eid}" data-file="{html.escape(path)}" '
          f'data-start="{first}" data-hl="{hl}"')
    print(f'       data-note="{html.escape(note, quote=True)}">')
    print(f'    <pre class="fv-src">{html.escape(body)}</pre>')
    print('  </div>')


if __name__ == "__main__":
    want = sys.argv[1:] or [e[0] for e in EXCERPTS]
    for e in EXCERPTS:
        if e[0] in want:
            emit(*e)
            print()
