"""Emit .file-view blocks straight from the real source files.

Hand-copying source into HTML drifts the moment the file changes, and escaping
it by hand invites mistakes. This reads the actual lines, escapes them, and
writes markup the fileview.js component renders with a gutter and highlights.

Usage:  python tools/make_fileviews.py > /tmp/fileviews.html
"""
import html
import os
import sys

# The package lives beside this repo rather than inside it. Point PKG_ROOT
# elsewhere if the two are ever merged; nothing else needs to change.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_ROOT = os.environ.get(
    "SNNSEARCH_ROOT",
    os.path.join(os.path.dirname(ROOT), "SNNConfig"))

# (id, path, first line, last line, highlighted lines relative to the excerpt, note)
EXCERPTS = [
    ("claims", "snnsearch/spaces.py", 77, 83, "2-5",
     "The narrowing and its evidence sit on the same screen, so the reason a range "
     "exists is readable from the file."),
    ("epoch-report", "snnsearch/search.py", 124, 128, "2-3",
     "Reported every epoch. val_accuracy is a placeholder here, holding the same "
     "float number as float_val_accuracy."),
    ("deploy-report", "snnsearch/search.py", 137, 158, "6,10-11",
     "Reported once, after training. This is where val_accuracy stops being a "
     "placeholder and becomes the converted accuracy, and an undeployable "
     "configuration is scored zero."),
    ("synops", "snnsearch/synops.py", 150, 175, "13-22",
     "Four ways to combine accuracy with energy. The mode is a decision the user "
     "makes; nothing here can guess the trade-off for them."),
    ("encoders", "snnsearch/encoders.py", 118, 145, "6-13",
     "Poisson coding: a spike per timestep with probability equal to intensity. "
     "Every coding shares the resize step and differs only in what it emits."),
]


def emit(eid, path, first, last, hl, note):
    full = os.path.join(PKG_ROOT, path)
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
