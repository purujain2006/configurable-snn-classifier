"""Catch the errors that a successful import does not.

Python binds names when a line runs, not when a module loads. So a module can
import cleanly while every function in it references a name that was never
imported, and the failure only appears when that function is finally called.
For a training path that is twenty minutes into a run, on a machine holding a
GPU. Splitting Practice2.py into a package produced exactly that: 28 undefined
names across 7 modules, none of which the import check could see.

Two checks, both static, both run without torch:

  undefined names     -- pyflakes, if it is installed
  call signatures     -- every call to a function defined in this package,
                         against that function's parameter list

    python tools/check_static.py
"""
import ast
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "snnsearch")

# pyflakes reports these as findings, but they are choices rather than errors:
# a re-export, a name kept for a caller, a loop variable used for its count.
IGNORE = ("imported but unused", "unable to detect undefined names",
          "redefinition of unused", "is assigned to but never used")


def py_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for f in sorted(filenames):
            if f.endswith(".py"):
                yield os.path.join(dirpath, f)


def undefined_names():
    """Run pyflakes over the package. Absent pyflakes is reported, not fatal."""
    # tools/ is included because it is code the user runs. Leaving it out let a
    # bad edit ship: two functions inserted mid-function silently absorbed the
    # rest of main(), and check_env.py died on a name that had been local to it.
    targets = [PKG, os.path.join(ROOT, "main.py"), os.path.join(ROOT, "tools")]
    try:
        out = subprocess.run([sys.executable, "-m", "pyflakes"] + targets,
                             capture_output=True, text=True, timeout=120)
    except Exception as exc:
        print(f"  pyflakes unavailable ({type(exc).__name__}); skipping.")
        print("  pip install pyflakes")
        return None
    if "No module named" in out.stderr:
        print("  pyflakes not installed; skipping.  pip install pyflakes")
        return None
    hits = [ln for ln in out.stdout.splitlines()
            if ln.strip() and not any(s in ln for s in IGNORE)]
    return hits


def collect_defs():
    """name -> signature, for module-level functions defined in the package.

    Only names defined exactly once are checked. A name defined twice cannot be
    resolved statically at the call site, so guessing would produce noise.
    """
    defs = {}
    for path in py_files(PKG):
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = node.args
                defs.setdefault(node.name, []).append({
                    "path": os.path.relpath(path, ROOT), "line": node.lineno,
                    "params": [x.arg for x in a.posonlyargs + a.args],
                    "ndefault": len(a.defaults),
                    "star": a.vararg is not None or a.kwarg is not None,
                })
    return {k: v[0] for k, v in defs.items() if len(v) == 1}


def bad_calls(defs):
    """Calls that cannot succeed: a required parameter with nothing to fill it."""
    problems = []
    for path in py_files(PKG):
        rel = os.path.relpath(path, ROOT)
        tree = ast.parse(open(path, encoding="utf-8").read())

        # A method call like bundle.describe() shares a name with a module
        # function. Skipping attribute calls avoids blaming the wrong callee.
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            d = defs.get(node.func.id)
            if d is None or d["star"]:
                continue
            if any(k.arg is None for k in node.keywords):   # **kwargs at the call
                continue
            given = set(d["params"][:len(node.args)]) | {k.arg for k in node.keywords}
            required = d["params"][:len(d["params"]) - d["ndefault"]]
            missing = [p for p in required if p not in given]
            if missing or len(node.args) > len(d["params"]):
                problems.append(
                    f"{rel}:{node.lineno}  {node.func.id}(...)  "
                    f"missing {missing}" if missing else
                    f"{rel}:{node.lineno}  {node.func.id}(...)  too many positional args"
                    f"  [defined {d['path']}:{d['line']}]")
    return problems


def broken_intra_imports():
    """`from .mod import name` where mod does not define name.

    pyflakes will not catch this: it checks names inside a file, not whether
    another module in the package actually provides what is being imported.
    And when the import sits inside a function -- which it must, wherever two
    modules would otherwise import in a circle -- nothing fails until that
    function is called. A reporting helper imported this way went missing from
    its module and the ImportError arrived after a 28-minute training run, at
    the one moment the results were about to be written.
    """
    provided, problems = {}, []
    for path in py_files(PKG):
        mod = os.path.relpath(path, ROOT)[:-3].replace(os.sep, ".")
        tree = ast.parse(open(path, encoding="utf-8").read())
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    names.add(a.asname or a.name.split(".")[0])
        provided[mod] = names

    for path in py_files(PKG):
        here = os.path.relpath(path, ROOT)[:-3].replace(os.sep, ".")
        pkg_parts = here.split(".")[:-1]
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            base = pkg_parts[: len(pkg_parts) - node.level + 1]
            target = ".".join(base + ([node.module] if node.module else []))
            if target not in provided:
                continue                      # a package __init__, not checkable here
            for alias in node.names:
                if alias.name == "*":
                    continue
                if alias.name not in provided[target]:
                    problems.append(
                        f"{os.path.relpath(path, ROOT)}:{node.lineno}  "
                        f"from {'.' * node.level}{node.module or ''} import "
                        f"{alias.name}  ->  {target} does not define it")
    return problems


def main():
    print("=" * 74)
    print("Static checks: what a successful import cannot tell you")
    print("=" * 74)

    print("\n-- undefined names --")
    hits = undefined_names()
    if hits is None:
        flake_ok = True                      # unavailable, not failed
    else:
        flake_ok = not hits
        print("\n".join(f"  {h}" for h in hits) if hits else "  none")

    print("\n-- call signatures --")
    problems = bad_calls(collect_defs())
    print("\n".join(f"  {p}" for p in problems) if problems else "  none")

    print("\n-- imports between package modules --")
    imports = broken_intra_imports()
    print("\n".join(f"  {p}" for p in imports) if imports else "  none")

    print("\n" + "=" * 74)
    ok = flake_ok and not problems and not imports
    print("STATIC CHECKS PASSED" if ok else "STATIC CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
