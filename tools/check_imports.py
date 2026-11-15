"""
Import-graph check for packages/.

Enforces, on every import in every module under packages/*/src (including lazy imports
inside functions, which is where hidden couplings live):

  1. No Streamlit imports anywhere in packages/.
  2. No imports of the prototype's top-level scripts (app, replay, preprocess). They are
     not ported; code that needs something from them must own a local equivalent.
  3. matrix_engine never imports plaiq_engine (one-way dependency: plaiq -> matrix).
  4. The package-level import graph has no cycles.

Exit code 0 when clean, 1 on violations. Run from the repo root:

    python tools/check_imports.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

FORBIDDEN_ROOTS = {
    "streamlit": "UI framework; engines must stay UI-agnostic",
    "app": "prototype Streamlit app; not ported",
    "replay": "prototype matplotlib demo; not ported",
    "preprocess": "prototype data-prep script; not ported",
}
# (importer, imported): forbidden package-level edges.
FORBIDDEN_EDGES = {("matrix_engine", "plaiq_engine"): "matrix_engine must never depend on plaiq_engine"}


def iter_modules(packages_dir: Path):
    for pkg_dir in sorted(p for p in packages_dir.iterdir() if p.is_dir()):
        src = pkg_dir / "src"
        if not src.is_dir():
            continue
        for f in sorted(src.rglob("*.py")):
            if "__pycache__" in f.parts:
                continue
            yield pkg_dir.name, f


def imports_in(path: Path):
    """Yield (lineno, absolute top-level module name or dotted name) for every import."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.lineno, node.module


def find_cycle(graph: dict[str, set[str]]):
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(n: str):
        state[n] = 1
        stack.append(n)
        for m in sorted(graph.get(n, ())):
            if state.get(m, 0) == 1:
                return stack[stack.index(m):] + [m]
            if state.get(m, 0) == 0:
                cyc = visit(m)
                if cyc:
                    return cyc
        stack.pop()
        state[n] = 2
        return None

    for node in sorted(graph):
        if state.get(node, 0) == 0:
            cyc = visit(node)
            if cyc:
                return cyc
    return None


def check(packages_dir: Path) -> list[str]:
    own = {p.name for p in packages_dir.iterdir() if p.is_dir() and (p / "src").is_dir()}
    problems: list[str] = []
    graph: dict[str, set[str]] = {name: set() for name in own}

    for pkg, f in iter_modules(packages_dir):
        rel = f.relative_to(packages_dir.parent)
        for lineno, name in imports_in(f):
            root = name.split(".")[0]
            if root in FORBIDDEN_ROOTS:
                problems.append(f"{rel}:{lineno}: imports '{name}' ({FORBIDDEN_ROOTS[root]})")
            if root in own and root != pkg:
                graph[pkg].add(root)
                reason = FORBIDDEN_EDGES.get((pkg, root))
                if reason:
                    problems.append(f"{rel}:{lineno}: imports '{name}' ({reason})")

    cycle = find_cycle(graph)
    if cycle:
        problems.append("package import cycle: " + " -> ".join(cycle))
    return problems


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    packages_dir = repo / "packages"
    problems = check(packages_dir)
    if problems:
        print("Import-graph check FAILED:")
        for p in problems:
            print(f"  {p}")
        return 1
    print("Import-graph check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
