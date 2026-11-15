"""Tests for tools/check_imports.py: the checker must catch each rule it claims to enforce."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_imports", Path(__file__).resolve().parents[1] / "tools" / "check_imports.py"
)
check_imports = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_imports)


def _make(tmp_path: Path, files: dict[str, str]) -> Path:
    packages = tmp_path / "packages"
    for rel, body in files.items():
        f = packages / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body)
    return packages


def test_clean_one_way_dependency_passes(tmp_path):
    packages = _make(tmp_path, {
        "matrix_engine/src/matrix_engine/a.py": "import numpy\n",
        "plaiq_engine/src/plaiq_engine/b.py": "from matrix_engine import a\n",
    })
    assert check_imports.check(packages) == []


def test_matrix_importing_plaiq_is_flagged(tmp_path):
    packages = _make(tmp_path, {
        "matrix_engine/src/matrix_engine/a.py": "from plaiq_engine import b\n",
        "plaiq_engine/src/plaiq_engine/b.py": "import numpy\n",
    })
    problems = check_imports.check(packages)
    assert any("must never depend on plaiq_engine" in p for p in problems)


def test_lazy_import_inside_a_function_is_still_caught(tmp_path):
    # The prototype hid its roster -> replay coupling in a function body.
    packages = _make(tmp_path, {
        "matrix_engine/src/matrix_engine/a.py": "def f():\n    from plaiq_engine import b\n",
        "plaiq_engine/src/plaiq_engine/b.py": "x = 1\n",
    })
    assert check_imports.check(packages)


@pytest.mark.parametrize("stmt", ["import streamlit as st", "from replay import load_game", "import preprocess", "import app"])
def test_forbidden_roots_are_flagged(tmp_path, stmt):
    packages = _make(tmp_path, {"plaiq_engine/src/plaiq_engine/b.py": f"{stmt}\n"})
    assert check_imports.check(packages)


def test_cycle_is_reported(tmp_path):
    # Two packages importing each other: caught as a forbidden edge and as a cycle.
    packages = _make(tmp_path, {
        "matrix_engine/src/matrix_engine/a.py": "import plaiq_engine\n",
        "plaiq_engine/src/plaiq_engine/b.py": "import matrix_engine\n",
    })
    problems = check_imports.check(packages)
    assert any("cycle" in p for p in problems)


def test_real_repo_is_clean():
    repo = Path(__file__).resolve().parents[1]
    assert check_imports.check(repo / "packages") == []
