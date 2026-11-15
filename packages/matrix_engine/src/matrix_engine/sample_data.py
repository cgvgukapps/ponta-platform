"""
Locate the open sample data that development tools and tests read.

The sample data (Metrica sample games, SkillCorner open data) is never copied into
this repository: redistribution terms are still open. It is read from outside the
repo. Resolution order:

1. the ``PONTA_SAMPLE_DATA`` environment variable, if set;
2. ``../PONTA-PROTOTYPE/sample-data`` relative to the repository root.

The directory returned holds ``data/`` (so Metrica games live under
``<root>/data/Sample_Game_N``).
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "PONTA_SAMPLE_DATA"


def repo_root() -> Path:
    """The ponta-platform repository root: the nearest parent holding ``pytest.ini`` and ``packages/``."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pytest.ini").is_file() and (parent / "packages").is_dir():
            return parent
    raise RuntimeError("Could not locate the ponta-platform repository root from " + str(here))


def sample_data_root() -> str:
    override = os.environ.get(ENV_VAR)
    if override:
        return os.path.abspath(override)
    return str((repo_root().parent / "PONTA-PROTOTYPE" / "sample-data").resolve())


def sample_data_dir() -> str:
    """``<sample data root>/data``."""
    return os.path.join(sample_data_root(), "data")


def processed_dir() -> str:
    """``<sample data root>/data/processed``: where the prototype wrote rosters and processed CSVs."""
    return os.path.join(sample_data_dir(), "processed")
