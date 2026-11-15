"""Repo-level pytest config: shared fixture builders live in tests/support."""

import sys
from pathlib import Path

_SUPPORT = str(Path(__file__).resolve().parent / "tests" / "support")
if _SUPPORT not in sys.path:
    sys.path.insert(0, _SUPPORT)
