"""
field_dictionary.py — write the AI field dictionary's docs from ``plaiq_engine.field_dictionary``.

    python tools/field_dictionary.py

Writes ``docs/ai_fields.json`` (machine-readable, checked by ``tests/cross_engine/test_field_dictionary.py``) and
``docs/ai_fields.md`` (for people). Edit the module, never the generated files.
"""

from __future__ import annotations

import json
from pathlib import Path

from plaiq_engine import field_dictionary as FD

DOCS = Path(__file__).resolve().parent.parent / "docs"

STATUS_WORDS = {
    "analyst_validated": "validated by analysts", "reference_agrees": "agrees with reference data",
    "reference_partial": "agrees in part", "reference_disagrees": "disagrees with reference data: not quotable",
    "outcome_measured": "measured against outcomes", "no_reference": "not yet measured", "not_applicable": "",
}


def markdown() -> str:
    out = [f"# AI field dictionary (version {FD.DICTIONARY_VERSION})", "",
           "> Generated from `packages/plaiq_engine/src/plaiq_engine/field_dictionary.py` by `tools/field_dictionary.py`; do not "
           "edit by hand. Figures are measured on a development set, not held out. Only observed values are player-facing "
           "until an analyst validates an estimated one (Gate 2b).", ""]
    for table in FD.TABLES:
        out += [f"## {table}", "", "| Column | Definition | Unit | Kind | Validation | Data check | Quotable | Player-facing |",
                "|---|---|---|---|---|---|---|---|"]
        for f in FD.fields(table):
            v = STATUS_WORDS[f.validation] + (f" ({f.evidence})" if f.evidence else "")
            out.append(f"| `{f.name}` | {f.definition} | {f.unit or ''} | {f.kind} | {v} | {f.data_check or ''} | "
                       f"{'yes' if f.quotable else 'no'} | {'yes' if f.player_facing else 'no'} |")
        out.append("")
    return "\n".join(out)


def main() -> int:
    DOCS.mkdir(exist_ok=True)
    (DOCS / "ai_fields.json").write_text(json.dumps(FD.as_json(), indent=1) + "\n")
    (DOCS / "ai_fields.md").write_text(markdown() + "\n")
    print(f"wrote docs/ai_fields.json and docs/ai_fields.md ({len(FD.fields())} fields)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
