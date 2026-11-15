"""Guard against kinematic-default drift between pass_option_value and pitch_control.

`analytics/pitch_control.py`'s DEFAULT_PROFILE (v_max=9.0, a_max=5.5,
reaction_time=0.7) is the canonical player physical profile, matching the
appendix constants table in ANALYTICS_SPECIFICATION.md (§Reference
Constants: DEFAULT_V_MAX=9.0, DEFAULT_REACTION=0.7). `analytics/
pass_option_value.py` has its own `.get(..., default)` fallbacks scattered
across several functions that build/consume the same kind of profile dict —
these must agree, or two code paths silently model players with different
top speeds depending on which function happened to supply the default.

This test greps the module source for every `v_max`/`reaction_time` fallback
literal so a future edit that reintroduces a mismatched default (as
`_build_defender_data`/`targeted_pcz` once did with 8.0/0.2) fails CI instead
of silently drifting.
"""
import ast
import inspect

from plaiq_engine.analytics import pass_option_value, pitch_control

CANONICAL_V_MAX = pitch_control.DEFAULT_PROFILE["v_max"]
CANONICAL_REACTION_TIME = pitch_control.DEFAULT_PROFILE["reaction_time"]


def _fallback_literals(module, key: str) -> list:
    """Return every literal default used in a `dict.get(key, default)` call
    (or `dict[key]`-style `.get(key, ...)`) found in `module`'s source."""
    tree = ast.parse(inspect.getsource(module))
    literals = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == key
            and isinstance(node.args[1], ast.Constant)
        ):
            literals.append(node.args[1].value)
    return literals


def test_pass_option_value_v_max_fallbacks_match_canonical_profile():
    literals = _fallback_literals(pass_option_value, "v_max")
    assert literals, "expected at least one v_max fallback in pass_option_value.py"
    assert all(v == CANONICAL_V_MAX for v in literals), (
        f"pass_option_value.py has v_max fallback(s) {literals} that disagree "
        f"with pitch_control.DEFAULT_PROFILE['v_max']={CANONICAL_V_MAX}"
    )


def test_pass_option_value_reaction_time_fallbacks_match_canonical_profile():
    literals = _fallback_literals(pass_option_value, "reaction_time")
    assert literals, "expected at least one reaction_time fallback in pass_option_value.py"
    assert all(v == CANONICAL_REACTION_TIME for v in literals), (
        f"pass_option_value.py has reaction_time fallback(s) {literals} that "
        f"disagree with pitch_control.DEFAULT_PROFILE['reaction_time']="
        f"{CANONICAL_REACTION_TIME}"
    )


def test_canonical_profile_matches_documented_spec_appendix():
    # ANALYTICS_SPECIFICATION.md's Reference Constants table documents these
    # as the intended canonical values; this pins pitch_control.py to them.
    assert CANONICAL_V_MAX == 9.0
    assert CANONICAL_REACTION_TIME == 0.7
