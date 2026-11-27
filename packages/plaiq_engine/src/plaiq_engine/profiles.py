"""
profiles.py — team match profiles and baselines (Stage E, ``ponta-blueprint/context-profiles-task.md``).

**A profile** is one team's match as measures, in long form: one row per team, match and measure, with the value, its
denominator and sample, its label (observed or estimated), its basis and source, the Data check state of its inputs,
and the versions it was built from. Measures follow the founder's decision order (23 November 2026): published
definitions first (FIFA EFI; Trainor's PPDA), common public definitions next (possession share, field tilt, named as
such), then Ponta's own measures from the context packages, labelled as Ponta's.

**A baseline** compares a team's match with the team's other matches and with the set (the same competition and season
where known), refuses profiles built from different versions, leaves Blocked matches out, and pulls a team's mean
towards the set's in proportion to its sample (empirical Bayes shrinkage; the weight of the set is the within-team
variance over the between-team variance). **A comparison** gives a verdict only when the team has at least the
measure's minimum number of matches (``MIN_MATCHES``, measured, not assumed) and the match falls outside the team's
usual range, a prediction interval set for a stated false-alarm rate (``pf_verdict_false_alarm``, 10% two-sided); the
comparison returns that range beside the word. Otherwise it says why not.

Labels and aggregates only: no coordinates, cells, names or provider ids.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from plaiq_engine.context.structure import OTHER, TEAMS
from plaiq_engine.context.thresholds import THRESHOLDS

PROFILE_VERSION = "0.3.0"  # 0.3.0 (27 November 2026, review backlog B, AI-04): match, team, competition and season are opaque keys
#                            from the tenant registry; values unchanged.
#                            0.2.0 (25 November 2026, review backlog A): out_share_* from the team's own block (AN-02); PPDA's
#                            mapping (AN-01); Blocked rows null (AI-05); minimums from the thresholds config with source (GC-03)
PHASES = ("build_up", "create", "finish", "direct", "quick_break", "transition", "set_play", "chaotic")
BLOCKS = ("high_block", "mid_block", "low_block")
P4_BLOCK_LABEL = {"high_block": "high_block", "mid_block": "medium_block", "low_block": "low_block"}
LANES = ("left_wing", "left_half_space", "centre", "right_half_space", "right_wing")
ENGAGEMENTS = ("recovery", "counter_press", "press", "pressure")
#: The matches a team needs before a comparison gives a verdict, per measure: from the intraclass correlation lifted by
#: Spearman-Brown to pf_reliability_target (tools/context_profiles.py, CONTEXT_PROFILES_REPORT.md). Kept in the
#: thresholds config with source and licence (thresholds 0.27.0, review GC-03, AI-14): ``MIN_MATCHES`` holds the values
#: measured on licensed data (SkillCorner's open data), the default; ``MIN_MATCHES_UNLICENSED`` the values measured on
#: PFF's World Cup 2022 set, used only when ``pf_allow_unlicensed_derived`` is on (internal development; never for
#: anything a customer, pilot or demo sees before PFF's written approval). Every other measure gives "minimum not
#: measured yet" until a club's own matches show it is stable.
MIN_MATCHES: dict[str, int] = {
    k[len("pf_min_matches_"):]: int(t.value) for k, t in THRESHOLDS.items() if k.startswith("pf_min_matches_") and not k.endswith("_pff")
}
MIN_MATCHES_UNLICENSED: dict[str, int] = {
    k[len("pf_min_matches_"):-len("_pff")]: int(t.value) for k, t in THRESHOLDS.items() if k.startswith("pf_min_matches_") and k.endswith("_pff")
}
MIN_SOURCE_LICENSED = THRESHOLDS["pf_min_matches_out_share_high_block"].licence
MIN_SOURCE_UNLICENSED = THRESHOLDS["pf_min_matches_out_share_high_block_pff"].licence


def min_matches(measure: str, cfg: dict | None = None) -> tuple[int | None, str | None]:
    """A measure's minimum matches and the licence of the data it was measured on (the larger when unlicensed is allowed)."""
    lic = MIN_MATCHES.get(measure)
    if cfg and cfg.get("pf_allow_unlicensed_derived", 0) >= 1 and measure in MIN_MATCHES_UNLICENSED:
        un = MIN_MATCHES_UNLICENSED[measure]
        if lic is None or un > lic:
            return un, MIN_SOURCE_UNLICENSED
    return lic, (MIN_SOURCE_LICENSED if lic is not None else None)


_STATE_RANK = {"blocked": 0, "degraded": 1, "full": 2}
#: Measures that read how chains start and end: Degraded when the feed gives no restarts (P1 ``dead_ball_known`` false),
#: since every stoppage then reads as a loss and a regain (review AN-03).
DEAD_BALL_MEASURES = frozenset({"chains_lost_share", "chains_dead_ball_share", "efi_ball_recovery_time_s", "efi_forced_turnovers_p90",
                                "counter_press_share_out", "engagements_counter_press_per_min_out"})


@dataclass(frozen=True)
class Measure:
    name: str
    definition: str
    denominator: str
    basis: str        # published | common | ponta
    source: str
    label: str        # observed | estimated
    inputs: tuple[tuple[str, str], ...]  # Data check entries (package, metric) the measure inherits


def _m(name, definition, denominator, basis, source, inputs, label="estimated"):
    return Measure(name, definition, denominator, basis, source, label, tuple(inputs))


_EFI = "FIFA Enhanced Football Intelligence (EFI) definitions, as computed by P5 (relationships.efi_aggregates)"
_ZONES = (("p3", "relative_zones"), ("p5", "line_breaks"))
CATALOGUE: tuple[Measure, ...] = (
    _m("efi_receptions_behind_midfield_p90", "Completed passes received beyond the defending midfield line.", "per 90 minutes of play", "published", _EFI, _ZONES),
    _m("efi_receptions_behind_defence_p90", "Completed passes received beyond the defending defensive line.", "per 90 minutes of play", "published", _EFI, _ZONES),
    _m("efi_final_third_entries_p90", "Possession chains whose ball first reaches the attacking third.", "per 90 minutes of play", "published", _EFI, (("p3", "relative_zones"),)),
    *(_m(f"efi_final_third_entries_{ln}_share", f"The share of final-third entries made in the {ln.replace('_', ' ')} lane.", "share of entries", "published", _EFI, (("p3", "relative_zones"),)) for ln in LANES),
    *(_m(f"efi_def_line_m_out_{th}", f"Median defensive line height out of possession with the ball in the team's {th} third (mean of the halves).", "metres from own goal line", "published", _EFI, (("p2", "team_shape"),)) for th in ("defensive", "middle", "attacking")),
    _m("efi_length_m_in_middle", "Median team length in possession with the ball in the middle third (mean of the halves).", "metres", "published", _EFI, (("p2", "team_shape"),)),
    _m("efi_length_m_out_middle", "Median team length out of possession with the ball in the middle third (mean of the halves).", "metres", "published", _EFI, (("p2", "team_shape"),)),
    _m("efi_ball_recovery_time_s", "Median time from losing the ball to the team's next possession (mean of the halves).", "seconds", "published", _EFI, (("p1", "possession"),)),
    _m("efi_forced_turnovers_p90", "The other team's possessions lost with its carrier under high pressure.", "per 90 minutes of play", "published", _EFI, (("p1", "possession"), ("p5", "relationships"))),
    _m("ppda", "The other team's passes in its own 60% of the pitch over this team's defensive actions there (challenges and fouls while the other team had the ball, interceptions), from the logged feed. Only for feeds whose defender-side actions are mapped (PPDA_FEEDS) and with at least PPDA_MIN_ACTIONS actions; otherwise Blocked.", "passes per defensive action", "published", "PPDA as published by Colin Trainor (2014); defensive actions mapped per provider", (("p1", "possession"),)),
    _m("possession_share", "The team's share of in-play time in possession (P1).", "share of in-play time", "common", "Time-based possession share, a common public definition", (("p1", "possession"),)),
    _m("field_tilt", "The team's share of both teams' completed passes that end in their attacking third, from the logged feed.", "share", "common", "Field tilt, a common public definition", (("p1", "possession"),)),
    *(_m(f"phase_share_{ph}", f"The share of the team's possession time with a phase spent in P4's {ph.replace('_', ' ')}.", "share of possession time with a phase", "ponta", "Ponta P4 (SkillCorner's phase vocabulary)", (("p4", "phase_of_play"),)) for ph in PHASES),
    *(_m(f"out_share_{b}", f"The share of the team's time out of possession spent defending established play in a {b.replace('_', ' ')}: its own block (the mean depth of its three deepest outfield players, P4), pressing included (press_share_out reports the press).", "share of out-of-possession time", "ponta", "Ponta P4 (the defending team's own block; FIFA's block names)", (("p4", "phase_of_play"),)) for b in BLOCKS),
    _m("press_share_out", "The share of the team's time out of possession in a high, mid or low press (P4v).", "share of out-of-possession time", "ponta", "Ponta P4v", (("p4v", "engagements"),)),
    _m("counter_press_share_out", "The share of the team's time out of possession in a counter-press (P4v).", "share of out-of-possession time", "ponta", "Ponta P4v", (("p4v", "engagements"),)),
    _m("build_up_opposed_share", "The share of the team's build-up time that is opposed (P4v: carrier pressure 6 or more on the same carrier).", "share of build-up time", "ponta", "Ponta P4v", (("p4v", "fifa_view"),)),
    _m("presses_per_min_out", "Presses (two or more defenders engaging in turn) per minute out of possession.", "per minute out of possession", "ponta", "Ponta P4v", (("p4v", "engagements"),)),
    *(_m(f"engagements_{e}_per_min_out", f"Engagements of type {e.replace('_', ' ')} per minute out of possession.", "per minute out of possession", "ponta", "Ponta P4v", (("p4v", "engagements"),)) for e in ENGAGEMENTS),
    _m("chains_lost_share", "The share of the team's possession chains that end with the other team taking the ball.", "share of chains", "ponta", "Ponta P1", (("p1", "possession"),)),
    _m("chains_shot_share", "The share of the team's possession chains that end in a shot or goal.", "share of chains", "ponta", "Ponta P1", (("p1", "possession"),)),
    _m("chains_dead_ball_share", "The share of the team's possession chains that end in a dead ball.", "share of chains", "ponta", "Ponta P1", (("p1", "possession"),)),
    _m("runs_in_behind_per_min_in", "Off-ball runs ending in behind the last line per minute in possession (P6).", "per minute in possession", "ponta", "Ponta P6 (FIFA's area categories)", (("p6", "movement"),)),
    _m("carrier_pressure_median_in", "Median pressure on the team's ball carrier (P5, Plaiq's 1 to 10 scale).", "1 to 10", "ponta", "Ponta P5 (Plaiq pressure)", (("p5", "relationships"),)),
    _m("carrier_pressure_median_out", "Median pressure the team puts on the other team's ball carrier (P5, 1 to 10).", "1 to 10", "ponta", "Ponta P5 (Plaiq pressure)", (("p5", "relationships"),)),
    _m("open_high_value_share_median", "Median share of the team's high-value space the other team does not control (P7).", "share", "ponta", "Ponta P7 (pitch control and threat on the honeycomb)", (("p7", "space"),)),
    _m("gain_median", "Median Gain (decision quality version 1, 0 to 1.5) of the team's scored passes. Never combined with Choice.", "0 to 1.5", "ponta", "Plaiq Gain (version 1)", (("p5", "line_breaks"),)),
    _m("choice_median", "Median Choice (decision quality version 2, 0 to 1) of the team's scored passes. Never combined with Gain.", "0 to 1", "ponta", "Plaiq Choice (version 2)", (("p5", "line_breaks"),)),
)
MEASURES = {m.name: m for m in CATALOGUE}
PROFILE_COLUMNS = ("match_id", "team", "team_key", "opp_team_key", "competition", "season", "measure", "value", "denominator",
                   "sample_minutes", "sample_n", "label", "basis", "source", "data_check", "has_feed", "versions")


#: Feeds whose defender-side actions are mapped and checked for PPDA (review AN-01, 25 November 2026): measured levels
#: within the usual published range (Metrica 5 to 8 on games 1 to 3, the German league 5 to 19 on three of its matches). Elsewhere PPDA is Blocked:
#: SkillCorner's dynamic events log no defender-side tackles or interceptions; PFF's feed logs far more contests as
#: challenges (3 to 7 even with the rule below); the StatsBomb and Genius mappings are not checked.
PPDA_FEEDS = ("metrica", "idsse")
#: Engineering: with fewer defensive actions in the zone a single match's PPDA is not quoted.
PPDA_MIN_ACTIONS = 10


def defensive_actions(typ: np.ndarray, sub: np.ndarray, team: np.ndarray, poss: np.ndarray, t: str) -> np.ndarray:
    """
    The team's defensive actions for PPDA (Trainor 2014: tackles, interceptions, challenges, fouls), per event row.

    A challenge or a foul counts only while the other team had the ball (``poss``, P1 at the event, not contested): feeds
    that log a duel for both players (Metrica) or for its winner (the German league) would otherwise count the team in
    possession's duels, and a contested moment would count both sides. An interception counts as logged. The team's
    fouls are the other team's FAULT RECEIVED.
    """
    o = OTHER[t]
    off_ball = poss == o
    challenge = (typ == "CHALLENGE") & (team == t) & off_ball
    interception = (typ == "RECOVERY") & (np.char.find(sub.astype(str), "INTERCEPTION") >= 0) & (team == t)
    foul = (typ == "FAULT RECEIVED") & (team == o) & off_ball
    return challenge | interception | foul


def possession_at(f: pd.DataFrame, frames: np.ndarray) -> np.ndarray:
    """P1's possession label at each native frame (the nearest grid point at or before it)."""
    nf = f["native_frame"].to_numpy(dtype=float)
    poss = f["p1_possession"].astype(object).to_numpy()
    order = np.argsort(nf, kind="stable")
    nf, poss = nf[order], poss[order]
    ok = np.isfinite(frames)
    i = np.clip(np.searchsorted(nf, np.where(ok, frames, 0), side="right") - 1, 0, max(len(nf) - 1, 0))
    out = np.full(len(frames), None, dtype=object)
    if len(nf):
        out[ok] = poss[i[ok]]
    return out


def match_profiles(match, res, identity, semantic_events: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    One team's match as measures (long form), for both teams. ``res`` is the ContextResult with P1 to P8; ``identity`` the
    match's opaque keys (``TeamRegistry.identity(match)``, profiles 0.3.0): ``match_id``, team, competition and season in
    every row are those keys, never the provider's ids.
    """
    from plaiq_engine.teams import MatchIdentity, PseudonymError

    if not isinstance(identity, MatchIdentity):
        raise PseudonymError("match_profiles needs the match's MatchIdentity (TeamRegistry.identity(match))")
    team_keys = identity.team_keys
    f, out = res.frames, res.outputs
    n = len(f)
    step = float(np.median(np.diff(f["frame_ts"].to_numpy(float)))) if n > 1 else 0.2
    poss = f["p1_possession"].astype(object).to_numpy()
    in_play_min = float(np.isin(poss, [*TEAMS, "contested"]).sum()) * step / 60.0
    dc = {(e.package, e.metric): e.state for k in out for e in out[k].entries}
    from matrix_engine.parity import SCHEMA_VERSION as DATA_CHECK_SCHEMA

    from plaiq_engine.semantic_builder import SCHEMA_VERSION as SEMANTIC_SCHEMA
    from plaiq_engine.semantic_builder import _engine_version

    # Engines and schemas too (review SE-07), so profiles from before and after a loader or engine change never mix.
    versions = json.dumps({**res.versions, "thresholds": res.thresholds_version, "profiles": PROFILE_VERSION, "engine": _engine_version(),
                           "semantic_schema": SEMANTIC_SCHEMA, "data_check_schema": DATA_CHECK_SCHEMA}, sort_keys=True)
    ev = match.events
    has_feed = bool(len(ev)) and {"Type", "Team", "Start X"} <= set(ev.columns)
    L = match.pitch.length_m
    ppda_feed = str(getattr(match, "provider", "")).lower() in PPDA_FEEDS
    dead_known = bool(out["p1"].measurements.get("dead_ball_known", False))
    ev_poss = possession_at(f, pd.to_numeric(ev["Start Frame"], errors="coerce").to_numpy(float)) if has_feed and "Start Frame" in ev else \
        np.full(len(ev), None, dtype=object)
    efi = out["p5"].segments.get("efi", pd.DataFrame(columns=["team", "period", "metric", "value"]))
    rows = []
    for t in TEAMS:
        o = OTHER[t]
        min_in = float((poss == t).sum()) * step / 60.0
        min_out = float((poss == o).sum()) * step / 60.0
        vals: dict[str, tuple[float | None, float, int]] = {}  # measure -> (value, sample minutes, sample n)

        def put(name, value, minutes, count, vals=vals):
            v = None if value is None or (isinstance(value, float) and not np.isfinite(value)) else round(float(value), 4)
            vals[name] = (v, round(minutes, 2), int(count))

        # FIFA EFI, from P5's per-half aggregates.
        e = efi.loc[efi["team"] == t]

        def efi_sum(metric, e=e):
            v = e.loc[e["metric"] == metric, "value"]
            return float(v.sum()) if len(v) else None

        def efi_mean(metric, e=e):
            v = e.loc[e["metric"] == metric, "value"]
            return float(v.mean()) if len(v) else None

        p90 = 90.0 / in_play_min if in_play_min > 0 else np.nan
        for metric in ("receptions_behind_midfield", "receptions_behind_defence", "final_third_entries", "forced_turnovers"):
            s = efi_sum(metric)
            put(f"efi_{metric}_p90", None if s is None else s * p90, in_play_min, int(s or 0))
        entries = efi_sum("final_third_entries") or 0
        for ln in LANES:
            s = efi_sum(f"final_third_entries_{ln}")
            put(f"efi_final_third_entries_{ln}_share", (s / entries) if entries and s is not None else None, in_play_min, int(entries))
        for th in ("defensive", "middle", "attacking"):
            put(f"efi_def_line_m_out_{th}", efi_mean(f"def_line_m_out_{th}"), min_out, 0)
        put("efi_length_m_in_middle", efi_mean("length_m_in_middle"), min_in, 0)
        put("efi_length_m_out_middle", efi_mean("length_m_out_middle"), min_out, 0)
        put("efi_ball_recovery_time_s", efi_mean("ball_recovery_time_s"), min_out, 0)

        # Published and common event measures (need the logged feed).
        put("possession_share", min_in / (min_in + min_out) if (min_in + min_out) > 0 else None, in_play_min, 0)
        if has_feed:
            typ = ev["Type"].fillna("").astype(str).str.upper()
            sub = ev["Subtype"].fillna("").astype(str).str.upper() if "Subtype" in ev else pd.Series("", index=ev.index)
            team = ev["Team"].fillna("").astype(str).str.lower()
            sign = {"home": 1.0, "away": -1.0}
            depth = {k: sign[k] * ev["Start X"].to_numpy(float) + L / 2 for k in TEAMS}  # in each team's own frame
            opp_passes = ((typ == "PASS") & (team == o)).to_numpy() & (depth[o] < 0.6 * L)
            defs = defensive_actions(typ.to_numpy(str), sub.to_numpy(str), team.to_numpy(str), ev_poss, t) & (depth[t] > 0.4 * L)
            nd = int(defs.sum())
            ppda_ok = ppda_feed and nd >= PPDA_MIN_ACTIONS
            put("ppda", opp_passes.sum() / nd if ppda_ok else None, min_out, nd)
            if "End X" in ev and ev["End X"].notna().any():
                end_depth = {k: sign[k] * ev["End X"].to_numpy(float) + L / 2 for k in TEAMS}
                ft = {k: int((((typ == "PASS") & (team == k)).to_numpy() & (end_depth[k] >= 2 * L / 3)).sum()) for k in TEAMS}
                put("field_tilt", ft[t] / (ft[t] + ft[o]) if (ft[t] + ft[o]) else None, in_play_min, ft[t] + ft[o])
            else:
                put("field_tilt", None, in_play_min, 0)
        else:
            put("ppda", None, min_out, 0)
            put("field_tilt", None, in_play_min, 0)

        # Ponta measures.
        p4t, ph = f["p4_team"].astype(object).to_numpy(), f["p4_phase"].astype(object).to_numpy()
        mine = (p4t == t) & pd.notna(ph)
        n_ph = int(mine.sum())
        for p in PHASES:
            put(f"phase_share_{p}", float((ph[mine] == p).mean()) if n_ph else None, n_ph * step / 60, n_ph)
        outv = f["p4v_out_phase"].astype(object).to_numpy()
        theirs = (p4t == o) & pd.notna(outv)
        n_out = int(theirs.sum())
        # The team's own block in established play (P4 0.6.0, review AN-02), pressing included; P4v's press labels
        # would otherwise take high-block time away from a team that presses from it.
        out4 = f["p4_out_phase"].astype(object).to_numpy()
        for b in BLOCKS:
            put(f"out_share_{b}", float((out4[theirs] == P4_BLOCK_LABEL[b]).mean()) if n_out else None, n_out * step / 60, n_out)
        put("press_share_out", float(np.isin(outv[theirs], ["high_press", "mid_press", "low_press"]).mean()) if n_out else None, n_out * step / 60, n_out)
        put("counter_press_share_out", float((outv[theirs] == "counter_press").mean()) if n_out else None, n_out * step / 60, n_out)
        fv = f["p4v_phase"].astype(object).to_numpy()
        bu = (p4t == t) & np.isin(fv, ["build_up_opposed", "build_up_unopposed"])
        put("build_up_opposed_share", float((fv[bu] == "build_up_opposed").mean()) if bu.any() else None, bu.sum() * step / 60, int(bu.sum()))
        pr = out["p4v"].segments["presses"]
        en = out["p4v"].segments["engagements"]
        put("presses_per_min_out", (pr["team"] == t).sum() / min_out if min_out else None, min_out, int((pr["team"] == t).sum()))
        for kind in ENGAGEMENTS:
            k = int(((en["team"] == t) & (en["type"] == kind)).sum())
            put(f"engagements_{kind}_per_min_out", k / min_out if min_out else None, min_out, k)
        ch = out["p1"].segments["chains"]
        ct = ch.loc[ch["team"] == t]
        nc = len(ct)
        put("chains_lost_share", float((ct["end_type"] == "lost").mean()) if nc else None, min_in, nc)
        put("chains_shot_share", float(ct["end_type"].isin(["shot", "goal"]).mean()) if nc else None, min_in, nc)
        put("chains_dead_ball_share", float((ct["end_type"] == "dead_ball").mean()) if nc else None, min_in, nc)
        runs = out["p6"].segments["runs"]
        rb = int(((runs["team"] == t) & runs["in_possession"].astype(bool) & (runs["area_end"] == "in_behind")).sum())
        put("runs_in_behind_per_min_in", rb / min_in if min_in else None, min_in, rb)
        cp = f["p5_carrier_pressure"].to_numpy(dtype=float, na_value=np.nan)
        att = f["p3_attacking_team"].astype(object).to_numpy() if "p3_attacking_team" in f else p4t
        cin, cout = cp[(att == t) & np.isfinite(cp)], cp[(att == o) & np.isfinite(cp)]
        put("carrier_pressure_median_in", float(np.median(cin)) if len(cin) else None, min_in, len(cin))
        put("carrier_pressure_median_out", float(np.median(cout)) if len(cout) else None, min_out, len(cout))
        hv = f[f"p7_{t}_high_value_open"].to_numpy(dtype=float, na_value=np.nan)
        hv = hv[np.isfinite(hv)]
        put("open_high_value_share_median", float(np.median(hv)) if len(hv) else None, in_play_min, len(hv))
        if semantic_events is not None and len(semantic_events):
            sp = semantic_events.loc[(semantic_events["action_type"] == "pass") & (semantic_events["actor_team"] == t)]
            g, c = sp["dq_score"].dropna().astype(float), sp["dq_v2_score"].dropna().astype(float)
            put("gain_median", float(g.median()) if len(g) else None, in_play_min, len(g))
            put("choice_median", float(c.median()) if len(c) else None, in_play_min, len(c))
        else:
            put("gain_median", None, in_play_min, 0)
            put("choice_median", None, in_play_min, 0)

        for name, (v, mins, cnt) in vals.items():
            meas = MEASURES[name]
            states = [dc.get(i, "blocked") for i in meas.inputs]
            if name in ("ppda", "field_tilt") and not has_feed or name == "ppda" and not ppda_feed:
                states.append("blocked")
            if name in DEAD_BALL_MEASURES and not dead_known:
                states.append("degraded")
            state = min(states, key=lambda s: _STATE_RANK.get(s, 0)) if states else "full"
            if v is None or state == "blocked":
                v, state = None, "blocked"  # a Blocked row never carries a value (review AI-05)
            rows.append({"match_id": identity.match_key, "team": t, "team_key": team_keys.get(t),
                         "opp_team_key": team_keys.get(o), "competition": identity.competition_key,
                         "season": identity.season_key, "measure": name, "value": v, "denominator": meas.denominator,
                         "sample_minutes": mins, "sample_n": cnt, "label": meas.label, "basis": meas.basis, "source": meas.source,
                         "data_check": state, "has_feed": has_feed, "versions": versions})
    return pd.DataFrame(rows, columns=list(PROFILE_COLUMNS))


def worst_state(states) -> str | None:
    """The worst Data check state among ``states`` (blocked, then degraded, then full); None when there are none."""
    s = [x for x in states if isinstance(x, str)]
    return min(s, key=lambda x: _STATE_RANK.get(x, 0)) if s else None


def variance_components(means: np.ndarray, counts: np.ndarray, values: np.ndarray, team_vars: np.ndarray) -> tuple[float, float, int]:
    """
    Within-team and between-team variance by the one-way ANOVA method of moments (review ST-06, 25 November 2026): the
    between component is (MSB - MSW) / n0, not the variance of the observed team means, which also carries each mean's
    own noise and so under-shrinks teams with few matches. Returns (within, between, within degrees of freedom).
    """
    g, N = len(means), float(counts.sum())
    if g < 2 or N - g < 1:
        return np.nan, np.nan, 0
    grand = float(values.mean())
    msw = float(((counts - 1) * team_vars).sum() / (N - g))
    msb = float((counts * (means - grand) ** 2).sum() / (g - 1))
    n0 = (N - float((counts ** 2).sum()) / N) / (g - 1)
    return msw, max(0.0, (msb - msw) / n0), int(N - g)


def baseline(profiles: pd.DataFrame, team_key: str, measure: str, exclude_match: str | None = None) -> dict:
    """
    A team's baseline for one measure against the set's, with shrinkage; refuses mixed versions and leaves Blocked out.
    The set is the profiles of the same competition and season as the team's rows when known, else all profiles given.
    """
    p = profiles.loc[(profiles["measure"] == measure) & (profiles["data_check"] != "blocked") & profiles["value"].notna()]
    if exclude_match is not None:
        p = p.loc[p["match_id"] != exclude_match]
    if p["versions"].nunique() > 1:
        return {"refused": "profiles built from different package or threshold versions; rebuild them from the current versions"}
    team = p.loc[p["team_key"] == team_key]
    if len(team) and team["competition"].notna().all():
        p = p.loc[(p["competition"] == team["competition"].iloc[0]) & (p["season"] == team["season"].iloc[0])]
    if team.empty:
        return {"refused": "no usable matches for this team and measure"}
    by_team = p.groupby("team_key")["value"]
    means, counts = by_team.mean(), by_team.count()
    n, t_mean = int(len(team)), float(team["value"].mean())
    set_mean = float(p["value"].mean())
    within, between, df_within = variance_components(means.to_numpy(float), counts.to_numpy(float), p["value"].to_numpy(float),
                                                     by_team.var(ddof=1).fillna(0).to_numpy(float))
    k = within / between if np.isfinite(within) and np.isfinite(between) and between > 0 else np.inf
    shrunk = (n * t_mean + (k * set_mean if np.isfinite(k) else 0)) / (n + k) if np.isfinite(k) else set_mean
    return {"measure": measure, "team_key": team_key, "data_check": worst_state(team["data_check"]), "team_matches": n, "team_mean": round(t_mean, 4), "set_matches": int(len(p)),
            "set_teams": int(len(means)), "set_mean": round(set_mean, 4), "within_sd": round(float(np.sqrt(within)), 4) if np.isfinite(within) else None,
            "shrunk": round(float(shrunk), 4), "set_weight": None if not np.isfinite(k) else round(float(k), 3),
            "within_df": df_within, "baseline_n": None if not np.isfinite(k) else round(float(n + k), 3),
            "blocked_left_out": int(((profiles["measure"] == measure) & (profiles["team_key"] == team_key) & (profiles["data_check"] == "blocked")).sum())}


def compare(profiles: pd.DataFrame, match_id: str, team_key: str, measure: str, cfg: dict) -> dict:
    """One match's value against the team's baseline (the match left out), with a verdict only when the evidence allows."""
    row = profiles.loc[(profiles["match_id"] == match_id) & (profiles["team_key"] == team_key) & (profiles["measure"] == measure)]
    if row.empty or pd.isna(row["value"].iloc[0]) or row["data_check"].iloc[0] == "blocked":
        return {"verdict": None, "reason": "no value for this match", "data_check": "blocked"}
    value = float(row["value"].iloc[0])
    b = baseline(profiles, team_key, measure, exclude_match=match_id)
    if "refused" in b:
        return {"value": value, "verdict": None, "reason": b["refused"], "data_check": row["data_check"].iloc[0]}
    need, need_source = min_matches(measure, cfg)
    # The worst Data check state behind the comparison, which the AI must disclose (review AI-05).
    out = {"value": value, **b, "data_check": worst_state([row["data_check"].iloc[0], b["data_check"]]), "minimum_matches": need, "minimum_source": need_source}
    if need is None:
        return {**out, "verdict": None, "reason": "the minimum number of matches for this measure is not measured yet"}
    if b["team_matches"] < need:
        return {**out, "verdict": "not enough matches yet", "reason": f"{b['team_matches']} of the {need} matches needed"}
    if b["within_sd"] is None or b["within_sd"] == 0 or not b["within_df"]:
        return {**out, "verdict": None, "reason": "no within-team spread to compare against"}
    # A prediction interval for one new match around the team's baseline (review ST-05): the within-team spread plus the
    # baseline's own uncertainty, at a stated two-sided false-alarm rate, with a t quantile on the within-team degrees of
    # freedom. Outside it is "above" or "below your usual"; under pure noise that happens at about that rate.
    from scipy.stats import t as student_t

    n_eff = b["baseline_n"] if b["baseline_n"] else b["team_matches"]
    spread = b["within_sd"] * np.sqrt(1.0 + 1.0 / n_eff)
    q = float(student_t.ppf(1.0 - cfg["pf_verdict_false_alarm"] / 2.0, b["within_df"]))
    lo, hi = b["shrunk"] - q * spread, b["shrunk"] + q * spread
    verdict = "above your usual" if value > hi else "below your usual" if value < lo else "about your usual"
    return {**out, "verdict": verdict, "difference_in_sd": round(float((value - b["shrunk"]) / b["within_sd"]), 2),
            "usual_range": [round(float(lo), 4), round(float(hi), 4)], "false_alarm_rate": cfg["pf_verdict_false_alarm"]}
