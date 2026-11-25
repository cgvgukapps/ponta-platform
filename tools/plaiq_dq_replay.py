"""
plaiq_dq_replay.py — a blind replay check of decision quality (pilot of the expert agreement study).

``ponta-blueprint/engine-governance.md`` asks for decisions rated blind (the pass and the alternatives,
no engine score shown) and agreement reported with the engine's band and rank. This writes the pilot:

- **completed passes** scored by Plaiq (``MatchAnalysis.event_pov``, cached by ``context_line_breaks.py``)
  on SkillCorner's 20 open-data matches, which the bands were not tuned on;
- drawn **evenly across the four bands** (poor, suboptimal, good, optimal) with a fixed seed, so
  agreement can be read per band (Plaiq's own spread is 5%, 18%, 33% and 44%);
- each case replays ``--before-s`` before the pass and ``--after-s`` after it, the passer's team in blue
  attacking left to right, the passer ringed; no score or band is shown;
- the judge rates the decision (poor, suboptimal, good, optimal) and may name a better option in a note;
- the key (Plaiq's score and band per case) goes to ``--key``, never into the page.

``--score`` reads the copied results and the key: exact and adjacent band agreement, quadratic-weighted
kappa, and rank correlation between the judge's rating and Plaiq's score. Positions from sample data,
so both files are written outside the repository.

    python tools/plaiq_dq_replay.py --data ~/ponta-reference-data/skillcorner-opendata --cache /tmp/ponta-context/sc-cache \
        --out /tmp/ponta-context/dq_replay.html --key /tmp/ponta-context/dq_replay_key.json
    python tools/plaiq_dq_replay.py --score results.json --key /tmp/ponta-context/dq_replay_key.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matrix_engine.sample_data import repo_root
from plaiq_engine.context.grid import build_grid
from plaiq_engine.context.structure import attack_signs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_tag import tag
from context_possession_replay import PAGE as _POSSESSION_PAGE
from context_reference_set import _load
from plaiq_pressure import weighted_kappa
from plaiq_references import spearman

BANDS = ("poor", "suboptimal", "good", "optimal")
EDGES = (0.55, 0.80, 0.95)  # Plaiq's unified bands: poor < 0.55 <= suboptimal < 0.80 <= good < 0.95 <= optimal


def band_of(dq: float) -> str:
    return BANDS[int(np.searchsorted(EDGES, dq, side="right"))]


def _outside(path: Path) -> bool:
    root = repo_root()
    return not (path.resolve() == root or root in path.resolve().parents)


def candidates(data: Path, cache: Path) -> pd.DataFrame:
    rows = []
    for d in sorted((data / "matches").iterdir()):
        p = cache / f"pov_skillcorner_{d.name}.{tag()}.pkl"
        if not p.exists():
            continue
        pov = pd.read_pickle(p)
        m = _load(d, cache)
        act = pov.loc[pov["decision_quality"].notna() & pov["v_actual_carry"].isna()]
        for r in act.itertuples():
            e = m.events.loc[r.event_idx] if r.event_idx in m.events.index else None
            if e is None or e["Type"] != "PASS":
                continue
            rows.append({"match": d.name, "event_row": int(r.event_idx), "frame": int(e["Start Frame"]), "team": str(e["Team"]).lower(),
                         "passer": str(e["From"]), "dq": float(r.decision_quality), "band": band_of(float(r.decision_quality))})
    return pd.DataFrame(rows)


def case(m, f: int, team: str, passer: str, before_s: float, after_s: float, cid: str, header: str) -> dict:
    step = max(1, int(round(m.fps * 0.2)))
    frames = [x for x in range(f - int(before_s * m.fps), f + int(after_s * m.fps) + 1, step)]
    known = set(m.frames["frame"])
    frames = [x for x in frames if x in known]
    g = build_grid(m, 5.0)
    period = int(m.frames.loc[m.frames["frame"] == f, "period"].iloc[0])
    sign = {(int(r.period), r.team): r.attack_sign for r in attack_signs(m, g).itertuples()}.get((period, team), 1)
    L, W = m.pitch.length_m, m.pitch.width_m
    tr = m.tracking.loc[m.tracking["frame"].isin(frames), ["frame", "player_id", "team", "x", "y"]]
    ball = m.ball.set_index("frame")
    other = "away" if team == "home" else "home"
    ids = {t: sorted(tr.loc[tr["team"] == t, "player_id"].unique()) for t in (team, other)}
    pid = f"{team}_{passer}"
    rows = []
    for x in frames:
        at = tr.loc[tr["frame"] == x].set_index("player_id")
        row = {"t": round((x - f) / m.fps, 2)}
        for t, key in ((team, "a"), (other, "d")):
            row[key] = [[round(float(sign * at.at[p, "x"] + L / 2), 1), round(float(sign * at.at[p, "y"]), 1)] if p in at.index else None for p in ids[t]]
        row["b"] = [round(float(sign * ball.at[x, "x"] + L / 2), 1), round(float(sign * ball.at[x, "y"]), 1)] if x in ball.index else None
        rows.append(row)
    moment = min(range(len(frames)), key=lambda i: abs(frames[i] - f))
    return {"id": cid, "L": L, "W": W, "moment": moment, "frames": rows, "header": header,
            "passer": ids[team].index(pid) if pid in ids[team] else -1}


def _rep(s: str, a: str, b: str) -> str:
    if a not in s:
        raise ValueError(f"page template changed; cannot find: {a[:60]}")
    return s.replace(a, b)


def page(cases: list[dict], before_s: float, after_s: float) -> str:
    s = _rep(_POSSESSION_PAGE, "<title>Possession replay check</title>", "<title>Decision replay check</title>")
    s = _rep(s, "<h1>Possession replay check</h1>", "<h1>Decision replay check</h1>")
    s = _rep(s, 
        "Home is <b style=\"color:var(--home)\">blue</b>, attacking left to right; away is <b style=\"color:var(--away)\">red</b>. No label is shown. Press Play, watch, then say <b>who has the ball at the marked moment</b> (the ball is ringed when you reach it): home, away, or nobody yet (the ball is in flight or loose).",
        "The passing team is <b style=\"color:var(--home)\">blue</b>, attacking left to right; the opponents are <b style=\"color:var(--away)\">red</b>. The passer is ringed. No score is shown. Press Play, watch the build-up and the pass, then rate <b>the decision at the moment of the pass</b>, given the options the passer had: poor, suboptimal, good or optimal. If a better option was on, say which in the note.")
    s = _rep(s, "['home','away','nobody'].map(v=>`<label><input type=\"radio\" name=\"c${ci}\" value=\"${v}\"> ${v==='nobody'?'Nobody yet':v==='home'?'Home (blue)':'Away (red)'}</label>`).join('')",
                  "['poor','suboptimal','good','optimal'].map(v=>`<label><input type=\"radio\" name=\"c${ci}\" value=\"${v}\"> ${v[0].toUpperCase()+v.slice(1)}</label>`).join('')")
    s = _rep(s, "const ball=el('circle',{r:4,class:'ball'},svg)",
                  "if(c.passer>=0){A[c.passer].setAttribute('stroke','var(--hi)');A[c.passer].setAttribute('stroke-width','3');}const ball=el('circle',{r:4,class:'ball'},svg)")
    s = _rep(s, 'placeholder="Note (optional)"', 'placeholder="Note (optional): a better option, or why"')
    return s.replace("__BEFORE__", f"{before_s:g}").replace("__AFTER__", f"{after_s:g}").replace("__DATA__", json.dumps(cases, separators=(",", ":")).replace("</", "<\\/"))


def build(args) -> None:
    rng = np.random.default_rng(20261119)
    cand = candidates(args.data, args.cache)
    per = int(np.ceil(args.n / len(BANDS)))
    picks = pd.concat([g.iloc[rng.permutation(len(g))[:per]] for _, g in cand.groupby("band")]).iloc[rng.permutation(per * len(BANDS))][: args.n]
    cases, key, loaded = [], {}, {}
    for r in picks.itertuples():
        if r.match not in loaded:
            loaded[r.match] = _load(args.data / "matches" / r.match, args.cache)
        m = loaded[r.match]
        cid = f"{r.match}-{r.event_row}"
        clock = float(m.frames.loc[m.frames["frame"] == r.frame, "time_s"].iloc[0])
        cases.append(case(m, r.frame, r.team, r.passer, args.before_s, args.after_s, cid, f"{int(clock // 60)}:{int(clock % 60):02d}"))
        key[cid] = {"dq": round(r.dq, 4), "band": r.band}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page(cases, args.before_s, args.after_s))
    args.key.write_text(json.dumps(key, indent=1))
    print(f"wrote {len(cases)} cases to {args.out} (bands {picks['band'].value_counts().to_dict()}); key to {args.key}")


def score(args) -> None:
    calls = {r["id"]: r["call"] for r in json.loads(Path(args.score).read_text()) if r.get("call")}
    key = json.loads(args.key.read_text())
    ids = [c for c in key if c in calls]
    eng = np.array([BANDS.index(key[c]["band"]) for c in ids])
    judge = np.array([BANDS.index(calls[c]) for c in ids])
    out = {"cases": len(ids), "exact": round(float((eng == judge).mean()), 4), "within_one_band": round(float((abs(eng - judge) <= 1).mean()), 4),
           "weighted_kappa": weighted_kappa(judge, eng, k=4), "spearman_score_vs_rating": spearman([key[c]["dq"] for c in ids], judge),
           "by_engine_band": {b: {"n": int((eng == i).sum()), "judge": {bb: int(((eng == i) & (judge == j)).sum()) for j, bb in enumerate(BANDS)}}
                              for i, b in enumerate(BANDS)}}
    print(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--before-s", type=float, default=5.0)
    ap.add_argument("--after-s", type=float, default=2.0)
    ap.add_argument("--score", type=Path)
    args = ap.parse_args()
    for p in [args.key] + [x for x in (args.out, args.cache, args.data) if x]:
        if not _outside(p):
            sys.exit(f"refusing to read or write sample-data positions inside the repository: {p}")
    score(args) if args.score else build(args)


if __name__ == "__main__":
    main()
