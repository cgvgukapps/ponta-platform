"""
context_spotcheck.py — a spot-check page for P3 zones and P4 phases (Gate 2a, zones).

Samples grid points where one team has the ball (deterministic, stratified by the ball's band
against the defending lines) and draws each on a pitch in the attacking team's frame (attacking
left to right): both teams' players, the ball, the defending team's line centroids and its
hull, with Ponta's labels. Tick whether the zone and the phase look right; the page counts the
ticks and copies them as JSON.

The page holds positions from sample data, so it is written **outside the repository** and is
not for sharing. No player ids or names are drawn.

    python tools/context_spotcheck.py --cache /tmp/ponta-context/cache --out /tmp/ponta-context/spotcheck.html
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import numpy as np
from matrix_engine.sample_data import repo_root
from plaiq_engine.context.phase import PhasePackage
from plaiq_engine.context.possession import PossessionPackage, ball_on_grid
from plaiq_engine.context.registry import Registry
from plaiq_engine.context.structure import (
    OTHER,
    StructurePackage,
    line_members,
    team_frames,
    team_states,
    window_index,
    windows_from_segments,
)
from plaiq_engine.context.thresholds import values
from plaiq_engine.context.zones import BANDS, ZonesPackage
from scipy.spatial import ConvexHull, QhullError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_measure import load

MATCHES = ("metrica_1", "metrica_2", "metrica_3", "skillcorner_1886347")
SCALE = 4.0  # px per metre


def sample(frames, per_match: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    band = frames["p3_ball_band"].astype(object).to_numpy()
    out: list[int] = []
    per_band = int(np.ceil(per_match / len(BANDS)))
    for b in BANDS:
        rows = np.where(band == b)[0]
        if len(rows):
            out += sorted(rng.choice(rows, size=min(per_band, len(rows)), replace=False).tolist())
    return sorted(out)[:per_match]


def card(key: str, i: int, m, res, tf, lines) -> str:
    f = res.frames.iloc[i]
    att = f["p3_attacking_team"]
    dfd = OTHER[att]
    L, W = m.pitch.length_m, m.pitch.width_m
    ball = ball_on_grid(m, res.frames.iloc[[i]])

    sign = {(int(r.period), r.team): r.attack_sign for r in res.outputs["p2"].segments["orientation"].itertuples()}[(int(f["period"]), att)]

    def px(depth_att: float, width_att: float) -> tuple[float, float]:
        return depth_att * SCALE, (W / 2 - width_att) * SCALE  # attacking left (+width) at the top

    svg = [f'<svg viewBox="0 0 {L * SCALE:.0f} {W * SCALE:.0f}" role="img" aria-label="Pitch at grid point {i}">',
           f'<rect width="{L * SCALE:.0f}" height="{W * SCALE:.0f}" class="pitch"/>',
           f'<line x1="{L * SCALE / 2:.0f}" y1="0" x2="{L * SCALE / 2:.0f}" y2="{W * SCALE:.0f}" class="mark"/>']
    for x0 in (0, L - 16.5):
        svg.append(f'<rect x="{x0 * SCALE:.0f}" y="{(W / 2 - 20.16) * SCALE:.0f}" width="{16.5 * SCALE:.0f}" height="{40.32 * SCALE:.0f}" class="mark" fill="none"/>')
    for y in (20.16, 9.16, -9.16, -20.16):
        svg.append(f'<line x1="0" y1="{(W / 2 - y) * SCALE:.0f}" x2="{L * SCALE:.0f}" y2="{(W / 2 - y) * SCALE:.0f}" class="lane"/>')
    # Defending team in the attacking frame: depth L - d, width -w.
    D, Wd, Ld = tf[dfd].depth[i], tf[dfd].width[i], lines[dfd][i]
    outf = np.isfinite(D) & (Ld != 3)
    pts = np.column_stack([L - D[outf], -Wd[outf]])
    if len(pts) >= 3:
        try:
            hull = ConvexHull(pts)
            poly = " ".join(f"{px(*pts[v])[0]:.1f},{px(*pts[v])[1]:.1f}" for v in hull.vertices)
            svg.append(f'<polygon points="{poly}" class="hull"/>')
        except QhullError:
            pass
    for k, name in enumerate(("def", "mid", "att")):
        sel = outf & (Ld == k)
        if sel.any():
            x = (L - D[sel].mean()) * SCALE
            svg.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{W * SCALE:.0f}" class="line"/><text x="{x + 3:.1f}" y="{12 + 12 * k}" class="lbl">{name}</text>')
    for team, cls in ((att, "att"), (dfd, "dfd")):
        Dt, Wt = tf[team].depth[i], tf[team].width[i]
        for d, w in zip(Dt, Wt, strict=True):
            if np.isfinite(d):
                a_d, a_w = (d, w) if team == att else (L - d, -w)
                x, y = px(a_d, a_w)
                svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" class="{cls}"/>')
    bx, by = ball["ball_x"].iloc[0], ball["ball_y"].iloc[0]
    if np.isfinite(bx):
        x, y = px(sign * bx + L / 2, sign * by)
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" class="ball"/>')
    svg.append("</svg>")
    labels = {"Lane": f["p3_ball_lane"], "Third": f["p3_ball_third"], "Band": f["p3_ball_band"], "In hull": f["p3_ball_in_hull"],
              "Pocket": f["p3_ball_in_pocket"], "Attackers between lines": f["p3_att_between_lines"],
              "Attackers behind": f["p3_att_behind_defence"], "Phase": f["p4_phase"], "Block": f["p4_block"],
              "Formation (att, in)": f[f"p2_{att}_formation_in"], "Formation (def, out)": f[f"p2_{dfd}_formation_out"]}
    rows = "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(str(v))}</dd>" for k, v in labels.items())
    cid = f"{key}-{i}"
    return (f'<article class="card" data-id="{cid}"><header>{html.escape(key)} · period {int(f["period"])} · '
            f'{int(f["clock_s"] // 60)}:{int(f["clock_s"] % 60):02d}</header>{"".join(svg)}<dl>{rows}</dl>'
            f'<fieldset><label><input type="checkbox" data-k="zone"> Zone right</label>'
            f'<label><input type="checkbox" data-k="phase"> Phase right</label></fieldset></article>')


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Zones spot check</title><style>
:root{--bg:#f6f7f9;--ink:#14171c;--muted:#5b6370;--card:#fff;--pitch:#e8efe6;--mark:#9aa79a;--att:#2f6fdf;--dfd:#d24a4a;--ball:#111}
@media (prefers-color-scheme:dark){:root{--bg:#0e1116;--ink:#e8ebf0;--muted:#9aa3b0;--card:#171b22;--pitch:#1d2a22;--mark:#4b5b4f;--att:#6c9cff;--dfd:#ff7b7b;--ball:#fff}}
body{margin:0;padding:16px;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}
h1{font-size:18px;margin:0 0 4px} p{color:var(--muted);margin:0 0 12px;max-width:70ch}
.bar{position:sticky;top:0;background:var(--bg);padding:8px 0;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,420px),1fr));gap:12px}
.card{background:var(--card);border-radius:8px;padding:10px} header{font-weight:600;margin-bottom:6px}
svg{width:100%;height:auto;display:block} .pitch{fill:var(--pitch)} .mark{stroke:var(--mark);stroke-width:1.5}
.lane{stroke:var(--mark);stroke-dasharray:3 4;stroke-width:1} .line{stroke:var(--dfd);stroke-width:1.5;stroke-dasharray:6 3}
.hull{fill:var(--dfd);fill-opacity:.08;stroke:var(--dfd);stroke-opacity:.5} .att{fill:var(--att)} .dfd{fill:var(--dfd)}
.ball{fill:var(--ball);stroke:var(--card);stroke-width:1.5} .lbl{fill:var(--dfd);font-size:11px}
dl{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;margin:8px 0} dt{color:var(--muted)} dd{margin:0}
fieldset{border:0;padding:0;display:flex;gap:16px} button{font:inherit;padding:6px 12px}
</style></head><body>
<h1>Zones and phases spot check</h1>
<p>Each card is one moment, drawn with the attacking team (blue) going left to right. Red dashed lines are the defending team's line centroids; the shaded area is its hull. Tick a box when Ponta's label looks right to you.</p>
<div class="bar"><span id="count"></span><button id="copy">Copy results</button><span id="copied" aria-live="polite"></span></div>
<textarea id="out" hidden readonly rows="4" style="width:100%;font:12px ui-monospace,monospace"></textarea>
<div class="grid">__CARDS__</div>
<script>
const boxes=[...document.querySelectorAll('input[type=checkbox]')];
function tally(){const z=boxes.filter(b=>b.dataset.k==='zone'),p=boxes.filter(b=>b.dataset.k==='phase');
document.getElementById('count').textContent=`Zones right: ${z.filter(b=>b.checked).length} of ${z.length} · Phases right: ${p.filter(b=>b.checked).length} of ${p.length}`;}
boxes.forEach(b=>b.addEventListener('change',tally));tally();
document.getElementById('copy').addEventListener('click',()=>{const out=[...document.querySelectorAll('.card')].map(c=>({id:c.dataset.id,
zone:c.querySelector('[data-k=zone]').checked,phase:c.querySelector('[data-k=phase]').checked}));
const txt=JSON.stringify(out),box=document.getElementById('out');box.value=txt;box.hidden=false;box.select();
const done=m=>{document.getElementById('copied').textContent=m;};
try{navigator.clipboard.writeText(txt).then(()=>done('Copied. Paste it back to Claude.'),()=>done('Copy blocked: select the text below and copy it.'));}catch(e){done('Copy blocked: select the text below and copy it.');}});
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--total", type=int, default=50)
    args = ap.parse_args()
    if repo_root() in args.out.resolve().parents:
        sys.exit("refusing to write sample-data positions inside the repository")
    cards = []
    per_match = int(np.ceil(args.total / len(MATCHES)))
    for n, key in enumerate(MATCHES):
        m = load(key, args.cache)
        res = Registry([PossessionPackage(), StructurePackage(), ZonesPackage(), PhasePackage()]).run(m)
        g = res.frames
        p1, p2 = res.outputs["p1"], res.outputs["p2"]
        tf = team_frames(m, g, p2.segments["orientation"])
        cfg = values()
        lines = line_members(g, tf, team_states(g, p1.frames, p1.segments["chains"], cfg), window_index(g, cfg),
                             windows_from_segments(p2.segments))
        for i in sample(g, per_match, seed=n + 1):
            cards.append(card(key, i, m, res, tf, lines))
    cards = cards[: args.total]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(PAGE.replace("__CARDS__", "".join(cards)))
    print(f"wrote {len(cards)} cards to {args.out}")


if __name__ == "__main__":
    main()
