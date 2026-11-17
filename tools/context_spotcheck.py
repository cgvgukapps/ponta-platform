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

``--replay`` writes the phase check instead: the same moments, each as a short replay (by default
5 s before and 2 s after) with Ponta's phase shown as it plays, a phase tick box and a note field.
A still frame rarely shows a phase, since a phase depends on how the possession began.

    python tools/context_spotcheck.py --replay --cache /tmp/ponta-context/cache --out /tmp/ponta-context/phase_replay.html
"""

from __future__ import annotations

import argparse
import html
import json
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


def _r(v: float) -> float | None:
    return round(float(v), 1) if np.isfinite(v) else None


def replay(key: str, i: int, m, res, tf, lines, before_s: float, after_s: float) -> dict:
    """
    One replay case: the grid points from ``before_s`` before to ``after_s`` after point ``i``,
    within its period, drawn in the attacking team's frame at point ``i`` (attacking left to
    right). Positions in metres, rounded to 0.1 m; no player ids.
    """
    g = res.frames
    f = g.iloc[i]
    att = f["p3_attacking_team"]
    dfd = OTHER[att]
    L, W = m.pitch.length_m, m.pitch.width_m
    hz = 1.0 / float(np.median(np.diff(g["clock_s"].to_numpy()[:50])))
    per = g["period"].to_numpy()
    idx = [j for j in range(i - int(round(before_s * hz)), i + int(round(after_s * hz)) + 1)
           if 0 <= j < len(g) and per[j] == per[i]]
    sign = {(int(r.period), r.team): r.attack_sign for r in res.outputs["p2"].segments["orientation"].itertuples()}[(int(f["period"]), att)]
    ball = ball_on_grid(m, g.iloc[idx])
    bx, by = ball["ball_x"].to_numpy(float), ball["ball_y"].to_numpy(float)
    frames = []
    for k, j in enumerate(idx):
        D, Wd, Ld = tf[dfd].depth[j], tf[dfd].width[j], lines[dfd][j]
        outf = np.isfinite(D) & (Ld != 3)
        pts = np.column_stack([L - D[outf], -Wd[outf]])
        hull = []
        if len(pts) >= 3:
            try:
                hull = [[_r(pts[v][0]), _r(pts[v][1])] for v in ConvexHull(pts).vertices]
            except QhullError:
                hull = []
        lx = [(_r(L - D[outf & (Ld == n)].mean()) if (outf & (Ld == n)).any() else None) for n in range(3)]
        a = [[_r(d), _r(w)] if np.isfinite(d) else None for d, w in zip(tf[att].depth[j], tf[att].width[j], strict=True)]
        d = [[_r(L - d), _r(-w)] if np.isfinite(d) else None for d, w in zip(tf[dfd].depth[j], tf[dfd].width[j], strict=True)]
        b = [_r(sign * bx[k] + L / 2), _r(sign * by[k])] if np.isfinite(bx[k]) else None
        ph = g["p4_phase"].iloc[j]
        frames.append({"t": round((j - i) / hz, 2), "a": a, "d": d, "b": b, "h": hull, "l": lx,
                       "ph": None if ph is None or (isinstance(ph, float) and np.isnan(ph)) else str(ph)})
    labels = {"Phase": f["p4_phase"], "Defending": f["p4_out_phase"], "Block": f["p4_block"], "Lane": f["p3_ball_lane"],
              "Third": f["p3_ball_third"], "Band": f["p3_ball_band"]}
    return {"id": f"{key}-{i}", "L": L, "W": W, "moment": idx.index(i), "frames": frames,
            "header": f'{key} · period {int(f["period"])} · {int(f["clock_s"] // 60)}:{int(f["clock_s"] % 60):02d}',
            "labels": {k: str(v) for k, v in labels.items()}}


REPLAY_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phase replay check</title><style>
:root{--bg:#f6f7f9;--ink:#14171c;--muted:#5b6370;--card:#fff;--pitch:#e8efe6;--mark:#9aa79a;--att:#2f6fdf;--dfd:#d24a4a;--ball:#111;--hi:#e0a100}
@media (prefers-color-scheme:dark){:root{--bg:#0e1116;--ink:#e8ebf0;--muted:#9aa3b0;--card:#171b22;--pitch:#1d2a22;--mark:#4b5b4f;--att:#6c9cff;--dfd:#ff7b7b;--ball:#fff;--hi:#ffc83d}}
body{margin:0;padding:16px;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}
h1{font-size:18px;margin:0 0 4px} p{color:var(--muted);margin:0 0 12px;max-width:75ch}
.bar{position:sticky;top:0;z-index:2;background:var(--bg);padding:8px 0;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,460px),1fr));gap:12px}
.card{background:var(--card);border-radius:8px;padding:10px} header{font-weight:600;margin-bottom:6px}
svg{width:100%;height:auto;display:block} .pitch{fill:var(--pitch)} .mark{stroke:var(--mark);stroke-width:1.5;fill:none}
.lane{stroke:var(--mark);stroke-dasharray:3 4;stroke-width:1} .line{stroke:var(--dfd);stroke-width:1.5;stroke-dasharray:6 3}
.hull{fill:var(--dfd);fill-opacity:.08;stroke:var(--dfd);stroke-opacity:.5} .att{fill:var(--att)} .dfd{fill:var(--dfd)}
.ball{fill:var(--ball);stroke:var(--card);stroke-width:1.5} .ring{fill:none;stroke:var(--hi);stroke-width:3}
.ctl{display:flex;gap:8px;align-items:center;margin:6px 0} .ctl input[type=range]{flex:1}
.now{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);min-height:1.5em} .now b{color:var(--ink)}
.badge{background:var(--hi);color:#111;border-radius:4px;padding:0 6px;font-weight:600;visibility:hidden}
dl{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;margin:8px 0} dt{color:var(--muted)} dd{margin:0}
fieldset{border:0;padding:0;display:flex;gap:12px;align-items:center;flex-wrap:wrap} button{font:inherit;padding:4px 10px}
input[type=text]{flex:1;min-width:12ch;font:inherit;padding:3px 6px}
</style></head><body>
<h1>Phase replay check</h1>
<p>Each card replays the __BEFORE__ seconds before a moment and __AFTER__ seconds after it, with the attacking team (blue) going left to right and its left at the top. Red dashed lines are the defending team's lines; the shaded area is its shape. The moment Ponta labelled is marked on the slider and ringed on the ball when you reach it. Press Play, watch, then judge Ponta's phase <b>at the moment</b>. Ponta's phase is shown live as it plays.</p>
<div class="bar"><span id="count"></span><button id="copy">Copy results</button><span id="copied" aria-live="polite"></span></div>
<textarea id="out" hidden readonly rows="4" style="width:100%;font:12px ui-monospace,monospace"></textarea>
<div class="grid" id="grid"></div>
<script>
const CASES=__DATA__;const S=4,NS='http://www.w3.org/2000/svg';
function el(t,a,p){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e;}
function lerp(a,b,u){return a&&b?[a[0]+(b[0]-a[0])*u,a[1]+(b[1]-a[1])*u]:(u<.5?a:b);}
CASES.forEach(c=>{
 const card=document.createElement('article');card.className='card';card.dataset.id=c.id;
 const W=c.W,L=c.L,X=x=>x*S,Y=y=>(W/2-y)*S;
 card.innerHTML=`<header>${c.header}</header>`;
 const svg=el('svg',{viewBox:`0 0 ${L*S} ${W*S}`,role:'img','aria-label':'Replay of '+c.id},card);
 el('rect',{width:L*S,height:W*S,class:'pitch'},svg);el('line',{x1:L*S/2,y1:0,x2:L*S/2,y2:W*S,class:'mark'},svg);
 for(const x0 of [0,L-16.5])el('rect',{x:x0*S,y:Y(20.16),width:16.5*S,height:40.32*S,class:'mark'},svg);
 for(const y of [20.16,9.16,-9.16,-20.16])el('line',{x1:0,y1:Y(y),x2:L*S,y2:Y(y),class:'lane'},svg);
 const hull=el('polygon',{class:'hull'},svg),lines=[0,1,2].map(()=>el('line',{y1:0,y2:W*S,class:'line'},svg));
 const n=c.frames[0].a.length,m=c.frames[0].d.length;
 const A=[...Array(n)].map(()=>el('circle',{r:5,class:'att'},svg)),D=[...Array(m)].map(()=>el('circle',{r:5,class:'dfd'},svg));
 const ball=el('circle',{r:4,class:'ball'},svg),ring=el('circle',{r:11,class:'ring',visibility:'hidden'},svg);
 const ctl=document.createElement('div');ctl.className='ctl';
 ctl.innerHTML=`<button data-a="play">Play</button><button data-a="moment">Go to moment</button><input type="range" min="0" max="${c.frames.length-1}" step="0.05" value="0" list="tk-${c.id}"><datalist id="tk-${c.id}"><option value="${c.moment}"></option></datalist>`;
 card.appendChild(ctl);
 const now=document.createElement('div');now.className='now';card.appendChild(now);
 const dl=document.createElement('dl');dl.innerHTML=Object.entries(c.labels).map(([k,v])=>`<dt>${k} at the moment</dt><dd>${v}</dd>`).join('');card.appendChild(dl);
 const fs=document.createElement('fieldset');fs.innerHTML=`<label><input type="checkbox" data-k="phase"> Phase right at the moment</label><input type="text" data-k="note" placeholder="Note (optional): what it should be">`;card.appendChild(fs);
 const range=ctl.querySelector('input');
 function draw(pos){const k=Math.min(Math.floor(pos),c.frames.length-1),u=pos-k,f=c.frames[k],g=c.frames[Math.min(k+1,c.frames.length-1)];
  f.a.forEach((p,j)=>{const q=lerp(p,g.a[j],u);A[j].setAttribute('visibility',q?'visible':'hidden');if(q){A[j].setAttribute('cx',X(q[0]));A[j].setAttribute('cy',Y(q[1]));}});
  f.d.forEach((p,j)=>{const q=lerp(p,g.d[j],u);D[j].setAttribute('visibility',q?'visible':'hidden');if(q){D[j].setAttribute('cx',X(q[0]));D[j].setAttribute('cy',Y(q[1]));}});
  const b=lerp(f.b,g.b,u);ball.setAttribute('visibility',b?'visible':'hidden');if(b){ball.setAttribute('cx',X(b[0]));ball.setAttribute('cy',Y(b[1]));ring.setAttribute('cx',X(b[0]));ring.setAttribute('cy',Y(b[1]));}
  const at=Math.abs(pos-c.moment)<0.5;ring.setAttribute('visibility',at&&b?'visible':'hidden');
  hull.setAttribute('points',f.h.map(p=>`${X(p[0])},${Y(p[1])}`).join(' '));
  f.l.forEach((x,j)=>{lines[j].setAttribute('visibility',x==null?'hidden':'visible');if(x!=null){lines[j].setAttribute('x1',X(x));lines[j].setAttribute('x2',X(x));}});
  const t=f.t+(g.t-f.t)*u;now.innerHTML=`<span>${t>=0?'+':''}${t.toFixed(1)} s</span><span>Ponta phase now: <b>${f.ph??'none'}</b></span><span class="badge" style="visibility:${at?'visible':'hidden'}">Labelled moment</span>`;}
 let pos=0,playing=false,last=0;const hz=1/((c.frames[1]?.t??0.2)-c.frames[0].t||0.2);
 function tick(ts){if(!playing)return;if(last)pos+=(ts-last)/1000*hz;last=ts;if(pos>=c.frames.length-1){pos=c.frames.length-1;playing=false;btn.textContent='Replay';}
  range.value=pos;draw(pos);if(playing)requestAnimationFrame(tick);}
 const btn=ctl.querySelector('[data-a=play]');
 btn.onclick=()=>{if(playing){playing=false;btn.textContent='Play';return;}if(pos>=c.frames.length-1)pos=0;playing=true;last=0;btn.textContent='Pause';requestAnimationFrame(tick);};
 ctl.querySelector('[data-a=moment]').onclick=()=>{playing=false;btn.textContent='Play';pos=c.moment;range.value=pos;draw(pos);};
 range.oninput=()=>{playing=false;btn.textContent='Play';pos=+range.value;draw(pos);};
 draw(0);document.getElementById('grid').appendChild(card);});
const boxes=[...document.querySelectorAll('input[type=checkbox]')];
function tally(){document.getElementById('count').textContent=`Phases right: ${boxes.filter(b=>b.checked).length} of ${boxes.length}`;}
boxes.forEach(b=>b.addEventListener('change',tally));tally();
document.getElementById('copy').addEventListener('click',()=>{const out=[...document.querySelectorAll('.card')].map(c=>({id:c.dataset.id,
phase:c.querySelector('[data-k=phase]').checked,note:c.querySelector('[data-k=note]').value.trim()}));
const txt=JSON.stringify(out),box=document.getElementById('out');box.value=txt;box.hidden=false;box.select();
const done=m=>{document.getElementById('copied').textContent=m;};
try{navigator.clipboard.writeText(txt).then(()=>done('Copied. Paste it back to Claude.'),()=>done('Copy blocked: select the text below and copy it.'));}catch(e){done('Copy blocked: select the text below and copy it.');}});
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--total", type=int, default=50)
    ap.add_argument("--replay", action="store_true", help="write the phase replay page instead of still cards")
    ap.add_argument("--before-s", type=float, default=5.0)
    ap.add_argument("--after-s", type=float, default=2.0)
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
            cards.append(replay(key, i, m, res, tf, lines, args.before_s, args.after_s) if args.replay
                         else card(key, i, m, res, tf, lines))
    cards = cards[: args.total]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.replay:
        page = (REPLAY_PAGE.replace("__BEFORE__", f"{args.before_s:g}").replace("__AFTER__", f"{args.after_s:g}")
                .replace("__DATA__", json.dumps(cards, separators=(",", ":")).replace("</", "<\\/")))
    else:
        page = PAGE.replace("__CARDS__", "".join(cards))
    args.out.write_text(page)
    print(f"wrote {len(cards)} cards to {args.out}")


if __name__ == "__main__":
    main()
