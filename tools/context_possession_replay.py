"""
context_possession_replay.py — a blind replay check of possession: P1 against the league's possession.

On the IDSSE matches P1 and the league's frame-level possession disagree on about 12% of in-play time.
The league's possession is anchored to its event log, so neither is ground truth; a football judge
decides. This writes a replay page of moments for the founder to judge **blind** (no label shown):
at the marked moment, who has the ball (home, away, or nobody yet: the ball in flight or loose)?

- ``--disagree`` moments where P1 and the league name different teams for at least 1 s (the middle
  of each run), and ``--agree`` control moments where they name the same team; shuffled together,
  drawn evenly from the seven matches with a fixed seed;
- each case replays ``--before-s`` before and ``--after-s`` after the moment, home in blue attacking
  left to right;
- the answer key (P1's and the league's team per case) goes to ``--key``, never into the page.

Both files hold sample-data positions, so they are written outside the repository.
``--score`` reads the founder's results and the key and prints how often each source agrees.

    python tools/context_possession_replay.py --idsse ~/ponta-reference-data/idsse --cache /tmp/ponta-context/idsse-cache \
        --out /tmp/ponta-context/possession_replay.html --key /tmp/ponta-context/possession_replay_key.json
    python tools/context_possession_replay.py --score results.json --key /tmp/ponta-context/possession_replay_key.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from matrix_engine.sample_data import repo_root
from plaiq_engine.context.possession import PossessionPackage
from plaiq_engine.context.registry import Registry

sys.path.insert(0, str(Path(__file__).resolve().parent))
from idsse import MATCH_IDS, load_idsse_match

HZ = 5.0


def _outside(path: Path) -> bool:
    root = repo_root()
    return not (path.resolve() == root or root in path.resolve().parents)


def _r(v: float) -> float:
    return round(float(v), 1)


def runs(mask: np.ndarray, per: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for i in range(len(mask) + 1):
        on = i < len(mask) and mask[i] and (start is None or per[i] == per[start])
        if on and start is None:
            start = i
        elif not on and start is not None:
            out.append((start, i - 1))
            start = i if i < len(mask) and mask[i] else None
    return out


def case(mid: str, m, grid, i: int, before_s: float, after_s: float) -> dict:
    L, W = m.pitch.length_m, m.pitch.width_m
    per = grid["period"].to_numpy()
    idx = [j for j in range(i - int(before_s * HZ), i + int(after_s * HZ) + 1) if 0 <= j < len(grid) and per[j] == per[i]]
    nf = grid["native_frame"].to_numpy()[idx]
    tr = m.tracking.loc[m.tracking["frame"].isin(nf), ["frame", "player_id", "team", "x", "y"]]
    ball = m.ball.set_index("frame")
    ids = {t: sorted(tr.loc[tr["team"] == t, "player_id"].unique()) for t in ("home", "away")}
    frames = []
    for k, f in enumerate(nf):
        at = tr.loc[tr["frame"] == f].set_index("player_id")
        row = {"t": round((idx[k] - i) / HZ, 2)}
        for t, key in (("home", "a"), ("away", "d")):
            row[key] = [[_r(at.at[p, "x"] + L / 2), _r(at.at[p, "y"])] if p in at.index else None for p in ids[t]]
        row["b"] = [_r(ball.at[f, "x"] + L / 2), _r(ball.at[f, "y"])] if f in ball.index else None
        frames.append(row)
    clock = grid["clock_s"].iloc[i]
    return {"id": f"{mid}-{i}", "L": L, "W": W, "moment": idx.index(i), "frames": frames,
            "header": f"Match {MATCH_IDS.index(mid) + 1} · period {int(per[i])} · {int(clock // 60)}:{int(clock % 60):02d}"}


def build(args) -> None:
    rng = np.random.default_rng(20261118)
    per_match_d = int(np.ceil(args.disagree / len(MATCH_IDS)))
    per_match_a = int(np.ceil(args.agree / len(MATCH_IDS)))
    cases, key = [], {}
    for mid in MATCH_IDS:
        m, off = load_idsse_match(args.idsse, mid, args.cache)
        res = Registry([PossessionPackage()]).run(m)
        g = res.frames
        ob = g[["native_frame"]].merge(off.ball.drop(columns=["period"]).rename(columns={"frame": "native_frame"}), on="native_frame", how="left")
        live, team = (ob["ball_status"] == 1).to_numpy(), ob["ball_team"].astype(object).to_numpy()
        p1 = g["p1_possession"].astype(object).to_numpy()
        per = g["period"].to_numpy()
        named = live & np.isin(team, ["home", "away"]) & np.isin(p1, ["home", "away"])
        dis = [(s, e) for s, e in runs(named & (p1 != team), per) if (e - s + 1) / HZ >= 1.0]
        agr = [(s, e) for s, e in runs(named & (p1 == team), per) if (e - s + 1) / HZ >= 3.0]
        picks = [("disagree", r) for r in rng.permutation(len(dis))[:per_match_d].tolist()]
        picks += [("agree", r) for r in rng.permutation(len(agr))[:per_match_a].tolist()]
        for kind, r in picks:
            s, e = (dis if kind == "disagree" else agr)[r]
            i = (s + e) // 2
            c = case(mid, m, g, i, args.before_s, args.after_s)
            cases.append(c)
            key[c["id"]] = {"kind": kind, "p1": p1[i], "league": team[i], "run_s": round((e - s + 1) / HZ, 1)}
        print(mid, len(dis), "disagreement runs,", len(agr), "agreement runs", flush=True)
    pools = {kind: [c for c in cases if key[c["id"]]["kind"] == kind] for kind in ("disagree", "agree")}
    chosen = [pools["disagree"][k] for k in rng.permutation(len(pools["disagree"]))[:args.disagree]]
    chosen += [pools["agree"][k] for k in rng.permutation(len(pools["agree"]))[:args.agree]]
    chosen = [chosen[k] for k in rng.permutation(len(chosen))]
    page = PAGE.replace("__BEFORE__", f"{args.before_s:g}").replace("__AFTER__", f"{args.after_s:g}") \
               .replace("__DATA__", json.dumps(chosen, separators=(",", ":")).replace("</", "<\\/"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page)
    args.key.write_text(json.dumps({c["id"]: key[c["id"]] for c in chosen}, indent=1))
    print(f"wrote {len(chosen)} cases to {args.out}; key to {args.key}")


def score(args) -> None:
    calls = {r["id"]: r for r in json.loads(Path(args.score).read_text())}
    key = json.loads(args.key.read_text())
    out = {"disagree": {"p1": 0, "league": 0, "nobody": 0, "n": 0}, "agree": {"both": 0, "nobody": 0, "n": 0}}
    for cid, k in key.items():
        c = calls.get(cid, {}).get("call")
        if not c:
            continue
        o = out[k["kind"]]
        o["n"] += 1
        if c == "nobody":
            o["nobody"] += 1
        elif k["kind"] == "disagree":
            o["p1"] += c == k["p1"]
            o["league"] += c == k["league"]
        else:
            o["both"] += c == k["p1"]
    print(json.dumps(out, indent=1))


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Possession replay check</title><style>
:root{--bg:#f6f7f9;--ink:#14171c;--muted:#5b6370;--card:#fff;--pitch:#e8efe6;--mark:#9aa79a;--home:#2f6fdf;--away:#d24a4a;--ball:#111;--hi:#e0a100}
@media (prefers-color-scheme:dark){:root{--bg:#0e1116;--ink:#e8ebf0;--muted:#9aa3b0;--card:#171b22;--pitch:#1d2a22;--mark:#4b5b4f;--home:#6c9cff;--away:#ff7b7b;--ball:#fff;--hi:#ffc83d}}
body{margin:0;padding:16px;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}
h1{font-size:18px;margin:0 0 4px} p{color:var(--muted);margin:0 0 12px;max-width:75ch}
.bar{position:sticky;top:0;z-index:2;background:var(--bg);padding:8px 0;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,460px),1fr));gap:12px}
.card{background:var(--card);border-radius:8px;padding:10px} header{font-weight:600;margin-bottom:6px}
svg{width:100%;height:auto;display:block} .pitch{fill:var(--pitch)} .mark{stroke:var(--mark);stroke-width:1.5;fill:none}
.home{fill:var(--home)} .away{fill:var(--away)} .ball{fill:var(--ball);stroke:var(--card);stroke-width:1.5} .ring{fill:none;stroke:var(--hi);stroke-width:3}
.ctl{display:flex;gap:8px;align-items:center;margin:6px 0} .ctl input[type=range]{flex:1}
.now{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);min-height:1.5em}
.badge{background:var(--hi);color:#111;border-radius:4px;padding:0 6px;font-weight:600}
fieldset{border:0;padding:0;display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:6px} button{font:inherit;padding:4px 10px}
input[type=text]{flex:1;min-width:12ch;font:inherit;padding:3px 6px}
</style></head><body>
<h1>Possession replay check</h1>
<p>Each card replays the __BEFORE__ seconds before a moment and __AFTER__ seconds after it. Home is <b style="color:var(--home)">blue</b>, attacking left to right; away is <b style="color:var(--away)">red</b>. No label is shown. Press Play, watch, then say <b>who has the ball at the marked moment</b> (the ball is ringed when you reach it): home, away, or nobody yet (the ball is in flight or loose).</p>
<div class="bar"><span id="count"></span><button id="copy">Copy results</button><span id="copied" aria-live="polite"></span></div>
<textarea id="out" hidden readonly rows="4" style="width:100%;font:12px ui-monospace,monospace"></textarea>
<div class="grid" id="grid"></div>
<script>
const CASES=__DATA__;const S=4,NS='http://www.w3.org/2000/svg';
function el(t,a,p){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e;}
function lerp(a,b,u){return a&&b?[a[0]+(b[0]-a[0])*u,a[1]+(b[1]-a[1])*u]:(u<.5?a:b);}
CASES.forEach((c,ci)=>{
 const card=document.createElement('article');card.className='card';card.dataset.id=c.id;
 const W=c.W,L=c.L,X=x=>x*S,Y=y=>(W/2-y)*S;
 card.innerHTML=`<header>Case ${ci+1} · ${c.header}</header>`;
 const svg=el('svg',{viewBox:`0 0 ${L*S} ${W*S}`,role:'img','aria-label':'Replay of case '+(ci+1)},card);
 el('rect',{width:L*S,height:W*S,class:'pitch'},svg);el('line',{x1:L*S/2,y1:0,x2:L*S/2,y2:W*S,class:'mark'},svg);
 el('circle',{cx:L*S/2,cy:W*S/2,r:9.15*S,class:'mark'},svg);
 for(const x0 of [0,L-16.5])el('rect',{x:x0*S,y:Y(20.16),width:16.5*S,height:40.32*S,class:'mark'},svg);
 const n=c.frames[0].a.length,m=c.frames[0].d.length;
 const A=[...Array(n)].map(()=>el('circle',{r:5,class:'home'},svg)),D=[...Array(m)].map(()=>el('circle',{r:5,class:'away'},svg));
 const ball=el('circle',{r:4,class:'ball'},svg),ring=el('circle',{r:11,class:'ring',visibility:'hidden'},svg);
 const ctl=document.createElement('div');ctl.className='ctl';
 ctl.innerHTML=`<button data-a="play">Play</button><button data-a="moment">Go to moment</button><input type="range" min="0" max="${c.frames.length-1}" step="0.05" value="0" list="tk-${ci}"><datalist id="tk-${ci}"><option value="${c.moment}"></option></datalist>`;
 card.appendChild(ctl);
 const now=document.createElement('div');now.className='now';card.appendChild(now);
 const fs=document.createElement('fieldset');
 fs.innerHTML=['home','away','nobody'].map(v=>`<label><input type="radio" name="c${ci}" value="${v}"> ${v==='nobody'?'Nobody yet':v==='home'?'Home (blue)':'Away (red)'}</label>`).join('')+`<input type="text" data-k="note" placeholder="Note (optional)">`;
 card.appendChild(fs);
 const range=ctl.querySelector('input');
 function draw(pos){const k=Math.min(Math.floor(pos),c.frames.length-1),u=pos-k,f=c.frames[k],g=c.frames[Math.min(k+1,c.frames.length-1)];
  f.a.forEach((p,j)=>{const q=lerp(p,g.a[j],u);A[j].setAttribute('visibility',q?'visible':'hidden');if(q){A[j].setAttribute('cx',X(q[0]));A[j].setAttribute('cy',Y(q[1]));}});
  f.d.forEach((p,j)=>{const q=lerp(p,g.d[j],u);D[j].setAttribute('visibility',q?'visible':'hidden');if(q){D[j].setAttribute('cx',X(q[0]));D[j].setAttribute('cy',Y(q[1]));}});
  const b=lerp(f.b,g.b,u);ball.setAttribute('visibility',b?'visible':'hidden');if(b){ball.setAttribute('cx',X(b[0]));ball.setAttribute('cy',Y(b[1]));ring.setAttribute('cx',X(b[0]));ring.setAttribute('cy',Y(b[1]));}
  const at=Math.abs(pos-c.moment)<0.5;ring.setAttribute('visibility',at&&b?'visible':'hidden');
  const t=f.t+(g.t-f.t)*u;now.innerHTML=`<span>${t>=0?'+':''}${t.toFixed(1)} s</span>`+(at?'<span class="badge">The moment</span>':'');}
 let pos=0,playing=false,last=0;const hz=1/((c.frames[1]?.t??0.2)-c.frames[0].t||0.2);
 function tick(ts){if(!playing)return;if(last)pos+=(ts-last)/1000*hz;last=ts;if(pos>=c.frames.length-1){pos=c.frames.length-1;playing=false;btn.textContent='Replay';}
  range.value=pos;draw(pos);if(playing)requestAnimationFrame(tick);}
 const btn=ctl.querySelector('[data-a=play]');
 btn.onclick=()=>{if(playing){playing=false;btn.textContent='Play';return;}if(pos>=c.frames.length-1)pos=0;playing=true;last=0;btn.textContent='Pause';requestAnimationFrame(tick);};
 ctl.querySelector('[data-a=moment]').onclick=()=>{playing=false;btn.textContent='Play';pos=c.moment;range.value=pos;draw(pos);};
 range.oninput=()=>{playing=false;btn.textContent='Play';pos=+range.value;draw(pos);};
 draw(0);document.getElementById('grid').appendChild(card);});
function tally(){const done=[...document.querySelectorAll('.card')].filter(c=>c.querySelector('input[type=radio]:checked')).length;
 document.getElementById('count').textContent=`Judged: ${done} of ${CASES.length}`;}
document.addEventListener('change',tally);tally();
document.getElementById('copy').addEventListener('click',()=>{const out=[...document.querySelectorAll('.card')].map(c=>({id:c.dataset.id,
call:(c.querySelector('input[type=radio]:checked')||{}).value||null,note:c.querySelector('[data-k=note]').value.trim()}));
const txt=JSON.stringify(out),box=document.getElementById('out');box.value=txt;box.hidden=false;box.select();
const done=m=>{document.getElementById('copied').textContent=m;};
try{navigator.clipboard.writeText(txt).then(()=>done('Copied. Paste it back to Claude.'),()=>done('Copy blocked: select the text below and copy it.'));}catch(e){done('Copy blocked: select the text below and copy it.');}});
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--idsse", type=Path)
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--disagree", type=int, default=40)
    ap.add_argument("--agree", type=int, default=10)
    ap.add_argument("--before-s", type=float, default=5.0)
    ap.add_argument("--after-s", type=float, default=2.0)
    ap.add_argument("--score", type=Path, help="the founder's copied results (JSON); prints agreement per source")
    args = ap.parse_args()
    for p in [args.key] + [x for x in (args.out, args.cache) if x]:
        if not _outside(p):
            sys.exit(f"refusing to write sample-data positions inside the repository: {p}")
    if args.score:
        score(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
