"""
context_states_review.py — the founder's episode review (Stage D, ``ponta-blueprint/context-states-task.md``, deliverable 5).

P8 cuts every match into episodes and writes a summary for each from code (``plaiq_engine/context/states.py``). This
writes a replay page of episodes for the founder to judge: **do they start and end where a passage of play starts and
ends**, and **is the summary accurate**? The summary is shown (the review judges it), so this is not blind. The founder's
calls are one expert's reading, an indication beside SkillCorner's phases of play, never ground truth.

- Episodes come from the German league's matches (IDSSE: full-pitch tracking, every player in view), ``--per-phase``
  from each P4 phase, with a fixed seed, at least 20 s apart within a match, between ``--min-s`` and ``--max-s`` long;
- each card replays ``--before-s`` before the episode to ``--after-s`` after it;
- the answer key (each case's episode row, without positions) goes to ``--key``; the page holds positions.

Both files hold sample-data positions or derived rows, so they are written outside the repository. ``--score`` reads the
founder's results and the key and prints the tallies by phase.

    python tools/context_states_review.py --idsse ~/ponta-reference-data/idsse --cache /tmp/ponta-context/idsse-cache \
        --out /tmp/ponta-context/episode_review.html --key /tmp/ponta-context/episode_review_key.json
    python tools/context_states_review.py --score results.json --key /tmp/ponta-context/episode_review_key.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from matrix_engine.sample_data import repo_root
from plaiq_engine.match_analysis import MatchAnalysis

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_golden import registry
from context_possession_replay import HZ, case
from idsse import MATCH_IDS, load_idsse_match

PHASES = ("build_up", "create", "finish", "direct", "quick_break", "transition", "set_play", "chaotic")
OPTS = [["good", "Good: starts and ends with the passage"], ["starts_early", "Starts too early"], ["starts_late", "Starts too late"],
        ["ends_early", "Ends too early"], ["ends_late", "Ends too late"], ["split", "Should be two episodes"],
        ["merge", "Should be part of the next or previous one"], ["unclear", "Cannot tell"]]
SUM = [["accurate", "Accurate"], ["partly", "Partly accurate"], ["wrong", "Wrong"], ["unclear", "Cannot tell"]]


def _outside(path: Path) -> bool:
    root = repo_root()
    return not (path.resolve() == root or root in path.resolve().parents)


def build(args) -> None:
    rng = np.random.default_rng(20261123)
    pool = []
    for mid in MATCH_IDS[: args.matches]:
        m, _ = load_idsse_match(args.idsse, mid, args.cache)
        if "End X" not in m.events:  # the league's event table has no end points; Plaiq's event tagging expects the columns
            m = dataclasses.replace(m, events=m.events.assign(**{"End X": np.nan, "End Y": np.nan}))
        res = registry().run(m, extras={"analysis": MatchAnalysis(m)})
        st = res.outputs["p8"].segments["states"]
        st = st.loc[(st["duration_s"] >= args.min_s) & (st["duration_s"] <= args.max_s)]
        ts = res.frames["frame_ts"].to_numpy(float)
        for r in st.to_dict("records"):
            pool.append((mid, m, res.frames, r, int(np.searchsorted(ts, r["start_s"])), int(np.searchsorted(ts, r["end_s"]) - 1)))
        print(mid, len(st), "episodes in range", flush=True)
    cases, key = [], []
    for ph in PHASES:
        cands = [p for p in pool if p[3]["phase"] == ph]
        rng.shuffle(cands)
        taken: list = []
        for mid, m, grid, r, a, b in cands:
            if len(taken) >= args.per_phase:
                break
            if any(t[0] == mid and abs(t[3]["start_s"] - r["start_s"]) < 20 for t in taken):
                continue
            taken.append((mid, m, grid, r, a, b))
        for mid, m, grid, r, a, b in taken:
            c = case(mid, m, grid, a, args.before_s, (b - a) / HZ + args.after_s)
            c["run_from"], c["run_to"] = c["moment"], c["moment"] + (b - a)
            c["summary"] = r["summary"]
            cases.append(c)
            key.append({"id": c["id"], **{k: r[k] for k in ("state_id", "phase", "fifa_phase", "duration_s", "end_reason", "summary")}})
    order = rng.permutation(len(cases))
    cases, key = [cases[i] for i in order], [key[i] for i in order]
    html = (PAGE.replace("__DATA__", json.dumps(cases)).replace("__OPTS__", json.dumps(OPTS)).replace("__SUM__", json.dumps(SUM))
            .replace("__BEFORE__", f"{args.before_s:g}").replace("__AFTER__", f"{args.after_s:g}"))
    args.out.write_text(html)
    args.key.write_text(json.dumps(key, indent=1))
    print(len(cases), "cases;", Counter(k["phase"] for k in key))


def score(args) -> None:
    calls = {c["id"]: c for c in json.loads(args.score.read_text())}
    key = json.loads(args.key.read_text())
    rows = [(k, calls.get(k["id"], {})) for k in key]
    judged = [(k, c) for k, c in rows if c.get("call") or c.get("summary")]
    print(len(judged), "of", len(key), "judged (one expert's reading: agreement with the founder's call, not accuracy)")
    print("boundaries:", dict(Counter(c.get("call") for _, c in judged)))
    print("summaries:", dict(Counter(c.get("summary") for _, c in judged)))
    for ph in PHASES:
        sub = [(k, c) for k, c in judged if k["phase"] == ph]
        if sub:
            print(f"  {ph}: boundaries good {sum(c.get('call') == 'good' for _, c in sub)}/{len(sub)}, "
                  f"summary accurate {sum(c.get('summary') == 'accurate' for _, c in sub)}/{len(sub)}")
    for k, c in judged:
        if c.get("note"):
            print("-", k["phase"], k["state_id"], "|", c.get("call"), c.get("summary"), "|", c["note"])


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Episode review</title><style>
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
fieldset{border:0;padding:0;display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,200px),1fr));gap:4px 12px;margin-top:6px} button{font:inherit;padding:4px 10px}
.sum{color:var(--ink);margin:0 0 6px;font-size:13px} legend{font-weight:600;grid-column:1/-1;padding:0}
input[type=text]{width:100%;font:inherit;padding:3px 6px;box-sizing:border-box;margin-top:6px}
</style></head><body>
<h1>Episode review</h1>
<p>Each card replays one episode, from __BEFORE__ second before it starts until __AFTER__ second after it ends. Home is <b style="color:var(--home)">blue</b>, attacking left to right; away is <b style="color:var(--away)">red</b>. The <span class="badge">Episode</span> badge shows while the episode lasts, and the summary Ponta wrote for it is above the pitch. Judge two things: <b>does it start and end where a passage of play starts and ends</b>, and <b>is the summary accurate</b> (nothing wrong, nothing invented)? Your calls are one reading among others, not a verdict.</p>
<div class="bar"><span id="count"></span><button id="copy">Copy results</button><span id="copied" aria-live="polite"></span></div>
<textarea id="out" hidden readonly rows="4" style="width:100%;font:12px ui-monospace,monospace"></textarea>
<div class="grid" id="grid"></div>
<script>
const CASES=__DATA__;const OPTS=__OPTS__;const SUM=__SUM__;const S=4,NS='http://www.w3.org/2000/svg';
function el(t,a,p){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e;}
function lerp(a,b,u){return a&&b?[a[0]+(b[0]-a[0])*u,a[1]+(b[1]-a[1])*u]:(u<.5?a:b);}
CASES.forEach((c,ci)=>{
 const card=document.createElement('article');card.className='card';card.dataset.id=c.id;
 const W=c.W,L=c.L,X=x=>x*S,Y=y=>(W/2-y)*S;
 card.innerHTML=`<header>Case ${ci+1} · ${c.header}</header><p class="sum">${c.summary}</p>`;
 const svg=el('svg',{viewBox:`0 0 ${L*S} ${W*S}`,role:'img','aria-label':'Replay of case '+(ci+1)},card);
 el('rect',{width:L*S,height:W*S,class:'pitch'},svg);el('line',{x1:L*S/2,y1:0,x2:L*S/2,y2:W*S,class:'mark'},svg);
 el('circle',{cx:L*S/2,cy:W*S/2,r:9.15*S,class:'mark'},svg);
 for(const x0 of [0,L-16.5])el('rect',{x:x0*S,y:Y(20.16),width:16.5*S,height:40.32*S,class:'mark'},svg);
 const n=c.frames[0].a.length,m=c.frames[0].d.length;
 const A=[...Array(n)].map(()=>el('circle',{r:5,class:'home'},svg)),D=[...Array(m)].map(()=>el('circle',{r:5,class:'away'},svg));
 const ball=el('circle',{r:4,class:'ball'},svg);
 const ctl=document.createElement('div');ctl.className='ctl';
 ctl.innerHTML=`<button data-a="play">Play</button><input type="range" min="0" max="${c.frames.length-1}" step="0.05" value="0" aria-label="Replay position">`;
 card.appendChild(ctl);
 const now=document.createElement('div');now.className='now';card.appendChild(now);
 const fs=document.createElement('fieldset');
 fs.innerHTML='<legend>Boundaries</legend>'+OPTS.map(([v,name])=>`<label><input type="radio" name="c${ci}" value="${v}"> ${name}</label>`).join('');
 card.appendChild(fs);
 const fs2=document.createElement('fieldset');
 fs2.innerHTML='<legend>Summary</legend>'+SUM.map(([v,name])=>`<label><input type="radio" name="s${ci}" value="${v}"> ${name}</label>`).join('');
 card.appendChild(fs2);
 const note=document.createElement('input');note.type='text';note.dataset.k='note';note.placeholder='Note (optional)';card.appendChild(note);
 const range=ctl.querySelector('input');
 function draw(pos){const k=Math.min(Math.floor(pos),c.frames.length-1),u=pos-k,f=c.frames[k],g=c.frames[Math.min(k+1,c.frames.length-1)];
  f.a.forEach((p,j)=>{const q=lerp(p,g.a[j],u);A[j].setAttribute('visibility',q?'visible':'hidden');if(q){A[j].setAttribute('cx',X(q[0]));A[j].setAttribute('cy',Y(q[1]));}});
  f.d.forEach((p,j)=>{const q=lerp(p,g.d[j],u);D[j].setAttribute('visibility',q?'visible':'hidden');if(q){D[j].setAttribute('cx',X(q[0]));D[j].setAttribute('cy',Y(q[1]));}});
  const b=lerp(f.b,g.b,u);ball.setAttribute('visibility',b?'visible':'hidden');if(b){ball.setAttribute('cx',X(b[0]));ball.setAttribute('cy',Y(b[1]));}
  const on=pos>=c.run_from&&pos<=c.run_to;
  const t=f.t+(g.t-f.t)*u;now.innerHTML=`<span>${t>=0?'+':''}${t.toFixed(1)} s</span>`+(on?'<span class="badge">Episode</span>':'');}
 let pos=0,playing=false,last=0;const hz=1/((c.frames[1]?.t??0.2)-c.frames[0].t||0.2);
 const btn=ctl.querySelector('[data-a=play]');
 function tick(ts){if(!playing)return;if(last)pos+=(ts-last)/1000*hz;last=ts;if(pos>=c.frames.length-1){pos=c.frames.length-1;playing=false;btn.textContent='Replay';}
  range.value=pos;draw(pos);if(playing)requestAnimationFrame(tick);}
 btn.onclick=()=>{if(playing){playing=false;btn.textContent='Play';return;}if(pos>=c.frames.length-1)pos=0;playing=true;last=0;btn.textContent='Pause';requestAnimationFrame(tick);};
 range.oninput=()=>{playing=false;btn.textContent='Play';pos=+range.value;draw(pos);};
 draw(0);document.getElementById('grid').appendChild(card);});
function tally(){const done=[...document.querySelectorAll('.card')].filter(c=>c.querySelector('input[name^=c]:checked')&&c.querySelector('input[name^=s]:checked')).length;
 document.getElementById('count').textContent=`Judged: ${done} of ${CASES.length}`;}
document.addEventListener('change',tally);tally();
document.getElementById('copy').addEventListener('click',()=>{const out=[...document.querySelectorAll('.card')].map(c=>({id:c.dataset.id,
call:(c.querySelector('input[name^=c]:checked')||{}).value||null,summary:(c.querySelector('input[name^=s]:checked')||{}).value||null,note:c.querySelector('[data-k=note]').value.trim()}));
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
    ap.add_argument("--matches", type=int, default=3, help="how many of the league's matches to draw from")
    ap.add_argument("--per-phase", type=int, default=5)
    ap.add_argument("--min-s", type=float, default=3.0)
    ap.add_argument("--max-s", type=float, default=30.0)
    ap.add_argument("--before-s", type=float, default=1.0)
    ap.add_argument("--after-s", type=float, default=1.0)
    ap.add_argument("--score", type=Path, help="the founder's copied results (JSON); prints the tallies")
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
