"""Training / evaluation report of the RL logs: for every mission one folder with

    <Mission>_train_dashboard.html   interactive training dashboard (open in any browser, works offline)
    <Mission>_eval_dashboard.html    interactive evaluation dashboard
    <Mission>_train_overview.png     static image of the training view (slides / reports)
    <Mission>_eval_overview.png      static image of the evaluation view
    SUMMARY.md                       key numbers in words
    data/train_log.csv, eval_log.csv the logs that were used (copies)

and a README.md with one line per mission.

    python scripts/plot_training.py                               # every mission in checkpoints/ -> outputs/Debug_Train_Eval/
    python scripts/plot_training.py --mission mission1            # one mission
    python scripts/plot_training.py --checkpoints arsip_v5/checkpoints --out outputs/Debug_Train_Eval_v5
    python scripts/plot_training.py --no-png                      # dashboards and summaries only

Input: checkpoints/<mission>/train_log.csv (one row per training episode) and eval_log.csv (one row per evaluation
of a checkpoint; a row with policy = "reference" is the reference controller alone, measured before training).
Old logs without the columns policy / s_per_fruit / sim_time_s work too. The script can be run while a training is
still going on.

The dashboards embed Chart.js (MIT licence) from scripts/vendor/chart.umd.min.js so that they work offline; without
that file they load it from the internet instead.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHART_JS = Path(__file__).resolve().parent / "vendor" / "chart.umd.min.js"
CHART_CDN = "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"

# ----------------------------------------------------------------------------- colours
# One colour per failure cause, the same in every chart, dashboard and mission (colour follows the cause, not its
# rank). The order is also the stacking order; neighbours were checked for colour-blind separation.
INK, INK2, MUTED, GRID, AXIS, PAGE, CARD = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f6f5f1", "#ffffff"
BLUE, GREY, OK_COL, ZERO_COL = "#2a78d6", "#b4b2a9", "#1baf7a", "#d85a30"
CAUSES = [("leaf_uprooted", "#2a78d6"), ("route_blocked", "#eb6834"), ("plant_hit", "#6250d6"), ("fruit_dropped", "#eda100"),
          ("fruit_hit", "#e87ba4"), ("fruit_knocked_off", "#e87ba4"), ("timeout", "#008300"), ("grasp_miss", "#e34948")]
CAUSE_COL = dict(CAUSES)
COMPONENT_NAMES = {"cross_track": "Cross-track", "leaf_damage": "Leaf damage", "fail": "Failure penalty"}


def nice(name: str) -> str:
    return name.replace("_", " ").strip().capitalize() if name else "Unknown"


def kfmt(v: float) -> str:
    """12000 -> '12k', 19300 -> '19.3k', 850 -> '850'."""
    return f"{v / 1000:.1f}".rstrip("0").rstrip(".") + "k" if abs(v) >= 1000 else f"{v:g}"


def approx(v: float) -> str:
    """Rounded for "about so many steps": 599,912 -> '600k', 59,800 -> '60k', 10,070 -> '10k'."""
    return kfmt(round(v, -3)) if v >= 10000 else kfmt(round(v, -2))


def num(x, default=np.nan) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


# ----------------------------------------------------------------------------- reading + statistics
def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def roll(y: np.ndarray, w: int, fn) -> np.ndarray:
    """fn (mean / median) over the last `w` values that are not NaN."""
    out = np.full(len(y), np.nan)
    for i in range(len(y)):
        seg = y[max(0, i - w + 1):i + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg):
            out[i] = fn(seg)
    return out


def bin_width(span: float, max_bins: int = 12) -> float:
    for e in range(0, 9):
        for m in (1, 2, 2.5, 5):
            if span / (m * 10 ** e) <= max_bins:
                return m * 10 ** e
    return span


def train_stats(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: num(r.get("timesteps"), 0))
    n = len(rows)
    steps = np.array([num(r.get("timesteps"), 0) for r in rows])
    ret = np.array([num(r.get("episode_return")) for r in rows])
    ok = np.array([str(r.get("success", "")).strip().lower() == "true" for r in rows])
    placed = np.array([num(r.get("n_placed"), 0) for r in rows])
    targets = np.array([num(r.get("n_targets"), 0) for r in rows])
    n_t = int(Counter(targets.astype(int)).most_common(1)[0][0]) if n else 1
    cause = ["success" if o else ((r.get("failure") or "unknown").split("@")[0] or "unknown") for r, o in zip(rows, ok)]
    w = int(min(100, max(5, n // 5)))                       # rolling window (100 episodes for long logs)
    k = 100 if n >= 200 else n // 2                         # "first vs last k episodes"
    i0 = min(n - 1, max(min(w, n) - 1, 4 if n > 5 else 0))
    idx = np.unique(np.linspace(i0, n - 1, min(60, n - i0)).round().astype(int))
    sr, rm = roll(ok.astype(float) * 100, w, np.mean), roll(ret, w, np.median)
    # time per harvested fruit (v6 logs)
    sim = np.array([num(r.get("sim_time_s")) for r in rows]) + np.nan_to_num(np.array([num(r.get("compute_s")) for r in rows]))
    with np.errstate(divide="ignore", invalid="ignore"):
        spf = np.where(placed > 0, sim / placed, np.nan)
    has_time = bool(np.isfinite(spf).sum() >= 3)
    tm = roll(spf, w, np.median)
    # failure causes per bin of training steps
    bw = bin_width(max(float(steps.max()), 1.0))
    nb = max(1, int(np.ceil(steps.max() / bw)))
    fails = Counter(c for c in cause if c != "success")
    shown = [c for c, _ in CAUSES if c in fails]
    other = [c for c in fails if c not in CAUSE_COL]
    stack = []
    for c in shown + (["__other__"] if other else []):
        cnt = [0] * nb
        for s, cc in zip(steps, cause):
            if cc == c or (c == "__other__" and cc in other):
                cnt[min(nb - 1, int(s // bw))] += 1
        stack.append((nice(c) if c != "__other__" else f"Other ({len(other)} type{'s' if len(other) != 1 else ''})",
                      CAUSE_COL.get(c, GREY), cnt))
    # reward components: median per episode, successful vs failed episodes
    comp_ok, comp_bad = {}, {}
    for r, o in zip(rows, ok):
        try:
            c = json.loads(r.get("components") or "{}")
        except json.JSONDecodeError:
            c = {}
        for key, v in c.items():
            (comp_ok if o else comp_bad).setdefault(key, []).append(num(v, 0.0))
    keys = sorted(set(comp_ok) | set(comp_bad), key=lambda q: -max(abs(float(np.median(comp_ok.get(q, [0])))),
                                                                   abs(float(np.median(comp_bad.get(q, [0]))))))[:9]
    top = fails.most_common(3)
    return dict(n=n, steps_max=float(steps.max()), envs=len({r.get("env") for r in rows}), n_targets=n_t,
                success=int(ok.sum()), success_pct=100 * ok.mean(), w=w, k=k,
                first_sr=100 * ok[:k].mean() if k else None, last_sr=100 * ok[-k:].mean() if k else None,
                first_ret=float(np.nanmedian(ret[:k])) if k else None, last_ret=float(np.nanmedian(ret[-k:])) if k else None,
                sr_min=float(np.nanmin(sr[idx])), sr_max=float(np.nanmax(sr[idx])),
                placed_mean=float(placed.mean()), placed_hist=[int((placed == i).sum()) for i in range(n_t + 1)],
                mismatch=int(((placed >= targets) & (targets > 0) != ok).sum()),
                x=[kfmt(s) for s in steps[idx]], xs=(steps[idx] / 1000).tolist(), sr=np.round(sr[idx], 1).tolist(),
                ret=np.round(rm[idx], 1).tolist(), has_time=has_time, spf=[None if np.isnan(v) else round(float(v), 1) for v in tm[idx]],
                spf_last=None if not has_time else float(np.nanmedian(spf[-max(k, 1):])),
                bins=[f"{kfmt(i * bw)}–{kfmt((i + 1) * bw)}" for i in range(nb)], stack=stack, fails=fails, top=top,
                comp_keys=[COMPONENT_NAMES.get(q, nice(q)) for q in keys],
                comp_ok=[round(float(np.median(comp_ok.get(q, [0]))), 2) for q in keys],
                comp_bad=[round(float(np.median(comp_bad.get(q, [0]))), 2) for q in keys],
                comp_raw=dict(ok={q: float(np.median(v)) for q, v in comp_ok.items()}, bad={q: float(np.median(v)) for q, v in comp_bad.items()}))


def eval_stats(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    ref = [r for r in rows if r.get("policy", "sac") == "reference"]
    sac = sorted([r for r in rows if r.get("policy", "sac") != "reference"], key=lambda r: num(r.get("timesteps"), 0))
    ref = ref[0] if ref else None
    if not sac and ref is None:
        return None
    steps = np.array([num(r.get("timesteps"), 0) for r in sac])
    col = lambda key, scale=1.0: np.array([num(r.get(key)) * scale for r in sac])
    sr, fr, rt, spf = col("success_rate", 100), col("fruit_rate", 100), col("mean_return"), col("s_per_fruit")
    outcome = lambda r: [(e.split("@")[0] or "success", e) for e in (r.get("failures") or "").split("|")]
    grid = [outcome(r) for r in sac]
    n_ep = max([len(g) for g in grid] + ([len(outcome(ref))] if ref else [0]))
    seen = Counter(c for g in grid + ([outcome(ref)] if ref else []) for c, _ in g)
    kinds = [("success", "Success", OK_COL)] + [(c, nice(c), col_) for c, col_ in CAUSES if c in seen]
    other = sorted(c for c in seen if c != "success" and c not in CAUSE_COL)
    if other:
        kinds.append(("__other__", f"Other ({len(other)} type{'s' if len(other) != 1 else ''})", GREY))
    kid = {k: i for i, (k, _, _) in enumerate(kinds)}
    code = lambda c: kid.get(c, kid.get("__other__", 0))
    k = min(10, len(sac) // 2)
    interval = float(Counter(np.diff(steps).round().tolist()).most_common(1)[0][0]) if len(steps) > 1 else (float(steps[0]) if len(steps) else 0.0)
    best = float(np.nanmax(sr)) if len(sr) else float("nan")
    refv = None if ref is None else dict(sr=num(ref.get("success_rate")) * 100, fr=num(ref.get("fruit_rate")) * 100, ret=num(ref.get("mean_return")),
                                         spf=num(ref.get("s_per_fruit")), grid=[code(c) for c, _ in outcome(ref)], raw=[e for _, e in outcome(ref)])
    # per evaluation episode: how often it succeeded, what it failed with
    per_ep = []
    for e in range(n_ep):
        cs = [g[e][0] for g in grid if e < len(g)]
        per_ep.append(dict(ok=sum(c == "success" for c in cs), n=len(cs), fails=Counter(c for c in cs if c != "success").most_common(2)))
    return dict(n=len(sac), n_ep=n_ep, interval=interval, steps=steps.tolist(), x=[kfmt(s) for s in steps], sr=np.round(sr, 1).tolist(),
                fr=np.round(fr, 1).tolist(), ret=np.round(rt, 2).tolist(),
                spf=[None if np.isnan(v) else round(float(v), 1) for v in spf], has_time=bool(np.isfinite(spf).any()),
                mean_sr=float(np.nanmean(sr)) if len(sr) else float("nan"), mean_fr=float(np.nanmean(fr)) if len(fr) else float("nan"),
                k=k, first_sr=float(np.nanmean(sr[:k])) if k else None, last_sr=float(np.nanmean(sr[-k:])) if k else None,
                first_ret=float(np.nanmean(rt[:k])) if k else None, last_ret=float(np.nanmean(rt[-k:])) if k else None,
                best=best, n_best=int(np.sum(np.isclose(sr, best))) if len(sr) else 0,
                best_ret=float(np.nanmax(rt)) if len(rt) else float("nan"), best_ret_at=kfmt(steps[int(np.nanargmax(rt))]) if len(rt) else "-",
                kinds=[(lab, c) for _, lab, c in kinds], grid=[[code(c) for c, _ in g] for g in grid],
                raw=[[e for _, e in g] for g in grid], ref=refv, per_ep=per_ep, counts=seen)


def verdict(ev: dict | None) -> str | None:
    """One sentence: does the SAC policy beat the reference controller on the evaluation episodes?"""
    if not ev or ev["ref"] is None or not ev["n"]:
        return None
    i = int(np.lexsort((-np.arange(ev["n"]), np.array(ev["sr"]), np.array(ev["fr"])))[-1])      # best; ties: the earliest
    f, s, rf, rs = ev["fr"][i], ev["sr"][i], ev["ref"]["fr"], ev["ref"]["sr"]
    word = "better than" if f > rf + 1e-6 or (abs(f - rf) < 1e-6 and s > rs + 1e-6) else \
        "the same as" if abs(f - rf) < 1e-6 and abs(s - rs) < 1e-6 else "worse than"
    return (f"Best SAC checkpoint ({ev['x'][i]} steps): {f:.0f}% of the fruits, {s:.0f}% of the episodes. Reference controller "
            f"alone: {rf:.0f}% / {rs:.0f}%. SAC is {word} the reference.")


# ----------------------------------------------------------------------------- HTML dashboards
CSS = """body{font-family:Inter,system-ui,-apple-system,Segoe UI,Arial,sans-serif;background:#f6f5f1;color:#0b0b0b;margin:0}
.wrap{max-width:1100px;margin:0 auto;padding:32px 24px 48px}h1{font-size:24px;font-weight:600;margin:0 0 4px}
.sub{color:#6b6a66;font-size:14px;margin-bottom:20px}.k{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:20px}
.m{background:#fff;border:1px solid #e4e2da;border-radius:10px;padding:14px 16px}.ml{font-size:13px;color:#6b6a66}.mv{font-size:24px;font-weight:600;margin:2px 0}
.card{background:#fff;border:1px solid #e4e2da;border-radius:12px;padding:16px 18px;margin-bottom:16px}.ct{font-size:14px;font-weight:600;margin-bottom:8px}
.lg{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:#52514e;margin-bottom:8px}.lg span{display:flex;align-items:center;gap:5px}
.sq{width:10px;height:10px;border-radius:2px;display:inline-block}.ln{width:16px;height:0;border-top:2px dashed #52514e;display:inline-block}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:16px}.hm{display:grid;gap:1px;align-items:center}.rl{font-size:12px;color:#52514e}
.note{font-size:13px;color:#52514e;line-height:1.6}.vd{background:#fff;border:1px solid #e4e2da;border-left:4px solid #2a78d6;border-radius:8px;padding:10px 14px;margin-bottom:20px;font-size:14px}
@media(max-width:760px){.k{grid-template-columns:repeat(2,minmax(0,1fr))}.g2{grid-template-columns:1fr}}"""
JS_HEAD = """Chart.defaults.color='#898781';Chart.defaults.font.size=12;Chart.defaults.font.family='Inter,system-ui,Arial,sans-serif';
const ax=o=>Object.assign({grid:{color:'#e1e0d9'},border:{display:false}},o||{});
const L=(c)=>({borderColor:c,borderWidth:2,pointRadius:0,pointHoverRadius:4,tension:0.25});
const D=(v,n,l)=>({data:Array(n).fill(v),borderColor:'#52514e',borderWidth:1.5,borderDash:[6,4],pointRadius:0,pointHoverRadius:0,label:l});
const O=(fmt,sc)=>({responsive:true,maintainAspectRatio:false,interaction:{mode:'index',intersect:false},plugins:{legend:{display:false},
tooltip:{callbacks:{label:c=>c.parsed.y==null?null:c.dataset.label+': '+fmt(c.parsed.y)}}},scales:sc});
const XS=()=>ax({grid:{display:false},ticks:{maxTicksLimit:10}});
"""


def page(title: str, sub: str, body: str, script: str) -> str:
    lib = f"<script>{CHART_JS.read_text(encoding='utf-8')}</script>" if CHART_JS.exists() else f'<script src="{CHART_CDN}"></script>'
    return (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n'
            f"<title>{title}</title><style>{CSS}</style>{lib}\n</head><body><div class=\"wrap\">\n<h1>{title}</h1><div class=\"sub\">{sub}</div>"
            f"{body}</div>\n<script>{JS_HEAD}{script}</script></body></html>")


def tiles(items) -> str:
    return '<div class="k">' + "".join(f'<div class="m"><div class="ml">{a}</div><div class="mv">{b}</div><div class="ml">{c}</div></div>'
                                       for a, b, c in items) + "</div>"


def card(title: str, cid: str, height: int, legend: str = "") -> str:
    return (f'<div class="card"><div class="ct">{title}</div>{legend}'
            f'<div style="position:relative;height:{height}px"><canvas id="{cid}"></canvas></div></div>')


def train_tiles(name: str, t: dict):
    fruit = f"{t['n_targets']} fruit{'s' if t['n_targets'] != 1 else ''}"
    arrow = "→"
    return [("Training episodes", f"{t['n']:,}", f"≈ {approx(t['steps_max'])} timesteps · {t['envs']} env{'s' if t['envs'] != 1 else ''}"),
            (f"Success ({fruit})", f"{t['success_pct']:.1f}%", f"{t['success']:,} episodes"),
            (f"First {arrow} last {t['k']} episodes", f"{t['first_sr']:.0f} {arrow} {t['last_sr']:.0f}%" if t["k"] else "–",
             f"median return {t['first_ret']:.1f} {arrow} {t['last_ret']:.1f}" if t["k"] else "too few episodes"),
            ("Fruits placed (mean)", f"{t['placed_mean']:.2f} / {t['n_targets']}",
             f"top failure: {nice(t['top'][0][0])} ({t['top'][0][1]})" if t["top"] else "no failed episode")]


def train_html(name: str, t: dict) -> str:
    J = json.dumps
    fruit = f"{t['n_targets']} target fruit{'s' if t['n_targets'] != 1 else ''} per episode"
    body = tiles(train_tiles(name, t))
    body += card(f"Success rate (rolling {t['w']} episodes) · dashed = overall mean {t['success_pct']:.1f}%", "c1", 240)
    body += card(f"Episode return (rolling median, {t['w']} episodes)", "c2", 220)
    if t["has_time"]:
        body += card(f"Time per harvested fruit, motion + computing (rolling median, {t['w']} episodes)", "c6", 220)
    body += card("Failure causes per timestep bin (episodes)", "c3", 280, '<div class="lg" id="lg3"></div>')
    lg = (f'<div class="lg"><span><i class="sq" style="background:{BLUE}"></i>Success ({t["success"]})</span>'
          f'<span><i class="sq" style="background:#eb6834"></i>Failure ({t["n"] - t["success"]})</span></div>')
    body += '<div class="g2">' + card(f"Fruits placed per episode (target = {t['n_targets']})", "c5", 260) + \
        card("Reward components: median per episode", "c4", 300, lg) + "</div>"
    ph = t["placed_hist"]
    pcol = [ZERO_COL] + [GREY] * (len(ph) - 2) + [OK_COL]
    js = f"const X={J(t['x'])},SR={J(t['sr'])},RT={J(t['ret'])},TM={J(t['spf'])},N={t['n']};\n"
    js += (f"new Chart(c1,{{type:'line',data:{{labels:X,datasets:[Object.assign({{data:SR,label:'Success'}},L('{BLUE}')),"
           f"D({t['success_pct']:.1f},X.length,'Mean')]}},options:O(v=>v.toFixed(1)+'%',{{x:XS(),y:ax({{ticks:{{callback:v=>v+'%'}}}})}})}});\n")
    js += (f"new Chart(c2,{{type:'line',data:{{labels:X,datasets:[Object.assign({{data:RT,label:'Median return'}},L('{BLUE}')),"
           "{data:X.map(()=>0),borderColor:'#898781',borderWidth:1,pointRadius:0,pointHoverRadius:0,label:'Zero'}]},"
           "options:O(v=>v.toFixed(1),{x:XS(),y:ax()})});\n")
    if t["has_time"]:
        js += (f"new Chart(c6,{{type:'line',data:{{labels:X,datasets:[Object.assign({{data:TM,label:'Median time per fruit',spanGaps:true}},L('{BLUE}'))]}},"
               "options:O(v=>v.toFixed(1)+' s',{x:XS(),y:ax({ticks:{callback:v=>v+' s'}})})});\n")
    js += f"const F={J(t['stack'])};\n"
    js += ("lg3.innerHTML=F.length?F.map(f=>'<span><i class=\"sq\" style=\"background:'+f[1]+'\"></i>'+f[0]+' \\u00b7 '+f[2].reduce((a,b)=>a+b,0)+'</span>').join(''):'no failed episode';\n"
           f"new Chart(c3,{{type:'bar',data:{{labels:{J(t['bins'])},datasets:F.map(f=>({{label:f[0],data:f[2],backgroundColor:f[1],borderColor:'#fff',"
           "borderWidth:{top:2},borderSkipped:false,maxBarThickness:28}))},options:{responsive:true,maintainAspectRatio:false,"
           "plugins:{legend:{display:false}},scales:{x:ax({stacked:true,grid:{display:false},ticks:{autoSkip:false,maxRotation:45}}),y:ax({stacked:true})}}});\n")
    js += (f"const P={J(ph)};\nnew Chart(c5,{{type:'bar',data:{{labels:P.map((_,i)=>i+' fruit'+(i===1?'':'s')),datasets:[{{data:P,backgroundColor:{J(pcol)},"
           "borderRadius:4,maxBarThickness:48}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},"
           "tooltip:{callbacks:{label:c=>c.parsed.y+' episodes ('+Math.round(c.parsed.y/N*100)+'%)'}}},scales:{x:ax({grid:{display:false}}),y:ax()}}});\n")
    js += (f"new Chart(c4,{{type:'bar',data:{{labels:{J(t['comp_keys'])},datasets:[{{label:'Success',data:{J(t['comp_ok'])},backgroundColor:'{BLUE}',"
           f"borderRadius:3,maxBarThickness:12}},{{label:'Failure',data:{J(t['comp_bad'])},backgroundColor:'#eb6834',borderRadius:3,maxBarThickness:12}}]}},"
           "options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:ax(),y:ax({grid:{display:false}})}}});\n")
    return page(f"{name} — training log", f"Source: train_log.csv · {t['n']:,} episodes · {fruit}", body, js)


def eval_tiles(e: dict):
    arrow = "→"
    ref = e["ref"]
    return [("Checkpoints", f"{e['n']}", f"every {kfmt(e['interval'])} · {e['n_ep']} episode{'s' if e['n_ep'] != 1 else ''} each"),
            ("Mean success rate", f"{e['mean_sr']:.1f}%" if e["n"] else "–", f"fruit rate {e['mean_fr']:.1f}%" if e["n"] else "no SAC evaluation yet"),
            (f"First {arrow} last {e['k']} checkpoints", f"{e['first_sr']:.0f} {arrow} {e['last_sr']:.0f}%" if e["k"] else "–",
             f"return {e['first_ret']:.1f} {arrow} {e['last_ret']:.1f}" if e["k"] else "too few checkpoints"),
            ("Best success rate", f"{e['best']:.1f}%" if e["n"] else "–",
             (f"reached at {e['n_best']} checkpoint{'s' if e['n_best'] != 1 else ''}" if e["n"] else "") +
             (f" · reference {ref['sr']:.1f}%" if ref else ""))]


def eval_html(name: str, e: dict) -> str:
    J = json.dumps
    ref = e["ref"]
    body = tiles(eval_tiles(e))
    v = verdict(e)
    if v:
        body = f'<div class="vd">{v}</div>' + body
    lg = f'<div class="lg"><span><i class="sq" style="background:{BLUE}"></i>SAC policy</span>' + \
        ('<span><i class="ln"></i>Reference controller alone (before training)</span>' if ref else "") + "</div>"
    body += card(f"Success rate per checkpoint ({e['n_ep']} eval episode{'s' if e['n_ep'] != 1 else ''})", "e1", 220, lg)
    body += card("Fruit rate per checkpoint (fruits in the crate / all fruits)", "e4", 200, lg)
    body += card("Mean return per checkpoint", "e2", 220, lg if ref else "")
    if e["has_time"]:
        body += card("Time per harvested fruit per checkpoint, motion + computing", "e3", 220, lg)
    body += ('<div class="card"><div class="ct">Outcome of each evaluation episode per checkpoint (left = first, right = last)</div>'
             '<div class="lg" id="lg"></div><div class="hm" id="hm"></div><div class="note" style="margin-top:10px">Episode order is read from the '
             'position of each entry in the <code>failures</code> column (separated by “|”; empty = success). The evaluation episodes '
             'are the same fixed scenes at every checkpoint.' + (' Column “ref” = reference controller alone.' if ref else "") + '</div></div>')
    n = max(1, e["n"])
    rd = lambda key, lab: f",D({ref[key]:.2f},{n},'{lab}')" if ref and np.isfinite(ref[key]) else ""
    js = f"const T={J(e['x'])},S={J(e['sr'])},FR={J(e['fr'])},R={J(e['ret'])},TM={J(e['spf'])};\n"
    pts = "pointRadius:T.length<=15?4:0"
    js += (f"new Chart(e1,{{type:'line',data:{{labels:T,datasets:[{{data:S,borderColor:'{BLUE}',backgroundColor:'{BLUE}',borderWidth:2,{pts},stepped:T.length>15?'middle':false,"
           f"label:'Success rate'}}{rd('sr', 'Reference')}]}},options:O(v=>v.toFixed(1)+'%',{{x:XS(),y:ax({{min:0,max:100,ticks:{{callback:v=>v+'%'}}}})}})}});\n")
    js += (f"new Chart(e4,{{type:'line',data:{{labels:T,datasets:[{{data:FR,borderColor:'{BLUE}',backgroundColor:'{BLUE}',borderWidth:2,{pts},stepped:T.length>15?'middle':false,"
           f"label:'Fruit rate'}}{rd('fr', 'Reference')}]}},options:O(v=>v.toFixed(1)+'%',{{x:XS(),y:ax({{min:0,max:100,ticks:{{callback:v=>v+'%'}}}})}})}});\n")
    js += (f"new Chart(e2,{{type:'line',data:{{labels:T,datasets:[{{data:R,borderColor:'{BLUE}',backgroundColor:'{BLUE}',borderWidth:2,{pts},tension:0.25,"
           f"label:'Mean return'}}{rd('ret', 'Reference')},{{data:T.map(()=>0),borderColor:'#898781',borderWidth:1,pointRadius:0,pointHoverRadius:0,label:'Zero'}}]}},"
           "options:O(v=>v.toFixed(2),{x:XS(),y:ax()})});\n")
    if e["has_time"]:
        js += (f"new Chart(e3,{{type:'line',data:{{labels:T,datasets:[{{data:TM,borderColor:'{BLUE}',backgroundColor:'{BLUE}',borderWidth:2,{pts},tension:0.25,"
               f"spanGaps:true,label:'Time per fruit'}}{rd('spf', 'Reference')}]}},options:O(v=>v.toFixed(1)+' s',{{x:XS(),y:ax({{ticks:{{callback:v=>v+' s'}}}})}})}});\n")
    cols = ([dict(label="ref", g=ref["grid"], raw=ref["raw"])] if ref else []) + \
        [dict(label=x, g=g, raw=r) for x, g, r in zip(e["x"], e["grid"], e["raw"])]
    js += f"const K={J(e['kinds'])},C={J(cols)},k={e['n_ep']},n=C.length,nr={1 if ref else 0};\n"
    js += ("const cnt=K.map((_,i)=>C.slice(nr).map(c=>c.g).flat().filter(v=>v===i).length);\n"
           "lg.innerHTML=K.map((c,i)=>'<span><i class=\"sq\" style=\"background:'+c[1]+'\"></i>'+c[0]+' \\u00b7 '+cnt[i]+'</span>').join('');\n"
           "hm.style.gridTemplateColumns='90px '+(nr?'minmax(14px,1fr) 6px ':'')+'repeat('+(n-nr)+',minmax(0,1fr)) 110px';\n"
           "let h='';for(let ep=0;ep<k;ep++){h+='<span class=\"rl\">Episode '+(ep+1)+'</span>';let ok=0;for(let c=0;c<n;c++){const v=C[c].g[ep];"
           "if(v===undefined){h+='<div></div>'}else{if(v===0&&c>=nr)ok++;const why=C[c].raw[ep]||'success';"
           "h+='<div style=\"height:26px;background:'+K[v][1]+'\" title=\"'+C[c].label+' \\u00b7 episode '+(ep+1)+': '+why.replace('@',' in phase ')+'\"></div>'}"
           "if(nr&&c===0)h+='<div></div>'}h+='<span class=\"rl\" style=\"padding-left:8px\">success '+ok+'/'+(n-nr)+'</span>'}\n"
           "h+='<span></span>';for(let c=0;c<n;c++){const show=c<nr||(n-nr)<=14||(c-nr)%Math.ceil((n-nr)/10)===0;"
           "h+='<span class=\"rl\" style=\"text-align:center;font-size:11px;overflow:visible;white-space:nowrap\">'+(show?C[c].label:'')+'</span>';if(nr&&c===0)h+='<span></span>'}"
           "h+='<span></span>';hm.innerHTML=h;\n")
    return page(f"{name} — evaluation log", f"Source: eval_log.csv · {e['n']} checkpoint{'s' if e['n'] != 1 else ''} · {e['n_ep']} evaluation "
                f"episode{'s' if e['n_ep'] != 1 else ''} per checkpoint" + (" · reference controller measured before training" if ref else ""), body, js)


# ----------------------------------------------------------------------------- static PNG overviews
def _axes(ax, title):
    ax.set_facecolor(CARD)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)


def _tiles(fig, title, items, note=None) -> float:
    """Title, the four key numbers and an optional sentence at the top of the figure (placed in inches, so the
    header looks the same on short and tall figures). Returns the top of the chart area (figure fraction)."""
    H = fig.get_figheight()
    y = lambda inch: 1 - inch / H
    fig.text(0.04, y(0.30), title, fontsize=17, fontweight="bold", color=INK, va="top")
    for i, (a, b, c) in enumerate(items):
        x = 0.04 + i * 0.235
        fig.text(x, y(0.80), a, fontsize=10.5, color=INK2, va="top")
        fig.text(x, y(1.05), b, fontsize=19, fontweight="bold", color=INK, va="top")
        fig.text(x, y(1.50), c, fontsize=9.5, color=INK2, va="top")
    if note:
        fig.text(0.04, y(1.92), note, fontsize=10.5, color=INK, va="top")
    return y(2.75 if note else 2.35)


def train_png(name: str, t: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig = plt.figure(figsize=(15, 14.6 if t["has_time"] else 12.0), facecolor=CARD)
    rows = 4 if t["has_time"] else 3
    top = _tiles(fig, f"{name} — training log", train_tiles(name, t))
    gs = fig.add_gridspec(rows, 2, left=0.06, right=0.975, top=top, bottom=0.06, hspace=0.55, wspace=0.18)
    xs = t["xs"]
    a = fig.add_subplot(gs[0, 0]); _axes(a, f"Success rate (rolling {t['w']} episodes) · dashed = overall mean")
    a.plot(xs, t["sr"], color=BLUE, lw=2); a.axhline(t["success_pct"], color=MUTED, lw=1, ls=(0, (4, 3)))
    a.set_ylabel("%", color=INK2, fontsize=9); a.set_xlabel("Timesteps (k)", color=INK2, fontsize=9)
    a = fig.add_subplot(gs[0, 1]); _axes(a, f"Episode return (rolling median, {t['w']} episodes)")
    a.plot(xs, t["ret"], color=BLUE, lw=2); a.axhline(0, color=MUTED, lw=1); a.set_xlabel("Timesteps (k)", color=INK2, fontsize=9)
    r = 1
    if t["has_time"]:
        a = fig.add_subplot(gs[1, :]); _axes(a, f"Time per harvested fruit in seconds, motion + computing (rolling median, {t['w']} episodes)")
        a.plot(xs, [np.nan if v is None else v for v in t["spf"]], color=BLUE, lw=2); a.set_xlabel("Timesteps (k)", color=INK2, fontsize=9)
        r = 2
    a = fig.add_subplot(gs[r, :]); _axes(a, "Failure causes per timestep bin (episodes)")
    bottom = np.zeros(len(t["bins"]))
    for lab, colr, cnt in t["stack"]:
        a.bar(range(len(cnt)), cnt, bottom=bottom, color=colr, width=0.6, edgecolor=CARD, linewidth=1.5)
        bottom += np.array(cnt)
    a.set_xticks(range(len(t["bins"]))); a.set_xticklabels(t["bins"], rotation=30, ha="right")
    if t["stack"]:
        a.set_ylim(0, max(1.0, bottom.max()) * 1.38)
        a.legend(handles=[Patch(color=c, label=f"{lab} · {sum(cnt)}") for lab, c, cnt in t["stack"]], loc="upper left", ncol=4, frameon=False,
                 fontsize=9.5, labelcolor=INK2)
    else:
        a.text(0.5, 0.5, "no failed episode", transform=a.transAxes, ha="center", color=INK2)
    a = fig.add_subplot(gs[r + 1, 0]); _axes(a, f"Fruits placed per episode (target = {t['n_targets']})")
    ph = t["placed_hist"]
    bars = a.bar(range(len(ph)), ph, color=[ZERO_COL] + [GREY] * (len(ph) - 2) + [OK_COL], width=0.6)
    for b, v in zip(bars, ph):
        a.text(b.get_x() + b.get_width() / 2, v, f"{v} ({100 * v / max(1, t['n']):.0f}%)", ha="center", va="bottom", fontsize=9.5, color=INK)
    a.set_xticks(range(len(ph))); a.set_xlabel("Fruits placed", color=INK2, fontsize=9); a.set_ylim(0, max(ph) * 1.15 + 1)
    a = fig.add_subplot(gs[r + 1, 1]); _axes(a, "Reward components: median per episode")
    a.grid(axis="y", visible=False); a.grid(axis="x", color=GRID, linewidth=0.9)
    y = np.arange(len(t["comp_keys"]))[::-1]
    a.barh(y + 0.19, t["comp_ok"], height=0.34, color=BLUE); a.barh(y - 0.19, t["comp_bad"], height=0.34, color="#eb6834")
    a.axvline(0, color=AXIS, lw=1); a.set_yticks(y); a.set_yticklabels(t["comp_keys"], color=INK2)
    a.legend(handles=[Patch(color=BLUE, label=f"Success ({t['success']})"), Patch(color="#eb6834", label=f"Failure ({t['n'] - t['success']})")],
             loc="lower left", frameon=False, fontsize=9.5, labelcolor=INK2)
    fig.savefig(path, dpi=130, facecolor=CARD)
    plt.close(fig)


def eval_png(name: str, e: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ref = e["ref"]
    panels = ["sr", "fr", "ret"] + (["spf"] if e["has_time"] else [])
    fig = plt.figure(figsize=(15, 3.1 * len(panels) + 6.4), facecolor=CARD)
    top = _tiles(fig, f"{name} — evaluation log", eval_tiles(e), verdict(e))
    gs = fig.add_gridspec(len(panels) + 1, 1, left=0.115, right=0.975, top=top, bottom=0.055, hspace=0.62,
                          height_ratios=[1] * len(panels) + [0.16 + 0.17 * max(2, e["n_ep"])])
    xs = np.array(e["steps"]) / 1000.0
    spec = dict(sr=(f"Success rate per checkpoint ({e['n_ep']} eval episode{'s' if e['n_ep'] != 1 else ''}), %", True, (0, 104)),
                fr=("Fruit rate per checkpoint (fruits in the crate / all fruits), %", True, (0, 104)),
                ret=("Mean return per checkpoint", False, None), spf=("Time per harvested fruit per checkpoint, seconds (motion + computing)", False, None))
    mk = dict(marker="o", ms=5, markeredgecolor=CARD) if e["n"] <= 15 else {}
    for i, key in enumerate(panels):
        a = fig.add_subplot(gs[i, 0]); _axes(a, spec[key][0])
        y = np.array([np.nan if q is None else q for q in e[key]], float)
        if len(xs):
            a.step(xs, y, where="mid", color=BLUE, lw=2) if spec[key][1] and e["n"] > 15 else a.plot(xs, y, color=BLUE, lw=2, **mk)
        if ref and np.isfinite(ref[key]):
            a.axhline(ref[key], color=INK2, lw=1.5, ls=(0, (5, 3)))
        if key == "ret":
            a.axhline(0, color=MUTED, lw=1)
        if spec[key][2]:
            a.set_ylim(*spec[key][2])
        a.set_xlim(0, max(1e-9, xs.max() if len(xs) else 1) * 1.03)
        a.set_xlabel("Timesteps (k)", color=INK2, fontsize=9)
        if i == 0:
            hs = [Line2D([0], [0], color=BLUE, lw=2, label="SAC policy")] + \
                ([Line2D([0], [0], color=INK2, lw=1.5, ls=(0, (5, 3)), label="Reference controller alone (before training)")] if ref else [])
            a.legend(handles=hs, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=9.5, labelcolor=INK2)
    a = fig.add_subplot(gs[-1, 0])
    a.set_facecolor(CARD)
    a.set_title("Outcome of each evaluation episode per checkpoint", loc="left", fontsize=12, color=INK, pad=24)
    cols = ([ref["grid"]] if ref else []) + e["grid"]
    G = np.full((e["n_ep"], len(cols)), -1.0)
    for c, g in enumerate(cols):
        G[:len(g), c] = g
    cmap = ListedColormap([c for _, c in e["kinds"]])
    a.imshow(np.ma.masked_less(G, 0), aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(e["kinds"]) - 0.5, interpolation="nearest")
    if ref:
        a.axvline(0.5, color=CARD, lw=5)
    nr = 1 if ref else 0
    a.set_yticks(range(e["n_ep"]))
    a.set_yticklabels([f"Episode {i + 1}  ({p['ok']}/{p['n']} ok)" for i, p in enumerate(e["per_ep"])], color=INK2, fontsize=9.5)
    stepk = max(1, int(np.ceil(e["n"] / 12)))
    ticks = list(range(nr, len(cols), stepk))
    a.set_xticks(([0] if ref else []) + ticks); a.set_xticklabels((["ref"] if ref else []) + [e["x"][i - nr] for i in ticks], color=MUTED, fontsize=9)
    a.set_xlabel("Timesteps", color=INK2, fontsize=9)
    a.tick_params(length=0)
    for s in a.spines.values():
        s.set_visible(False)
    cnt = Counter(v for g in e["grid"] for v in g)
    a.legend(handles=[Patch(color=c, label=f"{lab} · {cnt.get(i, 0)}") for i, (lab, c) in enumerate(e["kinds"])], loc="lower right",
             bbox_to_anchor=(1.0, 1.0), ncol=min(7, len(e["kinds"])), frameon=False, fontsize=9.5, labelcolor=INK2)
    fig.savefig(path, dpi=130, facecolor=CARD)
    plt.close(fig)


# ----------------------------------------------------------------------------- SUMMARY.md / README.md
def task_title(n_targets: int) -> str:
    return "single-fruit harvest (1 target per episode)" if n_targets == 1 else f"{n_targets} fruits per episode"


def summary_md(name: str, t: dict | None, e: dict | None, made: list[str]) -> str:
    arrow = "→"
    L = [f"# {name.replace('Mission', 'Mission ')} — {task_title(t['n_targets']) if t else 'training / evaluation logs'}", "", "## Files",
         "| File | What it is |", "|---|---|"]
    what = {"_train_dashboard.html": "Interactive training dashboard (open in any browser, works offline)",
            "_eval_dashboard.html": "Interactive evaluation dashboard", "_train_overview.png": "Static image of the training view (for slides / reports)",
            "_eval_overview.png": "Static image of the evaluation view"}
    L += [f"| `{f}` | {next(v for k, v in what.items() if f.endswith(k))} |" for f in made]
    L += ["| `data/train_log.csv`, `data/eval_log.csv` | The original logs used |", ""]
    if t:
        L += ["## Training log — key numbers",
              f"- **Episodes:** {t['n']:,} over ≈ {approx(t['steps_max'])} timesteps, {t['envs']} parallel environment{'s' if t['envs'] != 1 else ''}.",
              f"- **Success rate:** {t['success_pct']:.1f}% ({t['success']:,} episodes)."]
        if t["k"]:
            d = t["last_sr"] - t["first_sr"]
            trend = "a clear upward trend" if d >= 8 else "a clear downward trend" if d <= -8 else "no clear trend"
            L += [f"- **First {t['k']} {arrow} last {t['k']} episodes:** {t['first_sr']:.0f}% {arrow} {t['last_sr']:.0f}% success. The rolling rate "
                  f"({t['w']} episodes) stays between about {t['sr_min']:.0f}% and {t['sr_max']:.0f}%, with {trend}.",
                  f"- **Median episode return:** {t['first_ret']:+.1f} (first {t['k']}) {arrow} {t['last_ret']:+.1f} (last {t['k']})."]
        if t["top"]:
            n_fail = t["n"] - t["success"]
            rest = " and ".join(f"{nice(c).lower()} ({v})" for c, v in t["top"][1:])
            L += [f"- **Most common failure:** {nice(t['top'][0][0]).lower()} ({t['top'][0][1]} of {n_fail} failures)"
                  + (f", followed by {rest}." if rest else ".")]
        else:
            L += ["- **Failures:** none."]
        if t["has_time"]:
            L += [f"- **Time per harvested fruit:** median {t['spf_last']:.1f} s in the last {max(t['k'], 1)} episodes (motion + computing)."]
        ok_, bad_ = t["comp_raw"]["ok"], t["comp_raw"]["bad"]
        if "leaf" in ok_ and "leaf" in bad_:
            L += [f"- **Reward components:** median leaf penalty {bad_['leaf']:+.1f} in failed episodes vs {ok_['leaf']:+.1f} in successful ones; "
                  f"median progress {bad_.get('progress', float('nan')):+.1f} vs {ok_.get('progress', float('nan')):+.1f}."]
        L += [""]
    if e:
        L += ["## Evaluation log — key numbers"]
        if e["ref"]:
            r = e["ref"]
            L += [f"- **Reference controller alone (before training):** {r['sr']:.1f}% of the episodes, {r['fr']:.1f}% of the fruits, mean return {r['ret']:.1f}"
                  + (f", {r['spf']:.1f} s per fruit." if np.isfinite(r["spf"]) else ".")]
        if e["n"]:
            L += [f"- **Checkpoints:** {e['n']}, every {kfmt(e['interval'])} timesteps, with {e['n_ep']} evaluation episode{'s' if e['n_ep'] != 1 else ''} each.",
                  f"- **Mean success rate:** {e['mean_sr']:.1f}% (fruit rate {e['mean_fr']:.1f}%)."
                  + (f" From {e['first_sr']:.0f}% (first {e['k']} checkpoints) to {e['last_sr']:.0f}% (last {e['k']})." if e["k"] else ""),
                  f"- **Best success rate:** {e['best']:.1f}%, reached at {e['n_best']} checkpoint{'s' if e['n_best'] != 1 else ''}.",
                  f"- **Mean return:** best {e['best_ret']:.1f} at {e['best_ret_at']}"
                  + (f"; {e['first_ret']:.1f} {arrow} {e['last_ret']:.1f} (first vs last {e['k']} checkpoints)." if e["k"] else ".")]
            v = verdict(e)
            if v:
                L += [f"- **SAC vs reference:** {v}"]
            L += ["- **Per evaluation episode:**"]
            for i, p in enumerate(e["per_ep"]):
                how = "always succeeds" if p["ok"] == p["n"] else "never succeeds" if p["ok"] == 0 else f"succeeds at {p['ok']} of {p['n']} checkpoints"
                why = "" if not p["fails"] else "; fails with " + " or ".join(f"{nice(c).lower()} ({v})" for c, v in p["fails"])
                L += [f"  - Episode {i + 1} {how}{why}."]
        else:
            L += ["- No evaluation of the SAC policy yet."]
        L += [""]
    L += ["## Notes / assumptions",
          "- Episode order in the evaluation heat map comes from the position of each entry in the `failures` column (separated by `|`; an empty "
          "entry means success). The evaluation episodes are fixed scenes (same seeds at every checkpoint).",
          "- A failure written as `cause@phase` is counted under its cause; the phase is shown in the tooltip of the dashboard."]
    if t and t["mismatch"]:
        L += [f"- In {t['mismatch']} training episode{'s' if t['mismatch'] != 1 else ''}, `n_placed` does not match `success` exactly, so the "
              f"\"fruits placed\" counts differ slightly from the success count."]
    if e and e["n"] and e["n_ep"] <= 4:
        L += [f"- Only {e['n_ep']} evaluation episode{'s' if e['n_ep'] != 1 else ''} per checkpoint: one episode changes the success rate by "
              f"{100 / e['n_ep']:.0f} percentage points, so small differences mean little."]
    return "\n".join(L) + "\n"


def readme_md(done: list[tuple[str, dict | None, dict | None]]) -> str:
    L = ["# Pineapple harvesting robot — RL training and evaluation visualizations", "",
         "Made by `python scripts/plot_training.py`. Each mission has its own folder. Every folder contains:",
         "- two interactive HTML dashboards (training and evaluation), which open in any browser and work offline;",
         "- two static PNG images for slides and reports;", "- a `SUMMARY.md` with the key numbers;", "- the original logs in `data/`.", "",
         "| Folder | Task | Training | Evaluation |", "|---|---|---|---|"]
    for name, t, e in done:
        tr = f"{t['n']:,} episodes · ≈ {approx(t['steps_max'])} steps · {t['success_pct']:.1f}% success" if t else "–"
        ev = "–"
        if e and e["n"]:
            ev = f"{e['n']} checkpoints × {e['n_ep']} episode{'s' if e['n_ep'] != 1 else ''} · {e['mean_sr']:.1f}% mean success (best {e['best']:.1f}%)"
            if e["ref"]:
                ev += f" · reference controller {e['ref']['sr']:.1f}%"
        L += [f"| `{name}/` | {task_title(t['n_targets']) if t else '?'} | {tr} | {ev} |"]
    L += ["", "The missions use different tasks, numbers of evaluation episodes and training lengths, so their numbers are not directly comparable."]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", default="checkpoints", help="folder that holds <mission>/train_log.csv and eval_log.csv")
    ap.add_argument("--mission", action="append", help="mission folder name (repeatable); default: every folder with a CSV")
    ap.add_argument("--out", default="outputs/Debug_Train_Eval")
    ap.add_argument("--no-png", action="store_true", help="skip the static images (no matplotlib needed)")
    ap.add_argument("--no-html", action="store_true", help="skip the interactive dashboards")
    a = ap.parse_args()
    base = Path(a.checkpoints) if Path(a.checkpoints).is_absolute() else ROOT / a.checkpoints
    if not base.is_dir():
        sys.exit(f"folder not found: {base}")
    folders = [base / m for m in a.mission] if a.mission else \
        sorted(p for p in base.iterdir() if p.is_dir() and ((p / "train_log.csv").exists() or (p / "eval_log.csv").exists()))
    if not folders:
        sys.exit(f"no mission folder with train_log.csv / eval_log.csv in {base}")
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    if not a.no_html and not CHART_JS.exists():
        print(f"note: {CHART_JS} not found - the dashboards will load Chart.js from the internet")
    done = []
    for folder in folders:
        if not folder.is_dir():
            print(f"{folder.name}: folder not found in {base}")
            continue
        name = folder.name[:1].upper() + folder.name[1:]
        t, e = train_stats(read_csv(folder / "train_log.csv")), eval_stats(read_csv(folder / "eval_log.csv"))
        if t is None and e is None:
            print(f"{folder.name}: no rows in train_log.csv / eval_log.csv")
            continue
        d = out / name
        (d / "data").mkdir(parents=True, exist_ok=True)
        for f in ("train_log.csv", "eval_log.csv"):
            if (folder / f).exists():
                shutil.copyfile(folder / f, d / "data" / f)
        made = []
        for kind, st, html_fn, png_fn in (("train", t, train_html, train_png), ("eval", e, eval_html, eval_png)):
            if st is None:
                continue
            if not a.no_html:
                (d / f"{name}_{kind}_dashboard.html").write_text(html_fn(name, st), encoding="utf-8")
                made.append(f"{name}_{kind}_dashboard.html")
            if not a.no_png:
                png_fn(name, st, d / f"{name}_{kind}_overview.png")
                made.append(f"{name}_{kind}_overview.png")
        made.sort(key=lambda f: (f.endswith(".png"), "eval" in f))
        (d / "SUMMARY.md").write_text(summary_md(name, t, e, made), encoding="utf-8")
        done.append((name, t, e))
        line = f"{name}: " + (f"{t['n']:,} training episodes, {t['success_pct']:.1f}% success" if t else "no training log")
        if e and e["n"]:
            line += f"; {e['n']} checkpoints, best {e['best']:.1f}%" + (f" (reference {e['ref']['sr']:.1f}%)" if e["ref"] else "")
        print(line + f"\n   -> {d}")
        v = verdict(e)
        if v:
            print("   " + v)
    if done:
        (out / "README.md").write_text(readme_md(done), encoding="utf-8")
        print(f"README -> {out / 'README.md'}")


if __name__ == "__main__":
    main()
