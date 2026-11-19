"""Per-fruit timing of an evaluation episode (version 6).

All times are ROBOT time: control steps x 20 ms of simulated motion, plus `compute_s` = the wall-clock time the
perception / decision / planning code needed on the evaluation machine (the real robot waits while it runs).

  observe_to_basket_s   start of the fruit's session (arm leaves the default / survey posture for it; for the first
                        fruit of an episode this includes the survey) -> fruit verified in the crate   [USER target < 40 s]
  cycle_s               the same + return to the default posture (= one complete session)
  s_per_fruit           (whole episode: every motion, drive, failed attempt, compute) / fruits placed
"""
from __future__ import annotations

import numpy as np


def fruit_times(targets: list) -> list:
    """[(observe_to_basket_s, cycle_s, compute_s)] of every PLACED fruit of an episode's target records."""
    out = []
    for t in targets:
        if not t.get("placed"):
            continue
        t0, tp, te, c = t.get("t_start"), t.get("t_placed"), t.get("t_end"), float(t.get("compute_s") or 0.0)
        if t0 is None or tp is None or not np.isfinite(t0) or not np.isfinite(tp):
            continue
        out.append((float(tp - t0) + c, (float(te - t0) + c) if te is not None and np.isfinite(te) else float("nan"), c))
    return out


def summarize(episodes: list) -> dict:
    """episodes: [dict(targets=[...], sim_time_s, compute_s, n_placed)] -> timing summary."""
    ft = [x for e in episodes for x in fruit_times(e.get("targets", []))]
    placed = sum(int(e.get("n_placed", 0)) for e in episodes)
    total = sum(float(e.get("sim_time_s", 0.0)) + float(e.get("compute_s", 0.0)) for e in episodes)
    o2b = np.array([x[0] for x in ft]) if ft else np.zeros(0)
    cyc = np.array([x[1] for x in ft if np.isfinite(x[1])]) if ft else np.zeros(0)
    return dict(fruits_timed=len(ft),
                observe_to_basket_s_mean=float(o2b.mean()) if len(o2b) else None,
                observe_to_basket_s_median=float(np.median(o2b)) if len(o2b) else None,
                observe_to_basket_s_max=float(o2b.max()) if len(o2b) else None,
                observe_to_basket_under_40s=float((o2b < 40.0).mean()) if len(o2b) else None,
                cycle_s_mean=float(cyc.mean()) if len(cyc) else None,
                compute_s_mean=float(np.mean([x[2] for x in ft])) if ft else None,
                s_per_fruit=float(total / placed) if placed else None,
                motion_s_total=float(sum(float(e.get("sim_time_s", 0.0)) for e in episodes)),
                compute_s_total=float(sum(float(e.get("compute_s", 0.0)) for e in episodes)))
