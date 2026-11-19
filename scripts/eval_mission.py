"""Evaluate a mission over held-out seeds with the reference controller alone or with a residual policy.

  python scripts/eval_mission.py --mission mission1 --seeds 900000 20                 # reference only
  python scripts/eval_mission.py --mission mission1 --seeds 900000 20 --onnx models/mission1_v1.onnx
  python scripts/eval_mission.py --mission mission2 --seeds 910000 20 --model checkpoints/mission2/best_model.zip \
        --vecnorm checkpoints/mission2/best_vecnormalize.pkl
Writes outputs/eval/<mission>_<tag>.json with per-episode outcomes and summary (Wilson 95 % CI).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pinesim.config import PROJECT_ROOT
from pinesim.envs import make_env
from pinesim.rl.policy import load_policy


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="mission1")
    ap.add_argument("--seeds", nargs=2, type=int, default=[900000, 10], metavar=("FIRST", "N"))
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--vecnorm", default=None)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. limits.leaf_uproot_rad=1.0")
    a = ap.parse_args()
    over = {"noise": {"enabled": False}} if a.no_noise else {}
    import yaml
    for kv in a.set:
        k, v = kv.split("=", 1)
        cur = over
        parts = k.split(".")
        for q in parts[:-1]:
            cur = cur.setdefault(q, {})
        cur[parts[-1]] = yaml.safe_load(v)
    over = over or None
    env = make_env(a.mission, training=False, seed=a.seeds[0], overrides=over, level=a.level)
    policy, desc = load_policy(onnx=a.onnx, model=a.model, vecnorm=a.vecnorm)
    tag = a.tag or ("reference" if policy is None else Path(a.onnx or a.model).stem)
    rows = []
    t_all = time.time()
    for k in range(a.seeds[1]):
        seed = a.seeds[0] + k
        t0 = time.time()
        info = env.run_episode(policy, seed=seed)
        row = dict(seed=seed, success=bool(info["is_success"]), failure=info.get("failure", ""),
                   n_placed=info.get("n_placed", 0), n_targets=info.get("n_targets", 1), steps=info["steps_total"],
                   rl_steps=info["rl_steps"], ret=round(float(info["return"]), 2), wall_s=round(time.time() - t0, 1),
                   targets=info.get("targets", []), timing=info.get("timing", {}))
        for kk in ("stops", "vehicle_travel", "vehicle_drift_max", "damage_events", "skipped"):
            if kk in info:
                row[kk] = info[kk]
        rows.append(row)
        print(f"seed {seed}: success={row['success']} placed={row['n_placed']}/{row['n_targets']} fail={row['failure']} "
              f"steps={row['steps']} rl={row['rl_steps']} ret={row['ret']} ({row['wall_s']} s)", flush=True)
    n = len(rows)
    k = sum(r["success"] for r in rows)
    fruits = sum(r["n_placed"] for r in rows)
    fruits_n = sum(r["n_targets"] for r in rows)
    est = [t["est_error"] for r in rows for t in r["targets"] if t.get("est_error") == t.get("est_error")]
    summ = dict(mission=a.mission, policy=desc, n=n, success=k, success_rate=k / max(n, 1), ci95=wilson(k, n),
                fruits_placed=fruits, fruits_total=fruits_n, fruit_rate=fruits / max(fruits_n, 1),
                failures=dict(Counter(r["failure"] for r in rows if not r["success"])),
                est_error_cm=dict(mean=float(np.mean(est) * 100) if est else None,
                                  median=float(np.median(est) * 100) if est else None),
                mean_steps=float(np.mean([r["steps"] for r in rows])), wall_s=round(time.time() - t_all, 1),
                seeds=a.seeds, noise=not a.no_noise, level=env.level, overrides=a.set)
    out = PROJECT_ROOT / "outputs" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.mission}_{tag}.json"
    path.write_text(json.dumps(dict(summary=summ, episodes=rows), indent=1, default=str), encoding="utf-8")
    print(json.dumps(summ, indent=1, default=str))
    print("->", path)


if __name__ == "__main__":
    main()
