"""Run episodes and print a detailed failure diagnosis (developer tool).
  python scripts/diagnose.py --mission mission1 --seeds 105 113
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pinesim.envs import make_env
from pinesim.vendor.pose_utils import quat_angle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="mission1")
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    a = ap.parse_args()
    env = make_env(a.mission, training=False, seed=0)
    for s in a.seeds:
        labels = []
        stem = []

        def cb(label):
            labels.append(label)
            if env.target_slot is not None:
                brk, fe, te = env.sim.field.check_break(env.target_slot, env.sim.d)
                stem.append((env.step_count, label, round(fe, 1), round(te, 2)))
        env.frame_callback = cb
        info = env.run_episode(None, seed=s)
        print(f"=== seed {s}: success={info['is_success']} failure={info['failure']}  phases={dict(Counter(labels))}")
        for e in info["events"]:
            print("    ", e)
        if env.plan is not None:
            print("     plan:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in env.plan.info.items()},
                  "opening", round(env.plan.opening, 3))
        if env.route is not None and env.phase >= 0:
            sim = env.sim
            print("     phase", env.phase, "route idx", env.route.idx, "/", len(env.route.wps), "dist to end",
                  round(float(np.linalg.norm(sim.tcp_pos - env.route.end)), 3))
            if env.plan is not None:
                print("     orient err to line quat (deg)", round(np.degrees(quat_angle(sim.tcp_quat(), env.plan.line.quat)), 1),
                      "ref action", np.round(env._ref_action, 2))
        c = env._contacts
        if c:
            print("     contacts:", {k: (round(v, 1) if isinstance(v, float) else v) for k, v in c.items() if k != "wrist"})
        print("     leaf bend max (deg)", round(np.degrees(env._leaf_bend), 1))
        if stem:
            print("     last stem loads (step, phase, pull excess N, twist N m):", stem[-3:])
        if env.target_slot is not None:
            lay = env.sim.field.slots[env.target_slot]["layout"]
            print("     thresholds: pull", round(lay.break_force, 1), "twist", round(lay.break_torque, 2), "| plant",
                  {k: (round(v, 3) if isinstance(v, float) else v) for k, v in lay.summary().items()})


if __name__ == "__main__":
    main()
