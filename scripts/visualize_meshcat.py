"""Record one episode of any mission (light training scene by default) and export a standalone Drake-Meshcat HTML.

  python scripts/visualize_meshcat.py --mission mission1 --seed 900004 --html outputs/m1_light.html
  python scripts/visualize_meshcat.py --mission mission2 --seed 910000 --onnx models/mission2_v1.onnx --html outputs/m2.html
  python scripts/visualize_meshcat.py --mission mission1 --mode platform_detail --seed 900000 --html outputs/m1_real.html

For the realistic evaluations use scripts/eval_realistic.py (Missions 1-2) and scripts/eval_xgh.py /
eval_xgh_sections.py (Mission 3). The HTML has the same camera panel (free / top / gripper camera with bounding box,
window, shoulder, T2 candidates, T1, T2) and the status lamp.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinesim.envs import make_env
from pinesim.rl.policy import load_policy
from pinesim.viz.xgh_recorder import XghRecorder, export_html


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mission", default="mission1", choices=["mission1", "mission2", "mission3"])
    ap.add_argument("--seed", type=int, default=900000)
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--mode", default=None, help="override task.mode (platform, platform_detail, xgh_light)")
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--max-rl-steps", type=int, default=8000)
    ap.add_argument("--html", required=True)
    a = ap.parse_args()
    ov = {"task": {"mode": a.mode}} if a.mode else None
    env = make_env(a.mission, training=False, seed=a.seed, overrides=ov)
    policy, desc = load_policy(onnx=a.onnx)
    rec = XghRecorder(env.sim, every=a.every)
    env.frame_callback, env.overlay_callback = rec, rec.overlay
    info = env.run_episode(policy, seed=a.seed, max_rl_steps=a.max_rl_steps)
    out = Path(a.html)
    out.parent.mkdir(parents=True, exist_ok=True)
    rp = out.with_suffix(".npz")
    row = dict(success=bool(info["is_success"]), failure=info.get("failure", ""), n_placed=info.get("n_placed", 0),
               n_targets=info.get("n_targets", 1))
    rec.save(rp, result=row)
    lines = [f"<b>{a.mission.replace('mission', 'Misi ')}</b> (seed {a.seed}, adegan {env.cfg['task']['mode']})",
             f"buah di keranjang: {row['n_placed']} / {row['n_targets']}" + ("" if row["success"] else f" &nbsp; gagal: {row['failure']}"),
             f"policy: {desc}"]
    print(row, export_html(env.sim.m, rp, out, info_lines=lines))


if __name__ == "__main__":
    main()
