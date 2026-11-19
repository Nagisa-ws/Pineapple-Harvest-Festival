"""Debug helper: run one episode and save a contact sheet of external-camera frames (MuJoCo offscreen renderer).
  python scripts/debug_frames.py --mission mission1 --seed 100 --every 40 --out outputs/debug/m1_s100.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import mujoco
import numpy as np

from pinesim.envs import make_env
from pinesim.rl.policy import load_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="mission1")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--every", type=int, default=40)
    ap.add_argument("--out", default=None)
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--labels", nargs="*", default=None, help="only record frames with these phase labels")
    a = ap.parse_args()
    env = make_env(a.mission, training=False, seed=a.seed)
    m, d = env.sim.m, env.sim.d
    r = mujoco.Renderer(m, 240, 320)
    cam = mujoco.MjvCamera()
    frames, k = [], [0]
    last = [""]

    def cb(label):
        k[0] += 1
        if a.labels and label not in a.labels:
            return
        if k[0] % a.every == 0 or label != last[0]:
            base = env.sim.base_pos
            cam.lookat[:] = base + np.array([0.0, 0.0, -0.6])
            cam.distance, cam.azimuth, cam.elevation = 2.6, 135.0, -25.0
            r.update_scene(d, cam)
            img = r.render().copy()
            cv2.putText(img, f"{env.step_count} {label}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            frames.append(img)
        last[0] = label

    env.frame_callback = cb
    pol, _ = load_policy(onnx=a.onnx)
    info = env.run_episode(pol, seed=a.seed)
    print("result:", info["is_success"], info["failure"])
    for e in info["events"]:
        print("  ", e)
    if not frames:
        return
    cols = a.cols
    while len(frames) % cols:
        frames.append(np.zeros_like(frames[0]))
    rows = [np.hstack(frames[i:i + cols]) for i in range(0, len(frames), cols)]
    sheet = np.vstack(rows)
    out = Path(a.out or f"outputs/debug/{a.mission}_s{a.seed}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet[:, :, ::-1])
    print("->", out, sheet.shape)


if __name__ == "__main__":
    main()
