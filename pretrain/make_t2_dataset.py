"""Training data for the T2 diffusion policy: views of the realistic rig (random combination, fruit height, tilt,
camera pose), candidates + window features from the CAMERA, scores from the TRUE leaf geometry.

  python pretrain/make_t2_dataset.py --scenes 900 --seed0 7 --out outputs/pretrain/t2_data.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pinesim.config import PROJECT_ROOT, load_config
from pinesim.decision.window import build_candidates, true_scores
from pinesim.rig import StargazeRig
from pinesim.vision.perception import shoulder_in_view
from pinesim.vision.stargaze_cam import obstacle_points

MAXC = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=None)
    ap.add_argument("--views", type=int, default=None)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", default="outputs/pretrain/t2_data.npz")
    a = ap.parse_args()
    cfg = load_config("pretrain")
    ds = cfg["dataset"]
    n_sc = a.scenes or int(ds["n_scenes"])
    n_v = a.views or int(ds["views_per_scene"])
    rig = StargazeRig(cfg, seed=a.seed0, policy=False)
    rng = np.random.default_rng(a.seed0 + 99)
    F, Gl, U, V, CS, TS, NC, META = [], [], [], [], [], [], [], []
    t0 = time.time()
    for sc in range(n_sc):
        rig.new_plant(a.seed0 * 100000 + sc)
        gt = rig.plant
        for _ in range(n_v):
            az = rng.uniform(0, 2 * np.pi)
            el = np.radians(rng.uniform(*ds["elevation_deg_range"]))
            dist = rng.uniform(*ds["radius_range"])
            c = gt.fruit_center + rng.normal(0, ds["aim_jitter"], 3)
            head = gt.fruit_center + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            if head[2] < 0.15:
                continue
            rig.place_camera(head, c)
            fr = rig.capture()
            per, res = rig.perceive(fr)
            if not res.fruits:
                continue
            fruit = max(res.fruits, key=lambda f: f.n_points)
            if np.linalg.norm(fruit.center - gt.fruit_center) > 0.06:
                continue
            pts, rad = obstacle_points(fruit, res.leaf_points, cfg)
            sh = shoulder_in_view(per.views[0], fruit)
            cands, feats, glob, sh = build_candidates(fruit, per.views[0], pts, rad, cfg, sh)
            if len(cands) < 2:
                continue
            ts = true_scores(cands, gt.leaf_pts, gt.leaf_rad, cfg)
            n = min(len(cands), MAXC)
            u = np.zeros((MAXC, 2), np.float32)
            u[:n] = [c.uv for c in cands[:n]]
            cs = np.full(MAXC, -9.0, np.float32)
            cs[:n] = [c.score for c in cands[:n]]
            tsa = np.full(MAXC, -9.0, np.float32)
            tsa[:n] = ts[:n]
            F.append(feats); Gl.append(glob); U.append(u); CS.append(cs); TS.append(tsa); NC.append(n)
            META.append([gt.plant, gt.fruit, gt.crown, gt.zc, np.degrees(gt.tilt), np.degrees(el), dist])
        if sc % 25 == 0:
            print(f"scene {sc}/{n_sc}: {len(F)} samples, {time.time() - t0:.0f} s", flush=True)
    out = Path(a.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, feats=np.array(F), glob=np.array(Gl), uv=np.array(U), cam_score=np.array(CS),
                        true_score=np.array(TS), n=np.array(NC), meta=np.array(META))
    print("->", out, len(F), "samples", f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
