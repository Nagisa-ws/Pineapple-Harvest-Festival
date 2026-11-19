"""Camera self-check statistics of the Stargaze rig: many plants x camera poses, metrics vs ground truth.

  python pretrain/eval_rig.py --plants 20 --level 2            # -> outputs/pretrain/rig_eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pinesim.config import PROJECT_ROOT, load_config
from pinesim.rig import RigPose, StargazeRig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plants", type=int, default=20)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=500)
    ap.add_argument("--no-noise", action="store_true")
    a = ap.parse_args()
    cfg = load_config("pretrain")
    rows = []
    poses = [RigPose(az, r, h, 0.40) for az in (0, 60, 120, 180, 240, 300) for (r, h) in ((0.9, 0.95), (1.2, 1.4))]
    for k in range(a.plants):
        rig = StargazeRig(cfg, seed=a.seed0 + k, level=a.level, noise=not a.no_noise)
        for p in poses:
            rig.set_pose(p)
            res = rig.compute(rig.capture(), p)
            d = dict(res.diagnostics)
            rows.append(dict(plant=k, az=p.azimuth_deg, radius=p.radius, height=p.height, ok=res.ok, reason=res.reason,
                             face=res.face, ms=res.compute_s * 1000, **d))
        rig.cam.close()
    ok = [r for r in rows if r["ok"]]
    det = [r for r in rows if "fruit_center_error" in r]

    def stat(key, rr):
        v = np.array([r[key] for r in rr if key in r], float)
        return dict(mean=float(np.mean(v)), median=float(np.median(v)), p90=float(np.percentile(v, 90)), max=float(np.max(v))) if len(v) else None

    summ = dict(n_views=len(rows), plants=a.plants, level=a.level, noise=not a.no_noise,
                fruit_detected=len(det) / len(rows), line_found=len(ok) / len(rows),
                fruit_center_error_m=stat("fruit_center_error", det), radius_error_m=stat("radius_error", det),
                reprojection_px=stat("reprojection_px", det), fruit_surface_mean_abs_mm=stat("fruit_surface_mean_abs_mm", det),
                line_clearance_true_leaves_m=stat("line_clearance_true_leaves", ok),
                line_clearance_perceived_m=stat("line_clearance_perceived", ok),
                negative_true_clearance_frac=float(np.mean([r["line_clearance_true_leaves"] < 0 for r in ok])) if ok else None,
                faces={f: sum(r["face"] == f for r in ok) for f in ("front", "top", "left", "right")},
                compute_ms=stat("ms", rows))
    out = PROJECT_ROOT / "outputs" / "pretrain"
    out.mkdir(parents=True, exist_ok=True)
    (out / "rig_eval.json").write_text(json.dumps(dict(summary=summ, views=rows), indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
