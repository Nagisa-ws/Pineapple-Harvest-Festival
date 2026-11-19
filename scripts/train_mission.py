"""Train the residual SAC policy of one mission, transferring weights from the previous one.

  python scripts/train_mission.py --mission mission1                      # init: reference Mission 3 (abc_SAC_xyzBOT)
  python scripts/train_mission.py --mission mission2                      # init: checkpoints/mission1
  python scripts/train_mission.py --mission mission3                      # init: checkpoints/mission2 (XGH505H light scene)
  python scripts/train_mission.py --mission mission1 --resume --timesteps 50000
  python scripts/train_mission.py --mission mission1 --init zero          # no transfer (ablation)
  python scripts/train_mission.py --mission mission1 --n-envs 4           # parallel environments (more CPU cores)

Outputs (checkpoints/<mission>/): best_model.zip + best_vecnormalize.pkl (periodic held-out evaluation), final_*,
replay_buffer.pkl (resume), train_log.csv, eval_log.csv, init_info.json (what was transferred), TensorBoard logs (tb/).
After training: models/<mission>_v<N>.onnx + .json (normalisation baked in, parity-checked against PyTorch).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinesim.rl.train import train


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mission", required=True, choices=["mission1", "mission2", "mission3"])
    ap.add_argument("--init", default=None, help="reference_m3 | zero | <checkpoint dir>; default = previous mission")
    ap.add_argument("--out", default=None)
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--level", type=int, default=None, help="plant difficulty (0 sparse .. 2 reference)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-export", action="store_true")
    a = ap.parse_args()
    res = train(a.mission, out=a.out, timesteps=a.timesteps, init=a.init, resume=a.resume, n_envs=a.n_envs,
                seed=a.seed, level=a.level, export=not a.no_export)
    print(res)


if __name__ == "__main__":
    main()
