"""Export a trained checkpoint (default: best_model.zip selected by the periodic evaluation) to versioned ONNX.

  python scripts/export_onnx.py --mission mission1                     # checkpoints/mission1/best_*
  python scripts/export_onnx.py --mission mission2 --which final
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from pinesim.config import PROJECT_ROOT, load_config
from pinesim.envs import make_env
from pinesim.rl.train import export_onnx
from pinesim.vendor.trainer import ArmSAC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--which", default="best", choices=["best", "final"])
    a = ap.parse_args()
    cfg = load_config(a.mission)
    out = Path(a.ckpt or PROJECT_ROOT / "checkpoints" / a.mission)
    model = ArmSAC.load(str(out / f"{a.which}_model.zip"), device="cpu")
    venv = DummyVecEnv([lambda: make_env(a.mission, training=False, seed=0)])
    vn = VecNormalize.load(str(out / f"{a.which}_vecnormalize.pkl"), venv)
    meta = json.loads((out / "init_info.json").read_text()) if (out / "init_info.json").exists() else {}
    info_p = out / f"{a.which}_model_info.json"
    meta["selected"] = json.loads(info_p.read_text()) if info_p.exists() else f"{a.which}_model.zip"
    print(export_onnx(a.mission, model, vn, cfg, out, meta, 0.0))


if __name__ == "__main__":
    main()
