"""Re-export a Meshcat HTML from a saved recording (.npz) without re-running the simulation (e.g. after a viewer
change). The model is rebuilt exactly as the evaluation built it.

  python scripts/reexport_html.py --mission mission1 --mode platform_detail \
      --recording outputs/eval_realistic/mission1_mission1_v1/rec_900000.npz --html outputs/m1.html
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

from pinesim.envs import make_env
from pinesim.viz.xgh_recorder import export_html


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mission", required=True, choices=["mission1", "mission2", "mission3"])
    ap.add_argument("--mode", default=None, help="task.mode used for the recording (platform_detail for eval_realistic)")
    ap.add_argument("--recording", required=True)
    ap.add_argument("--html", required=True)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    env = make_env(a.mission, training=False, seed=0, overrides={"task": {"mode": a.mode}} if a.mode else None)
    res = json.loads(str(np.load(a.recording)["result"][0]))
    lines = [a.title or f"<b>{a.mission.replace('mission', 'Misi ')}</b> (seed {res.get('seed', '-')})",
             f"buah di keranjang: {res.get('n_placed', '-')} / {res.get('n_targets', '-')}"
             + ("" if res.get("success", True) else f" &nbsp; gagal: {res.get('failure', '')}"),
             "lampu: kuning = observasi, biru = aksi, hijau = sukses/standby, merah = gagal"]
    print(export_html(env.sim.m, a.recording, a.html, info_lines=lines))


if __name__ == "__main__":
    main()
