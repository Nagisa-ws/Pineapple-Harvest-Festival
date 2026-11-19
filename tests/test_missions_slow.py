"""Slow end-to-end checks (several minutes): reference controller (no RL) on the light scenes."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pinesim.envs import make_env


def test_mission1_full_cycle_physical():
    """Window decision -> tip T1 -> T2 -> open -> advance -> grip -> twist -> pull -> basket -> default posture."""
    env = make_env("mission1", training=False, seed=0)
    ok = 0
    for seed in (100, 105, 106):
        info = env.run_episode(None, seed=seed, max_rl_steps=6000)
        ev = " ".join(e for _, e in info["events"])
        if info["is_success"]:
            ok += 1
            assert "window" in ev and "fruit 0 in basket" in ev
            assert np.max(np.abs(env.sim.q_arm() - env.default_q_near())) < 0.05   # back in the default posture (+-2 pi)
    assert ok >= 2


def test_mission2_detects_two_fruits():
    env = make_env("mission2", training=False, seed=0)
    env.reset(seed=202)
    assert len(env.fruits_scan) >= 2
