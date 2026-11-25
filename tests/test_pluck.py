"""Pluck harvest (harvest.method = pluck): stem kink / tear-off rule, the method choice, one harvest by plucking."""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np


def _env(method="twist"):
    from pinesim.envs import make_env
    env = make_env("mission1", training=False, seed=777000, overrides={"harvest": {"method": method}})
    env.reset(seed=777000)
    return env


def test_stem_kinks_and_tears_off_when_bent():
    """Below the yield moment the stem is elastic; beyond it the bend stays (kink); beyond the plant's own angle
    (35-50 deg) the stem tears off."""
    env = _env()
    fld, d, m = env.sim.field, env.sim.d, env.sim.m
    i = env.target_slot if env.target_slot is not None and env.target_slot >= 0 else fld.active_slots()[0]
    s = fld.slots[i]
    a = s["stem_qadr"][1:3]
    assert np.radians(35.0) <= s["break_bend"] <= np.radians(50.0)
    lim = fld.bend_yield / s["k_bend"]                       # elastic bend at the yield moment (rad)
    d.qpos[a] = [0.5 * lim, 0.0]
    fld.update_stems(d)
    assert np.allclose(m.qpos_spring[a], 0.0)                # elastic: no kink
    d.qpos[a] = [np.radians(20.0), 0.0]
    fld.update_stems(d)
    assert abs(m.qpos_spring[a][0] - (np.radians(20.0) - lim)) < 1e-9      # kinked: the bend stays
    assert not fld.check_break(i, d)[0]
    d.qpos[a] = [s["break_bend"] + np.radians(1.0), 0.0]
    assert fld.check_break(i, d)[0]                          # torn off
    d.qpos[a] = 0.0
    fld.arm_stem(i, d)
    assert np.allclose(m.qpos_spring[a], 0.0)                # a fresh plant has no kink


def test_harvest_method_choice(monkeypatch):
    """--harvest wins, then PINESIM_HARVEST; without a terminal (tests, scripts) the configuration decides - no
    question is asked."""
    import io
    from pinesim.config import choose_harvest_method, load_config
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.delenv("PINESIM_HARVEST", raising=False)
    assert choose_harvest_method("pluck", "mission1") == "pluck"
    assert choose_harvest_method("twist", "mission3") == "twist"
    assert choose_harvest_method(None, "mission1") == load_config("mission1")["harvest"]["method"] == "twist"
    monkeypatch.setenv("PINESIM_HARVEST", "pluck")
    assert choose_harvest_method(None, "mission1") == "pluck"
    assert choose_harvest_method("twist", "mission1") == "twist"


def test_pluck_harvests_a_fruit():
    """Light Mission 1 scene: the fruit is tilted off its stalk (not twisted) and ends in the crate."""
    env = _env("pluck")
    info = env.run_episode(None, seed=777000)
    assert info["is_success"], info.get("failure")
    assert env.rec.method == "pluck" and env.rec.pluck_deg > 40.0 and env.rec.twist_deg == 0.0
    assert any("stem tore off at tilt" in e for _, e in info["events"])
