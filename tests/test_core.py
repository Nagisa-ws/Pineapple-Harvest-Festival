"""Fast tests (about 2-3 minutes): scenes, plants (fruit 60-70 cm, tilt 0-15 deg), stem physics, realistic rig
(shoulder / window / T2 decision) and its cancellation, XGH505H drive, environment API (default posture, closed
gripper, lamp), leaf uproot rule, transfer of the reference Mission 3 weights, ONNX export parity."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
import pytest

from pinesim.config import load_config
from pinesim.model.builder import build
from pinesim.plant import PlantField, PlantSampler


@pytest.fixture(scope="module")
def cfg():
    return load_config("mission1")


def test_models_build_and_load(cfg):
    for mission, mode, n in (("pretrain", "rig", 1), ("mission1", "platform", 1), ("mission1", "platform_detail", 1),
                             ("mission2", "platform", 2), ("mission3", "xgh_light", 0)):
        m = mujoco.MjModel.from_xml_path(str(build(mode, n, load_config(mission))))
        assert m.nbody > 10
        if mode != "rig":
            assert m.nu >= 8 and mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam") >= 0
            assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tip_site") >= 0


def test_platform_height_is_115cm(cfg):
    m = mujoco.MjModel.from_xml_path(str(build("platform", 1, cfg)))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    base = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")]
    assert abs(base[2] - 1.15) < 1e-9                       # base frame on the 1.15 m mounting surface


def test_plant_generator_varies(cfg):
    s = PlantSampler(cfg)
    s.set_level(2)
    rng = np.random.default_rng(0)
    lays = [s.sample(rng, np.zeros(3), np.array([1.0, 0, 0])) for _ in range(60)]
    n_ros = {sum(l.kind == "rosette" for l in L.leaves) for L in lays}
    assert len(n_ros) >= 3                                 # leaf count varies
    zc = [L.fruit_center[2] for L in lays]
    assert 0.58 < min(zc) and max(zc) < 0.72               # fruit 60-70 cm above the ground (tilt lowers it a bit)
    tilt = [L.tilt_deg for L in lays]
    assert max(tilt) <= 15.0 + 1e-6 and max(tilt) > 10.0   # plants tilted 0-15 deg
    d = [2 * L.fruit_r for L in lays]
    assert max(d) - min(d) > 0.01                          # fruit size varies
    sides = {sd for L in lays for sd in L.blocked_sides}
    assert {"top", "front", "back", "left", "right"} <= sides
    cols = {tuple(c) for L in lays for lf in L.leaves for c in lf.colours}
    assert len(cols) >= 2                                  # green and orange leaves


def test_stem_breaks_by_twist_not_by_rest(cfg):
    m = mujoco.MjModel.from_xml_path(str(build("platform", 1, cfg)))
    d = mujoco.MjData(m)
    f = PlantField(m)
    lay = PlantSampler(cfg).sample(np.random.default_rng(1), np.array([0.6, 0, 0]), np.array([-1.0, 0, 0]))
    lay.break_torque = 1.0
    lay.quat = np.array([1.0, 0, 0, 0])                    # upright for the twist-angle check
    mujoco.mj_forward(m, d)
    f.apply(0, lay, d)
    for _ in range(200):
        mujoco.mj_step(m, d)
    f.record_rest_load(0, d)
    assert not f.check_break(0, d)[0]                      # resting fruit does not break
    s = f.slots[0]
    broke = False
    for k in range(400):                                   # twist the fruit with an external torque about z
        d.xfrc_applied[s["fruit_body"]] = [0, 0, 0, 0, 0, 2.0]
        mujoco.mj_step(m, d)
        if f.check_break(0, d)[0]:
            broke = True
            break
    assert broke
    ang = abs(d.qpos[s["stem_qadr"][0]])
    assert 0.8 < ang < 1.3                                 # ~ break_torque / k_twist = 1.0 rad


def test_rig_shoulder_window_decision():
    """Realistic rig: the shoulder lies above the fruit centre, the window inside the visible-fruit box, several T2
    candidates on the shoulder, T1 projects inside the window, T1-T2 = line length."""
    from pinesim.rig import RigPose, StargazeRig
    cfg = load_config("pretrain")
    rig = StargazeRig(cfg, seed=3, noise=True)
    errs = []
    for az in (0, 120, 240):
        p = RigPose(az, 0.75, 1.0, 0.62)
        rig.set_pose(p)
        res = rig.compute(rig.capture(), p)
        assert res.ok, res.reason
        dec = res.decision
        errs.append(res.diagnostics["fruit_center_error"])
        u0, v0, u1, v1 = dec.window
        b0, c0, b1, c1 = dec.bbox
        assert b0 - 1 <= u0 and u1 <= b1 + 1 and c0 - 1 <= v0 and v1 <= c1 + 1
        assert len(dec.candidates) >= 2
        gt = rig.plant
        for c in dec.candidates:
            h = (c.t2 - gt.fruit_center) @ gt.axis
            assert -0.02 < h < gt.H / 2 + 0.02                 # T2 on the shoulder
            assert abs(np.linalg.norm(c.t1 - c.t2) - cfg["stargaze"]["line_length"]) < 1e-6
            m = cfg["stargaze"]["window_margin_px"] + 1
            assert u0 - m <= c.t1_uv[0] <= u1 + m and v0 - m <= c.t1_uv[1] <= v1 + m
    assert np.median(errs) < 0.03
    rig.cam.close()


def test_rig_cancels_when_camera_moves():
    from pinesim.rig import RigController, RigPose, StargazeRig
    cfg = load_config("pretrain")
    cfg["rig"]["debug_slow_s"] = 1.0
    rig = StargazeRig(cfg, seed=1, noise=False)
    ctrl = RigController(rig, settle_s=0.2)
    p = RigPose(30, 0.75, 1.0, 0.62)
    t0 = time.time()
    while time.time() - t0 < 0.6:                          # rest -> starts computing
        st = ctrl.tick(p)
        time.sleep(0.02)
    assert st == "COMPUTING"
    st = ctrl.tick(RigPose(200, 0.8, 1.05, 0.62))          # user moves to the other side
    assert st == "MOVING" and ctrl.n_cancelled == 1
    p2 = RigPose(200, 0.8, 1.05, 0.62)
    t0 = time.time()
    while time.time() - t0 < 8.0 and ctrl.state not in ("DONE", "FAILED"):
        ctrl.tick(p2)
        time.sleep(0.02)
    assert ctrl.state == "DONE" and ctrl.result.pose.azimuth_deg == 200
    rig.cam.close()


def test_vehicle_drives_and_holds():
    from pinesim.sim import Sim
    from pinesim.vehicle import VehicleController
    cfg = load_config("mission3")
    sim = Sim(cfg, "xgh_light", 8)
    mujoco.mj_forward(sim.m, sim.d)
    sim.brake()
    sim.step(40)
    vc = VehicleController(sim)
    vc.drive_to(0.30)
    for _ in range(1500):
        if vc.update():
            break
        sim.step(1)
    sim.step(20)
    p, yaw = sim.vehicle_pose()
    assert abs(p[0] - 0.30) < 0.03 and abs(p[1]) < 0.02 and abs(yaw) < 0.02
    x0 = p[0]
    sim.step(100)
    assert abs(sim.vehicle_pose()[0][0] - x0) < 0.002      # holds position


def test_default_posture_closed_gripper_lamp():
    """Every session starts in the default posture of the user's picture with the gripper closed."""
    from pinesim.envs import make_env
    env = make_env("mission1", training=False, seed=0)
    env._reset_state()
    env._scene_reset()
    q = env.sim.q_arm()
    assert np.max(np.abs(q - env.default_q())) < 1e-6
    assert abs(env.sim.gripper_gap() - env.cfg["gripper"]["min_opening"]) < 1e-6
    tcp, tip = env.sim.tcp_pos, env.sim.tip_pos
    assert abs(np.linalg.norm(tip - tcp) - env.sim.tip_offset) < 1e-6
    assert abs(env.sim.tcp_rot()[2, 2] + np.sin(np.radians(35.0))) < 0.05              # closed jaws point horizontally (user's picture)
    assert env.lamp_state == "observe"


def test_leaf_uproot_ends_session_touch_does_not():
    from pinesim.envs import make_env
    env = make_env("mission1", training=False, seed=0)
    env._reset_state()
    env._scene_reset()
    fld, d = env.sim.field, env.sim.d
    s = fld.slots[0]
    j = s["root_qadr"][0]
    d.qpos[j[0]] = 0.6                                     # bent leaf (touched): no failure
    env._update_leaf_state()
    assert not env._leaf_uprooted() and not env.done
    d.qpos[j[0]] = 1.3                                     # beyond the uproot limit: session ends
    env._update_leaf_state()
    assert env._leaf_uprooted() and env.failure == "leaf_uprooted" and env.lamp_state == "fail"


def test_env_api_and_reference_episode_runs():
    from pinesim.envs import make_env
    env = make_env("mission1", training=False, seed=0)
    obs, info = env.reset(seed=106)
    assert obs.shape == (63,) and np.all(np.isfinite(obs))
    n = 0
    done = env.done
    while not done and n < 400:
        obs, r, te, tr, info = env.step(np.zeros(6))
        assert np.all(np.isfinite(obs)) and np.isfinite(r)
        done = te or tr
        n += 1
    assert "phase" in info


def test_reference_mission3_transfer_shapes():
    import gymnasium as gym
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from pinesim.vendor.obs_layout import OBS_DIM

    class Dummy(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        action_space = gym.spaces.Box(-1, 1, (6,), np.float32)

        def reset(self, seed=None, options=None):
            return np.zeros(OBS_DIM, np.float32), {}

        def step(self, a):
            return np.zeros(OBS_DIM, np.float32), 0.0, False, False, {}

    ref = SAC.load(str(Path(__file__).resolve().parents[1] / "third_party/abc_mission3/best_model.zip"), device="cpu")
    vn = VecNormalize(DummyVecEnv([Dummy]))
    m = SAC("MlpPolicy", vn, policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256])), device="cpu")
    m.policy.load_state_dict(ref.policy.state_dict(), strict=True)
    assert ref.observation_space.shape == (OBS_DIM,) and ref.action_space.shape == (6,)


def test_onnx_export_parity(tmp_path):
    import gymnasium as gym
    from stable_baselines3 import SAC
    from pinesim.vendor.obs_layout import OBS_DIM
    from pinesim.vendor.to_onnx import export_policy_to_onnx
    from pinesim.vendor.verify_onnx import verify_onnx
    ref = SAC.load(str(Path(__file__).resolve().parents[1] / "third_party/abc_mission3/best_model.zip"), device="cpu")
    mean, var = np.random.default_rng(0).normal(size=OBS_DIM), np.ones(OBS_DIM) * 0.5
    p = tmp_path / "p.onnx"
    export_policy_to_onnx(ref, mean, var, 10.0, 1e-8, str(p), OBS_DIM)
    mx, mn, ok = verify_onnx(ref, mean, var, 10.0, 1e-8, str(p), OBS_DIM, tol=1e-4)
    assert ok and mx < 1e-4
