"""Mission 4 (XGH505H) checks: scenes build, light/detailed share the robot, field layout rules, LOD swaps.

    python -m pytest -q tests/test_mission4.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import pytest
import trimesh

from pinesim.config import PROJECT_ROOT, load_config
from pinesim.model.builder import build
from pinesim.model.builder_xgh import field_layout

CFG = load_config("mission4")


@pytest.fixture(scope="module")
def models():
    out = {}
    for mode in ("xgh_light", "xgh_detail"):
        p = build(mode, 8, CFG)
        out[mode] = mujoco.MjModel.from_xml_path(str(p))
    return out


def test_same_robot_collision(models):
    """Robot, gripper, vehicle boxes and crates: identical collision geometry in both scenes."""
    def coll(m):
        rows = []
        for g in range(m.ngeom):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
            robot = b in ("base", "shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link",
                          "wrist_3_link", "gripper_housing", "finger_left", "finger_right")
            veh = n.startswith(("chassis", "frame_", "basket")) and "_fill_" not in n
            if (robot or veh) and m.geom_contype[g] and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                rows.append((b, int(m.geom_type[g]), *np.round(m.geom_size[g], 5), *np.round(m.geom_pos[g], 5)))
        return sorted(rows)
    a, b = coll(models["xgh_light"]), coll(models["xgh_detail"])
    assert len(a) > 30 and a == b


def test_field_layout_rules():
    L = field_layout(CFG)
    assert len(L) == 100
    assert abs((L[-1]["x"] - L[0]["x"]) - 14.85) < 1e-6           # 15 m field
    for i, p in enumerate(L):
        for q in L[i + 1:]:
            if np.hypot(q["x"] - p["x"], q["y"] - p["y"]) < 0.65:
                assert q["kind"] != p["kind"]                     # never next to the same combination


def test_static_equals_segments():
    """The static plant mesh is exactly the rest pose of the articulated leaf segments (no jump at a LOD swap)."""
    from scipy.spatial import cKDTree
    D = PROJECT_ROOT / "assets_xgh" / "plants"
    info = json.loads((D / "p3.json").read_text())
    pts = []
    for leaf in info["leaves"]:
        for s in leaf["segments"]:
            v = trimesh.load(D / "leaves" / f"{s['mesh']}.stl", force="mesh").vertices
            pts.append(v @ np.array(s["R"]).T + np.array(s["p0"]))
    st = trimesh.load(D / "p3_static.stl", force="mesh")
    d, _ = cKDTree(st.vertices).query(np.vstack(pts))
    assert d.max() < 1e-5


def test_lod_swap_and_release():
    from pinesim.envs.xgh_env import XghEvalEnv
    env = XghEvalEnv("mission4", seed=1, overrides={"eval_field": {"section": [20, 6]}})
    env._reset_state()
    env._scene_reset()
    lod, sim = env.lod, env.sim
    inter = lod.interactive()
    assert len(inter) >= 8
    n = inter[0]
    i = lod.state[n]["slot"]
    p, _ = sim.field.fruit_pose(i, sim.d)
    lay = sim.field.slots[i]["layout"]
    assert np.linalg.norm(p - lay.fruit_center) < 0.01            # fruit sits on its stem
    assert sim.m.body_pos[lod.static_body[n]][2] < -10            # static copy hidden while interactive
    ev = lod.update(float(sim.base_pos[0]) + 1.0)                 # pretend the vehicle drove 1 m
    assert n in ev["released"]
    assert sim.m.body_pos[lod.static_body[n]][2] == 0.0           # static copy back
    assert lod.state[n]["slot"] is None and sim.field.slots[i]["plant"] != n   # slot free or reused elsewhere


def test_crate_freeze_and_unload():
    """A picked fruit lying in a crate is frozen into the crate fill when its slot is recycled; unload clears it."""
    from pinesim.envs.xgh_env import XghEvalEnv
    env = XghEvalEnv("mission4", seed=2, overrides={"eval_field": {"section": [30, 4]}})
    env._reset_state()
    env._scene_reset()
    lod, sim = env.lod, env.sim
    n = lod.interactive()[0]
    i = lod.state[n]["slot"]
    sim.field.detach(i, sim.d)
    c, _ = sim.basket_frame(0)                                    # drop the fruit into crate 0
    q = sim.field.slots[i]["qadr"]
    sim.d.qpos[q:q + 3] = c + np.array([0, 0, 0.02])
    sim.d.qvel[sim.field.slots[i]["dadr"]:sim.field.slots[i]["dadr"] + 6] = 0
    sim.step(60)
    assert sim.basket_of(i) == 0
    sim.baskets[0]["count"] = 1
    lod._release(n)
    k = sim.field.slots[i]["kind"]
    gs = lod.fill[(0, k)][0]
    assert sim.m.geom_pos[gs[0]][2] > 0.8 and sim.m.geom_contype[gs[0]] == 1     # frozen in the crate
    for b in sim.baskets:
        b["count"] = 8
    assert lod.crates_full()
    lod.unload()
    assert sim.m.geom_pos[gs[0]][2] < -10 and all(b["count"] == 0 for b in sim.baskets) and lod.unloads == 1
