"""Mission 3 (version 5, XGH505H) checks: scenes build, light/realistic share the robot, 15 m field layout rules,
three-plant level of detail, crate freeze/unload, colours of the realistic assets.

    python -m pytest -q tests/test_mission3.py
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

CFG = load_config("mission3")


@pytest.fixture(scope="module")
def models():
    return {mode: mujoco.MjModel.from_xml_path(str(build(mode, 8, CFG))) for mode in ("xgh_light", "xgh_detail")}


def _robot_collision(m):
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


def test_same_robot_collision(models):
    a, b = _robot_collision(models["xgh_light"]), _robot_collision(models["xgh_detail"])
    assert len(a) > 30 and a == b


def test_platform_light_and_detail_same_robot():
    c1 = load_config("mission1")
    a = _robot_collision(mujoco.MjModel.from_xml_path(str(build("platform", 1, c1))))
    b = _robot_collision(mujoco.MjModel.from_xml_path(str(build("platform_detail", 1, c1))))
    assert len(a) > 20 and a == b


def test_field_layout_rules():
    L = field_layout(CFG)
    assert len(L) == 60                                            # 2 x 30 plants
    assert abs((L[-1]["x"] - L[0]["x"]) - 14.75) < 1e-6            # ~15 m field
    xs = sorted(p["x"] for p in L)
    assert abs(min(np.diff(xs)) - 0.25) < 1e-6                     # zig-zag 0.25 m, rows 0.50 m
    for i, p in enumerate(L):
        assert 0.0 <= np.degrees(p["tilt"]) <= 15.0 + 1e-9
        for q in L[i + 1:]:
            if np.hypot(q["x"] - p["x"], q["y"] - p["y"]) < 0.65:
                assert q["kind"] != p["kind"]                     # never next to the same combination


def test_static_equals_segments():
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


def test_three_plant_rule():
    from pinesim.envs.xgh_env import XghEvalEnv
    env = XghEvalEnv("mission3", seed=1, overrides={"eval_field": {"section": [10, 6]}})
    env._reset_state()
    env._scene_reset()
    lod, sim = env.lod, env.sim
    L = lod.layout
    t = 12
    w = lod.wanted(t, float(sim.base_pos[0]))
    assert w[0] == t and len(w) == 2                               # start: target + next
    ev = lod.set_interactive(w)
    assert sorted(lod.interactive()) == sorted(w)
    for n in w:
        assert sim.m.body_pos[lod.static_body[n]][2] < -10        # static copies hidden
    i = lod.state[t]["slot"]
    p, _ = sim.field.fruit_pose(i, sim.d)
    from pinesim.model.builder_xgh import plant_quat
    M = np.zeros(9)
    mujoco.mju_quat2Mat(M, plant_quat(L[t]))
    assert np.linalg.norm(p[:2] - (np.array([L[t]["x"], L[t]["y"]]) + (M.reshape(3, 3) @ [0, 0, p[2]])[:2])) < 0.05
    lod.state[t]["fruit"] = "picked"
    lod.prev_plant = t
    nxt = w[1]
    w2 = lod.wanted(nxt, float(sim.base_pos[0]))
    assert w2[0] == nxt and t in w2 and len(w2) == 3               # target + next + previous
    lod.set_interactive(w2)
    assert len(lod.interactive()) == 3
    far = [n for n in range(len(L)) if n not in w2][0]
    assert sim.m.body_pos[lod.static_body[far]][2] == 0.0          # every other plant stays a static mesh


def test_crate_freeze_and_unload():
    from pinesim.envs.xgh_env import XghEvalEnv
    env = XghEvalEnv("mission3", seed=2, overrides={"eval_field": {"section": [20, 4]}})
    env._reset_state()
    env._scene_reset()
    lod, sim = env.lod, env.sim
    n = 21
    lod.set_interactive([n])
    i = lod.state[n]["slot"]
    sim.field.detach(i, sim.d)
    c, _ = sim.basket_frame(0)
    q = sim.field.slots[i]["qadr"]
    sim.d.qpos[q:q + 3] = c + np.array([0, 0, 0.02])
    sim.d.qpos[q + 3:q + 7] = [1, 0, 0, 0]
    sim.d.qvel[sim.field.slots[i]["dadr"]:sim.field.slots[i]["dadr"] + 6] = 0
    sim.step(60)
    assert sim.basket_of(i) == 0
    sim.baskets[0]["count"] = 1
    lod._release(n)
    k = sim.field.slots[i]["kind"]
    gs = lod.fill[(0, k)][0]
    assert sim.m.geom_pos[gs[0]][2] > 0.8 and sim.m.geom_contype[gs[0]] == 1
    for b in sim.baskets:
        b["count"] = 8
    assert lod.crates_full()
    lod.unload()
    assert sim.m.geom_pos[gs[0]][2] < -10 and all(b["count"] == 0 for b in sim.baskets) and lod.unloads == 1


def test_asset_colours_and_mesh_integrity():
    s = json.loads((PROJECT_ROOT / "assets_xgh" / "summary.json").read_text())
    assert all(c["degenerate"] == 0 and not c["nan"] for c in s["mesh_checks"])
    import colorsys
    h, sat, v = colorsys.rgb_to_hsv(*s["fruit_1"]["mean_color"])
    assert 40 / 360 < h < 65 / 360                                 # yellow fruit
    h, sat, v = colorsys.rgb_to_hsv(*s["crown_1"]["mean_color"])
    assert 90 / 360 < h < 150 / 360 and v < 0.5                     # dark green crown
    info = json.loads((PROJECT_ROOT / "assets_xgh" / "plants" / "p1.json").read_text())
    assert {"green", "orange"} <= set(info["static_classes"])      # green and orange leaves
