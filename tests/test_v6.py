"""Version 6 checks: static neighbour rows (Mission 3), crate drop pattern, per-fruit timing, view-move leaf guard.

    python -m pytest -q tests/test_v6.py
"""
from __future__ import annotations

import copy
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import pytest

from pinesim.config import load_config
from pinesim.model.builder import build
from pinesim.model.builder_xgh import side_rows_layout
from pinesim.timing import fruit_times, summarize

CFG = load_config("mission3")


def test_side_rows_default_gap_and_user_gap():
    """Edge gap = distance between the outer plant row of the harvested bed and the nearest row of the neighbour."""
    oy = float(CFG["field"]["row_offset_y"])
    lay = side_rows_layout(CFG, detail=True)
    assert lay, "side rows are on by default"
    assert CFG["field"]["side_rows"]["edge_gap"] == pytest.approx(0.50)
    near = min(abs(p["y"]) for p in lay)
    assert near - oy == pytest.approx(0.50, abs=1e-6)
    assert {p["side"] for p in lay} == {"L", "R"}
    cfg2 = copy.deepcopy(CFG)
    cfg2["field"]["side_rows"]["edge_gap"] = 0.80
    assert min(abs(p["y"]) for p in side_rows_layout(cfg2, detail=True)) - oy == pytest.approx(0.80, abs=1e-6)
    cfg2["field"]["side_rows"]["enabled"] = False
    assert side_rows_layout(cfg2, detail=True) == []


@pytest.mark.parametrize("mode", ["xgh_light", "xgh_detail"])
def test_side_rows_are_static_and_do_not_collide(mode):
    m = mujoco.MjModel.from_xml_path(str(build(mode, 8, CFG)))
    side = [b for b in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("splant_side_")]
    assert side, "neighbour-row bodies exist"
    for b in side:
        assert m.body_jntnum[b] == 0 and m.body_mocapid[b] < 0, "no joint, not a mocap body: a fixed mesh"
        assert m.body_weldid[b] == 0, "rigidly attached to the world"
    gs = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in set(side)]
    assert gs
    assert not np.any(m.geom_contype[gs]) and not np.any(m.geom_conaffinity[gs]), "visual only"


def test_crate_drop_pattern_has_no_repeated_place():
    """Every fruit of a crate filling gets its own place on the crate floor, inside the walls."""
    for name in ("mission1", "mission3"):
        cfg = load_config(name)
        b = cfg["basket"]
        pat = np.array(b["drop_pattern"], float)
        il, iw, _ = b["inner_size"]
        assert np.all(np.abs(pat[:, 0]) < il / 2 - 0.05) and np.all(np.abs(pat[:, 1]) < iw / 2 - 0.05)
        d = np.linalg.norm(pat[:, None, :] - pat[None, :, :], axis=2) + np.eye(len(pat))
        assert d.min() > 0.07, "two fruits are never released above the same place"
        if name == "mission3":
            assert len(pat) >= int(b["capacity"])


def test_timing_summary():
    tg = [dict(placed=True, t_start=10.0, t_placed=30.0, t_end=34.0, compute_s=2.0),
          dict(placed=False, t_start=40.0, t_placed=float("nan"), t_end=50.0, compute_s=1.0),
          dict(placed=True, t_start=50.0, t_placed=95.0, t_end=99.0, compute_s=3.0)]
    ft = fruit_times(tg)
    assert [round(x[0], 3) for x in ft] == [22.0, 48.0]
    s = summarize([dict(targets=tg, sim_time_s=100.0, compute_s=6.0, n_placed=2)])
    assert s["observe_to_basket_s_mean"] == pytest.approx(35.0)
    assert s["observe_to_basket_under_40s"] == pytest.approx(0.5)
    assert s["s_per_fruit"] == pytest.approx(53.0)


def test_view_move_guard_stops_before_a_leaf_is_torn_out():
    """Light Mission 2 scene whose first look move ran into an unseen leaf (it was torn out before version 6)."""
    from pinesim.envs import make_env
    env = make_env("mission2", training=False, seed=787000)
    info = env.run_episode(None, seed=787000)
    assert any("move: stop and go back" in e for _, e in info["events"])
    assert not (info.get("failure") == "leaf_uprooted" and info.get("fail_phase") in ("look", "scan"))


def test_scenery_is_html_only_and_keeps_the_field_free():
    """Dome / rocks / trees / hut of the Mission 3 HTML: not in the MuJoCo model, nothing inside the field."""
    from pinesim.viz import scenery
    cfg3 = copy.deepcopy(CFG)
    cfg3["field"]["side_rows"]["beds_per_side"] = 3
    m = mujoco.MjModel.from_xml_path(str(build("xgh_light", 8, cfg3)))
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)]
    assert not any(n.startswith(("scenery", "rock", "tree", "hut", "dome")) for n in names)
    x0, x1, y0, y1 = scenery.field_bounds(m)
    oy, gap = float(CFG["field"]["row_offset_y"]), float(CFG["field"]["side_rows"]["edge_gap"])
    assert y1 == pytest.approx(oy + gap + 2 * (2 * oy + gap) + 2 * oy, abs=1e-6) and y0 == pytest.approx(-y1, abs=1e-6)
    p = scenery.params(None)
    mods = scenery.models()                     # the user's converted meshes (trees, stumps, hut) when they are installed
    lay = scenery.layout((x0, x1, y0, y1), p, 40.0, mods)
    assert len(lay["rocks"]) == p["n_rocks"] and len(lay["trees"]) == p["n_trees"] and lay["hut"] is not None
    if any(v["kind"] == "stump" for v in mods.values()):
        assert len(lay["stumps"]) == p["n_stumps"]
        assert {o["model"] for o in lay["trees"]} == {k for k, v in mods.items() if v["kind"] == "tree"}
    for o in lay["rocks"] + lay["trees"] + lay["stumps"] + [lay["hut"]]:
        inside = (x0 - p["margin_ends"]) < o["x"] < (x1 + p["margin_ends"]) and (y0 - p["margin_side"]) < o["y"] < (y1 + p["margin_side"])
        assert not inside
        assert np.hypot(o["x"] - lay["center"][0], o["y"] - lay["center"][1]) < 40.0
