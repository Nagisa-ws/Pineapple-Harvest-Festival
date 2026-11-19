"""
pinesim/viz/scenery.py
----------------------
Decoration of the Mission 3 HTML [USER v6.1]: a dome with a 360 degree landscape photo, the soil widened to the dome
radius, rocks, trees, tree stumps and one hut.

The decoration exists ONLY in the Meshcat HTML. It is not part of the MuJoCo model: no bodies, no contacts, and the
robot cameras never see it - so physics, perception, timing and the evaluation results are exactly the same with and
without it. Assets: assets_xgh/scenery/ (scripts/make_scenery_assets.py).

Settings (optional `scenery:` block in configs/common.yaml; these are the defaults):

  scenery:
    enabled: true
    seed: 5                  # another seed = another arrangement of rocks / trees
    yaw_deg: 0.0             # turns the panorama around the field
    n_rocks: 34
    rock_scale: [0.35, 1.6]  # rock size range (x the 1.1 m rock)
    n_trees: 22              # trees: every model of assets_xgh/scenery/models whose kind is "tree", in turn
    tree_scale: [0.8, 1.25]  # tree size range (x the size of the model, 6 - 9 m)
    n_stumps: 8              # tree stumps (kind "stump")
    stump_scale: [0.7, 1.2]
    margin_side: 1.6         # m between the outermost plant row and the nearest rock
    margin_ends: 4.0         # m kept free before the first / after the last plant (vehicle path)
    tree_margin: 4.5         # trees stay this much further away than the rocks
    hut: true
    hut_xy: null             # [x, y] of the hut; null = beside the middle of the field, on the +y side
    hut_yaw_deg: null        # null = entrance turned to the field

Own meshes: assets_xgh/scenery/models/<name>.gltf + models.json (name -> kind "tree" / "stump" / "hut", size_m).
With that folder the trees, stumps and the hut are these models; without it the built-in low-poly tree and hut.
"""
from __future__ import annotations

import json

import mujoco
import numpy as np

from pinesim.config import PROJECT_ROOT

DIR = PROJECT_ROOT / "assets_xgh" / "scenery"
MODELS = DIR / "models"
DEFAULTS = dict(enabled=True, seed=5, yaw_deg=0.0, n_rocks=34, rock_scale=[0.35, 1.6], n_trees=22, tree_scale=[0.8, 1.25],
                n_stumps=8, stump_scale=[0.7, 1.2], margin_side=1.6, margin_ends=4.0, tree_margin=4.5, hut=True,
                hut_xy=None, hut_yaw_deg=None)


def models() -> dict:
    """Models converted from the user's own meshes (assets_xgh/scenery/models/models.json): name -> kind ("tree" /
    "stump" / "hut"), size. Without that folder the built-in low-poly tree and hut are used."""
    f = MODELS / "models.json"
    if not f.exists():
        return {}
    return {k: v for k, v in json.loads(f.read_text()).items() if (MODELS / f"{k}.gltf").exists()}


def params(cfg_block: dict | None) -> dict:
    p = dict(DEFAULTS)
    p.update(cfg_block or {})
    return p


def available() -> bool:
    return (DIR / "dome.gltf").exists() and (DIR / "scenery.json").exists()


def field_bounds(m: mujoco.MjModel):
    """x / y extent of every plant of the scene (harvested bed and neighbour beds), from the model."""
    xs, ys = [], []
    for b in range(1, m.nbody):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if nm.startswith("splant_") and int(m.body_parentid[b]) == 0:
            xs.append(float(m.body_pos[b][0])); ys.append(float(m.body_pos[b][1]))
    if not xs:
        return None
    return min(xs), max(xs), min(ys), max(ys)


def layout(bounds, p: dict, radius: float, mods: dict | None = None) -> dict:
    """Where the decoration goes: nothing inside the field rectangle (plants + margins, vehicle path), rocks from
    the field edge outwards (denser near the field), trees further out, the hut beside the middle of the field."""
    x0, x1, y0, y1 = bounds
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    rng = np.random.default_rng(int(p["seed"]))
    ms, me = float(p["margin_side"]), float(p["margin_ends"])
    r_out = 0.90 * radius

    def inside_field(x, y, extra=0.0):
        return (x0 - me - extra) < x < (x1 + me + extra) and (y0 - ms - extra) < y < (y1 + ms + extra)

    placed = []                                              # (x, y, keep-out radius)

    def free(x, y, r):
        return all((x - a) ** 2 + (y - b) ** 2 > (r + c) ** 2 for a, b, c in placed)

    mods = mods or {}
    by_kind = lambda kind: sorted(k for k, v in mods.items() if v.get("kind") == kind)
    half = lambda name, dflt: 0.5 * max(mods[name]["size_m"][:2]) if name in mods else dflt      # half width of a model
    out = dict(center=[cx, cy], radius=radius, rocks=[], trees=[], stumps=[], hut=None)
    if p.get("hut", True):
        hm = (by_kind("hut") or [None])[0]
        hr = half(hm, 1.6) * 1.35
        hx, hy = p["hut_xy"] if p.get("hut_xy") is not None else (cx - 1.5, y1 + ms + 3.5 + hr)
        yaw0 = float(mods[hm].get("yaw_to_field_deg", 180.0)) if hm else 180.0     # turns the entrance to the field (-y)
        yaw = np.radians(p["hut_yaw_deg"] if p.get("hut_yaw_deg") is not None else yaw0)
        out["hut"] = dict(x=float(hx), y=float(hy), yaw=float(yaw), scale=1.0, model=hm)
        placed.append((float(hx), float(hy), hr + 1.0))

    def scatter(n, s_lo, s_hi, extra, r_keep, near_bias, names=(None,)):
        res, tries = [], 0
        while len(res) < n and tries < 20000:
            tries += 1
            name = names[len(res) % len(names)]              # the models take turns
            # radius around the field centre: `near_bias` > 1 puts more objects near the field
            rr = (rng.random() ** near_bias) * r_out
            a = rng.uniform(0, 2 * np.pi)
            x, y = cx + rr * np.cos(a), cy + rr * np.sin(a)
            s = float(np.exp(rng.uniform(np.log(s_lo), np.log(s_hi))))
            rk = half(name, r_keep) * s                      # crown / root radius of this model
            if inside_field(x, y, extra + (rk if name else 0.0)) or not free(x, y, max(rk, r_keep * s if not name else 0.8)):
                continue
            placed.append((x, y, max(rk, 0.8)))
            res.append(dict(x=float(x), y=float(y), yaw=float(rng.uniform(0, 2 * np.pi)), scale=s, model=name))
        return res

    trees, stumps = by_kind("tree"), by_kind("stump")
    out["trees"] = scatter(int(p["n_trees"]), *(p["tree_scale"] if trees else [0.7, 1.5]), float(p["tree_margin"]), 2.4, 0.8,
                           trees or (None,))
    if stumps:
        out["stumps"] = scatter(int(p["n_stumps"]), *p["stump_scale"], 1.0, 1.0, 1.3, stumps)
    out["rocks"] = scatter(int(p["n_rocks"]), *p["rock_scale"], 0.0, 0.7, 1.5)
    return out


def add_to_meshcat(mc, model: mujoco.MjModel, cfg_block: dict | None = None) -> dict:
    """Put the decoration into a Meshcat scene (static objects under /scenery). Returns what was added."""
    from pydrake.geometry import Mesh
    from pydrake.math import RigidTransform, RollPitchYaw
    p = params(cfg_block)
    if not p.get("enabled", True):
        return dict(enabled=False)
    if not available():
        return dict(enabled=False, reason=f"no assets in {DIR} (python scripts/make_scenery_assets.py --panorama <photo>)")
    bounds = field_bounds(model)
    if bounds is None:
        return dict(enabled=False, reason="scene has no field")
    info = json.loads((DIR / "scenery.json").read_text())
    radius = float(info.get("radius", 40.0))
    mods = models()
    lay = layout(bounds, p, radius, mods)
    cx, cy = lay["center"]

    def put(path, name, x, y, z, yaw, scale=1.0, folder=DIR):
        mc.SetObject(path, Mesh(str(folder / f"{name}.gltf"), float(scale)))
        mc.SetTransform(path, RigidTransform(RollPitchYaw(0.0, 0.0, float(yaw)), [float(x), float(y), float(z)]))

    put("/scenery/dome", "dome", cx, cy, 0.0, np.radians(float(p.get("yaw_deg", 0.0))))
    put("/scenery/soil", "soil_disc", cx, cy, -0.035, 0.0)    # just under the lowest point of the field soil mesh
    if (DIR / "grass_ring.gltf").exists():
        put("/scenery/grass", "grass_ring", cx, cy, -0.035, 0.0)
    for i, o in enumerate(lay["rocks"]):
        put(f"/scenery/rock_{i}", "rock", o["x"], o["y"], -0.035, o["yaw"], o["scale"])
    for kind in ("trees", "stumps"):
        for i, o in enumerate(lay[kind]):
            if o.get("model"):
                put(f"/scenery/{kind}_{i}", o["model"], o["x"], o["y"], -0.035, o["yaw"], o["scale"], MODELS)
            else:
                put(f"/scenery/{kind}_{i}", "tree", o["x"], o["y"], -0.035, o["yaw"], o["scale"])
    if lay["hut"] is not None:
        o = lay["hut"]
        if o.get("model"):
            put("/scenery/hut", o["model"], o["x"], o["y"], -0.035, o["yaw"], 1.0, MODELS)
        else:
            put("/scenery/hut", "hut", o["x"], o["y"], -0.035, o["yaw"])
    return dict(enabled=True, radius=radius, rocks=len(lay["rocks"]), trees=len(lay["trees"]), stumps=len(lay["stumps"]),
                hut=lay["hut"] is not None, models=sorted(mods), center=[round(cx, 2), round(cy, 2)])
