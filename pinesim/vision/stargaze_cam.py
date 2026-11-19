"""
pinesim/vision/stargaze_cam.py
------------------------------
The user's Stargaze max-clearance concept applied to what the CAMERA sees.

  nucleus (end point T2)   = estimated fruit centre (body grip with the 20 cm gripper) [or crown neck, optional]
  constellation            = leaf voxels seen by the camera (rosette, blocking and crown leaves) around the fruit
  cube                     = cube of half-size L centred at the nucleus, rotated so that the FRONT face looks at the
                             viewer (robot base, or the pole camera in the pretraining rig); BACK (opposite) dropped,
                             BOTTOM forbidden -> faces top / front / left / right (stargaze rule)
  start point (T1)         = grid point on an allowed face whose straight segment T1 -> T2 has the largest minimum
                             distance to the leaf points (minus their radius)

The geometry is computed by the vendored reference planner (pinesim/vendor/harvest_planner.py, ApproachPlanner, which
in turn uses the reference stargaze.py segment-clearance). This module only adapts perception output to it and adds
cancellation and ground-truth diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pinesim.vendor.harvest_planner import ApproachPlanner, GripperSpec, ApproachPlan, grasp_quat
from pinesim.vendor.stargaze import ConstellationGenerator
from pinesim.vision.perception import FruitEstimate, _check


@dataclass
class PerceivedPlant:
    """Duck-typed stand-in for the reference PlantLayout (only what ApproachPlanner reads)."""
    fruit_center: np.ndarray
    fruit_r: float
    neck_center: np.ndarray
    neck_r: float = 0.022


def gripper_spec(cfg: dict) -> GripperSpec:
    g = cfg["gripper"]
    return GripperSpec(finger_offset=0.5 * g["min_opening"], stroke_per_finger=0.5 * (g["max_opening"] - g["min_opening"]),
                       grip_clearance=g.get("grip_clearance", 0.015))


def obstacle_points(fruit: FruitEstimate, leaf_points: np.ndarray, cfg: dict, other_fruits=()):
    """Leaf voxels near the target, minus the target itself; other fruits become obstacles too."""
    p = cfg["perception"]
    c = fruit.center
    pts = leaf_points
    if len(pts):
        d = np.linalg.norm(pts - c[None, :], axis=1)
        pts = pts[d < float(p["leaf_crop_radius"])]
        # points on / hugging the target fruit body are part of the target (fingers close around it)
        dh = np.linalg.norm(pts[:, :2] - c[None, :2], axis=1)
        inside = (dh < fruit.radius + float(p["target_keepout"])) & (pts[:, 2] > fruit.bottom_z - 0.02) & \
                 (pts[:, 2] < fruit.top_z + 0.01)
        pts = pts[~inside]
        # dense canopies (realistic mesh plants): the clearance of every candidate line is an (M points x N lines)
        # matrix; 50 k leaf points x 1 764 lines needed ~3.5 GB (measured: the evaluation was killed for memory).
        # Thin the cloud on a coarser grid until it is below max_obstacle_points (light scenes stay untouched).
        cap = int(p.get("max_obstacle_points", 6000))
        vox = float(p.get("voxel", 0.012))
        if len(pts) > cap:
            from pinesim.vision.perception import voxel_down
            while len(pts) > cap:
                vox *= 1.3
                pts = voxel_down(pts, vox)
    rad = np.full(len(pts), float(p["leaf_point_radius"]))
    extra_p, extra_r = [], []
    for of in other_fruits:
        zs = np.linspace(of.bottom_z, of.top_z, 4)
        for z in zs:
            extra_p.append([of.center[0], of.center[1], z])
            extra_r.append(of.radius)
    if extra_p:
        pts = np.vstack([pts, np.array(extra_p)])
        rad = np.concatenate([rad, np.array(extra_r)])
    return pts, rad


def stargaze_plans(fruit: FruitEstimate, leaf_points: np.ndarray, viewer_pos: np.ndarray, cfg: dict,
                   x_hint=np.array([1.0, 0.0, 0.0]), reach_filter=None, top_k: int = 6, other_fruits=(),
                   cancel=None, per_face: int = 1, min_sep: float = 0.08) -> tuple[list, np.ndarray, np.ndarray]:
    """Return (plans sorted by clearance, obstacle points, radii).

    per_face = 1 is exactly the reference planner (best line per face, faces sorted by clearance). per_face > 1 also
    returns the next-best lines of each face whose start points are at least `min_sep` apart (alternatives for the
    robot when the best line is not reachable)."""
    _check(cancel)
    pts, rad = obstacle_points(fruit, leaf_points, cfg, other_fruits)
    plant = PerceivedPlant(fruit.center.copy(), fruit.radius, fruit.neck_center)
    planner = ApproachPlanner(cfg["stargaze"], gripper_spec(cfg))
    _check(cancel)
    if per_face <= 1:
        plans = planner.plan(plant, pts, rad, np.asarray(viewer_pos, float), np.asarray(x_hint, float),
                             reach_filter=reach_filter, top_k=top_k)
        _check(cancel)
        return plans, pts, rad
    # same scoring as ApproachPlanner.plan (vendored), keeping several separated candidates per face
    N = plant.fruit_center
    S, names = planner._face_points(N, np.asarray(viewer_pos, float))
    if reach_filter is not None:
        keep = np.asarray(reach_filter(S), bool)
        S, names = S[keep], names[keep]
    if len(S) == 0:
        return [], pts, rad
    d = S - N[None, :]
    elev = np.degrees(np.arctan2(d[:, 2], np.linalg.norm(d[:, :2], axis=1)))
    ok = np.array([planner._mode_ok("body", e) for e in elev])
    S, names, elev = S[ok], names[ok], elev[ok]
    if len(S) == 0:
        return [], pts, rad
    _check(cancel)
    if len(pts):
        dist = ConstellationGenerator.segment_clearance(pts - N[None, :], S - N[None, :])
        clear = (dist - rad[:, None]).min(axis=0)
    else:
        clear = np.full(len(S), np.inf)
    plans = []
    for fname in dict.fromkeys(names.tolist()):
        idx = np.where(names == fname)[0]
        idx = idx[np.argsort(-clear[idx])]
        chosen = []
        for i in idx:
            if all(np.linalg.norm(S[i] - S[j]) >= min_sep for j in chosen):
                chosen.append(i)
            if len(chosen) >= per_face:
                break
        for i in chosen:
            plans.append(ApproachPlan(start=S[i].copy(), target=N.copy(), mode="body", face=str(names[i]),
                                      clearance=float(clear[i]), elevation_deg=float(elev[i]),
                                      quat=grasp_quat(N - S[i], np.asarray(x_hint, float))))
    plans.sort(key=lambda p: -p.clearance)
    _check(cancel)
    return plans[:top_k], pts, rad


def segment_clearance_to(points: np.ndarray, radii: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Minimum surface distance of point obstacles to segment a-b (for ground-truth diagnostics)."""
    if len(points) == 0:
        return float("inf")
    d = ConstellationGenerator.segment_clearance(points - b[None, :], (a - b)[None, :])[:, 0]
    return float(np.min(d - radii))


def face_corners(plan: ApproachPlan, viewer_pos: np.ndarray, L: float) -> dict:
    """Corners of the stargaze cube faces (for drawing), in world coordinates."""
    N = plan.target
    f = np.asarray(viewer_pos, float) - N
    f[2] = 0.0
    f /= np.linalg.norm(f) + 1e-12
    z = np.array([0.0, 0.0, 1.0])
    left = np.cross(z, f)
    out = {}
    for name, n in (("top", z), ("front", f), ("left", left), ("right", -left)):
        u = f if name == "top" else z
        v = np.cross(n, u)
        c = N + L * n
        out[name] = np.array([c + L * (su * u + sv * v) for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    return out
