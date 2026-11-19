"""
pinesim/viz/meshcat_viewer.py
-----------------------------
Drake Meshcat visualisation (headless, browser, Google Colab). MuJoCo is only used to simulate; Meshcat draws the
primitive geometry in the browser (light on a weak PC: no MuJoCo viewer window, no mesh rendering on the host).

  publish_model(meshcat, model, data)   geoms -> /scene/<body>/<geom> (boxes, capsules, cylinders, spheres, plane, meshes)
  set_body_poses(meshcat, model, data)  live update of body transforms
  Recording / record() / play()         episode recording (body poses + geom sizes/visibility + stargaze overlays),
                                        MeshcatAnimation playback, standalone HTML export
  draw_* helpers                        points (point cloud), spheres, polylines for Stargaze results
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

try:
    from pydrake.common.eigen_geometry import Quaternion
    from pydrake.geometry import Box, Capsule, Cylinder, Mesh, Meshcat, MeshcatAnimation, Rgba, Sphere
    from pydrake.math import RigidTransform, RotationMatrix
    from pydrake.perception import BaseField, Fields, PointCloud
    HAVE_DRAKE = True
except Exception:     # pragma: no cover
    HAVE_DRAKE = False


def start_meshcat(port: int | None = None):
    """Meshcat server (in Colab use pydrake.geometry.StartMeshcat() instead to get the forwarded URL)."""
    assert HAVE_DRAKE, "pydrake is required: pip install drake"
    from pydrake.geometry import MeshcatParams
    p = MeshcatParams()
    if port is not None:
        p.port = int(port)
    return Meshcat(p)


def _rt(pos, quat_wxyz=(1.0, 0.0, 0.0, 0.0)):
    q = np.asarray(quat_wxyz, float)
    q = q / np.linalg.norm(q)
    return RigidTransform(RotationMatrix(Quaternion(q[0], q[1], q[2], q[3])), np.asarray(pos, float))


def _rgba(c):
    c = list(c) + [1.0] * (4 - len(c))
    return Rgba(float(c[0]), float(c[1]), float(c[2]), float(c[3]))


def body_name(model, b):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"


def publish_model(meshcat, model: mujoco.MjModel, geom_rgba=None, geom_size=None, geom_pos=None, geom_quat=None,
                  asset_dir: str | Path | None = None, prefix: str = "/scene", skip_groups=(3, 4)) -> int:
    rgba_all = model.geom_rgba if geom_rgba is None else geom_rgba
    size_all = model.geom_size if geom_size is None else geom_size
    gpos_all = model.geom_pos if geom_pos is None else geom_pos
    gquat_all = model.geom_quat if geom_quat is None else geom_quat
    meshcat.Delete(prefix)
    n = 0
    mat_rgba = model.mat_rgba
    for g in range(model.ngeom):
        rgba = rgba_all[g].copy()
        mid = int(model.geom_matid[g])
        if mid >= 0 and np.allclose(model.geom_rgba[g], [0.5, 0.5, 0.5, 1.0]):
            rgba = mat_rgba[mid].copy()
            if rgba_all[g][3] <= 0.01:
                rgba[3] = 0.0
        if rgba[3] <= 0.01 or int(model.geom_group[g]) in skip_groups:
            continue
        t = int(model.geom_type[g])
        sz = size_all[g]
        if t == mujoco.mjtGeom.mjGEOM_BOX:
            shape = Box(2 * sz[0], 2 * sz[1], 2 * sz[2])
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            shape = Capsule(max(sz[0], 1e-4), max(2 * sz[1], 1e-4))
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            shape = Cylinder(sz[0], 2 * sz[1])
        elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
            shape = Sphere(sz[0])
        elif t == mujoco.mjtGeom.mjGEOM_PLANE:
            shape = Box(12.0, 12.0, 0.01)
            rgba = np.array([0.34, 0.27, 0.2, 1.0])        # textured soil (textures are not exported)
        elif t == mujoco.mjtGeom.mjGEOM_MESH and asset_dir is not None:
            mname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[g]))
            f = Path(asset_dir) / f"{mname}.obj"
            if not f.exists():
                continue
            shape = Mesh(str(f), 1.0)
        else:
            continue
        path = f"{prefix}/{body_name(model, int(model.geom_bodyid[g]))}/g{g}"
        meshcat.SetObject(path, shape, _rgba(rgba))
        pos = gpos_all[g].copy()
        if t == mujoco.mjtGeom.mjGEOM_PLANE:
            pos = pos - np.array([0, 0, 0.005])
        meshcat.SetTransform(path, _rt(pos, gquat_all[g]))
        n += 1
    return n


def set_body_poses(meshcat, model, data, prefix="/scene", bodies=None):
    for b in (range(1, model.nbody) if bodies is None else bodies):
        meshcat.SetTransform(f"{prefix}/{body_name(model, b)}", _rt(data.xpos[b], data.xquat[b]))


# ----------------------------------------------------------------------------- overlays
def draw_points(meshcat, path, pts, rgba=(1, 0.5, 0, 1), size=0.008):
    pts = np.asarray(pts, float).reshape(-1, 3)
    meshcat.Delete(path)
    if len(pts) == 0:
        return
    cloud = PointCloud(len(pts), Fields(BaseField.kXYZs))
    cloud.mutable_xyzs()[:] = pts.T
    meshcat.SetObject(path, cloud, point_size=size, rgba=_rgba(rgba))


def draw_sphere(meshcat, path, center, radius, rgba):
    meshcat.SetObject(path, Sphere(radius), _rgba(rgba))
    meshcat.SetTransform(path, _rt(center))


def draw_line(meshcat, path, pts, rgba=(1, 1, 1, 1), width=3.0):
    pts = np.asarray(pts, float).reshape(-1, 3)
    meshcat.SetLine(path, pts.T, width, _rgba(rgba))


def draw_stargaze(meshcat, prefix, t1, t2, plans=(), faces=None, gt_center=None, obstacle_pts=None, fruit_pts=None,
                  leaf_pts=None, camera_pos=None):
    """Stargaze result: T1 (green), T2 (red), line, alternative lines, cube faces, clouds, ground-truth centre."""
    meshcat.Delete(prefix)
    if leaf_pts is not None and len(leaf_pts):
        draw_points(meshcat, f"{prefix}/leaf_cloud", leaf_pts, (0.2, 0.9, 0.2, 1), 0.007)
    if obstacle_pts is not None and len(obstacle_pts):
        draw_points(meshcat, f"{prefix}/obstacles", obstacle_pts, (0.9, 0.9, 0.2, 1), 0.010)
    if fruit_pts is not None and len(fruit_pts):
        draw_points(meshcat, f"{prefix}/fruit_cloud", fruit_pts, (1.0, 0.45, 0.0, 1), 0.008)
    if t1 is not None and t2 is not None:
        draw_sphere(meshcat, f"{prefix}/T1", t1, 0.018, (0.1, 0.9, 0.2, 1))
        draw_sphere(meshcat, f"{prefix}/T2", t2, 0.018, (0.95, 0.1, 0.1, 1))
        draw_line(meshcat, f"{prefix}/line", [t1, t2], (1, 1, 1, 1), 5.0)
    for k, p in enumerate(list(plans)[1:6]):
        draw_line(meshcat, f"{prefix}/alt{k}", [p.start, p.target], (0.6, 0.6, 0.9, 0.6), 1.5)
    if faces:
        for name, c in faces.items():
            draw_line(meshcat, f"{prefix}/face_{name}", np.vstack([c, c[:1]]), (0.5, 0.8, 1.0, 0.8), 1.5)
    if gt_center is not None:
        draw_sphere(meshcat, f"{prefix}/gt_center", gt_center, 0.012, (1, 1, 1, 0.5))
    if camera_pos is not None and t2 is not None:
        draw_line(meshcat, f"{prefix}/view_ray", [camera_pos, t2], (0.3, 0.6, 1.0, 0.5), 1.0)


# ----------------------------------------------------------------------------- recording
@dataclass
class Recording:
    body_names: list
    times: np.ndarray
    xpos: np.ndarray
    xquat: np.ndarray
    geom_rgba: np.ndarray
    geom_size: np.ndarray
    geom_pos: np.ndarray
    geom_quat: np.ndarray
    labels: list = field(default_factory=list)          # phase label per frame
    overlays: list = field(default_factory=list)        # (frame index, dict(t1, t2, fruit_center, ...))
    result: dict = field(default_factory=dict)

    def save(self, path):
        import json
        ov = [(int(k), {kk: (np.asarray(vv).tolist() if vv is not None else None) for kk, vv in d.items()})
              for k, d in self.overlays]
        np.savez_compressed(path, body_names=np.array(self.body_names), times=self.times, xpos=self.xpos, xquat=self.xquat,
                            geom_rgba=self.geom_rgba, geom_size=self.geom_size, geom_pos=self.geom_pos,
                            geom_quat=self.geom_quat, labels=np.array(self.labels),
                            overlays=np.array([json.dumps(ov)]), result=np.array([json.dumps(self.result, default=str)]))

    @classmethod
    def load(cls, path):
        import json
        z = np.load(path, allow_pickle=False)
        ov = [(k, {kk: (np.array(vv) if vv is not None else None) for kk, vv in d.items()})
              for k, d in json.loads(str(z["overlays"][0]))]
        return cls(list(z["body_names"]), z["times"], z["xpos"], z["xquat"], z["geom_rgba"], z["geom_size"], z["geom_pos"],
                   z["geom_quat"], list(z["labels"]), ov, json.loads(str(z["result"][0])))


class Recorder:
    """Attach to an env (env.frame_callback = recorder) to grab body poses every `every` control steps."""

    def __init__(self, model, data, every: int = 4, dt: float = 0.02):
        self.m, self.d, self.every, self.dt = model, data, int(every), dt
        self.k = 0
        self.times, self.xp, self.xq, self.labels, self.overlays = [], [], [], [], []
        self.label = ""

    def __call__(self, label: str | None = None, force: bool = False):
        if label is not None:
            self.label = label
        if force or self.k % self.every == 0:
            self.times.append(self.k * self.dt)
            self.xp.append(self.d.xpos.copy())
            self.xq.append(self.d.xquat.copy())
            self.labels.append(self.label)
        self.k += 1

    def overlay(self, **kw):
        self.overlays.append((max(0, len(self.times) - 1), kw))

    def finish(self, result: dict | None = None) -> Recording:
        self(force=True)
        m = self.m
        names = [body_name(m, b) for b in range(m.nbody)]
        return Recording(names, np.array(self.times), np.array(self.xp), np.array(self.xq), m.geom_rgba.copy(),
                         m.geom_size.copy(), m.geom_pos.copy(), m.geom_quat.copy(), self.labels, self.overlays,
                         dict(result or {}))


def play(meshcat, model, rec: Recording, prefix="/scene", asset_dir=None, fps: float | None = None):
    """Publish the recorded geometry and set a MeshcatAnimation (play / scrub in the browser)."""
    publish_model(meshcat, model, rec.geom_rgba, rec.geom_size, rec.geom_pos, rec.geom_quat, asset_dir, prefix)
    nb = len(rec.body_names)
    dt = float(np.median(np.diff(rec.times))) if len(rec.times) > 1 else 0.1
    anim = MeshcatAnimation(frames_per_second=fps or (1.0 / max(dt, 1e-3)))
    for k in range(len(rec.times)):
        for b in range(1, nb):
            anim.SetTransform(k, f"{prefix}/{rec.body_names[b]}", _rt(rec.xpos[k, b], rec.xquat[k, b]))
    # stargaze overlays: visible from the frame they were computed
    for j, (k0, d) in enumerate(rec.overlays):
        p = f"/overlay/{j}"
        if d.get("t1") is not None and d.get("t2") is not None:
            draw_sphere(meshcat, f"{p}/T1", d["t1"], 0.018, (0.1, 0.9, 0.2, 1))
            draw_sphere(meshcat, f"{p}/T2", d["t2"], 0.018, (0.95, 0.1, 0.1, 1))
            draw_line(meshcat, f"{p}/line", [d["t1"], d["t2"]], (1, 1, 1, 1), 4.0)
        if d.get("leaf_points") is not None and len(d["leaf_points"]):
            draw_points(meshcat, f"{p}/leaf_cloud", d["leaf_points"], (0.2, 0.9, 0.2, 1), 0.006)
        if d.get("fruit_points") is not None and len(d["fruit_points"]):
            draw_points(meshcat, f"{p}/fruit_cloud", d["fruit_points"], (1.0, 0.45, 0.0, 1), 0.007)
        k1 = rec.overlays[j + 1][0] if j + 1 < len(rec.overlays) else len(rec.times)
        for k in range(len(rec.times)):
            anim.SetProperty(k, p, "visible", bool(k0 <= k < k1))
    meshcat.SetAnimation(anim)
    for b in range(1, nb):
        meshcat.SetTransform(f"{prefix}/{rec.body_names[b]}", _rt(rec.xpos[-1, b], rec.xquat[-1, b]))
