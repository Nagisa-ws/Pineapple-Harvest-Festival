"""
pinesim/viz/xgh_recorder.py
---------------------------
Recording + Drake Meshcat export of the REALISTIC scenes of version 5 (pretraining rig, Missions 1-2 platform,
Mission 3 15 m field).

  XghRecorder        env.frame_callback: every `every` control steps stores only the bodies that MOVED (> 1 mm /
                     0.5 deg) and the geoms whose pose / visibility changed (crate fill, picked fruits, the blinking
                     indicator lamp) -> compact .npz even for hours
  export_html()      standalone Meshcat HTML of a frame range: glTF meshes (vertex colours; Drake stores repeated
                     .bin buffers once), body animation, T1/T2 lines, and a CAMERA MODE panel injected into the page:
                     Bebas (free orbit) | Ikuti kendaraan | Kamera pojok | Kamera gripper (only the cameras the scene
                     has). In the GRIPPER-CAMERA view the decision of every target is drawn: bounding box of the
                     visible fruit, WINDOW, SHOULDER points, T2 candidates, T1 and T2 (+ a text box) [USER v5]
  CornerVideo        MP4 of the corner camera (picture-in-picture: wrist camera) during the harvesting phases

Mesh frames: MuJoCo recentres meshes (mesh_pos / mesh_quat); the glTF files keep the original coordinates, so every
mesh geom is drawn at  T_geom_compiled * inverse(T_mesh)  (the fix for the old track-belt display bug).
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from pinesim.config import PROJECT_ROOT

ASSETS = PROJECT_ROOT / "assets_xgh"
CAMERA_NODES = {"corner": ("vehicle", "corner_cam"), "follow": ("vehicle", "follow"), "wrist": ("gripper_housing", "wrist_cam"),
                "overview": ("world", "overview")}
RIG_CAMERA_NODES = {"wrist": ("rig_head", "rig_cam")}
HARVEST_LABELS = {"scan", "look", "open", "align", "advance", "approach", "line", "grasp", "twist", "pluck", "retreat", "lift",
                  "transport", "release", "verify", "return", "back_out", "rescue"}


def camera_nodes(m):
    """Camera-mode nodes the scene has: (key -> (body, camera))."""
    out = {}
    for key, (body, cam) in list(CAMERA_NODES.items()) + list(RIG_CAMERA_NODES.items()):
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam) >= 0 and key not in out:
            bid = m.cam_bodyid[m.camera(cam).id]
            out[key] = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or "world", cam)
    return out


def mesh_gltf(name: str) -> Path | None:
    """Meshcat file of a MuJoCo mesh asset (None = not drawn, e.g. collision hulls)."""
    if name.endswith("_col"):
        return None
    if name.startswith("plant") and "_static_" in name:          # rest-pose leaves: MuJoCo has one mesh per colour
        p, cls = name[5], name.split("_static_")[1]               # class; Meshcat draws the whole vertex-coloured
        f = ASSETS / f"plants/p{p}_static.gltf"                   # mesh once (on the "green" geom)
        return f if cls == "green" and f.exists() else None
    table = {"track_teeth_left": "vehicle/track_guide_teeth_left.gltf", "track_teeth_right": "vehicle/track_guide_teeth_right.gltf",
             "chassis_wheels": "vehicle/track_wheels_right.gltf", "bed": "vehicle/bed.gltf", "crate": "vehicle/basket-v1.gltf",
             "ground": "ground/ground.gltf", "Servo-v1": "ur10e/Servo-v1.gltf"}
    if name in table:
        rel = table[name]
    elif name.startswith(("fruit_", "crown_")):
        rel = f"pineapple/{name}.gltf"
    elif name.startswith("plant") and name.endswith(("_static", "_stem")):
        p, kind = name[5], name.split("_")[-1]
        rel = f"plants/p{p}_{kind}.gltf"
    elif name.startswith("p") and "_l" in name and "_s" in name:
        rel = f"plants/leaves/{name}.gltf"
    else:
        rel = f"ur10e/{name}.gltf"
    f = ASSETS / rel
    return f if f.exists() else None


def _qmul(a, b):
    o = np.zeros(4)
    mujoco.mju_mulQuat(o, a, b)
    return o


def _qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _qrot(q, v):
    o = np.zeros(3)
    mujoco.mju_rotVecQuat(o, np.asarray(v, float), np.asarray(q, float))
    return o


def visual_geoms(m: mujoco.MjModel) -> list:
    """Geoms that are drawn: visible groups 0-2, not fully transparent at compile time OR toggled at runtime."""
    out = []
    for g in range(m.ngeom):
        if int(m.geom_group[g]) > 2:
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, int(m.geom_dataid[g]))
            if mesh_gltf(mn) is None:
                continue
        elif m.geom_rgba[g, 3] < 0.01 and "_fill_" not in name and not name.startswith("lamp_"):
            continue
        out.append(g)
    return out


def geom_local_original(m: mujoco.MjModel, g: int, gpos=None, gquat=None):
    """Geom frame relative to its body in the ORIGINAL mesh coordinates (mesh recentring undone)."""
    p = (m.geom_pos[g] if gpos is None else gpos).copy()
    q = (m.geom_quat[g] if gquat is None else gquat).copy()
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        d = int(m.geom_dataid[g])
        qm, pm = m.mesh_quat[d], m.mesh_pos[d]
        q = _qmul(q, _qconj(qm))
        p = p - _qrot(q, pm)
    return p, q


class XghRecorder:
    def __init__(self, sim, every: int = 5, pos_tol: float = 1e-3, ang_tol: float = np.radians(0.5)):
        self.sim, self.m, self.d = sim, sim.m, sim.d
        self.every, self.pos_tol, self.cos_tol = int(every), pos_tol, np.cos(ang_tol / 2)
        m = self.m
        self.vgeoms = visual_geoms(m)
        self.bodies = sorted({int(m.geom_bodyid[g]) for g in self.vgeoms} |
                             {int(m.body(n).id) for n, _ in camera_nodes(m).values() if n != "world"})
        nm = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        self.watch_geoms = [g for g in self.vgeoms if "_fill_" in nm(g) or nm(g).endswith(("_vis", "_crownvis"))
                            or nm(g).startswith("lamp_")]
        self.k = 0
        self.frames, self.times, self.labels = [], [], []
        self.bchg = []                   # (frame, body, x y z, qw qx qy qz)
        self.gchg = []                   # (frame, geom, x y z, qw qx qy qz, alpha)
        self.last_b = {}
        self.last_g = {}
        self.overlays = []
        self.events = []
        self.label = ""
        self.vehicle_x = []

    def __call__(self, label: str | None = None, force: bool = False):
        if label is not None:
            self.label = label
        if force or self.k % self.every == 0:
            self._grab()
        self.k += 1

    def _grab(self):
        f = len(self.frames)
        d, m = self.d, self.m
        self.frames.append(self.k)
        self.times.append(self.k * self.sim.dt)
        self.labels.append(self.label)
        vb = getattr(self.sim, "vehicle_body", -1)
        self.vehicle_x.append(float(d.xpos[vb][0]) if vb is not None and vb >= 0 else 0.0)
        for b in self.bodies:
            p, q = d.xpos[b], d.xquat[b]
            last = self.last_b.get(b)
            if last is None or np.linalg.norm(p - last[0]) > self.pos_tol or abs(float(np.dot(q, last[1]))) < self.cos_tol:
                self.last_b[b] = (p.copy(), q.copy())
                self.bchg.append((f, b, *p, *q))
        for g in self.watch_geoms:
            st = (m.geom_pos[g].copy(), m.geom_quat[g].copy(), float(m.geom_rgba[g, 3]))
            last = self.last_g.get(g)
            if last is None or np.linalg.norm(st[0] - last[0]) > 1e-4 or abs(st[2] - last[2]) > 1e-3 or \
                    abs(float(np.dot(st[1], last[1]))) < 0.99999:
                self.last_g[g] = st
                self.gchg.append((f, g, *st[0], *st[1], st[2]))

    def overlay(self, **kw):
        keep = {}
        for k, v in kw.items():
            if k in ("t1", "t2", "fruit_center", "shoulder", "window", "bbox", "candidates"):
                keep[k] = None if v is None else np.round(np.asarray(v, float), 4).tolist()
            elif k == "info":
                keep[k] = v
        self.overlays.append((max(0, len(self.frames) - 1), keep))

    def save(self, path, result: dict | None = None):
        self(force=True)
        np.savez_compressed(path, frames=np.array(self.frames), times=np.array(self.times), labels=np.array(self.labels),
                            bchg=np.array(self.bchg, dtype=np.float64).reshape(-1, 9),
                            gchg=np.array(self.gchg, dtype=np.float64).reshape(-1, 10),
                            vehicle_x=np.array(self.vehicle_x), overlays=np.array([json.dumps(self.overlays)]),
                            result=np.array([json.dumps(result or {}, default=str)]), every=self.every, dt=self.sim.dt)
        return path


# ----------------------------------------------------------------------------- Meshcat export
CAMERA_PANEL_JS = r"""
<script>
(function () {
  // Camera modes + gripper-camera decision overlay (injected by pinesim/viz/xgh_recorder.py).
  const NODES = __NODES__;
  const OVL = __OVL__;          // [{f0, f1, text}] decision overlays (animation frames)
  const FPS = __FPS__;
  const LABELS = {free: "Free (orbit)", follow: "Follow Vehicle", corner: "Corner Camera",
                  overview: "Overview", wrist: "Wrist Camera"};
  const ORDER = ["free", "follow", "corner", "overview", "wrist"];
  let mode = "free";
  const panel = document.createElement("div");
  panel.style.cssText = "position:fixed;left:12px;bottom:12px;z-index:1000;background:rgba(20,24,28,.82);color:#eee;" +
    "font:13px/1.3 system-ui,sans-serif;padding:8px 10px;border-radius:8px;user-select:none";
  panel.innerHTML = "<div style='margin-bottom:6px;font-weight:600'>Camera Options</div>";
  const buttons = {};
  let keyn = 1; const keys = {};
  for (const k of ORDER) {
    if (k !== "free" && !NODES[k]) continue;
    const b = document.createElement("button");
    b.textContent = keyn + "  " + LABELS[k];
    keys[String(keyn)] = k; keyn += 1;
    b.style.cssText = "display:block;width:100%;margin:3px 0;padding:5px 8px;border:0;border-radius:5px;cursor:pointer;text-align:left";
    b.onclick = () => setMode(k);
    panel.appendChild(b);
    buttons[k] = b;
  }
  const hint = document.createElement("div");
  hint.style.cssText = "margin-top:6px;opacity:.75;font-size:11px;max-width:210px";
  hint.textContent = "Tombol angka juga bisa. Kamera gripper menampilkan bounding box (biru), window (putih), shoulder (titik magenta), kandidat T2, T1 (hijau), T2 (merah). Putar animasi lewat panel Animations (kanan atas).";
//  panel.appendChild(hint);
  document.body.appendChild(panel);
  const info = document.createElement("div");
  info.style.cssText = "position:fixed;left:12px;top:12px;z-index:1000;background:rgba(10,12,14,.78);color:#fff;" +
    "font:12px/1.35 ui-monospace,monospace;padding:7px 9px;border-radius:6px;display:none;max-width:360px;white-space:pre";
//  document.body.appendChild(info);
  let defaultFov = null;
  function paint() {
    for (const k in buttons) {
      buttons[k].style.background = (k === mode) ? "#f2a33a" : "#3a4046";
      buttons[k].style.color = (k === mode) ? "#111" : "#eee";
    }
  }
  function overlayVisible(on) {
    try { viewer.scene_tree.find(["wrist_overlay"]).object.visible = on; } catch (e) {}
  }
  function setMode(k) {
    mode = k;
    const cam = viewer.camera;
    if (defaultFov === null) defaultFov = cam.fov;
    if (k === "free") { cam.fov = defaultFov; cam.updateProjectionMatrix(); }
    overlayVisible(k === "wrist");
    info.style.display = (k === "wrist") ? "block" : "none";
    paint();
    viewer.set_dirty();
  }
  window.addEventListener("keydown", (e) => { if (keys[e.key]) setMode(keys[e.key]); });
  const T = MeshCat.THREE;
  function animTime() {      // clip time of the animation (the scrubber / seek position)
    try {
      const a = viewer.animator.actions;
      // tracks end at their own last keyframe (clamped): the longest-running track carries the clip time
      if (a && a.length) { let t = 0; for (const x of a) t = Math.max(t, x.time || 0); return t; }
      return viewer.animator.time || 0;
    } catch (e) { return 0; }
  }
  function apply() {
    const spec = NODES[mode];
    if (!spec) return;
    let node;
    try { node = viewer.scene_tree.find(spec.path).object; } catch (e) { return; }
    node.updateMatrixWorld(true);
    const p = new T.Vector3(), q = new T.Quaternion();
    node.getWorldPosition(p);
    node.getWorldQuaternion(q);
    const cam = viewer.camera, parent = cam.parent;
    parent.updateMatrixWorld(true);
    const pq = new T.Quaternion();
    parent.getWorldQuaternion(pq);
    cam.position.copy(parent.worldToLocal(p.clone()));
    cam.quaternion.copy(pq.clone().invert().multiply(q));
    const ahead = new T.Vector3(0, 0, -1).applyQuaternion(q).add(p);
    viewer.controls.target.copy(parent.worldToLocal(ahead));
    if (cam.fov !== spec.fov) { cam.fov = spec.fov; cam.updateProjectionMatrix(); }
    cam.updateMatrixWorld(true);
    if (mode === "wrist") {
      const f = animTime() * FPS;
      let txt = "Kamera gripper\nbelum ada keputusan T1/T2";
      for (const o of OVL) { if (f >= o.f0 && f < o.f1) txt = o.text; }
      info.textContent = txt;
    }
  }
  const prev = viewer.render_callback;
  viewer.render_callback = function () { if (prev) prev.call(viewer); if (mode !== "free") apply(); };
  setInterval(() => { if (mode !== "free") viewer.set_dirty(); }, 33);
  overlayVisible(false);
  paint();
})();
</script>
"""

INFO_JS = r"""
<div id="xgh-info" style="position:fixed;right:12px;bottom:12px;z-index:1000;background:rgba(20,24,28,.82);color:#eee;
 font:12px/1.35 system-ui,sans-serif;padding:8px 10px;border-radius:8px;max-width:360px">__INFO__</div>
"""


def _rt(pos, quat):
    from pydrake.common.eigen_geometry import Quaternion
    from pydrake.math import RigidTransform, RotationMatrix
    q = np.asarray(quat, float)
    q = q / np.linalg.norm(q)
    return RigidTransform(RotationMatrix(Quaternion(q[0], q[1], q[2], q[3])), np.asarray(pos, float))


def _shape(m, g):
    from pydrake.geometry import Box, Capsule, Cylinder, Ellipsoid, Mesh, Sphere
    t = m.geom_type[g]
    sz = m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return Box(2 * sz[0], 2 * sz[1], 2 * sz[2])
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return Capsule(max(sz[0], 1e-4), max(2 * sz[1], 1e-4))
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return Cylinder(sz[0], 2 * sz[1])
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return Sphere(sz[0])
    if t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return Ellipsoid(sz[0], sz[1], sz[2])
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        f = mesh_gltf(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, int(m.geom_dataid[g])))
        return None if f is None else Mesh(str(f), 1.0)
    return None


def _geom_rgba(m, g, alpha=None):
    from pydrake.geometry import Rgba
    c = m.geom_rgba[g].copy()
    mid = int(m.geom_matid[g])
    if mid >= 0 and np.allclose(c, [0.5, 0.5, 0.5, 1.0]):
        c = m.mat_rgba[mid].copy()
    return Rgba(float(c[0]), float(c[1]), float(c[2]), 1.0)


def overlay_text(d: dict) -> str:
    inf = d.get("info") or {}
    lines = ["Kamera gripper - keputusan T1/T2"]
    if inf:
        lines.append(f"pemilih T2      : {inf.get('method', '-')}")
        lines.append(f"kandidat T2     : {inf.get('n_candidates', '-')}")
        if inf.get("shoulder_ratio") is not None:
            lines.append(f"shoulder        : {inf.get('shoulder_px', 0)} px = {100 * inf['shoulder_ratio']:.0f} % buah terlihat")
        if inf.get("clearance") is not None:
            lines.append(f"jarak garis-daun: {100 * inf['clearance']:.1f} cm")
        if inf.get("votes") is not None:
            lines.append(f"suara diffusion : {inf['votes']:.0f}")
    if d.get("t1") is not None:
        lines.append("T1 hijau, T2 merah, window putih,")
        lines.append("bounding box biru, shoulder magenta")
    return "\n".join(lines)


def export_html(model: mujoco.MjModel, rec_path, out_html, frame_range=None, x_window=3.0, info_lines=(), fps=None,
                scenery=None):
    """Standalone Meshcat HTML of recorded frames [f0, f1) (None = all). `scenery`: None = no decoration; a dict
    (the `scenery:` block of the config, may be empty) = dome with the landscape photo, widened soil, rocks, trees,
    hut around the field (pinesim/viz/scenery.py; HTML only, not part of the simulation)."""
    from pydrake.geometry import Meshcat, MeshcatAnimation, MeshcatParams, Rgba, Sphere
    from pydrake.perception import BaseField, Fields, PointCloud
    z = np.load(rec_path, allow_pickle=False)
    nF = len(z["frames"])
    f0, f1 = (0, nF) if frame_range is None else (max(0, frame_range[0]), min(nF, frame_range[1]))
    m = model
    B = z["bchg"]
    G = z["gchg"]
    vx = z["vehicle_x"][f0:f1]
    has_vehicle = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "vehicle") >= 0
    x_lo, x_hi = float(vx.min()) - x_window, float(vx.max()) + 2.0 + x_window
    pose = {}
    for row in B[B[:, 0] <= f0]:
        pose[int(row[1])] = (row[2:5], row[5:9])
    gstate = {}
    for row in G[G[:, 0] <= f0]:
        gstate[int(row[1])] = (row[2:5], row[5:9], row[9])
    vg = visual_geoms(m)
    bname = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
    cams = camera_nodes(m)
    keep_bodies = set()
    for b in {int(m.geom_bodyid[g]) for g in vg} | {m.body(n).id for n, _ in cams.values() if n != "world"}:
        nm = bname(b)
        if has_vehicle and nm.startswith("splant_"):
            root = b
            while int(m.body_parentid[root]) > 0:
                root = int(m.body_parentid[root])
            p0 = pose.get(root, (m.body_pos[root], None))[0]
            if not (x_lo <= p0[0] <= x_hi):                 # static plants outside the window: not exported
                continue
        keep_bodies.add(b)
    mc = Meshcat(MeshcatParams())
    mc.Delete("/")
    mc.SetProperty("/Grid", "visible", False)
    mc.SetProperty("/Axes", "visible", False)
    mc.SetProperty("/Background", "top_color", [0.62, 0.76, 0.92])
    mc.SetProperty("/Background", "bottom_color", [0.93, 0.95, 0.97])
    mc.SetProperty("/Lights/AmbientLight/<object>", "intensity", 0.75)
    mc.SetProperty("/Lights/FillLight/<object>", "intensity", 0.9)
    x0 = float(vx[0])
    if has_vehicle:
        mc.SetCameraPose(np.array([x0 - 1.8, -3.4, 2.4]), np.array([x0 + 1.4, 0.0, 0.5]))
    elif mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "rig_head") >= 0:      # pretraining rig: close to the plant
        mc.SetCameraPose(np.array([1.05, -1.05, 1.15]), np.array([0.0, 0.0, 0.55]))
    else:
        mc.SetCameraPose(np.array([1.8, -1.9, 1.9]), np.array([0.0, 0.0, 0.8]))
    n_obj = 0
    scn = None
    if scenery is not None:
        from pinesim.viz.scenery import add_to_meshcat
        scn = add_to_meshcat(mc, m, scenery)
    for g in vg:
        b = int(m.geom_bodyid[g])
        if b not in keep_bodies and b != 0:
            continue
        sh = _shape(m, g)
        if sh is None:
            continue
        path = f"/scene/{bname(b)}/g{g}"
        gp, gq = (gstate[g][0], gstate[g][1]) if g in gstate else (None, None)
        p, q = geom_local_original(m, g, gp, gq)
        mc.SetObject(path, sh, _geom_rgba(m, g))
        mc.SetTransform(path, _rt(p, q))
        n_obj += 1
    nodes = {}
    for key, (body, cam) in cams.items():
        c = m.camera(cam).id
        path = f"/scene/{body}/cam_{key}"
        mc.SetObject(path, Sphere(0.001), Rgba(0, 0, 0, 0))
        mc.SetTransform(path, _rt(m.cam_pos[c], m.cam_quat[c]))
        nodes[key] = dict(path=["scene", body, f"cam_{key}"], fov=float(m.cam_fovy[c]))
    dt = float(z["dt"]) * int(z["every"])
    rate = fps or 1.0 / dt
    anim = MeshcatAnimation(frames_per_second=rate)
    for b in keep_bodies:
        if b in pose:
            anim.SetTransform(0, f"/scene/{bname(b)}", _rt(*pose[b]))
    # Meshcat interpolates linearly between keyframes: a pose that is held for a while and then jumps (a parked
    # plant placed in the rig, a crate lifted) gets a "hold" keyframe one frame before the jump, otherwise the
    # body would drift slowly towards the next pose (a rig plant sank into the ground between two scenes)
    lastk = {b: (0, pose[b]) for b in keep_bodies if b in pose}
    sel = B[(B[:, 0] > f0) & (B[:, 0] < f1)]
    for row in sel:
        b = int(row[1])
        if b in keep_bodies:
            k = int(row[0]) - f0
            if b in lastk and k - lastk[b][0] > 1:
                anim.SetTransform(k - 1, f"/scene/{bname(b)}", _rt(*lastk[b][1]))
            anim.SetTransform(k, f"/scene/{bname(b)}", _rt(row[2:5], row[5:9]))
            lastk[b] = (k, (row[2:5], row[5:9]))
    glast = {g: (0, geom_local_original(m, g, gstate[g][0], gstate[g][1])) for g in gstate}
    gsel = G[(G[:, 0] >= f0) & (G[:, 0] < f1)]
    for row in gsel:
        g = int(row[1])
        b = int(m.geom_bodyid[g])
        if (b not in keep_bodies and b != 0) or g not in vg:
            continue
        p, q = geom_local_original(m, g, row[2:5], row[5:9])
        k = max(0, int(row[0]) - f0)
        if g in glast and k - glast[g][0] > 1:
            anim.SetTransform(k - 1, f"/scene/{bname(b)}/g{g}", _rt(*glast[g][1]))
        glast[g] = (k, (p, q))
        anim.SetTransform(k, f"/scene/{bname(b)}/g{g}", _rt(p, q))
        anim.SetProperty(k, f"/scene/{bname(b)}/g{g}", "visible", bool(row[9] > 0.01 and p[2] > -10))
    for g in vg:                                    # initial visibility of toggled geoms (lamp starts dark)
        b = int(m.geom_bodyid[g])
        if (b not in keep_bodies and b != 0):
            continue
        if g in gstate:
            st = gstate[g]
            anim.SetProperty(0, f"/scene/{bname(b)}/g{g}", "visible", bool(st[2] > 0.01 and st[0][2] > -10))
        elif (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("lamp_") and m.geom_rgba[g, 3] < 0.01:
            anim.SetProperty(0, f"/scene/{bname(b)}/g{g}", "visible", False)
    # decision overlays: T1/T2 + line in every view; bbox / window / shoulder / candidates in the gripper view
    ov = [(k, d) for k, d in json.loads(str(z["overlays"][0])) if f0 <= k < f1]
    ovl_js = []
    for j, (k0, d) in enumerate(ov):
        k1 = ov[j + 1][0] if j + 1 < len(ov) else f1
        ovl_js.append(dict(f0=k0 - f0, f1=k1 - f0, text=overlay_text(d)))
        p = f"/overlay/{j}"
        w = f"/wrist_overlay/{j}"
        if d.get("t1") is not None and d.get("t2") is not None:
            mc.SetObject(f"{p}/T1", Sphere(0.014), Rgba(0.1, 0.9, 0.2, 1))
            mc.SetTransform(f"{p}/T1", _rt(d["t1"], [1, 0, 0, 0]))
            mc.SetObject(f"{p}/T2", Sphere(0.014), Rgba(0.95, 0.1, 0.1, 1))
            mc.SetTransform(f"{p}/T2", _rt(d["t2"], [1, 0, 0, 0]))
            mc.SetLine(f"{p}/line", np.array([d["t1"], d["t2"]]).T, 4.0, Rgba(1, 1, 1, 1))
            mc.SetObject(f"{w}/T1", Sphere(0.010), Rgba(0.1, 1.0, 0.2, 1))
            mc.SetTransform(f"{w}/T1", _rt(d["t1"], [1, 0, 0, 0]))
            mc.SetObject(f"{w}/T2", Sphere(0.008), Rgba(1.0, 0.1, 0.1, 1))
            mc.SetTransform(f"{w}/T2", _rt(d["t2"], [1, 0, 0, 0]))
        if d.get("window"):
            c = np.array(d["window"])
            mc.SetLine(f"{w}/window", np.vstack([c, c[:1]]).T, 3.0, Rgba(1, 1, 1, 1))
        if d.get("bbox"):
            c = np.array(d["bbox"])
            mc.SetLine(f"{w}/bbox", np.vstack([c, c[:1]]).T, 2.0, Rgba(0.2, 0.55, 1.0, 1))
        if d.get("shoulder"):
            X = np.array(d["shoulder"]).reshape(-1, 3)
            cloud = PointCloud(len(X), Fields(BaseField.kXYZs))
            cloud.mutable_xyzs()[:] = X.T
            mc.SetObject(f"{w}/shoulder", cloud, point_size=0.007, rgba=Rgba(1.0, 0.15, 0.85, 1))
        if d.get("candidates"):
            for ci, cpt in enumerate(np.array(d["candidates"]).reshape(-1, 3)[:12]):
                mc.SetObject(f"{w}/cand{ci}", Sphere(0.0045), Rgba(0.2, 1.0, 1.0, 1))
                mc.SetTransform(f"{w}/cand{ci}", _rt(cpt, [1, 0, 0, 0]))
        for path in (p, w):
            anim.SetProperty(0, path, "visible", False)
            anim.SetProperty(max(0, k0 - f0), path, "visible", True)
            anim.SetProperty(max(0, k1 - f0), path, "visible", False)
    mc.SetAnimation(anim)
    html = mc.StaticHtml()
    html = html.replace("</body>", "" +
                        CAMERA_PANEL_JS.replace("__NODES__", json.dumps(nodes)).replace("__OVL__", json.dumps(ovl_js))
                        .replace("__FPS__", f"{rate:.6f}") + "\n</body>")
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    Path(out_html).write_text(html, encoding="utf-8")
    return dict(objects=n_obj, frames=f1 - f0, size_mb=round(len(html) / 1e6, 1), bodies=len(keep_bodies),
                overlays=len(ov), cameras=list(nodes), **({"scenery": scn} if scn is not None else {}))


# ----------------------------------------------------------------------------- corner camera video
class CornerVideo:
    """MP4 of the corner camera with the wrist camera as an inset, written during harvesting phases only."""

    def __init__(self, sim, path, every: int = 5, size=(640, 480), fps=None):
        import imageio
        self.sim, self.every = sim, int(every)
        self.w, self.h = size
        self.r = mujoco.Renderer(sim.m, self.h, self.w)
        self.iw, self.ih = self.w // 3, self.h // 3
        self.ri = mujoco.Renderer(sim.m, self.ih, self.iw)
        self.writer = imageio.get_writer(str(path), fps=fps or 1.0 / (sim.dt * self.every), codec="libx264",
                                         quality=7, macro_block_size=8, ffmpeg_params=["-preset", "veryfast"])
        self.k = 0
        self.n = 0
        self.text = ""

    def __call__(self, label: str):
        self.k += 1
        if label not in HARVEST_LABELS or self.k % self.every:
            return
        from PIL import Image, ImageDraw
        self.r.update_scene(self.sim.d, "corner_cam")
        img = self.r.render().copy()
        self.ri.update_scene(self.sim.d, "wrist_cam")
        ins = self.ri.render()
        img[8:8 + self.ih, self.w - 8 - self.iw:self.w - 8] = ins
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        dr.rectangle([self.w - 9 - self.iw, 7, self.w - 7, 9 + self.ih], outline=(255, 255, 255))
        dr.text((10, 10), f"{label}  {self.text}", fill=(255, 255, 255))
        self.writer.append_data(np.asarray(im))
        self.n += 1

    def close(self):
        self.writer.close()
        for r in (self.r, self.ri):
            try:
                r.close()
            except Exception:
                pass
