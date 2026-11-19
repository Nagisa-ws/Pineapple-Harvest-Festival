"""
Pretraining - interactive Stargaze camera rig (version 5: realistic pineapples, shoulder / window / diffusion T2).

A still pineapple (the user's leaf, fruit and crown meshes; 5 x 5 x 5 combinations, fruit centre 0.60-0.70 m,
plant tilted 0-15 deg), a camera on a pole that orbits it. Move the pole; when the camera has been still for
`settle_s` seconds the frame is captured and the computation runs: SHOULDER (fruit pixels between the centre and the
crown base; leaf-covered parts excluded) -> WINDOW (bounding box of the shoulder) -> several T2 candidates in the
window, the safest line T1-T2 for each (T1 inside the window) -> the DIFFUSION POLICY picks the best T2.
Moving the pole again cancels the running computation.

  python pretrain/stargaze_viewer.py --viewer meshcat            # browser UI with sliders; button 'Camera view'
  python pretrain/stargaze_viewer.py --viewer mujoco             # MuJoCo viewer: keys / drag the pole
  python pretrain/stargaze_viewer.py --viewer demo --html outputs/pretrain/rig_demo.html   # scripted, headless

Lamp on the camera head: red = moving, yellow = settling (camera still, waiting), blue = computing,
green = result, dark = no result. In the camera view (Meshcat button / HTML 'Kamera gripper'): bounding box of the
visible fruit (blue), window (white), shoulder (magenta points), T2 candidates (cyan), T1 (green), T2 (red).
MuJoCo viewer keys: LEFT/RIGHT azimuth -/+ 5 deg, UP/DOWN camera height +/- 5 cm, PAGE_UP/PAGE_DOWN radius
+/- 5 cm, HOME/END aim height +/- 5 cm, N new plant.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

if "mujoco" not in sys.argv:                 # headless rendering unless the MuJoCo viewer (needs a display) is used
    os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from pinesim.config import PROJECT_ROOT, load_config
from pinesim.rig import RigController, RigPose, StargazeRig

OUT = PROJECT_ROOT / "outputs" / "pretrain"
LAMP_OF_STATE = {"MOVING": "fail", "SETTLING": "observe", "COMPUTING": "action", "DONE": "success", "FAILED": None}


def describe(res) -> str:
    if res is None:
        return ""
    if not res.ok:
        return f"no plan: {res.reason}"
    d = res.diagnostics
    return (f"T1={np.round(res.t1, 3).tolist()} T2={np.round(res.t2, 3).tolist()} | shoulder {d.get('shoulder_pixels', 0)} px "
            f"({100 * d.get('shoulder_ratio', 0):.0f} % of the visible fruit), {d.get('n_candidates', 0)} T2 candidates, "
            f"decision {d.get('decision', '-')} | clearance perceived {res.clearance * 100:.1f} cm, true "
            f"{d.get('true_clearance_chosen', np.nan) * 100:.1f} cm | camera check: centre err "
            f"{d.get('fruit_center_error', np.nan) * 100:.1f} cm, axis err {d.get('axis_error_deg', np.nan):.1f} deg, "
            f"reprojection {d.get('reprojection_px', np.nan):.1f} px | {res.compute_s * 1000:.0f} ms")


def set_lamp(rig: StargazeRig, state: str, t: float):
    m = rig.model
    on = (t * 2.0) % 1.0 < 0.5 or state in ("DONE",)
    want = LAMP_OF_STATE.get(state)
    for k in ("observe", "action", "success", "fail"):
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"lamp_{k}")
        if g >= 0:
            m.geom_rgba[g, 3] = 1.0 if (k == want and on) else 0.0


def overlay_of(res) -> dict:
    """Recorder overlay dict of a rig result (same keys as the mission environments)."""
    dec = res.decision
    if dec is None:
        return {}
    sh = dec.shoulder or {}
    X = sh.get("X", np.zeros((0, 3)))
    if len(X) > 400:
        X = X[np.random.default_rng(0).choice(len(X), 400, replace=False)]
    d = res.diagnostics
    info = dict(method=dec.method, n_candidates=len(dec.candidates), shoulder_ratio=float(sh.get("ratio", 0.0)),
                shoulder_px=int(sh.get("n_shoulder", 0)), clearance=float(res.clearance) if res.ok else None,
                votes=float(dec.candidates[0].votes) if dec.candidates else None)
    return dict(t1=res.t1, t2=res.t2, fruit_center=None if res.fruit is None else res.fruit.center, shoulder=X,
                window=dec.window_corners, bbox=dec.bbox_corners,
                candidates=np.array([c.t2 for c in dec.candidates]) if dec.candidates else None, info=info)


def save_debug_image(rig: StargazeRig, res, path: Path):
    """Camera image | colour classes + shoulder, window (white), bounding box (blue), candidates, T1, T2."""
    import cv2
    from pinesim.vision.perception import Perceiver
    fr = res.frame
    img = fr["rgb"].copy()
    fruit, leaf = Perceiver(rig.cfg).masks(img)
    over = img.copy()
    over[fruit] = (0.5 * over[fruit] + 0.5 * np.array([255, 120, 0])).astype(np.uint8)
    over[leaf] = (0.6 * over[leaf] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
    dec = res.decision
    if dec is not None and dec.shoulder and len(dec.shoulder.get("us", [])):
        over[dec.shoulder["vs"], dec.shoulder["us"]] = [255, 40, 215]
    if dec is not None and dec.window is not None:
        u0, v0, u1, v1 = [int(round(x)) for x in dec.window]
        cv2.rectangle(over, (u0, v0), (u1, v1), (255, 255, 255), 1)
    if dec is not None and dec.bbox is not None:
        u0, v0, u1, v1 = [int(round(x)) for x in dec.bbox]
        cv2.rectangle(over, (u0, v0), (u1, v1), (60, 140, 255), 1)
    if dec is not None:
        for c in dec.candidates:
            u, v, _ = rig.cam.project_world(c.t2[None, :], fr["R"], fr["t"])
            cv2.circle(over, (int(u[0]), int(v[0])), 2, (0, 255, 255), -1)
    if res.ok:
        u, v, z = rig.cam.project_world(np.vstack([res.t1, res.t2]), fr["R"], fr["t"])
        p = [(int(round(a)), int(round(b))) for a, b in zip(u, v)]
        cv2.line(over, p[0], p[1], (255, 255, 255), 1)
        cv2.circle(over, p[0], 4, (0, 220, 0), -1)
        cv2.circle(over, p[1], 4, (230, 0, 0), -1)
    big = cv2.resize(np.hstack([img, over]), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), big[:, :, ::-1])


class RigSim:
    """Adapter so that the mission recorder (pinesim/viz/xgh_recorder.XghRecorder) records the rig."""

    def __init__(self, rig: StargazeRig, dt: float = 0.05):
        self.m, self.d, self.dt, self.vehicle_body = rig.model, rig.data, dt, -1


# ----------------------------------------------------------------------------- Meshcat front-end (live)
class MeshcatFront:
    def __init__(self, rig: StargazeRig, ctrl: RigController, meshcat=None, port=None):
        from pinesim.viz import meshcat_viewer as mv
        self.mv = mv
        self.rig, self.ctrl = rig, ctrl
        self.mc = meshcat or mv.start_meshcat(port)
        self.publish()
        p = rig.pose
        self.mc.AddSlider("azimuth_deg", 0.0, 360.0, 1.0, p.azimuth_deg)
        self.mc.AddSlider("radius_m", 0.4, 1.6, 0.01, p.radius)
        self.mc.AddSlider("height_m", 0.2, 1.8, 0.01, p.height)
        self.mc.AddSlider("aim_height_m", 0.0, 1.2, 0.01, p.aim_height)
        self.mc.AddSlider("settle_s", 0.2, 5.0, 0.1, ctrl.settle_s)
        self.mc.AddButton("New plant")
        self.mc.AddButton("Camera view")
        self.mc.AddButton("Quit")
        self._new_clicks, self._cam_clicks, self._cam_view, self._last_state = 0, 0, False, None

    def publish(self):
        """Realistic meshes (glTF) + primitives of the rig model."""
        from pinesim.viz.xgh_recorder import _geom_rgba, _rt, _shape, geom_local_original, visual_geoms
        m = self.rig.model
        self.mc.Delete()
        self.mc.SetProperty("/Grid", "visible", False)
        for g in visual_geoms(m):
            sh = _shape(m, g)
            if sh is None:
                continue
            b = int(m.geom_bodyid[g])
            path = f"/scene/{self.mv.body_name(m, b)}/g{g}"
            p, q = geom_local_original(m, g)
            self.mc.SetObject(path, sh, _geom_rgba(m, g))
            self.mc.SetTransform(path, _rt(p, q))
        self.mv.set_body_poses(self.mc, m, self.rig.data)

    def read_pose(self) -> RigPose:
        g = self.mc.GetSliderValue
        return RigPose(g("azimuth_deg"), g("radius_m"), g("height_m"), g("aim_height_m"))

    def update(self, state: str):
        rig, m = self.rig, self.rig.model
        set_lamp(rig, state, time.time())
        self.mv.set_body_poses(self.mc, m, rig.data)
        for k in ("observe", "action", "success", "fail"):
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"lamp_{k}")
            self.mc.SetProperty(f"/scene/rig_head/g{g}", "visible", bool(m.geom_rgba[g, 3] > 0.5))
        if state != self._last_state:
            if state in ("MOVING", "SETTLING"):
                self.mc.Delete("/decision")
            if state in ("DONE", "FAILED"):
                self.show_result(self.ctrl.result)
            self._last_state = state

    def show_result(self, res):
        if res is None:
            return
        mv, mc = self.mv, self.mc
        o = overlay_of(res)
        if o.get("t1") is not None:
            mv.draw_sphere(mc, "/decision/T1", o["t1"], 0.012, (0.1, 0.9, 0.2, 1))
            mv.draw_sphere(mc, "/decision/T2", o["t2"], 0.010, (0.95, 0.1, 0.1, 1))
            mv.draw_line(mc, "/decision/line", [o["t1"], o["t2"]], (1, 1, 1, 1), 4.0)
        if o.get("window") is not None:
            c = np.asarray(o["window"])
            mv.draw_line(mc, "/decision/window", np.vstack([c, c[:1]]), (1, 1, 1, 1), 3.0)
        if o.get("bbox") is not None:
            c = np.asarray(o["bbox"])
            mv.draw_line(mc, "/decision/bbox", np.vstack([c, c[:1]]), (0.2, 0.55, 1.0, 1), 2.0)
        if o.get("shoulder") is not None and len(o["shoulder"]):
            mv.draw_points(mc, "/decision/shoulder", o["shoulder"], (1.0, 0.15, 0.85, 1), 0.007)
        if o.get("candidates") is not None:
            for k, c in enumerate(o["candidates"]):
                mv.draw_sphere(mc, f"/decision/cand{k}", c, 0.0045, (0.2, 1.0, 1.0, 1))
        print(describe(res), flush=True)
        if res.frame is not None:
            save_debug_image(self.rig, res, OUT / "last_view.png")

    def set_camera_view(self, on: bool):
        from pydrake.geometry import Meshcat
        PerspectiveCamera = Meshcat.PerspectiveCamera
        self._cam_view = on
        for path in ("/scene/rig_head", "/scene/rig_pole"):
            self.mc.SetProperty(path, "visible", not on)
        if on:
            w, h = self.rig.cam.width, self.rig.cam.height
            self.mc.SetCamera(PerspectiveCamera(fov=float(self.rig.cam.fovy), aspect=w / h, near=0.02, far=20.0))
            self.follow_camera()
        else:
            self.mc.SetCamera(PerspectiveCamera(fov=75.0))
            c = np.asarray(self.rig.plant.fruit_center, float)
            self.mc.SetCameraPose(c + np.array([1.4, -1.4, 0.9]), c)

    def follow_camera(self):
        rig = self.rig
        cid = rig.cam.cam_id
        p = rig.data.cam_xpos[cid].copy()
        fwd = -rig.data.cam_xmat[cid].reshape(3, 3)[:, 2]
        self.mc.SetCameraPose(p, p + fwd)

    def run(self):
        print(f"Meshcat: {self.mc.web_url()}  (sliders on the right; 'Quit' to stop)", flush=True)
        while self.mc.GetButtonClicks("Quit") == 0:
            self.ctrl.settle_s = self.mc.GetSliderValue("settle_s")
            if self.mc.GetButtonClicks("New plant") > self._new_clicks:
                self._new_clicks = self.mc.GetButtonClicks("New plant")
                self.ctrl.cancel_running()
                self.rig.new_plant(int(time.time()) % 100000)
                print("plant:", self.rig.plant.summary(), flush=True)
                self.publish()
                if self._cam_view:
                    self.set_camera_view(True)
                self.ctrl.state, self.ctrl._t_still, self._last_state = "SETTLING", time.time(), None
            n = self.mc.GetButtonClicks("Camera view")
            if n != self._cam_clicks:
                self._cam_clicks = n
                self.set_camera_view(not self._cam_view)
            state = self.ctrl.tick(self.read_pose())
            self.update(state)
            if self._cam_view:
                self.follow_camera()
            time.sleep(0.05)
        self.ctrl.cancel_running()


# ----------------------------------------------------------------------------- MuJoCo viewer front-end
class MujocoFront:
    def __init__(self, rig: StargazeRig, ctrl: RigController):
        self.rig, self.ctrl = rig, ctrl
        self.pose = RigPose(**vars(rig.pose))
        self._new_plant = False
        self._drag_ref = rig.data.mocap_pos[rig.pole_mocap].copy()

    def key(self, keycode: int):
        step = {262: ("azimuth_deg", +5), 263: ("azimuth_deg", -5), 265: ("height", +0.05), 264: ("height", -0.05),
                266: ("radius", +0.05), 267: ("radius", -0.05), 268: ("aim_height", +0.05), 269: ("aim_height", -0.05)}
        if keycode in step:
            k, dv = step[keycode]
            setattr(self.pose, k, getattr(self.pose, k) + dv)
            self.pose.azimuth_deg %= 360.0
        elif keycode == ord("N"):
            self._new_plant = True

    def draw(self, viewer, state):
        scn = viewer.user_scn
        scn.ngeom = 0
        res = self.ctrl.result

        def add(type_, size, pos, rgba):
            if scn.ngeom >= scn.maxgeom:
                return
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], type_, np.asarray(size, float), np.asarray(pos, float),
                                np.eye(3).ravel(), np.asarray(rgba, np.float32))
            scn.ngeom += 1

        if state in ("DONE", "FAILED") and res is not None and res.ok:
            o = overlay_of(res)
            add(mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0], o["t1"], (0.1, 0.9, 0.2, 1))
            add(mujoco.mjtGeom.mjGEOM_SPHERE, [0.010, 0, 0], o["t2"], (0.95, 0.1, 0.1, 1))
            for c in (o.get("candidates") if o.get("candidates") is not None else []):
                add(mujoco.mjtGeom.mjGEOM_SPHERE, [0.005, 0, 0], c, (0.2, 1.0, 1.0, 1))
            for p in (o.get("shoulder") if o.get("shoulder") is not None else [])[::4]:
                add(mujoco.mjtGeom.mjGEOM_SPHERE, [0.003, 0, 0], p, (1.0, 0.15, 0.85, 1))

    def run(self):
        import mujoco.viewer
        rig = self.rig
        with mujoco.viewer.launch_passive(rig.model, rig.data, key_callback=self.key) as viewer:
            last = None
            while viewer.is_running():
                with viewer.lock():
                    p = rig.data.mocap_pos[rig.pole_mocap]
                    if np.linalg.norm(p - self._drag_ref) > 1e-4:
                        dp = rig.pose_from_mocap()
                        self.pose.azimuth_deg, self.pose.radius = dp.azimuth_deg, dp.radius
                    if self._new_plant:
                        self._new_plant = False
                        self.ctrl.cancel_running()
                        rig.new_plant(int(time.time()) % 100000)
                        self.ctrl.state, self.ctrl._t_still = "SETTLING", time.time()
                    state = self.ctrl.tick(RigPose(**vars(self.pose)))
                    set_lamp(rig, state, time.time())
                    self._drag_ref = rig.data.mocap_pos[rig.pole_mocap].copy()
                    self.draw(viewer, state)
                if state != last and state in ("DONE", "FAILED"):
                    print(describe(self.ctrl.result), flush=True)
                    if self.ctrl.result is not None and self.ctrl.result.frame is not None:
                        save_debug_image(rig, self.ctrl.result, OUT / "last_view.png")
                last = state
                viewer.sync()
                time.sleep(0.02)
        self.ctrl.cancel_running()


# ----------------------------------------------------------------------------- scripted demo (headless) -> HTML
def run_demo(rig: StargazeRig, ctrl: RigController, html: str | None, slow_s: float = 0.0, plants=(0, 1, 2)):
    """Simulated user, for several plants: move the pole, wait (compute), move during a computation (must cancel),
    wait again. Recorded with the mission recorder -> standalone Meshcat HTML with the camera-mode panel."""
    from pinesim.viz.xgh_recorder import XghRecorder, export_html
    rig.cfg.setdefault("rig", {})["debug_slow_s"] = slow_s
    dt = 0.05
    rsim = RigSim(rig, dt)
    rec = XghRecorder(rsim, every=1)
    results = []
    k = 0
    for pi in plants:
        rig.new_plant(1000 + pi)
        print("plant:", rig.plant.summary(), flush=True)
        p0 = RigPose(20.0 + 90 * pi, 0.75, 1.05, 0.62)
        script = []
        for j in range(12):
            script.append((dt, RigPose(p0.azimuth_deg + 5 * (j + 1), p0.radius, p0.height, p0.aim_height)))
        a1 = p0.azimuth_deg + 60
        script.append((ctrl.settle_s + 2.5, RigPose(a1, 0.7, 1.0, 0.62)))                # still -> compute
        script.append((ctrl.settle_s + 0.15, RigPose(a1 + 120, 0.65, 0.92, 0.62)))       # other side ...
        script.append((dt * 3, RigPose(a1 + 128, 0.65, 0.92, 0.62)))                     # ... moved while computing
        script.append((ctrl.settle_s + 2.5, RigPose(a1 + 128, 0.65, 0.92, 0.62)))       # still -> compute
        for dur, pose in script:
            t_end = time.time() + dur
            while True:
                st = ctrl.tick(pose)
                set_lamp(rig, st, k * dt)
                rec(st)
                k += 1
                if st in ("DONE", "FAILED") and (not results or results[-1] is not ctrl.result):
                    results.append(ctrl.result)
                    print(f"[{st}] az={ctrl.result.pose.azimuth_deg:.0f}  {describe(ctrl.result)}", flush=True)
                    if ctrl.result.ok:
                        rec.overlay(**overlay_of(ctrl.result))
                    if ctrl.result.frame is not None:
                        save_debug_image(rig, ctrl.result, OUT / f"view_p{pi}_az{int(ctrl.result.pose.azimuth_deg)}.png")
                if time.time() >= t_end:
                    break
                time.sleep(0.02)
    print(f"started={ctrl.n_started} cancelled={ctrl.n_cancelled} finished={ctrl.n_done}")
    for t, msg in ctrl.log:
        print(f"  {t - ctrl.log[0][0]:6.2f}s  {msg}")
    if html:
        rp = OUT / "rig_demo_recording.npz"
        rec.save(rp, result=dict(started=ctrl.n_started, cancelled=ctrl.n_cancelled, finished=ctrl.n_done))
        ok = [r for r in results if r.ok]
        lines = ["<b>Pretraining - rig kamera (mesh realistis)</b>",
                 f"komputasi: dimulai {ctrl.n_started}, dibatalkan karena kamera bergerak {ctrl.n_cancelled}, selesai {ctrl.n_done}",
                 f"hasil T1/T2: {len(ok)}; pemilih T2: {ok[0].diagnostics.get('decision', '-') if ok else '-'}",
                 "lampu di kepala kamera: merah = bergerak, kuning = diam/menunggu, biru = menghitung, hijau = hasil",
                 "pilih 'Kamera gripper' (kamera rig) untuk melihat bounding box, window, shoulder, T1, T2"]
        st = export_html(rig.model, rp, html, info_lines=lines, fps=1.0 / dt)
        print("standalone HTML ->", html, st)
    return results, ctrl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--viewer", choices=["meshcat", "mujoco", "demo"], default="meshcat")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--settle", type=float, default=None)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--html", default=None, help="demo: write a standalone Meshcat HTML")
    ap.add_argument("--slow", type=float, default=0.0, help="demo: artificial compute time to show cancellation")
    a = ap.parse_args()
    if a.viewer != "mujoco":
        os.environ.setdefault("MUJOCO_GL", "egl")
    cfg = load_config("pretrain")
    rig = StargazeRig(cfg, seed=a.seed, noise=not a.no_noise)
    ctrl = RigController(rig, a.settle if a.settle is not None else cfg["rig"]["settle_s"])
    print("plant:", rig.plant.summary(), "| decision:", "diffusion policy" if rig.policy is not None else "heuristic (no model)")
    if a.viewer == "meshcat":
        MeshcatFront(rig, ctrl, port=a.port).run()
    elif a.viewer == "mujoco":
        MujocoFront(rig, ctrl).run()
    else:
        run_demo(rig, ctrl, a.html, slow_s=a.slow)


if __name__ == "__main__":
    main()
