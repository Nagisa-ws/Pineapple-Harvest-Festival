"""
pinesim/rig.py
--------------
Pretraining / camera debugging: the Stargaze camera rig (version 5: REALISTIC pineapples).

A pineapple stands still at the origin: one of the user's 5 plant meshes (rest-pose leaves), 5 fruit meshes and 5
crown meshes (125 combinations), fruit centre 0.60-0.70 m above the ground, the whole plant tilted 0-15 deg in any
direction. A camera sits on top of a pole; the pole runs on a circular track around the plant and the camera always
looks at the plant (aim height adjustable). Logic required by the user (unchanged):

  * while the pole moves                     -> state MOVING, nothing is computed; any running computation is CANCELLED
  * pole still for `settle_s` seconds        -> capture an RGB-D frame and compute (in a worker thread):
        perception (fruit cluster + axis, leaf cloud) -> SHOULDER (fruit pixels between the centre and the crown base,
        leaf-covered parts excluded) -> WINDOW (its bounding box) -> several T2 candidates in the window, the safest
        line T1-T2 for each (T1 inside the window) -> DIFFUSION POLICY picks the best T2 (-> T1, T2)
  * user moves the pole again (other side)   -> the computation stops immediately; the cycle restarts

Camera self-check: every result carries diagnostics against the simulator's ground truth (used ONLY for these
numbers): fruit centre / radius / tilt error, re-projection error, clearance of the chosen line to the TRUE leaves,
best true clearance among the candidates (how good the decision was).
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from pinesim.config import PROJECT_ROOT, load_config
from pinesim.model.builder import build
from pinesim.model.builder_xgh import ASSETS, plant_quat
from pinesim.vision.camera import RGBDCamera, look_at_quat_mj
from pinesim.vision.perception import Cancelled, Perceiver, _check


@dataclass
class RigPose:
    azimuth_deg: float = 0.0
    radius: float = 0.75
    height: float = 1.05
    aim_height: float = 0.62

    def close_to(self, o: "RigPose", tol: float = 1e-4) -> bool:
        return (abs(self.azimuth_deg - o.azimuth_deg) < 1e-2 and abs(self.radius - o.radius) < tol and
                abs(self.height - o.height) < tol and abs(self.aim_height - o.aim_height) < tol)


@dataclass
class RigPlant:
    """Ground truth of the shown combination (diagnostics / pretraining labels only)."""
    plant: int
    fruit: int
    crown: int
    zc: float
    D: float
    H: float
    yaw: float
    tilt: float
    tilt_az: float
    quat: np.ndarray
    leaf_pts: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 3)))
    leaf_rad: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))

    @property
    def R(self):
        M = np.zeros(9)
        mujoco.mju_quat2Mat(M, self.quat)
        return M.reshape(3, 3)

    @property
    def axis(self):
        return self.R[:, 2]

    @property
    def fruit_center(self):
        return self.R @ np.array([0.0, 0.0, self.zc])

    @property
    def fruit_r(self):
        return self.D / 2

    @property
    def fruit_half(self):
        return self.H / 2 - self.D / 2

    def summary(self):
        return dict(plant=self.plant, fruit=self.fruit, crown=self.crown, fruit_center_z=round(float(self.fruit_center[2]), 3),
                    fruit_diameter=self.D, tilt_deg=round(float(np.degrees(self.tilt)), 1))


@dataclass
class RigResult:
    pose: RigPose
    ok: bool
    reason: str = ""
    t1: np.ndarray | None = None
    t2: np.ndarray | None = None
    clearance: float = float("nan")          # line clearance vs perceived leaves (m)
    decision: object = None                  # pinesim.decision.window.WindowDecision
    fruit: object = None
    fruit_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    leaf_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    obstacle_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    camera_pos: np.ndarray | None = None
    diagnostics: dict = field(default_factory=dict)
    compute_s: float = 0.0
    frame: dict | None = None
    view: dict | None = None


def _seg_points(pinfo, R, per_seg=4):
    pts, rad = [], []
    for leaf in pinfo["leaves"]:
        for sg in leaf["segments"]:
            p0, p1 = np.array(sg["p0"]), np.array(sg["p1"])
            for t in (np.arange(per_seg) + 0.5) / per_seg:
                pts.append(R @ (p0 + t * (p1 - p0)))
                rad.append(max(sg["radius"], 0.5 * np.linalg.norm(p1 - p0) / per_seg))
    return np.array(pts), np.array(rad)


class StargazeRig:
    def __init__(self, cfg: dict | None = None, seed: int = 0, level: int | None = 2, noise: bool = True, policy=None):
        self.cfg = cfg or load_config("pretrain")
        path = build("rig", 1, self.cfg)
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.cam = RGBDCamera(m, self.data, "rig_cam", self.cfg, seed=seed, noise=noise)
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.pole_mocap = int(m.body_mocapid[bid("rig_pole")])
        self.head_mocap = int(m.body_mocapid[bid("rig_head")])
        self.plant_mocap = {p: int(m.body_mocapid[bid(f"rplant{p}")]) for p in range(1, 6)}
        self.fruit_mocap = {f: int(m.body_mocapid[bid(f"rfruit{f}")]) for f in range(1, 6)}
        self.crown_mocap = {c: int(m.body_mocapid[bid(f"rcrown{c}")]) for c in range(1, 6)}
        self.pinfo = {p: json.loads((ASSETS / "plants" / f"p{p}.json").read_text()) for p in range(1, 6)}
        self.summ = json.loads((ASSETS / "summary.json").read_text())
        self.crown_pts = {}
        import trimesh
        for c in range(1, 6):
            v = trimesh.load(ASSETS / "pineapple" / f"crown_{c}.stl", force="mesh").vertices
            self.crown_pts[c] = v[np.random.default_rng(c).choice(len(v), min(120, len(v)), replace=False)]
        self.rng = np.random.default_rng(seed)
        self.policy = policy
        if policy is None:
            from pinesim.decision.diffusion import load_policy
            self.policy = load_policy(self.cfg)
        elif policy is False:                           # dataset generation: no policy needed
            self.policy = None
        self.pose = RigPose(**{k: v for k, v in self.cfg.get("rig", {}).get("initial_pose", {}).items()})
        self.plant: RigPlant | None = None
        self.new_plant(seed)

    # ------------------------------------------------------------------ scene
    def new_plant(self, seed: int | None = None, combo=None):
        """Show a random combination (plant, fruit, crown) at a random fruit height (0.60-0.70 m) and tilt."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        rng, cfg = self.rng, self.cfg
        p, f, c = combo if combo is not None else (int(rng.integers(1, 6)), int(rng.integers(1, 6)), int(rng.integers(1, 6)))
        D, H, _ = cfg["realistic"]["fruit"][str(f)]
        zc = float(rng.uniform(*cfg["plant"]["fruit"]["center_height_range"]))
        t0, t1 = np.radians(cfg["plant"].get("tilt_deg_range", [0, 15]))
        pl = dict(yaw=float(rng.uniform(0, 2 * np.pi)), tilt=float(rng.uniform(t0, t1)), tilt_az=float(rng.uniform(0, 2 * np.pi)))
        q = plant_quat(pl)
        d, m = self.data, self.model
        mujoco.mj_resetData(m, d)
        for k in range(1, 6):
            d.mocap_pos[self.plant_mocap[k]] = [0, 0, -10]
            d.mocap_pos[self.fruit_mocap[k]] = [0, 0, -10]
            d.mocap_pos[self.crown_mocap[k]] = [0, 0, -10]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        R = R.reshape(3, 3)
        zb = zc - H / 2
        d.mocap_pos[self.plant_mocap[p]] = [0, 0, 0]
        d.mocap_quat[self.plant_mocap[p]] = q
        fq = np.zeros(4)
        mujoco.mju_mulQuat(fq, q, np.array([np.cos(pl["yaw"]), 0, 0, np.sin(pl["yaw"])]))
        fpos = R @ np.array([0.0, 0.0, zb])
        d.mocap_pos[self.fruit_mocap[f]] = fpos
        d.mocap_quat[self.fruit_mocap[f]] = fq
        crown_z = self.summ[f"fruit_{f}"]["top"] - 0.012
        d.mocap_pos[self.crown_mocap[c]] = fpos + R @ np.array([0, 0, crown_z])
        d.mocap_quat[self.crown_mocap[c]] = fq
        lp, lr = _seg_points(self.pinfo[p], R)
        Rf = np.zeros(9)
        mujoco.mju_quat2Mat(Rf, fq)
        cp = (self.crown_pts[c] + np.array([0, 0, crown_z])) @ Rf.reshape(3, 3).T + fpos
        self.plant = RigPlant(p, f, c, zc, float(D), float(H), pl["yaw"], pl["tilt"], pl["tilt_az"], q,
                              np.vstack([lp, cp]), np.r_[lr, np.full(len(cp), 0.006)])
        self.set_pose(self.pose)
        return self.plant

    @property
    def layout(self):
        return self.plant

    def set_pose(self, pose: RigPose):
        r = self.cfg.get("rig", {})
        pose.radius = float(np.clip(pose.radius, *r.get("radius_range", [0.4, 1.6])))
        pose.height = float(np.clip(pose.height, *r.get("height_range", [0.2, 1.8])))
        self.pose = pose
        a = np.radians(pose.azimuth_deg)
        foot = np.array([pose.radius * np.cos(a), pose.radius * np.sin(a), 0.0])
        head = foot + np.array([0.0, 0.0, pose.height])
        target = np.array([0.0, 0.0, pose.aim_height])
        self.place_camera(head, target, pole_foot=foot)

    def place_camera(self, head, target, pole_foot=None):
        d = self.data
        if pole_foot is None:
            pole_foot = np.array([head[0], head[1], 0.0])
        a = np.arctan2(pole_foot[1], pole_foot[0])
        d.mocap_pos[self.pole_mocap] = pole_foot
        d.mocap_quat[self.pole_mocap] = [np.cos(a / 2), 0, 0, np.sin(a / 2)]
        d.mocap_pos[self.head_mocap] = head
        d.mocap_quat[self.head_mocap] = look_at_quat_mj(np.asarray(head, float), np.asarray(target, float))
        g = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "rig_pole_geom")
        h = float(head[2] - pole_foot[2])
        self.model.geom_size[g, 1] = 0.5 * h
        self.model.geom_pos[g, 2] = 0.5 * h
        mujoco.mj_forward(self.model, d)

    def pose_from_mocap(self) -> RigPose:
        p = self.data.mocap_pos[self.pole_mocap]
        return RigPose(float(np.degrees(np.arctan2(p[1], p[0])) % 360.0), float(np.hypot(p[0], p[1])),
                       self.pose.height, self.pose.aim_height)

    # ------------------------------------------------------------------ computation
    def capture(self) -> dict:
        return self.cam.capture(segmentation=False)

    def perceive(self, frame, cancel=None):
        per = Perceiver(self.cfg)
        per.add_view(frame, cancel)
        res = per.result(cancel)
        return per, res

    def compute(self, frame: dict, pose: RigPose, cancel=None, method: str | None = None) -> RigResult:
        """Perception + shoulder/window/T2 decision on one captured frame. Raises Cancelled if cancel() becomes true."""
        from pinesim.decision.window import decide
        from pinesim.vision.stargaze_cam import obstacle_points
        t0 = time.time()
        slow = float(self.cfg.get("rig", {}).get("debug_slow_s", 0.0))
        per, res = self.perceive(frame, cancel)
        if slow > 0:                                    # optional artificial work (to exercise cancellation)
            t_end = time.time() + slow
            while time.time() < t_end:
                _check(cancel)
                time.sleep(0.01)
        out = RigResult(pose=pose, ok=False, camera_pos=frame["t"].copy(), leaf_points=res.leaf_points, frame=frame)
        if not res.fruits:
            out.reason = "no fruit visible"
            out.compute_s = time.time() - t0
            return out
        fruit = max(res.fruits, key=lambda f: f.n_points)
        out.fruit, out.fruit_points = fruit, fruit.points
        pts, rad = obstacle_points(fruit, res.leaf_points, self.cfg)
        out.obstacle_points = pts
        _check(cancel)
        cfg = self.cfg
        if method is not None:
            cfg = dict(cfg)
            cfg["stargaze"] = dict(cfg["stargaze"], decision=method)
        dec = decide(fruit, per.views, pts, rad, cfg, policy=self.policy, rng=np.random.default_rng(0))
        _check(cancel)
        out.decision, out.view = dec, per.views[dec.view_index] if dec.view_index >= 0 else None
        if not dec.ok:
            out.reason = dec.reason
        else:
            best = dec.candidates[0]
            out.ok, out.t1, out.t2, out.clearance = True, best.t1, best.t2, best.clear_line
        out.diagnostics = self.diagnostics(frame, fruit, out)
        out.compute_s = time.time() - t0
        return out

    def diagnostics(self, frame: dict, fruit, out: RigResult) -> dict:
        from pinesim.decision.window import true_scores
        from pinesim.vision.stargaze_cam import segment_clearance_to
        gt = self.plant
        gt_c = gt.fruit_center
        dg = dict(fruit_center_error=float(np.linalg.norm(fruit.center - gt_c)),
                  fruit_center_error_xy=float(np.linalg.norm(fruit.center[:2] - gt_c[:2])),
                  fruit_center_error_z=float(fruit.center[2] - gt_c[2]),
                  radius_error=float(fruit.radius - gt.fruit_r), fruit_points=int(fruit.n_points),
                  arc_deg=float(fruit.arc_deg), tilt_true_deg=float(np.degrees(gt.tilt)), tilt_est_deg=fruit.tilt_deg,
                  axis_error_deg=float(np.degrees(np.arccos(np.clip(fruit.axis @ gt.axis, -1, 1)))))
        u, v, _ = self.cam.project_world(gt_c[None, :], frame["R_true"], frame["t_true"])
        ue, ve, _ = self.cam.project_world(fruit.center[None, :], frame["R_true"], frame["t_true"])
        dg["reprojection_px"] = float(np.hypot(u[0] - ue[0], v[0] - ve[0]))
        P = fruit.points
        h = (P - gt_c[None, :]) @ gt.axis
        hh = np.clip(h, -gt.fruit_half, gt.fruit_half)
        surf = np.linalg.norm(P - gt_c[None, :] - hh[:, None] * gt.axis[None, :], axis=1) - gt.fruit_r
        dg["fruit_surface_mean_abs_mm"] = float(1000 * np.mean(np.abs(surf)))
        dec = out.decision
        if dec is not None and dec.shoulder:
            dg["shoulder_pixels"] = int(dec.shoulder.get("n_shoulder", 0))
            dg["visible_fruit_pixels"] = int(dec.shoulder.get("n_visible", 0))
            dg["shoulder_ratio"] = float(dec.shoulder.get("ratio", 0.0))
        if out.ok:
            ts = true_scores(dec.candidates, gt.leaf_pts, gt.leaf_rad, self.cfg)
            dg["n_candidates"] = len(dec.candidates)
            dg["true_clearance_chosen"] = float(segment_clearance_to(gt.leaf_pts, gt.leaf_rad, out.t1, out.t2))
            dg["true_score_chosen"] = float(ts[0])
            dg["true_score_best"] = float(ts.max())
            dg["line_clearance_perceived"] = float(out.clearance)
            dg["t2_on_shoulder_h"] = float((out.t2 - gt_c) @ gt.axis)
            dg["decision"] = dec.method
        return dg


class RigController:
    """Settle-timer state machine with a cancellable worker thread (front-end independent).

    states: MOVING -> SETTLING -> COMPUTING -> DONE (or FAILED); any pose change -> MOVING (+ cancel)."""

    def __init__(self, rig: StargazeRig, settle_s: float = 1.5):
        self.rig = rig
        self.settle_s = float(settle_s)
        self.state = "SETTLING"
        self._last_pose = RigPose(**vars(rig.pose))
        self._t_still = time.time()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._job_result: RigResult | None = None
        self._job_error: str | None = None
        self.result: RigResult | None = None
        self.n_started = self.n_cancelled = self.n_done = 0
        self.log: list = []

    def _event(self, msg: str):
        self.log.append((time.time(), msg))

    def cancel_running(self):
        if self._thread is not None and self._thread.is_alive():
            self._cancel.set()
            self._thread.join(timeout=2.0)
            self.n_cancelled += 1
            self._event("computation cancelled (camera moved)")
        self._thread = None

    def tick(self, pose: RigPose, now: float | None = None) -> str:
        now = time.time() if now is None else now
        if not pose.close_to(self._last_pose):
            if self.state == "COMPUTING":
                self.cancel_running()
            self.rig.set_pose(pose)
            self._last_pose = RigPose(**vars(self.rig.pose))
            self._t_still = now
            self.result = None
            self.state = "MOVING"
            return self.state
        if self.state == "MOVING":
            self.state = "SETTLING"
        if self.state == "SETTLING" and now - self._t_still >= self.settle_s:
            frame = self.rig.capture()               # rendering stays in the caller's (GL) thread
            self._cancel.clear()
            self._job_result, self._job_error = None, None
            p = RigPose(**vars(self.rig.pose))

            def work():
                try:
                    self._job_result = self.rig.compute(frame, p, cancel=self._cancel.is_set)
                except Cancelled:
                    self._job_error = "cancelled"
                except Exception as e:                # pragma: no cover - surfaced in the UI
                    self._job_error = f"error: {e!r}"

            self._thread = threading.Thread(target=work, daemon=True)
            self._thread.start()
            self.n_started += 1
            self.state = "COMPUTING"
            self._event("camera still -> computing shoulder / window / T2")
        if self.state == "COMPUTING" and self._thread is not None and not self._thread.is_alive():
            self._thread = None
            if self._job_result is not None:
                self.result = self._job_result
                self.n_done += 1
                self.state = "DONE" if self.result.ok else "FAILED"
                self._event(f"result: {'ok' if self.result.ok else self.result.reason}")
            else:
                self.state = "FAILED"
                self._event(self._job_error or "failed")
        return self.state
