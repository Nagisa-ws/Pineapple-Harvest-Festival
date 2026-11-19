"""
pinesim/vision/camera.py
------------------------
RGB-D camera on any named MuJoCo camera (wrist camera of the robot, or the pole camera of the pretraining rig).

Conventions [PREV]
  * Pixel (u, v) = (column, row), row 0 at the top; pinhole intrinsics fy = (H/2)/tan(fovy/2), fx = fy, c = (W/2, H/2).
  * Camera frame = OpenCV (x right, y down, z forward). MuJoCo cameras look along -z with +y up, so
    R_world_cv = R_world_mj @ diag(1, -1, -1).
  * Depth = metres along the optical axis (MuJoCo's linear depth), invalid = 0.
  * Extrinsics given to perception = true pose (+ a fixed calibration error per camera instance, if enabled), so
    calibration error propagates into every 3D estimate exactly as on a real robot.
"""
from __future__ import annotations

import atexit
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])


class RGBDCamera:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, camera_name: str, cfg: dict, seed: int = 0,
                 noise: bool | None = None):
        self.m, self.d = model, data
        c = cfg["camera"]
        self.width, self.height = int(c["image_width"]), int(c["image_height"])
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.cam_id < 0:
            raise ValueError(f"camera '{camera_name}' not in model")
        self.fovy = float(model.cam_fovy[self.cam_id])
        fy = (self.height / 2.0) / np.tan(np.radians(self.fovy) / 2.0)
        self.intr = Intrinsics(fy, fy, self.width / 2.0, self.height / 2.0, self.width, self.height)
        self.far = float(c["far_clip"])
        self.ncfg = dict(cfg.get("noise", {}))
        if noise is not None:
            self.ncfg["enabled"] = bool(noise)
        self.rng = np.random.default_rng(seed)
        self.renderer = mujoco.Renderer(model, self.height, self.width)
        atexit.register(self.close)
        try:                     # warm-up: the very first RGB render of a GL context differs from later ones
            self.render_raw(segmentation=True)
        except Exception:
            pass
        self._calib_R, self._calib_t = self._sample_calibration()

    def _sample_calibration(self):
        n = self.ncfg
        if not n.get("enabled", False) or (n.get("calib_rot_deg", 0) == 0 and n.get("calib_trans_m", 0) == 0):
            return np.eye(3), np.zeros(3)
        axis = self.rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        q = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q, axis, np.radians(self.rng.normal(0, n["calib_rot_deg"])))
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        return R.reshape(3, 3), self.rng.normal(0, n["calib_trans_m"], size=3)

    # ------------------------------------------------------------ poses
    def true_pose_cv(self):
        R_mj = self.d.cam_xmat[self.cam_id].reshape(3, 3)
        return R_mj @ np.diag([1.0, -1.0, -1.0]), self.d.cam_xpos[self.cam_id].copy()

    def calibrated_pose_cv(self):
        R, t = self.true_pose_cv()
        return R @ self._calib_R, t + self._calib_t

    # ------------------------------------------------------------ rendering
    def render_raw(self, segmentation: bool = True):
        r = self.renderer
        r.disable_depth_rendering()
        r.disable_segmentation_rendering()
        r.update_scene(self.d, camera=self.cam_id)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(self.d, camera=self.cam_id)
        depth = r.render().copy().astype(np.float32)
        r.disable_depth_rendering()
        seg = None
        if segmentation:
            r.enable_segmentation_rendering()
            r.update_scene(self.d, camera=self.cam_id)
            s = r.render().copy()
            r.disable_segmentation_rendering()
            seg = np.where(s[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM), s[:, :, 0], -1).astype(np.int32)
        return rgb, depth, seg

    def apply_noise(self, rgb, depth):
        n = self.ncfg
        if not n.get("enabled", False):
            return rgb.copy(), depth.copy()
        rng = self.rng
        img = rgb.astype(np.float32) * rng.uniform(*n.get("brightness_range", [1, 1]))
        img += rng.normal(0, n.get("rgb_noise_std", 0.0), size=img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)
        dep = depth.copy()
        valid = (dep > 0) & (dep < self.far)
        dep = dep + rng.normal(0, 1, size=dep.shape) * (n.get("depth_noise_rel", 0) * dep + n.get("depth_noise_abs", 0))
        drop = rng.random(dep.shape) < n.get("depth_invalid_frac", 0.0)
        dep[~valid | drop] = 0.0
        return img, dep.astype(np.float32)

    def capture(self, segmentation: bool = True) -> dict:
        """Noisy RGB-D frame + the (calibrated) extrinsics used by perception + ground truth for metrics."""
        rgb, depth, seg = self.render_raw(segmentation)
        rgb_n, depth_n = self.apply_noise(rgb, depth)
        R, t = self.calibrated_pose_cv()
        Rt, tt = self.true_pose_cv()
        return dict(rgb=rgb_n, depth=depth_n, rgb_clean=rgb, depth_clean=depth, seg=seg, R=R, t=t, R_true=Rt, t_true=tt,
                    intr=self.intr)

    # ------------------------------------------------------------ geometry
    def backproject(self, u, v, z):
        x = (u - self.intr.cx) / self.intr.fx * z
        y = (v - self.intr.cy) / self.intr.fy * z
        return np.stack([x, y, z], axis=-1)

    def project_world(self, X_world, R=None, t=None):
        """World points -> (u, v, z_cam) with the true pose (or the given one)."""
        if R is None:
            R, t = self.true_pose_cv()
        Xc = (np.atleast_2d(X_world) - t) @ R
        z = Xc[:, 2]
        return self.intr.fx * Xc[:, 0] / z + self.intr.cx, self.intr.fy * Xc[:, 1] / z + self.intr.cy, z

    def close(self):
        try:
            if self.renderer is not None:
                self.renderer.close()
                self.renderer = None
        except Exception:
            pass


def look_at_quat_mj(cam_pos: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Quaternion (wxyz) of a MuJoCo camera frame at cam_pos looking at target (camera -z = viewing direction)."""
    f = np.asarray(target, float) - np.asarray(cam_pos, float)
    f /= np.linalg.norm(f) + 1e-12
    z = -f
    x = np.cross(np.asarray(up, float), z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([x, y, z]).ravel())
    return q
