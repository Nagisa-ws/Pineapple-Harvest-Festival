"""
pinesim/control/reference.py
----------------------------
Reference (route-following) controller of the residual structure, generalised from the reference repository's
shared/harvest/reference.py (Mission 4-5) to every arm phase of the harvest:

  position : follow the active route (list of waypoints) with a look-ahead; the waypoint index only moves forward
  rotation : rotate the target orientation toward the route orientation at the look-ahead waypoint
  slowdown : translation is slowed while the actual orientation lags the requested one (wrist slower than translation)
  action   = clip(gain * error / max_delta), position and rotation norms each <= 1

The controller knows nothing about leaves: steering around leaves that are still in the way is the job of the
residual SAC correction (as in the reference project).
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from pinesim.vendor.ik_solver import _quat_to_axis_angle_error
from pinesim.vendor.pose_utils import quat_angle


@dataclass
class ReferenceConfig:
    gain: float = 0.8
    lookahead: int = 2
    rot_gain: float = 0.8
    lag_free_deg: float = 8.0
    lag_span_deg: float = 25.0
    lag_min_scale: float = 0.15
    speed_scale: float = 1.0

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class Route:
    def __init__(self, wps: np.ndarray, quats: np.ndarray, name: str = ""):
        self.wps = np.asarray(wps, float).reshape(-1, 3)
        self.quats = np.asarray(quats, float).reshape(-1, 4)
        self.name = name
        self.idx = 0

    @property
    def start(self):
        return self.wps[0]

    @property
    def end(self):
        return self.wps[-1]

    def advance(self, tcp: np.ndarray, window: int = 6):
        hi = min(len(self.wps), self.idx + window)
        d = np.linalg.norm(self.wps[self.idx:hi] - tcp[None, :], axis=1)
        self.idx = self.idx + int(np.argmin(d))
        return self.idx


class ReferenceController:
    def __init__(self, cfg: ReferenceConfig | None = None):
        self.cfg = cfg or ReferenceConfig()

    def act(self, tcp: np.ndarray, actual_quat: np.ndarray, target_quat: np.ndarray, route: Route,
            max_pos_delta: float, max_rot_delta: float, speed_scale: float = 1.0) -> np.ndarray:
        c = self.cfg
        a = np.zeros(6)
        i = route.advance(tcp)
        j = min(i + c.lookahead, len(route.wps) - 1)
        tgt = route.wps[j]
        v = (tgt - tcp) / max_pos_delta * c.gain
        n = float(np.linalg.norm(v))
        a[:3] = v if n <= 1.0 else v / n
        a[:3] *= speed_scale * c.speed_scale
        q_des = route.quats[j]
        lag = np.degrees(quat_angle(actual_quat, q_des))
        a[:3] *= float(np.clip(1.0 - (lag - c.lag_free_deg) / c.lag_span_deg, c.lag_min_scale, 1.0))
        err = _quat_to_axis_angle_error(target_quat, q_des)
        w = err / max_rot_delta * c.rot_gain
        n = float(np.linalg.norm(w))
        a[3:] = (w if n <= 1.0 else w / n) * min(1.0, speed_scale * c.speed_scale)
        return a
