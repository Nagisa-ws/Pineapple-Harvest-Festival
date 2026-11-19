"""
pinesim/vehicle.py
------------------
Vehicle controller, separate from the manipulator policy [PREV]. Differential-drive semantics: left/right belt speed
commands (m/s) -> hidden drive-wheel velocity actuators (Level A drive model).

  DRIVE : proportional speed toward a target x along the row; heading/lateral correction through the left/right
          speed difference; ARRIVED when inside the stop tolerance and nearly at rest
  HOLD  : wheel angles locked by the brake actuators while the arm works
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VehicleConfig:
    kp_x: float = 2.5                 # [v6] was 1.2 (a 0.25 m move between two fruits took ~5 s)
    v_min: float = 0.05
    stop_tolerance: float = 0.015     # the fruit only has to be inside the arm's work area
    rest_speed: float = 0.015
    arrive_steps: int = 5             # control steps at rest inside the tolerance before the brakes hold
    k_yaw: float = 0.4
    k_y: float = 0.5
    max_heading_cmd: float = 0.15


class VehicleController:
    def __init__(self, sim, cfg: VehicleConfig | None = None):
        self.sim = sim
        self.cfg = cfg or VehicleConfig()
        self.target_x = None
        self.state = "HOLD"
        self._arrived = 0
        self.vmax = float(sim.cfg["vehicle"]["max_track_speed"])

    def drive_to(self, x: float):
        self.target_x = float(x)
        self.state = "DRIVE"
        self._arrived = 0

    def hold(self):
        self.state = "HOLD"
        self.sim.brake()

    def update(self) -> bool:
        """Apply one command; True once arrived and holding."""
        sim, c = self.sim, self.cfg
        if self.state != "DRIVE":
            return True
        p, yaw = sim.vehicle_pose()
        err = self.target_x - p[0]
        v = float(np.clip(c.kp_x * err, -self.vmax, self.vmax))
        v = float(np.sign(err) * max(abs(v), c.v_min)) if abs(err) > c.stop_tolerance else 0.0
        heading_des = float(np.clip(-c.k_y * p[1], -c.max_heading_cmd, c.max_heading_cmd))
        dv = c.k_yaw * (heading_des - yaw)
        sim.drive(v - dv, v + dv)
        vx = float(sim.d.qvel[sim.vehicle_dadr])
        if abs(err) < c.stop_tolerance and abs(vx) < c.rest_speed:
            self._arrived += 1
            if self._arrived >= self.cfg.arrive_steps:
                self.hold()
                return True
        else:
            self._arrived = 0
        return False
