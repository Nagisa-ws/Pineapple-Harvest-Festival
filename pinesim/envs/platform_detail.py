"""
pinesim/envs/platform_detail.py
-------------------------------
Missions 1-2 REALISTIC evaluation (mode platform_detail): the same robot, column, basket, observation and action as
the light training scene, but the plants are the user's mesh pineapples - per-leaf articulated slots (5 kinds,
pinesim/lod.DetailField), tilted 0-15 deg, fruit centre 0.60-0.70 m, ground mesh - and the robot / gripper meshes.
Never used for training.
"""
from __future__ import annotations

import numpy as np

from pinesim.envs.harvest_env import HarvestEnv


class PlatformDetailEnv(HarvestEnv):
    def __init__(self, mission: str = "mission1", **kw):
        ov = dict(kw.pop("overrides", None) or {})
        ov.setdefault("task", {})["mode"] = "platform_detail"
        kw["overrides"] = ov
        kw.setdefault("training", False)
        super().__init__(mission, **kw)
        self.sim.field.configure(self.sim.meta, self.cfg)
        self.resample_on_plan_fail = False

    def _place_plants(self):
        """Mission 1: one plant; Mission 2: plus a neighbour in a random direction (different kinds). Basket 180 deg."""
        sim, t, rc = self.sim, self.task, self.cfg["realistic"]
        pl = t["placement"]
        base = sim.base_pos
        az = self.rng.uniform(*np.radians(pl.get("azimuth_deg_range", [0, 360])))
        R = self.rng.uniform(*pl["radius_range"])
        grounds = [np.array([base[0] + R * np.cos(az), base[1] + R * np.sin(az), 0.0])]
        if int(t["n_plants"]) >= 2:
            for _ in range(80):
                d = self.rng.uniform(*pl["neighbour_distance_range"])
                ang = self.rng.uniform(0, 2 * np.pi)
                g2 = grounds[0] + np.array([d * np.cos(ang), d * np.sin(ang), 0.0])
                r2 = np.hypot(g2[0] - base[0], g2[1] - base[1])
                if pl["radius_range_neighbour"][0] <= r2 <= pl["radius_range_neighbour"][1]:
                    grounds.append(g2)
                    break
        az_mean = np.arctan2(np.mean([g[1] - base[1] for g in grounds]), np.mean([g[0] - base[0] for g in grounds]))
        b = self.cfg["basket"]
        ab = az_mean + np.pi
        sim.m.body_pos[sim.basket_body] = [base[0] + b["radius_platform"] * np.cos(ab), base[1] + b["radius_platform"] * np.sin(ab), 0.0]
        fld = sim.field
        for i in range(fld.n):
            fld.hide(i, sim.d)
        kinds = self.rng.permutation(len(rc["kinds"]))[:len(grounds)]
        t0, t1 = np.radians(self.cfg["plant"].get("tilt_deg_range", [0, 15]))
        self.plants = []
        for g, k in zip(grounds, kinds):
            slot = next(i for i in fld.slots_of_kind(int(k)) if fld.slots[i]["layout"] is None)
            p = dict(index=len(self.plants), x=float(g[0]), y=float(g[1]), kind=int(k),
                     yaw=float(self.rng.uniform(0, 2 * np.pi)), fruit_yaw=float(self.rng.uniform(0, 2 * np.pi)),
                     tilt=float(self.rng.uniform(t0, t1)), tilt_az=float(self.rng.uniform(0, 2 * np.pi)),
                     break_force=float(self.rng.uniform(*rc["stem_break_force_range"])),
                     break_torque=float(self.rng.uniform(*rc["stem_break_torque_range"])))
            fld.assign(slot, p, sim.d)
            self.plants.append(dict(p, slot=slot))
        self.basket_azimuth = ab

    def _tick(self, label: str):
        super()._tick(label)
        from pinesim.vendor.robot_geometry import world_segments
        a, b = world_segments(self.arm.caps, self.sim.d)
        self.sim.field.gate_leaves(self.sim.d, a, b, self.arm.caps.radii)

    def _get_info(self) -> dict:
        info = super()._get_info()
        info["plants"] = [dict(kind=p["kind"], tilt_deg=round(float(np.degrees(p["tilt"])), 1)) for p in getattr(self, "plants", [])]
        return info
