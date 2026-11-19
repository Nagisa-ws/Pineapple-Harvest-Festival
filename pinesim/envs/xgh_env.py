"""
pinesim/envs/xgh_env.py
-----------------------
Mission 3 (version 5 = old Missions 3 + 4 merged): end-to-end harvest of a zig-zag double row from the ATEX XGH505H
crawler with the UR10e + U-jaw gripper and two crates on the bed.

  XghHarvestEnv   training / quick evaluation on the LIGHT scene (procedural plants, short row segment)
  XghEvalEnv      realistic evaluation: 15 m field of the user's mesh pineapples (2 x 30 plants, spacing 0.50 m,
                  zig-zag 0.25 m, tilted 0-15 deg), per-leaf articulated plants; only THREE plants are interactive at
                  a time (target + next before the first fruit is placed, then target + previous + next; pinesim/lod.py),
                  crate capacity with operator unloads. Same robot, gripper, vehicle collision, observation and action
                  as training.

Both crates are used: every fruit goes to the crate with fewer fruits (the arm alternates left / right).
"""
from __future__ import annotations

import mujoco
import numpy as np

from pinesim.envs.vehicle_env import VehicleHarvestEnv
from pinesim.vision.perception import merge_fruits


class XghHarvestEnv(VehicleHarvestEnv):
    MISSION = 3

    def __init__(self, mission: str = "mission3", **kw):
        super().__init__(mission, **kw)

    # ------------------------------------------------------------------ crates
    def _scene_reset(self):
        for b in self.sim.baskets:
            b["count"] = 0
            b["drops"] = 0
        self.sim.active_basket = 0
        self.fruit_memory = []
        super()._scene_reset()

    def _self_filter(self):
        """Camera points on the robot's own vehicle (green bed = leaf colour) are dropped: the robot knows its body."""
        x = self.cfg["xgh"]
        m = float(self.cfg["perception"].get("self_filter_margin", 0.03))
        sim = self.sim

        def on_vehicle(P):
            pv, yaw = sim.vehicle_pose()
            c, s_ = np.cos(yaw), np.sin(yaw)
            rel = P - pv[None, :]
            lx = c * rel[:, 0] + s_ * rel[:, 1]
            ly = -s_ * rel[:, 0] + c * rel[:, 1]
            return ((lx > x["bed_x"][0] - m) & (lx < x["bed_x"][1] + m) & (np.abs(ly) < x["bed_half_y"] + m) &
                    (rel[:, 2] > x["bed_bottom"] - 0.30) & (rel[:, 2] < x["deck_top"] + 0.35))
        return on_vehicle

    def _known(self, f) -> bool:
        """Harvested / skipped already? 16 cm: in the dense mesh canopy a partly hidden fruit is estimated up to
        ~10 cm off between surveys; the nearest other fruit is 0.39 m away (zig-zag 0.25 / 0.30 m)."""
        for p in self.harvested_xy + self.skipped_xy:
            if np.linalg.norm(f.center[:2] - p) < 0.16:
                return True
        return False

    # ------------------------------------------------------------------ recovery after skipping a fruit
    def _allow_back_out(self):
        """Skipping a fruit inside the canopy: the leaf bend that is already there is allowed while backing out and
        returning to the default posture (a little further bending must not end the whole row)."""
        if self.failure and self.failure not in self.hard_failures:
            self._bend_allow = max(self._bend_allow, getattr(self, "_root_bend", 0.0) +
                                   float(self.task.get("recovery_bend_margin", 0.5)))

    def _soft_after(self, r: float) -> float:
        self._allow_back_out()
        return super()._soft_after(r)

    def _soft_recover(self, fruit) -> bool:
        self._allow_back_out()
        return super()._soft_recover(fruit)

    # ------------------------------------------------------------------ fruit memory
    def _scan(self) -> bool:
        """Survey + memory of fruits seen in earlier surveys / close views (world frame, the vehicle odometry is
        known). After the vehicle drove a fruit close to the base it lies below the survey camera's field of view."""
        super()._scan()
        if self.done:
            return False
        self._remember(list(self.fruits_scan))
        return len(self.fruits_scan) > 0

    def _remember(self, fruits: list):
        """Merge new estimates into the memory (new estimates win over old ones of the same fruit)."""
        base = self.sim.base_pos
        # every fruit that is still on its plant stays in the memory as an OBSTACLE - also skipped ones and those the
        # vehicle has driven past (they were dropped before v6: the arm then swung through a skipped fruit);
        # which fruit is harvested next is decided by _ahead()
        old = [f for f in getattr(self, "fruit_memory", []) if self._fruit_present(f)
               and f.center[0] - base[0] > -1.5
               and all(np.linalg.norm(f.center[:2] - g.center[:2]) > 0.12 for g in fruits)]
        for f in old:
            f._done = False
        self.fruit_memory = merge_fruits(list(fruits) + old)
        self.fruits_scan = list(self.fruit_memory)

    def _on_look(self, coarse, fruit, seen: list):
        """Close view: the refined target estimate replaces the remembered one; other fruits seen are remembered."""
        keep = [f for f in self.fruit_memory if f is not coarse
                and np.linalg.norm(f.center[:2] - coarse.center[:2]) > 0.08]
        self.fruit_memory = merge_fruits([fruit] + list(seen) + [f for f in keep if all(
            np.linalg.norm(f.center[:2] - g.center[:2]) > 0.12 for g in [fruit] + list(seen))])
        for f in self.fruit_memory:
            f._done = f is not fruit and getattr(f, "_done", True)
        fruit._done = True
        self.fruits_scan = [fruit] + [f for f in self.fruit_memory if f is not fruit]

    def _forget(self, fruit):
        self.fruit_memory = [f for f in self.fruit_memory if f is not fruit
                             and np.linalg.norm(f.center[:2] - fruit.center[:2]) > 0.08]
        self.fruits_scan = list(self.fruit_memory)

    def _drop_index(self) -> int:
        return int(self.sim.baskets[self.sim.active_basket].get("drops", 0))

    def _count_drop(self):
        b = self.sim.baskets[self.sim.active_basket]
        b["drops"] = int(b.get("drops", 0)) + 1

    def _choose_basket(self):
        counts = [b["count"] for b in self.sim.baskets]
        self.sim.active_basket = int(np.argmin(counts))

    def _prepare_target(self) -> bool:
        self._choose_basket()
        return super()._prepare_target()

    def _release_and_verify(self) -> bool:
        ok = super()._release_and_verify()
        if ok:
            k = self.sim.basket_of(self.target_slot)
            if k is not None:
                self.sim.baskets[k]["count"] += 1
                self._event(f"fruit in crate {self.sim.baskets[k]['prefix']} ({self.sim.baskets[k]['count']})")
        return ok

    def _get_info(self) -> dict:
        info = super()._get_info()
        info["crate_counts"] = [int(b["count"]) for b in self.sim.baskets]
        return info


class XghEvalEnv(XghHarvestEnv):
    """Realistic 15 m evaluation with the three-plant level of detail (never used for training)."""

    def __init__(self, mission: str = "mission3", **kw):
        ov = dict(kw.pop("overrides", None) or {})
        ov.setdefault("task", {})["mode"] = "xgh_detail"
        sec = (ov.get("eval_field") or {}).get("section")
        if sec:                                         # vehicle starts 0.83 m (base) before the section's first plant
            from pinesim.config import load_config
            from pinesim.model.builder_xgh import field_layout
            cfg0 = load_config(mission, ov)
            first = field_layout(cfg0)[int(sec[0])]["x"]
            bx = cfg0["xgh"]["base_pos_in_vehicle"][0]
            ov.setdefault("xgh", {})["start_x"] = float(first - 0.83 - bx)
            ov.setdefault("field", {})["vehicle_start_x"] = float(first - 0.83 - bx)
        kw["overrides"] = ov
        kw.setdefault("training", False)
        from pinesim.lod import FieldLOD
        super().__init__(mission, **kw)
        self.sim.field.configure(self.sim.meta, self.cfg)
        self.lod = FieldLOD(self.sim, self.cfg)
        self.resample_on_plan_fail = False
        self.max_steps = 10_000_000                    # the whole field (section) is one episode

    # ------------------------------------------------------------------ field / section
    def section_plants(self):
        sec = self.cfg["eval_field"].get("section")
        L = self.sim.meta["layout"]
        if not sec:
            return L
        k0, n = int(sec[0]), int(sec[1])
        return L[k0:k0 + n]

    def _n_targets(self) -> int:
        return len(self.section_plants())

    def _tick_gate(self):
        from pinesim.vendor.robot_geometry import world_segments
        mujoco.mj_kinematics(self.sim.m, self.sim.d)
        a, b = world_segments(self.arm.caps, self.sim.d)
        self.sim.field.gate_leaves(self.sim.d, a, b, self.arm.caps.radii)

    def _tick(self, label: str):
        super()._tick(label)
        from pinesim.vendor.robot_geometry import world_segments
        a, b = world_segments(self.arm.caps, self.sim.d)
        self.n_gated = self.sim.field.gate_leaves(self.sim.d, a, b, self.arm.caps.radii)

    def _field_end(self) -> float:
        """x up to which fruits are harvested: half a plant spacing (zig-zag pitch 0.25 m) beyond the last plant of
        a section - the next section's first plant must not be picked by this run - or 0.35 m beyond the last plant
        of the whole field (plants lean up to 15 deg)."""
        sec = self.cfg["eval_field"].get("section")
        last = max(p["x"] for p in self.section_plants())
        whole = (not sec) or int(sec[0]) + int(sec[1]) >= len(self.sim.meta["layout"])
        return last + (0.35 if whole else 0.12)

    def _scan(self) -> bool:
        ok = super()._scan()
        if self.done:
            return False
        return len(self.fruits_scan) > 0     # (fruits beyond the section end stay in the memory as obstacles only)

    def _place_plants(self):
        self.lod.reset()
        self.basket_azimuth = np.pi

    def _scene_reset(self):
        super()._scene_reset()
        self.lod.set_render_window(float(self.sim.base_pos[0]))
        self._target_plant = None
        self.unloads = 0

    def _drive(self, x_target: float) -> bool:
        ok = super()._drive(x_target)
        if ok:
            self.lod.set_render_window(float(self.sim.base_pos[0]))
        return ok

    def _prepare_target(self) -> bool:
        """Before the close views of a new target: three-plant rule (the arm is in a survey pose above the canopy)."""
        if self.lod.crates_full():
            self.lod.unload()
            self._event(f"crates full: operator unload #{self.lod.unloads}")
        remaining = [f for f in self.fruits_scan if not getattr(f, "_done", False)]
        if remaining:
            coarse = self._priority(remaining)[0]
            n = self.lod.plant_near(coarse.center)
            self._target_plant = n
            want = self.lod.wanted(n, float(self.sim.base_pos[0]))
            ev = self.lod.set_interactive(want, "target")
            self._tick_gate()
            self._event(f"LOD: interactive {ev['interactive']} (+{len(ev['assigned'])} / -{len(ev['released'])})")
        return super()._prepare_target()

    def _release_and_verify(self) -> bool:
        ok = super()._release_and_verify()
        if ok:
            n = self.lod.mark(self.target_slot, "picked")
            if n is not None:
                self.lod.prev_plant = n
        return ok

    def _get_info(self) -> dict:
        info = super()._get_info()
        info["lod_swaps"] = int(self.lod.swaps)
        info["lod_deferred"] = int(self.lod.deferred)
        info["lod_max_interactive"] = int(self.lod.max_interactive)
        info["unloads"] = int(self.lod.unloads)
        info["interactive_plants"] = len(self.lod.interactive())
        info["n_harvested_total"] = int(self.n_placed)
        return info
