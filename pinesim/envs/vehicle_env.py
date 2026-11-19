"""
pinesim/envs/vehicle_env.py
---------------------------
Mission 3 base: the arm on a tracked vehicle, harvesting a whole row segment end to end (used by the ATEX XGH505H
environments of pinesim/envs/xgh_env.py).

Loop (one episode = the whole row):
    SURVEY (forward wrist-camera scan from the stopped vehicle)
 -> choose the next fruit (nearest ahead; already harvested positions are remembered)
 -> fruit beyond the arm's work area?  -> DRIVE the vehicle forward until it is inside, HOLD, survey again
 -> inside the work area                -> Mission 1 harvest (look, shoulder / window / T2 decision, approach,
                                           line, open + advance, grip, twist, pull, lift, transport to a crate)
 -> back to the DEFAULT posture (arm above the canopy) -> survey again ... until no fruit is detected ahead and
    the vehicle reached the row end. The vehicle only drives with the arm in the default posture.

One fruit = one session. Hard failures (collision, floor contact, timeout, numerical blow-up) end the episode; any
other failure of one fruit (leaf uprooted, no route, grasp miss, slip, drop, basket miss, perception lost) ends that
fruit's session: it is recorded, the fruit is skipped and the robot continues with the next one, as an operator would
expect in the field. Success = every fruit of the row placed without plant damage.
"""
from __future__ import annotations

import numpy as np
import mujoco

from pinesim.envs.harvest_env import HarvestEnv
from pinesim.model.builder import field_positions
from pinesim.vehicle import VehicleController


DAMAGE = {"leaf_uprooted", "plant_hit", "fruit_hit", "fruit_knocked_off"}   # plant damage (reported per row)


class VehicleHarvestEnv(HarvestEnv):
    MISSION = 3

    def __init__(self, mission: str = "mission3", **kw):
        super().__init__(mission, **kw)
        self.vc = VehicleController(self.sim)
        self.hard_failures = set(self.task.get("hard_failures", ["collision", "floor_contact", "timeout", "sim_unstable"]))
        self.reach = dict(self.task["reach_window"])

    # ------------------------------------------------------------------ scene
    def _place_plants(self):
        sim = self.sim
        self.sampler.set_level(self.level)
        pos = field_positions(self.cfg)
        for i, (x, y) in enumerate(pos):
            g = np.array([x, y, 0.0])
            front = np.array([-1.0, -np.sign(y) * 0.5, 0.0])    # facing the oncoming vehicle / the arm
            lay = self.sampler.sample(self.rng, g, front)
            sim.field.apply(i, lay, sim.d)
        self.basket_azimuth = np.pi

    def _scene_reset(self):
        sim = self.sim
        mujoco.mj_resetData(sim.m, sim.d)
        q = sim.vehicle_qadr
        sim.d.qpos[q:q + 7] = [float(self.cfg["field"]["vehicle_start_x"]), 0.0, 0.0, 1, 0, 0, 0]
        mujoco.mj_forward(sim.m, sim.d)
        self._place_plants()
        self._go_home_instant()
        mujoco.mj_forward(sim.m, sim.d)
        for i in sim.field.active_slots():
            sim.field.arm_stem(i, sim.d)
        sim.brake()
        sim.step(40)                      # settle the vehicle on its wheels
        for i in sim.field.active_slots():
            sim.field.record_rest_load(i, sim.d)
        self.vc.hold()
        self.harvested_xy = []
        self.skipped_xy = []
        self._retried_xy = []
        self._lost_xy = []
        self._recovering = False
        self.skip_log = []                # (reason) of every skipped fruit; damage reasons are reported separately
        self.stops = 0
        self.travel = 0.0
        self._x0 = float(sim.vehicle_pose()[0][0])
        self.drift_max = 0.0

    def _home_azimuth(self):
        return self.sim.vehicle_pose()[1] if self.sim.vehicle_body >= 0 else 0.0

    def _scan_azimuths(self):
        yaw = self._home_azimuth()
        return [yaw + np.radians(a) for a in self.cfg["planning"]["scan"]["azimuths_deg"]]

    def _n_targets(self) -> int:
        return self.sim.field.n

    # ------------------------------------------------------------------ vehicle
    def _drive(self, x_target: float) -> bool:
        """Drive to x_target. Leaves brushed by the tracks/frames while driving (rows pass under the vehicle) are
        recorded as a separate damage event (`vehicle_leaf_overbend`) instead of aborting the drive; collisions of
        the robot still end the episode."""
        sim = self.sim
        x_start = float(sim.vehicle_pose()[0][0])
        self.vc.drive_to(x_target)
        sim.hold_arm()
        allow0, self._bend_allow, bend_max, ok = self._bend_allow, np.inf, 0.0, True
        try:
            for _ in range(int(self.task.get("max_drive_steps", 1500))):
                arrived = self.vc.update()
                if not self._sim(1, "drive"):
                    ok = False
                    break
                bend_max = max(bend_max, self._leaf_bend)
                if arrived:
                    break
            if ok:
                self.vc.hold()
                ok = self._sim(int(self.task.get("drive_settle_steps", 5)), "drive")
        finally:
            self._bend_allow = allow0
            self._hold_x = float(sim.vehicle_pose()[0][0])      # drift is measured from where the vehicle stopped
        self.stops += 1
        self.travel += float(sim.vehicle_pose()[0][0]) - x_start
        if bend_max > float(self.lim.get("leaf_damage_rad", 0.9)):
            self._event(f"leaves bent {np.degrees(bend_max):.0f} deg while driving")
        self._event(f"vehicle at x={self._hold_x:.3f} (target {x_target:.3f})")
        return ok

    def _tick(self, label: str):
        super()._tick(label)
        if label != "drive" and getattr(self, "_hold_x", None) is not None:
            self.drift_max = max(self.drift_max, abs(float(self.sim.vehicle_pose()[0][0]) - self._hold_x))

    # ------------------------------------------------------------------ episode flow
    def _known(self, f) -> bool:
        for p in self.harvested_xy + self.skipped_xy:
            if np.linalg.norm(f.center[:2] - p) < 0.10:
                return True
        return False

    def _fruit_present(self, f) -> bool:
        if getattr(f, "_picked", False):
            return False
        return all(np.linalg.norm(f.center[:2] - p) > 0.12 for p in self.harvested_xy)

    def _start_episode(self) -> bool:
        self._hold_x = float(self.sim.vehicle_pose()[0][0])
        return self._survey_and_next()

    def _ahead(self) -> list:
        """Remembered fruits ahead of the arm in the own row that are neither harvested nor skipped."""
        base, rw = self.sim.base_pos, self.reach
        end = self._field_end()                      # (fruits of the next section are left for that run)
        return [f for f in self.fruits_scan if not self._known(f) and abs(f.center[1] - base[1]) < rw["lateral_max"]
                and f.center[0] - base[0] > rw["dx_min"] and f.center[0] <= end]

    def _look_side(self, coarse):
        """Close views from the corridor side of the fruit's row (away from the bed centre line = the vehicle axis)."""
        if not bool(self.cfg.get("planning", {}).get("look", {}).get("prefer_outside", False)):
            return None
        pv, yaw = self.sim.vehicle_pose()
        left = np.array([-np.sin(yaw), np.cos(yaw)])
        lat = float((coarse.center[:2] - pv[:2]) @ left)
        return left * (1.0 if lat >= 0 else -1.0)

    def _scan_known_count(self):
        return None                                  # a row: the number of fruits in view is not known

    def _scan_need(self) -> int:
        return int(self.cfg.get("planning", {}).get("scan", {}).get("need", 1))

    def _scan_accept(self, fruits: list) -> list:
        base, rw = self.sim.base_pos, self.reach
        end = self._field_end()
        return [f for f in fruits if not self._known(f) and abs(f.center[1] - base[1]) < rw["lateral_max"]
                and f.center[0] - base[0] > rw["dx_min"] and f.center[0] <= end]

    def _survey_and_next(self) -> bool:
        """Next fruit of the row. False = row finished / failed.

        [v6] The fruits seen so far are remembered in world coordinates (the vehicle odometry is known), so the arm
        does NOT survey before every fruit: it goes from the default posture straight to the ideal close view of the
        next remembered fruit (driving first when it is beyond the work area). A survey from the default-side
        camera pose is made only when no remembered fruit is left ahead, or when a remembered fruit was not found."""
        sim, rw = self.sim, self.reach
        field_end = self._field_end()
        need_scan = not self._ahead()
        for _ in range(int(self.task.get("max_stops", 20))):
            if self.done:
                return False
            if need_scan:
                self._scan()
                if self.done:
                    return False
            base = sim.base_pos
            ahead = self._ahead()
            if not ahead:
                if not need_scan:                       # memory used up: look before driving on
                    need_scan = True
                    continue
                # nothing detected ahead: drive one step forward if the row is not finished yet
                if base[0] + rw["dx_max"] < field_end:
                    if not self._to_default("drive") or not self._drive(float(sim.vehicle_pose()[0][0]) + rw["advance_blind"]):
                        return False
                    continue
                self.done = True
                self._event("row finished")
                return False
            need_scan = False
            nxt = min(ahead, key=lambda f: f.center[0])
            dx = nxt.center[0] - base[0]
            if dx > rw["dx_max"]:
                target_x = float(sim.vehicle_pose()[0][0]) + (dx - rw["dx_target"])
                if not self._to_default("drive") or not self._drive(target_x):
                    return False
                continue
            # reachable: harvest this one (others stay in the list as obstacles)
            self.fruits_scan = [nxt] + [f for f in self.fruits_scan if f is not nxt]
            nxt._done = False
            for f in self.fruits_scan[1:]:
                f._done = True
            ok = self._prepare_target()
            if ok:
                return True
            if self.failure == "perception_lost" and self.failure not in self.hard_failures:
                # the remembered position was wrong (seen from far away / half hidden): forget it and survey again
                # from here; a place that fails twice is skipped
                n_lost = sum(np.linalg.norm(nxt.center[:2] - p) < 0.20 for p in self._lost_xy)
                self._lost_xy.append(nxt.center[:2].copy())
                if n_lost < int(self.task.get("lost_retries", 1)):
                    self._event("remembered fruit not found: survey again")
                    self.fruits_scan = [f for f in self.fruits_scan if f is not nxt]
                    self._forget(nxt)
                    self.failure, self.done = "", False
                    self.phase, self.plan, self.carry, self.target_slot = -1, None, None, None
                    need_scan = True
                    continue
            # no feasible Stargaze route from this vehicle position: move closer once and re-plan from new views
            shift = float(rw.get("replan_shift", 0.10))
            retried = any(np.linalg.norm(nxt.center[:2] - p) < 0.10 for p in self._retried_xy)
            if (self.failure == "plan_no_feasible_route" and not retried and shift > 0
                    and dx - shift >= rw["dx_min"] + 0.04):
                self._retried_xy.append(nxt.center[:2].copy())
                self._event(f"no route at dx={dx:.2f}: move vehicle {shift:.2f} m and re-plan")
                self.failure, self.done = "", False
                self.phase, self.plan, self.target_slot = -1, None, None
                if not self._to_default("drive") or not self._drive(float(sim.vehicle_pose()[0][0]) + shift):
                    return False
                continue
            if not self._soft_recover(nxt):
                return False
        self._fail("too_many_stops")
        return False

    def _forget(self, fruit):
        """Drop a remembered fruit (hook: Mission 3 fruit memory)."""

    def _to_default(self, label: str) -> bool:
        """Arm to the default posture (above the canopy) before the vehicle moves."""
        sim = self.sim
        if np.max(np.abs(sim.q_arm() - self.default_q_near())) < 0.02:
            return True
        sim.set_gripper(self.cfg["gripper"]["min_opening"])
        mv = self._move_safe(self.default_q(), label)
        if mv is None:                                 # the IK work posture is also above the canopy
            if np.max(np.abs(sim.q_arm() - self._home_q)) < 0.05:
                return True
            mv = self._move_safe(self._home_q, label)
            if mv is None:
                mv = self._move_joints(self._home_q, label)
        return bool(mv)

    def _field_end(self) -> float:
        return max(x for x, _ in field_positions(self.cfg)) + 0.2

    def _soft_recover(self, fruit) -> bool:
        """A soft failure on one fruit: remember it as skipped and continue with the row."""
        if not self.done or self.failure in self.hard_failures:
            return False
        self.skip_log.append(self.failure)
        self._event(f"skip fruit ({self.failure})")
        self.skipped_xy.append(fruit.center[:2].copy())
        self.failure, self.done = "", False
        self._fail_charged = False
        self.phase, self.plan, self.carry, self.target_slot = -1, None, None, None
        return self._go_home_after_skip()

    def _go_home_after_skip(self) -> bool:
        """Back out of the plant and return to the default posture after a fruit was given up. The plant damage of
        that fruit is already counted; while the arm leaves the plant, further leaf / stalk contact of the same kind
        must not end the whole row (measured: a second `plant_hit` during the back-out ended the run), and when the
        checked return fails softly the arm is taken home by a plain joint move. False only on a hard failure."""
        self._in_recovery = True
        try:
            self.sim.set_gripper(self.cfg["gripper"]["max_opening"])
            ok = self._back_out() and not self.done
            self.phase, self.plan, self.carry, self.target_slot, self.rec = -1, None, None, None, None
            if ok:
                ok = self._return_home() and not self.done
            if not ok and self.failure not in self.hard_failures:
                if self.failure:
                    self._event(f"recovery: {self.failure} while leaving the plant (ignored), plain move home")
                self.failure, self.done, self._fail_charged = "", False, False
                self.lamp("observe")
                ok = self._force_home()
        finally:
            self._in_recovery = False
        if ok:
            self._t_session = self.now()
        return ok

    def _force_home(self) -> bool:
        """Plain joint move to the default posture (hard failures still end the episode)."""
        sim = self.sim
        sim.set_gripper(self.cfg["gripper"]["min_opening"])
        target = self.default_q_near()
        for qq in self.arm.joint_path(sim.q_arm(), target, 0.02):
            sim.d.ctrl[sim.arm_act] = qq
            sim.step(1)
            self._tick("return")
            if sim.unstable:
                self._fail("sim_unstable")
                return False
            c = sim.classify_contacts(None)
            if c["self_or_env"] or c["floor"]:
                self._event(f"collision: {c['env_names'][:3]} during the plain move home")
                self._fail("collision" if c["self_or_env"] else "floor_contact")
                return False
        for _ in range(25):
            sim.step(1)
            self._tick("return")
        self._bend_allow = 0.0
        self.arm.target_quat = sim.tcp_quat()
        return True

    def _return_home_safe(self) -> bool:
        return self._go_home_after_skip()

    def _after_transport(self) -> float:
        if not self._release_and_verify():
            if not self.done:
                self._fail("basket_miss")
            return self._soft_after(self._w("drop_penalty", -30.0))
        r = self._w("place_reward", 40.0)
        self.harvested_xy.append(self.plan.fruit.center[:2].copy())
        if not self._return_home():
            return r
        self.phase = -1
        if not self._survey_and_next():
            if not self.failure and self.done:
                pass                                         # row finished
            elif not self.done:
                self._fail("next_target_failed")
        return r

    def _soft_after(self, r: float) -> float:
        """After a soft failure during RL phases: skip the fruit and continue the row (episode not terminated)."""
        if self.failure in self.hard_failures:
            return r
        fruit = self.plan.fruit if self.plan is not None else None
        slot = self.target_slot
        if fruit is not None:
            self.skipped_xy.append(fruit.center[:2].copy())
        if slot is not None and self.sim.field.slots[slot]["detached"] and not self.sim.field.slots[slot].get("lost"):
            self.sim.field.slots[slot]["harvested"] = True          # lost fruit: no longer on the plant
            if not self.sim.fruit_in_basket(slot):
                self.sim.set_gripper(self.cfg["gripper"]["max_opening"])
                self.sim.field.remove_lost(slot, self.sim.d)
        self.skip_log.append(self.failure)
        self._event(f"skip fruit ({self.failure})")
        self.failure, self.done = "", False
        self._fail_charged = False
        if not self._go_home_after_skip():
            return r
        if not self._survey_and_next() and not self.done:
            self._fail("next_target_failed")
        return r

    def _fail(self, reason: str):
        super()._fail(reason)

    def step(self, action):
        obs, r, te, tr, info = super().step(action)
        # soft failures raised inside the RL phases (grasp miss, slip, drop, basket miss, ...) -> continue the row
        # (also soft failures raised by scripted moves after a placement: return home / survey)
        final = self.hard_failures | {"too_many_stops", "next_target_failed"}
        n_rec = 0
        while self.done and self.failure and self.failure not in final and n_rec < 6:
            # (a soft failure of the NEXT fruit's scripted preparation inside the recovery is recovered as well)
            n_rec += 1
            r = self._soft_after(r)
        if n_rec:
            obs = self._get_obs()
            info = self._get_info()
            te = bool(self.done)
            tr = False
        return obs, r, te, tr, info

    def _get_info(self) -> dict:
        info = super()._get_info()
        info["n_placed"] = int(self.n_placed)
        info["stops"] = int(getattr(self, "stops", 0))
        info["vehicle_travel"] = float(getattr(self, "travel", 0.0))
        info["vehicle_drift_max"] = float(getattr(self, "drift_max", 0.0))
        sl = list(getattr(self, "skip_log", []))
        info["skipped"] = sl
        info["damage_events"] = int(sum(r in DAMAGE for r in sl) + (self.failure in DAMAGE))
        if self.done:
            ok = (not self.failure) and self.n_placed >= self._n_targets() and info["damage_events"] == 0
            info["is_success"] = bool(ok)
            info["termination_reason"] = "success" if ok else (self.failure or "row_finished_incomplete")
            if not self.failure and not ok:
                info["failure"] = "row_finished_incomplete"
        return info
