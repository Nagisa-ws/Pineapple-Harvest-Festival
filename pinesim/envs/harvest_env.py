"""
pinesim/envs/harvest_env.py
---------------------------
Missions 1 and 2 (robot on a 1.15 m platform) as one Gymnasium environment with a residual-SAC action interface.

Mission 1 (version 5), camera-driven, UR10e + U-jaw gripper; every session starts and ends in the DEFAULT posture
(user's picture) with the gripper CLOSED; the tip of the closed jaws is the reference point:
    SCAN (wrist-camera survey; the simulator does NOT give the plant position)                    lamp: yellow
 -> LOOK (close views) -> SHOULDER (fruit pixels between centre and crown base, leaf-covered parts excluded)
 -> WINDOW (its bounding box) -> T2 candidates in the window, safest line T1-T2 each (T1 inside the window)
 -> DIFFUSION POLICY picks the best T2 -> arm validation (IK, collisions)
 -> phase 0 APPROACH (current pose -> tip at T1)                    [reference + residual SAC]    lamp: blue
 -> phase 1 LINE (tip T1 -> T2 along the max-clearance line)        [reference + residual SAC]
 -> OPEN the jaws at T2 -> ADVANCE until the fruit axis is between the pads -> GRIP -> TWIST about the fruit axis
    until the stem breaks                                                                         [scripted]
 -> phase 2 RETREAT (pull out along the line)                       [reference + residual SAC]
 -> phase 3 LIFT                                                    [reference + residual SAC]
 -> phase 4 TRANSPORT to the basket (180 deg away), FASTER than v4  [reference + residual SAC]
 -> RELEASE -> VERIFY (fruit stays in the basket) -> back to the DEFAULT posture, jaws closed     lamp: green
Leaves: touching / bending them is penalised (the policy keeps avoiding them) but never ends the session; a leaf torn
out of the rosette (root hinge beyond `limits.leaf_uproot_rad`) ends it (lamp red), as do a dropped fruit, a
collision, a knocked-off fruit or the time limit.
Mission 2 = Mission 1 + a neighbour plant at an unknown position (left/right/front/back): both fruits are detected,
the first one (priority: nearest to the robot base) is harvested and placed, then the robot looks at the neighbour
again, re-plans and harvests it.

Physics honesty: the fruit is a free body held by a breakable weld (stem) that is released only when the measured
constraint load exceeds the plant's thresholds; it is carried by finger friction only (no weld to the gripper, no
teleport). Ground truth is used only to judge outcomes (success/failure, metrics), never as planner/policy input.

Observation: the reference 63-value layout (pinesim/vendor/obs_layout.py) -> Mission 3 weights of the reference
repository transfer directly; obstacle slots = nearest PERCEIVED leaf points, 4th slot = wrist force (F/T).
Action: residual correction a_rl in [-1,1]^6; executed a = clip(a_ref + g * scale * a_rl) with the reference
Mission 4 progress floor and target gate.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from pinesim.config import load_config
from pinesim.control.arm import Arm, quat_mul, quat_z, unit
from pinesim.control.planner import Planner
from pinesim.control.reference import ReferenceConfig, ReferenceController, Route
from pinesim.plant import PlantSampler
from pinesim.sim import Sim
from pinesim.vendor.obs_layout import N_NEAREST_OBSTACLES, OBS_DIM, OBS_FAR, OBS_SLICES
from pinesim.vendor.pose_utils import perpendicular_distance_to_segment, project_point_on_segment, quat_angle
from pinesim.vendor.robot_geometry import world_segments
from pinesim.vision.camera import RGBDCamera
from pinesim.vision.perception import Perceiver, merge_fruits
from pinesim.decision.window import decide
from pinesim.decision.diffusion import load_policy


def quat_axis(u, th: float) -> np.ndarray:
    u = np.asarray(u, float)
    u = u / (np.linalg.norm(u) + 1e-12)
    return np.r_[np.cos(th / 2), np.sin(th / 2) * u]

PHASES = ["approach", "line", "retreat", "lift", "transport"]
PHASE_ID = {p: i for i, p in enumerate(PHASES)}


@dataclass
class TargetRecord:
    slot_gt: int = -1                 # ground-truth slot of the estimate (metrics only)
    est_error: float = float("nan")   # |estimated - true fruit centre| (metrics only)
    plan_reason: str = ""
    face: str = ""
    clearance: float = float("nan")
    grasped: bool = False
    detached: bool = False
    placed: bool = False
    twist_deg: float = 0.0
    pluck_deg: float = 0.0            # tilt of the hand at which the stem tore off (harvest.method = pluck)
    method: str = "twist"             # how this fruit was taken off the stalk: twist | pluck | pull    
    stem_force: float = 0.0
    stem_torque: float = 0.0
    advance_cm: float = 0.0
    failure: str = ""
    # timing of this fruit's session (seconds of robot motion; compute_s = measured perception + decision + planning)
    t_start: float = float("nan")     # session start: the arm leaves the default / survey posture for this fruit
    t_decided: float = float("nan")   # T1 / T2 decided and validated (observation finished)
    t_grasped: float = float("nan")
    t_placed: float = float("nan")    # fruit verified in the crate
    t_end: float = float("nan")       # arm back in the default posture
    compute_s: float = 0.0
    n_views: int = 0                  # gripper-camera views used for the decision
    t1_shift_cm: float = float("nan") # correction of the fruit estimate by the gripper camera at T1
    advance_corrections: int = 0
    grip_drop_cm: float = 0.0         # pads moved this far along the fruit axis to grip the widest section
    regrips: int = 0                  # closed-loop grip height corrections (pad contact position)
    carry_guards: int = 0             # pull-out re-planned because a leaf was being dragged


class HarvestEnv(gym.Env):
    metadata = {"render_modes": []}
    MISSION = 1

    def __init__(self, mission: str = "mission1", training: bool = True, seed: int | None = None,
                 overrides: dict | None = None, level: int | None = None):
        super().__init__()
        self.cfg = cfg = load_config(mission, overrides)
        self.task = task = cfg["task"]
        self.training = training
        self.mission = mission
        self.sim = Sim(cfg, task.get("mode", "platform"), int(task["n_plants"]))
        self.arm = Arm(self.sim, cfg)
        self.planner = Planner(self.sim, self.arm, cfg)
        self.cam = RGBDCamera(self.sim.m, self.sim.d, "wrist_cam", cfg, seed=0 if seed is None else seed,
                              noise=bool(cfg["noise"].get("enabled", True)))
        self.sampler = PlantSampler(cfg)
        self.level = task.get("level", 2) if level is None else level
        self.reference = ReferenceController(ReferenceConfig.from_dict(cfg["control"]["reference"]))
        ps = cfg.get("policy", {})
        self.residual_scale = np.array([ps.get("residual_scale_pos", 0.5)] * 3 + [ps.get("residual_scale_rot", 0.5)] * 3)
        self.progress_min = float(ps.get("progress_min", 0.4))
        g = ps.get("target_gate", [0.04, 0.10])
        self.target_gate = None if g is None else (float(g[0]), float(g[1]))
        self.carry_speed = float(ps.get("carry_speed_scale", 0.6))
        self.W = dict(cfg.get("reward", {}))
        self.lim = dict(cfg.get("limits", {}))
        self.max_steps = int(task.get("max_steps", 3000))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.frame_callback = None          # recorder hook: called every control step with a label
        self.overlay_callback = None        # recorder hook: stargaze overlays
        self.policy_fn = None               # optional: residual policy used during evaluation helpers
        self.resample_on_plan_fail = bool(task.get("resample_on_plan_fail", training))
        self.t2_policy = load_policy(cfg)                 # diffusion decision maker (None -> heuristic)
        mujoco.mj_forward(self.sim.m, self.sim.d)
        self._home_q = None
        self._reset_state()
        # IK work posture solved once on the empty scene (basket at its default place, behind the robot): solved
        # after a scene reset it could collide with a basket placed at azimuth 0 (measured: evaluation env crash)
        self._go_home_instant()

    # ================================================================== bookkeeping
    def _reset_state(self):
        self.shield_events = 0
        self.step_count = 0
        self.phase = -1
        self.route: Route | None = None
        self.plan = None
        self.carry = None
        self.target_slot = None
        self.records: list[TargetRecord] = []
        self.rec: TargetRecord | None = None
        self.fruits_scan = []
        self.leaf_cloud = np.zeros((0, 3))
        self.scan_leaf = np.zeros((0, 3))
        self._plan_ctx = None
        self.obs_pts = np.zeros((0, 3))
        self.obs_rad = np.zeros(0)
        self._prev_action = np.zeros(6)
        self._cur_action = np.zeros(6)
        self._ref_action = np.zeros(6)
        self._gate = 1.0
        self._prev_dist = 0.0
        self._rc = {}
        self._contacts = {}
        self._leaf_bend = 0.0
        self._leaf_energy = 0.0
        self._bend_allow = 0.0            # > limit only while backing out after a soft leaf failure (Mission 3)
        self._min_clear = np.inf
        self._clear = np.zeros(0)
        self._witness = np.zeros((0, 3))
        self.failure = ""
        self.done = False
        self.n_placed = 0
        self.timing = {}
        self.events = []
        self.leaf_damage = 0              # leaves bent beyond limits.leaf_damage_rad (counted once per leaf)
        self.root_bend_max = 0.0
        self.decision = None
        self._t_session = 0.0             # robot time at which the current fruit's session started
        self.compute_by = {}
        self.compute_s = 0.0              # wall time of perception / decision / planning (robot waits meanwhile)
        self._shield_replans = 0
        self.residual_off_events = 0
        self._residual_off = False
        self._phase_shields = 0
        self._prog = [-1, np.inf, 0]
        self._in_recovery = False         # Mission 3: backing out / going home after a fruit was given up
        self._closed_xy = []              # Missions 1-2: places whose fruit was picked (not surveyed again)
        self._last_az = None
        self._drops = 0                   # fruits released into the crate (Missions 1-2: this episode)
        self._q_hist = []                 # arm postures of the last policy steps of the active phase (shield backtrack)
        self._residual_used = False       # the residual policy has moved the arm for the current fruit
        self._residual_ban = False        # ... and was banned for the rest of this fruit after a shield backtrack
        self._stalks = []                 # (top point, axis) of the bare stalks of fruits picked in this episode
        self._fail_charged = False        # the terminal failure penalty is given once per failure
        self.fail_phase = ""              # label of the control phase in which the episode failed (diagnostics)
        self._label = ""
        self.lamp("observe")

    def _w(self, k, default=0.0):
        return float(self.W.get(k, default))

    def _event(self, msg):
        self.events.append((self.step_count, msg))

    def now(self) -> float:
        """Robot time of this episode (s): control steps x 20 ms."""
        return self.step_count * self.sim.dt

    class _Timer:
        def __init__(self, env, tag="other"):
            self.env, self.tag = env, tag

        def __enter__(self):
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, *exc):
            dt = time.perf_counter() - self.t0
            self.env.compute_s += dt
            self.env.compute_by[self.tag] = self.env.compute_by.get(self.tag, 0.0) + dt
            if self.env.rec is not None:
                self.env.rec.compute_s += dt
            return False

    def _compute(self, tag: str = "other"):
        """Context manager: wall time of perception / decision / planning code. On the real robot the arm waits
        while this runs, so it is added to the per-fruit time (reported separately from the motion time)."""
        return HarvestEnv._Timer(self, tag)

    def lamp(self, state: str):
        """Indicator lamp: observe (yellow) / action (blue) / success (green) / fail (red)."""
        self.lamp_state = state
        self.sim.set_lamp(state)

    # ================================================================== scene
    def _place_plants(self):
        """Mission 1: one plant; Mission 2: plus a neighbour in a random direction. Basket at 180 deg."""
        sim, t = self.sim, self.task
        pl = t["placement"]
        base = sim.base_pos
        az = self.rng.uniform(*np.radians(pl.get("azimuth_deg_range", [0, 360])))
        R = self.rng.uniform(*pl["radius_range"])
        grounds = [np.array([base[0] + R * np.cos(az), base[1] + R * np.sin(az), 0.0])]
        if int(t["n_plants"]) >= 2:
            for _ in range(50):
                d = self.rng.uniform(*pl["neighbour_distance_range"])
                ang = self.rng.uniform(0, 2 * np.pi)          # left / right / front / back: any direction
                g2 = grounds[0] + np.array([d * np.cos(ang), d * np.sin(ang), 0.0])
                r2 = np.hypot(g2[0] - base[0], g2[1] - base[1])
                if pl["radius_range_neighbour"][0] <= r2 <= pl["radius_range_neighbour"][1]:
                    grounds.append(g2)
                    break
        az_mean = np.arctan2(np.mean([g[1] - base[1] for g in grounds]), np.mean([g[0] - base[0] for g in grounds]))
        b = self.cfg["basket"]
        ab = az_mean + np.pi
        sim.m.body_pos[sim.basket_body] = [base[0] + b["radius_platform"] * np.cos(ab), base[1] + b["radius_platform"] * np.sin(ab), 0.0]
        self.sampler.set_level(self.level)
        for i in range(sim.field.n):
            if i < len(grounds):
                lay = self.sampler.sample(self.rng, grounds[i], base - grounds[i])
                sim.field.apply(i, lay, sim.d)
            else:
                sim.field.hide(i, sim.d)
        self.basket_azimuth = ab

    def _scene_reset(self):
        sim = self.sim
        mujoco.mj_resetData(sim.m, sim.d)
        mujoco.mj_forward(sim.m, sim.d)
        self._place_plants()
        self._go_home_instant()
        mujoco.mj_forward(sim.m, sim.d)
        for i in sim.field.active_slots():
            sim.field.arm_stem(i, sim.d)
        sim.step(5)
        for i in sim.field.active_slots():          # resting stem load (fruit weight) = zero point of the break test
            sim.field.record_rest_load(i, sim.d)

    def default_q(self) -> np.ndarray:
        """DEFAULT posture of every session (user's picture), pan turned to the robot's forward direction."""
        q = np.array(self.cfg["robot"]["default_q"], float)
        q[0] = np.radians(float(self.cfg["robot"].get("default_pan_deg", 0.0)))
        if self.sim.vehicle_body >= 0:
            q[0] += self.sim.vehicle_pose()[1]
        return q

    def _go_home_instant(self):
        """Arm in the DEFAULT posture, gripper CLOSED (start of a session). The IK work posture (tool down, used as
        the canonical branch of camera views and plans) is solved once."""
        sim = self.sim
        if self._home_q is None:
            sim.set_arm(self.default_q())
            mujoco.mj_forward(sim.m, sim.d)
            pos, quat = self.planner.home_pose(self._home_azimuth())
            seed = sim.d.qpos.copy()
            seed[sim.arm_qadr] = [-3.54, -1.83, 2.49, -2.23, -1.57, -1.97]
            if self.sim.vehicle_body >= 0:
                seed[sim.arm_qadr[0]] += self.sim.vehicle_pose()[1]
            q, ok = self.arm.solve_pose(pos, quat, q_seed_full=seed, tries=10, rng=np.random.default_rng(1))
            if not ok:
                raise RuntimeError("IK work posture infeasible")
            self._home_q = q.copy()
            self.planner.home_q = q.copy()
        sim.set_arm(self.default_q())
        sim.set_gripper(self.cfg["gripper"]["min_opening"], instant=True)
        mujoco.mj_forward(sim.m, sim.d)
        self.arm.target_quat = sim.tcp_quat()

    def _home_azimuth(self):
        return 0.0

    # ================================================================== stepping helpers
    def _tick(self, label: str):
        self.step_count += 1
        self._label = label
        self.sim.field.update_muted(self.sim.d)
        if self.frame_callback is not None:
            self.frame_callback(label)

    def _sim(self, n: int, label: str, monitor: bool = True) -> bool:
        self._label = label
        for _ in range(n):
            self.sim.step(1)
            self._tick(label)
            if monitor and not self._monitor_ok():
                return False
            if self.step_count >= self.max_steps:
                self._fail("timeout")
                return False
        return True

    def _monitor_ok(self) -> bool:
        """Safety monitor for scripted motion: robot collisions and leaf damage end the episode."""
        if self.sim.unstable:                       # numerical blow-up: reported as such, never as a collision
            self._fail("sim_unstable")
            return False
        c = self.sim.classify_contacts(self.target_slot)
        self._contacts = c
        if c["self_or_env"]:
            self._event(f"collision: {c['env_names'][:3]} during '{getattr(self, '_label', '?')}'")
            self._fail("collision")
            return False
        if c["floor"]:
            self._fail("floor_contact")
            return False
        self._update_leaf_state()
        if self._leaf_uprooted():
            return False
        if c["stalk"] > float(self.lim.get("stalk_force", 10.0)) and not self._in_recovery:
            self._event(f"plant hit: stalk {c['stalk']:.0f} N during '{self._label}', robot contacts {self._robot_contacts()[:4]}")
            self._fail("plant_hit")
            return False
        if not self._stems_intact():
            return False
        return True

    def _robot_contacts(self) -> list:
        """(robot geom, other geom) names of the current robot contacts (diagnostics)."""
        sim, out = self.sim, []
        for i in range(sim.d.ncon):
            c = sim.d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if sim.robot_geom[g1] != sim.robot_geom[g2]:
                rg, og = (g1, g2) if sim.robot_geom[g1] else (g2, g1)
                if og != sim.floor_geom:
                    out.append((sim._gname(rg), sim._gname(og)))
        return out

    def _stems_intact(self) -> bool:
        """A stem that breaks before the fruit is gripped (robot knocked the plant) ends the episode."""
        fld = self.sim.field
        for i in fld.active_slots():
            s = fld.slots[i]
            if s["detached"] or s["harvested"]:
                continue
            if self.rec is not None and i == self.target_slot and self.rec.grasped:
                continue
            brk, fe, te = fld.check_break(i, self.sim.d)
            if brk:
                self._event(f"stem broken: slot {i} ({'target' if i == self.target_slot else 'other'}), "
                            f"force {fe:.0f} N, torque {te:.1f} Nm, during '{getattr(self, '_label', '?')}', "
                            f"robot contacts {self._robot_contacts()[:4]}")
                pk, _ = fld.fruit_pose(i, self.sim.d)
                seen = [float(np.linalg.norm(f.center - pk)) for f in getattr(self, "fruits_scan", [])]
                self._event(f"knocked fruit at {np.round(pk, 2).tolist()} (base {np.round(self.sim.base_pos, 2).tolist()}): "
                            f"nearest perceived fruit {min(seen) * 100 if seen else float('nan'):.0f} cm away, "
                            f"{len(seen)} fruits in memory")
                fld.detach(i, self.sim.d)
                fld.remove_lost(i, self.sim.d)
                self._fail("fruit_knocked_off")
                return False
        return True

    def _update_leaf_state(self):
        """Leaf bend (max hinge), ROOT bend (max over leaves of the root hinge pair), spring energy, damage events."""
        f = self.sim.field
        b = e = rb = 0.0
        new_damage = 0
        lim_d = float(self.lim.get("leaf_damage_rad", 0.9))
        for i in f.active_slots():
            b = max(b, f.leaf_bend_max(i, self.sim.d))
            rb = max(rb, f.leaf_root_bend_max(i, self.sim.d))
            e += f.leaf_energy(i, self.sim.d)
            if b > lim_d:
                new_damage += f.new_leaf_damage(i, self.sim.d, lim_d)
        self._root_rate = rb - getattr(self, "_root_bend", 0.0)      # rad per control step
        self._leaf_bend, self._leaf_energy, self._root_bend = b, e, rb
        self.root_bend_max = max(self.root_bend_max, rb)
        if new_damage:
            self.leaf_damage += new_damage
            self._new_damage = getattr(self, "_new_damage", 0) + new_damage

    def _leaf_uprooted(self) -> bool:
        """A leaf torn out of the rosette ends the session [USER v5] (touching / bending leaves does not)."""
        if getattr(self, "_root_bend", 0.0) > max(float(self.lim.get("leaf_uproot_rad", 1.15)), self._bend_allow):
            self._event(f"leaf uprooted: root bend {np.degrees(self._root_bend):.0f} deg")
            self._fail("leaf_uprooted")
            return True
        return False

    def _leaf_guard(self, key: str = "line_guard_frac") -> bool:
        """Leaf guard of the scripted / guarded motions: True when the most bent leaf root is beyond the guard fraction
        of the uproot limit OR will be within `leaf_guard_lead_steps` control steps at its current rate [v6: a leaf
        that was whipped from 43 to 60 deg within 80 ms passed the old threshold test and was torn out]."""
        lim = float(self.lim.get("leaf_uproot_rad", 1.15))
        frac = float(self.lim.get(key, self.lim.get("line_guard_frac", 0.85)))
        lead = float(self.lim.get("leaf_guard_lead_steps", 5.0))
        rb = getattr(self, "_root_bend", 0.0)
        return rb + lead * max(0.0, getattr(self, "_root_rate", 0.0)) > frac * lim

    def _known_leaves(self) -> np.ndarray:
        """Leaf points perceived so far (survey + close look), coarse grid: obstacles for scripted moves."""
        parts = [c for c in (getattr(self, "scan_leaf", np.zeros((0, 3))), self.leaf_cloud) if len(c)]
        if not parts:
            return np.zeros((0, 3))
        from pinesim.vision.perception import voxel_down
        pts = np.vstack(parts)
        # only leaves the arm can reach (base to jaw tip < 1.7 m): the survey sees the canopy far ahead as well,
        # and the clearance of every candidate path is a (points x capsules x samples) computation
        base = self.sim.base_pos
        pts = pts[np.linalg.norm(pts - base[None, :], axis=1) < float(self.cfg.get("planning", {}).get("move_leaf_reach", 1.7))]
        vox, cap = float(self.cfg.get("planning", {}).get("move_leaf_voxel", 0.04)), int(self.cfg.get("planning", {}).get("move_leaf_max_points", 1200))
        pts = voxel_down(pts, vox)
        while len(pts) > cap:                    # dense canopy: coarser grid (the cost is points x capsules x samples)
            vox *= 1.3
            pts = voxel_down(pts, vox)
        return pts

    def _fruit_present(self, f) -> bool:
        """False for a remembered fruit that has been harvested since (it is no obstacle any more)."""
        return not getattr(f, "_picked", False)

    def _path_fruit_clear(self, qs, margin: float = 0.02) -> bool:
        """Scripted joint moves keep away from every PERCEIVED fruit (survey estimates): a neighbour fruit knocked off
        by a look-view move was measured in Mission 2. A path may start closer than the margin (the default posture
        above a row of crowns, Mission 3) as long as it does not get closer than it starts."""
        fr = [f for f in getattr(self, "fruits_scan", []) if self._fruit_present(f)]
        if not fr:
            return True
        hp, hr = [], []
        for f in fr:
            (p_, r_), _, _ = self.planner.hard_points(f, n=4)
            hp.append(p_); hr.append(r_)
            # the crown above the fruit: pushing it pulls the stalk (a neighbour's stalk snapped when a jaw brushed
            # its crown during a look move, measured in Mission 3)
            # (the target's own crown is an obstacle of the PATH as well: the jaws swept through it on the way to
            #  the close view and twisted the target fruit off its stalk, measured in Mission 3 [v6])
            ax = getattr(f, "axis", np.array([0.0, 0.0, 1.0]))
            hh = getattr(f, "half_height", 0.5 * (f.top_z - f.bottom_z))
            hs = hh + np.array([0.03, 0.08, 0.13, 0.18])
            # (the crown widens upward: a jaw that brushed a tuft tip twisted the neighbour's fruit off, Mission 3)
            hp.append(f.center[None, :] + hs[:, None] * ax[None, :]); hr.append(np.array([0.045, 0.05, 0.055, 0.06]))
        hp, hr = np.vstack(hp), np.concatenate(hr)
        chk = self.arm.checker
        full = self.sim.d.qpos.copy()
        chk._data.qpos[:] = full
        mujoco.mj_kinematics(self.sim.m, chk._data)
        start = chk.hard_clearance([world_segments(self.arm.caps, chk._data)], hp, hr)
        segs = []
        for q in qs[::2]:
            full[self.sim.arm_qadr] = q
            chk._data.qpos[:] = full
            mujoco.mj_kinematics(self.sim.m, chk._data)
            segs.append(world_segments(self.arm.caps, chk._data))
        return chk.hard_clearance(segs, hp, hr) >= min(margin, start - 0.005)

    def _path_leaf_intrusion(self, qs, pts) -> float:
        """Sum over perceived leaf points of max(0, margin - clearance) of the whole arm along joint configurations."""
        if len(pts) == 0:
            return 0.0
        chk = self.arm.checker
        full = self.sim.d.qpos.copy()
        segs = []
        for q in qs:
            full[self.sim.arm_qadr] = q
            chk._data.qpos[:] = full
            mujoco.mj_kinematics(self.sim.m, chk._data)
            segs.append(world_segments(self.arm.caps, chk._data))
        r = np.full(len(pts), float(self.cfg["perception"].get("leaf_point_radius", 0.015)))
        clr = chk.min_clearance_along(segs, pts, r)
        return float(np.sum(np.clip(float(self.cfg.get("planning", {}).get("move_leaf_margin", 0.03)) - clr, 0.0, None)))

    def _lifted(self, q_arm, z_min: float):
        """Same arm configuration family with the TCP raised to z_min (orientation kept); None if not reachable."""
        chk = self.arm.checker
        full = self.sim.d.qpos.copy()
        full[self.sim.arm_qadr] = q_arm
        chk._data.qpos[:] = full
        mujoco.mj_kinematics(self.sim.m, chk._data)
        p = chk._data.site_xpos[self.arm.ik.site_id].copy()
        if p[2] >= z_min:
            return np.array(q_arm, float)
        R = chk._data.site_xmat[self.arm.ik.site_id].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R.flatten())
        q, info = self.arm.ik.solve(full, np.array([p[0], p[1], z_min]), quat)
        if info["position_error"] > 0.01 or np.max(np.abs(q - q_arm)) > 1.5:
            return None
        return q

    def _move_safe(self, q_to, label: str):
        """Joint move avoiding robot/environment collisions AND the leaves perceived so far: candidate paths
        (direct, via home, lifted over the plants) are checked; the collision-free one with the least leaf
        intrusion is executed. Returns True (moved), False (failed while moving: episode over) or None (skipped)."""
        sim = self.sim
        t_plan = time.perf_counter()
        q0 = sim.q_arm().copy()
        pts = self._known_leaves()
        unwrap = self.arm.ik.unwrap_to_current
        # same pose, nearest joint representation first (a 2*pi detour of the base swept through the basket)
        # (the planner's representation first - its later IK steps assume it -; the unwrapped one as alternative)
        targets = [np.array(q_to, float)]
        qu = unwrap(np.array(q_to, float), q0)
        if np.max(np.abs(qu - q_to)) > 1e-6:
            targets.append(qu)
        # other representations of the wrist joints (+-2 pi): turning wrist 1 the other way round avoids folding the
        # long U-jaws into the forearm (measured self-collision on the way back to the default posture)
        lim = self.sim.arm_range
        for j in (3, 5):
            for qb in list(targets):
                for sgn in (2 * np.pi, -2 * np.pi):
                    qv = qb.copy()
                    qv[j] += sgn
                    if lim[j, 0] <= qv[j] <= lim[j, 1] and all(np.max(np.abs(qv - t)) > 1e-6 for t in targets):
                        targets.append(qv)
        targets = targets[:6]
        z_min = (float(np.max(pts[:, 2])) + float(self.cfg.get("planning", {}).get("move_lift_clearance", 0.15))
                 if len(pts) else None)
        # candidates in order of simplicity: the direct move for every joint representation first, then the detours
        # (via the IK work posture, base-turn-last, lifted over the plants)
        home = unwrap(self._home_q.copy(), q0)
        hc = home.copy()                         # IK work posture at the CURRENT azimuth, then turn, then go
        hc[0] = q0[0]
        stages = [[], [], [], [], [], []]
        for qt in targets:
            hq = home.copy()
            hq[0] = qt[0]
            # the target posture first at the current azimuth, then turn the base only (the compact default
            # posture keeps the jaws above the basket rim and away from the column while turning)
            dc = qt.copy()
            dc[0] = q0[0]
            dc = unwrap(dc, q0)
            dc[0] = q0[0]
            stages[0].append([qt])
            stages[1] += [[hq, qt], [dc, qt]]
            stages[2] += [[home, qt], [hc, hq, qt], [hc, dc, qt]]
            if z_min is not None:
                l0, l1 = self._lifted(q0, z_min), self._lifted(qt, z_min)
                if l0 is not None and l1 is not None:
                    stages[3] += [[l0, l1, qt], [l0, home, qt]]
        cands = [c for st in stages for c in st]
        pc = self.cfg.get("planning", {})
        max_ok, accept = int(pc.get("move_max_candidates", 5)), float(pc.get("move_leaf_accept", 0.02))
        seg_ok = {}

        def path_ok(a, b):                       # (segments are shared by many candidates)
            key = (np.asarray(a, float).round(5).tobytes(), np.asarray(b, float).round(5).tobytes())
            if key not in seg_ok:
                seg_ok[key] = self.arm.joint_path_ok(a, b)
            return seg_ok[key]
        best, best_cost, n_ok = None, np.inf, 0
        rej = {"contact": 0, "fruit": 0}
        for k, path in enumerate(cands):
            chain = [q0] + list(path)
            if not all(path_ok(a, b) for a, b in zip(chain[:-1], chain[1:])):
                rej["contact"] += 1
                continue
            qs = np.vstack([self.arm.joint_path(a, b, 0.10) for a, b in zip(chain[:-1], chain[1:])])
            if label in ("look", "scan") and not self._path_fruit_clear(qs, margin=0.01):
                rej["fruit"] += 1
                continue
            n_ok += 1
            cost = self._path_leaf_intrusion(qs, pts) + 1e-3 * k        # tie-break: simpler path first
            if cost < best_cost:
                best, best_cost = path, cost
            if cost < accept + 1e-3 * k or n_ok >= max_ok:              # (nearly) leaf-free, or enough alternatives
                break
        dt_plan = time.perf_counter() - t_plan
        self.compute_s += dt_plan
        self.compute_by["move_plan"] = self.compute_by.get("move_plan", 0.0) + dt_plan
        if self.rec is not None:
            self.rec.compute_s += dt_plan
        if best is None:
            chk = self.arm.checker
            if chk.self_pairs or chk.env_pairs:
                # scripted moves are slow (small servo lag): retry with the plain contact test, without the 1 cm
                # clearance margins that guard the faster policy-driven phases
                saved = (chk.self_pairs, chk.env_pairs)
                chk.self_pairs, chk.env_pairs = [], []
                try:
                    return self._move_safe(q_to, label)
                finally:
                    chk.self_pairs, chk.env_pairs = saved
            self._event(f"{label}: no collision-free joint path (view skipped; {len(cands)} candidates: "
                        f"{rej['contact']} robot/vehicle contact, {rej['fruit']} too close to a perceived fruit / crown)")
            return None
        return self._run_chain(list(best), label, guard=True)

    # ---------------------------------------------------------------- timed joint motion (scripted moves)
    def _perceived_fruit_points(self, include_target: bool = True):
        """Hard points (fruit body + crown above it) of every fruit the CAMERA has seen and that is still on its
        plant: the speed governor slows the arm down near them [USER: the gripper must never knock a fruit off]."""
        fr = [f for f in getattr(self, "fruits_scan", []) if self._fruit_present(f)]
        ctx = getattr(self, "_plan_ctx", None)
        tgt = ctx["fruit"] if ctx is not None else None
        if tgt is not None:
            # (the survey's coarse estimate of the target itself is not "another fruit")
            fr = [f for f in list(ctx.get("others", [])) + fr if np.linalg.norm(f.center - tgt.center) > 0.18
                  and self._fruit_present(f)]
            fr = merge_fruits(fr, dist=0.10)
            if include_target and not (self.rec is not None and self.rec.detached):
                fr.append(tgt)
        hp, hr = [], []
        for f in fr:
            ax = getattr(f, "axis", np.array([0.0, 0.0, 1.0]))
            hh = getattr(f, "half_height", 0.5 * (f.top_z - f.bottom_z))
            hs = np.linspace(-hh + 0.5 * f.radius, hh - 0.5 * f.radius, 3)
            hp.append(f.center[None, :] + hs[:, None] * ax[None, :]); hr.append(np.full(3, f.radius))
            hc = hh + np.array([0.04, 0.10, 0.16])
            hp.append(f.center[None, :] + hc[:, None] * ax[None, :]); hr.append(np.array([0.045, 0.05, 0.06]))
        # stalks: rigid, below every OTHER fruit that is still on its plant and where a fruit was picked before (the
        # bare stalk stays) - measured in Mission 3: during a twist the wrist swung into the rosette of the
        # neighbour and pressed its stalk with 530 N (plant_hit)
        stalks = [(f.center - getattr(f, "axis", np.array([0.0, 0.0, 1.0])) *
                   getattr(f, "half_height", 0.5 * (f.top_z - f.bottom_z)),
                   getattr(f, "axis", np.array([0.0, 0.0, 1.0]))) for f in fr if f is not tgt]
        stalks += [(b, ax) for b, ax in getattr(self, "_stalks", [])
                   if tgt is None or np.linalg.norm(b[:2] - tgt.center[:2]) > 0.15]
        base = self.sim.base_pos
        for bot, ax in stalks:
            if np.linalg.norm(bot[:2] - base[:2]) > 1.6:
                continue
            foot = bot - ax * (bot[2] / max(float(ax[2]), 0.5))
            t = np.array([0.0, 0.25, 0.5, 0.75])[:, None]
            sp = bot[None, :] * (1 - t) + foot[None, :] * t
            hp.append(sp); hr.append(np.full(len(sp), 0.025))
        if not hp:
            return np.zeros((0, 3)), np.zeros(0)
        return np.vstack(hp), np.concatenate(hr)

    def _speed_cap(self, d: float, v_fast: float, v_slow: float) -> float:
        """Speed limit as a function of the distance d (m) to the nearest perceived fruit / crown."""
        pc = self.cfg.get("planning", {})
        d0, d1 = float(pc.get("fruit_slow_dist", 0.06)), float(pc.get("fruit_free_dist", 0.30))
        return float(v_slow + (v_fast - v_slow) * np.clip((d - d0) / max(d1 - d0, 1e-6), 0.0, 1.0))

    def _joint_trajectory(self, chain, v_fast: float, v_slow: float):
        """Joint commands (one per 20 ms) through the waypoints `chain` (chain[0] = current posture): trapezoidal
        speed profile (bounded acceleration), slower at corners and near every perceived fruit, never stopping at
        intermediate waypoints. Speeds are in rad per control step of the two big joints (pan, lift); the elbow and
        wrist joints may turn `joint_speed_ratio` times faster (UR10e: 120 deg/s vs 180 deg/s)."""
        pc = self.cfg.get("planning", {})
        ratio = np.array(pc.get("joint_speed_ratio", [1.0, 1.0, 1.5, 1.5, 1.5, 1.5]), float)
        acc = float(pc.get("joint_accel_per_step2", 0.003))
        ds = 0.02
        pts = [np.asarray(chain[0], float)]
        cap = [v_fast]
        dirs = []
        for a, b in zip(chain[:-1], chain[1:]):
            a, b = np.asarray(a, float), np.asarray(b, float)
            L = float(np.max(np.abs(b - a) / ratio))
            if L < 1e-6:
                continue
            n = max(1, int(np.ceil(L / ds)))
            u = (b - a) / ratio
            u = u / (np.linalg.norm(u) + 1e-12)
            if dirs:                                         # corner: slow down in proportion to the turn
                cap[-1] = min(cap[-1], v_fast * float(np.clip(0.5 * (1.0 + float(dirs[-1] @ u)), 0.25, 1.0)))
            dirs.append(u)
            for k in range(1, n + 1):
                pts.append(a + (b - a) * k / n)
                cap.append(v_fast)
        if len(pts) < 2:
            return np.zeros((0, 6))
        P = np.array(pts)
        cap = np.array(cap)
        hp, hr = self._perceived_fruit_points()
        if len(hp):                                          # slower near perceived fruit
            chk = self.arm.checker
            full = self.sim.d.qpos.copy()
            for i in range(len(P)):
                full[self.sim.arm_qadr] = P[i]
                chk._data.qpos[:] = full
                mujoco.mj_kinematics(self.sim.m, chk._data)
                d = chk.hard_clearance([world_segments(self.arm.caps, chk._data)], hp, hr)
                cap[i] = min(cap[i], self._speed_cap(d, v_fast, v_slow))
        seg = np.max(np.abs(np.diff(P, axis=0)) / ratio[None, :], axis=1)
        cum = np.r_[0.0, np.cumsum(seg)]
        v = cap.copy()
        v[-1] = min(v[-1], float(pc.get("joint_end_speed", 0.006)))
        for i in range(len(v) - 2, -1, -1):                 # deceleration limit
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * acc * seg[i]))
        v[0] = min(v[0], max(acc, 0.004))
        for i in range(1, len(v)):                          # acceleration limit
            v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * acc * seg[i - 1]))
        out, s_ = [], 0.0
        while s_ < cum[-1] - 1e-9:
            s_ = min(cum[-1], s_ + max(float(np.interp(s_, cum, v)), 0.003))
            i = int(np.clip(np.searchsorted(cum, s_) - 1, 0, len(P) - 2))
            t = (s_ - cum[i]) / max(seg[i], 1e-12)
            out.append(P[i] + (P[i + 1] - P[i]) * min(t, 1.0))
        return np.array(out)

    def _run_chain(self, chain, label: str, v_fast: float | None = None, settle_tol: float = 0.01,
                   guard: bool = False):
        """Execute a joint path (waypoints after the current posture) with the timed profile, then settle.
        `guard` [v6]: a leaf the camera has not seen can lie in the path of a view move; when its root is about to be
        torn out (predictive leaf guard, `limits.move_guard_frac`) the arm stops, goes back the way it came and None
        is returned (the caller skips this view) - measured: a look move tore a leaf out and ended the episode."""
        sim = self.sim
        pc = self.cfg.get("planning", {})
        vf = float(pc.get("joint_speed_per_step", 0.025)) if v_fast is None else float(v_fast)
        vs = min(vf, float(pc.get("joint_speed_near_fruit", 0.012)))
        q_to = np.asarray(chain[-1], float)
        went = [sim.q_arm().copy()]
        for q in self._joint_trajectory([sim.q_arm()] + [np.asarray(c, float) for c in chain], vf, vs):
            sim.d.ctrl[sim.arm_act] = q
            if not self._sim(1, label):
                return False
            went.append(np.array(q, float))
            if guard and self._leaf_guard("move_guard_frac"):
                self._event(f"leaf root bent {np.degrees(self._root_bend):.0f} deg during the '{label}' move: "
                            f"stop and go back")
                for qb in went[-2::-1]:
                    sim.d.ctrl[sim.arm_act] = qb
                    if not self._sim(1, label):
                        return False
                for _ in range(int(pc.get("settle_steps", 30))):
                    if np.max(np.abs(sim.q_arm() - went[0])) < settle_tol:
                        break
                    if not self._sim(1, label):
                        return False
                self.arm.target_quat = sim.tcp_quat()
                return None
        sim.d.ctrl[sim.arm_act] = q_to
        for _ in range(int(pc.get("settle_steps", 30))):    # let the servos settle
            if np.max(np.abs(sim.q_arm() - q_to)) < settle_tol:
                break
            if not self._sim(1, label):
                return False
        self.arm.target_quat = sim.tcp_quat()
        return True

    def _move_joints(self, q_to, label: str, max_step: float | None = None) -> bool:
        return self._run_chain([q_to], label, v_fast=max_step)

    def _fail(self, reason: str):
        if not self.failure:
            self.failure = reason
            self.fail_phase = str(getattr(self, "_label", ""))
            if self.rec is not None and not self.rec.failure:
                self.rec.failure = reason
            self._event(f"FAIL {reason}")
            self.lamp("fail")
        self.done = True

    # ================================================================== perception
    def _capture(self, per: Perceiver):
        mujoco.mj_forward(self.sim.m, self.sim.d)
        frame = self.cam.capture(segmentation=False)
        per.add_view(frame)
        return frame

    def _self_filter(self):
        """Callable that marks camera points lying on the robot's own vehicle (None on the platform)."""
        return None

    def _perceiver(self) -> Perceiver:
        return Perceiver(self.cfg, self_filter=self._self_filter())

    def _scan(self, around: float | None = None) -> bool:
        """Survey with the wrist camera; returns True if at least one fruit was detected. The survey stops as soon as
        enough fruit has been seen (`planning.scan.early_stop`): the remaining azimuths are only visited when the
        first views show nothing."""
        t0 = time.time()
        sim = self.sim
        sc = self.cfg.get("planning", {}).get("scan", {})
        self.lamp("observe")
        per = self._perceiver()
        with self._compute("views"):
            views = self.planner.scan_views(sim.d.qpos.copy(), self._scan_azimuths())
        if around is not None:                       # start where the fruit is expected (re-survey)
            views.sort(key=lambda v: abs((v[0] - around + np.pi) % (2 * np.pi) - np.pi))
        need = self._scan_need() if bool(sc.get("early_stop", True)) else None
        min_px = 4 * int(self.cfg["perception"].get("min_fruit_pixels", 25))
        taken, extra, res, res_at = 0, 0, None, -1
        for a, q in views:
            mv = self._move_safe(q, "scan")
            if mv is None:
                continue
            if not mv:
                return False
            self._sim(int(sc.get("settle_steps", 3)), "scan")
            self._capture(per)
            taken += 1
            if need is None or len(per.views[-1]["us"]) < min_px and res is None:
                continue
            with self._compute("perception"):
                res, res_at = per.result(), taken
                found = self._scan_accept(merge_fruits([f for f in res.fruits if f.confidence > 0.15]))
            # only CONFIDENT detections end the survey early: weak ones (few points, radius at the bound of the
            # fit) were either leaf tips (36-60 cm from any fruit) or a fruit seen through leaves and located
            # 3-8 cm off (measured on 30 test scenes); with only weak ones the survey goes on [v6]
            sure = [f for f in found if f.confidence >= float(sc.get("stop_confidence", 0.85))]
            if len(sure) >= need:
                break
            if sure:                                    # some fruit, not all: a few more views, then go
                extra += 1
                if extra > int(sc.get("extra_views", 1)):
                    break
        if res is None or res_at != taken:
            with self._compute("perception"):
                res = per.result()
        self.scan_leaf = res.leaf_points
        self.fruits_scan = merge_fruits([f for f in res.fruits if f.confidence > 0.15])
        self.fruits_scan = [f for f in self.fruits_scan if self._scan_keep(f)]
        k = self._scan_known_count()
        if k is not None and len(self.fruits_scan) > k:
            # more detections than fruits that exist: the least confident ones are leaf tips, not fruit
            self.fruits_scan = sorted(self.fruits_scan, key=lambda f: -float(f.confidence))[:k]
        self.timing["scan_s"] = time.time() - t0
        self._event(f"scan: {taken} views, {len(self.fruits_scan)} fruit")
        return len(self.fruits_scan) > 0

    def _scan_azimuths(self):
        return None

    def _scan_need(self) -> int:
        """Number of detected fruits that ends the survey early."""
        return max(1, self._n_targets() - self.n_placed)

    def _scan_known_count(self):
        """Number of fruits that can still be on the plants when it is known (Missions 1-2), else None."""
        return max(1, self._n_targets() - self.n_placed - len(getattr(self, "_given_up_xy", [])))

    def _scan_accept(self, fruits: list) -> list:
        """Fruits of a survey that count for the early stop (Mission 3: only fruit ahead in the own row)."""
        return [f for f in fruits if self._scan_keep(f)]

    def _scan_keep(self, f) -> bool:
        """A re-survey must not return places where a fruit was already picked or given up (Missions 1-2)."""
        return all(np.linalg.norm(f.center[:2] - p) > 0.12 for p in getattr(self, "_closed_xy", []))

    def _acquire_target(self) -> bool:
        """Prepare the next fruit. When no fruit is known any more (the survey stopped at the first one) or the
        remembered fruit is not found in the close view, the arm surveys again - starting at the azimuth where the
        fruit is expected - and tries once more [v6: `next_target_failed` / `perception_lost` ended Mission 2
        episodes with a fruit still on its plant]."""
        base = self.sim.base_pos
        az = getattr(self, "_last_az", None)
        for attempt in range(2):
            if not [f for f in self.fruits_scan if not getattr(f, "_done", False)]:
                if not self._scan(around=az):
                    if not self.done:
                        self._fail("nothing_detected")
                    return False
            if self._prepare_target():
                return True
            if self.failure != "perception_lost" or attempt == 1:
                return False
            lost = self._look_target
            az = float(np.arctan2(lost[1] - base[1], lost[0] - base[0]))
            self._event("target not found in the close view: survey again")
            self.failure, self.done, self._fail_charged = "", False, False
            self.lamp("observe")
            self.fruits_scan = []
        return False

    def _priority(self, fruits):
        """'First' fruit = nearest to the robot base (horizontal), then most confident."""
        base = self.sim.base_pos
        return sorted(fruits, key=lambda f: (round(float(np.hypot(*(f.center[:2] - base[:2]))), 2), -f.confidence))

    def _look_next_view(self, per) -> bool | None:
        """Move the gripper camera to the next reachable close view of the target and capture it.
        True = captured, None = no view left, False = failed while moving (episode over)."""
        sim = self.sim
        for q in self._look_iter:
            mv = self._move_safe(q, "look")
            if mv is None:
                continue
            if not mv:
                return False
            if not self._sim(int(self.cfg.get("planning", {}).get("look", {}).get("settle_steps", 3)), "look"):
                return False
            self._capture(per)
            self._look_q.append(sim.q_arm().copy())
            return True
        return None

    def _look_side(self, coarse):
        """Preferred side of the offset close views (None = no preference; Mission 3: the corridor side)."""
        return None

    def _on_look(self, coarse, fruit, seen: list):
        """Hook: the close view refined the target and saw other fruits (Mission 3 updates its fruit memory)."""

    def _match_fruit(self, fruits: list, coarse):
        """The estimate of the target among the fruits of the close view(s) (None if it is not there)."""
        if not fruits:
            return None, []
        tol = float(self.cfg.get("planning", {}).get("look", {}).get("match_dist", 0.15))
        d = [np.linalg.norm(f.center[:2] - coarse.center[:2]) for f in fruits]
        k = int(np.argmin(d))
        if d[k] > tol:
            return None, []
        return fruits[k], [f for i, f in enumerate(fruits) if i != k and f.n_points > 40]

    def _gt_slot_of(self, center):
        best, bd = -1, 1e9
        for i in self.sim.field.active_slots():
            s = self.sim.field.slots[i]
            if s["harvested"] or s["detached"]:
                continue
            p, _ = self.sim.field.fruit_pose(i, self.sim.d)
            dd = float(np.linalg.norm(p - center))
            if dd < bd:
                best, bd = i, dd
        return best, bd

    # ================================================================== target cycle
    def _prepare_target(self) -> bool:
        """LOOK from the IDEAL gripper pose + window / T2 decision + validation for the next fruit.

        [USER] The gripper camera detects the fruit from the ideal pose: ONE close view on the robot's side of the
        fruit, at the elevation the approach lines start from (the approach then runs along the camera rays of this
        view). The decision is made from that single view; only when it gives no feasible line (fruit hidden, no
        shoulder, no reachable line) the camera moves on to the next view and the estimate is refined."""
        self._residual_used, self._residual_ban = False, False
        sim = self.sim
        remaining = [f for f in self.fruits_scan if not getattr(f, "_done", False)]
        if not remaining:
            return False
        coarse = self._priority(remaining)[0]
        coarse._done = True
        self.rec = TargetRecord()
        self._shield_replans = 0
        self.rec.t_start = float(getattr(self, "_t_session", 0.0))
        self.records.append(self.rec)
        self.lamp("observe")
        lk = self.cfg.get("planning", {}).get("look", {})
        n_max = int(lk.get("n_views", 3))
        per = self._perceiver()
        self._look_per = per
        self._look_target = np.asarray(coarse.center, float).copy()
        self._look_q = []                       # arm posture of every captured view (same order as per.views)
        with self._compute("views"):
            self._look_iter = iter(self.planner.look_views(coarse.center, sim.d.qpos.copy(), want=n_max + 4,
                                                           lateral_pref=self._look_side(coarse)))
        from pinesim.vision.stargaze_cam import obstacle_points
        from pinesim.vision.perception import shoulder_in_view
        base = sim.base_pos
        fruit, leaf, others, cloud, obs = None, np.zeros((0, 3)), [], np.zeros((0, 3)), (np.zeros((0, 3)), np.zeros(0))
        dec, plan, why = None, None, "lost_in_look"

        def try_view(vi):
            """Decision on view vi (+ validation by the arm planner); returns (decision, plan, reason)."""
            with self._compute("decision"):
                d_ = decide(fruit, per.views, obs[0], obs[1], self.cfg, policy=self.t2_policy, rng=self.planner.rng,
                            view_order=[vi])
            if not d_.ok:
                return d_, None, "window_" + d_.reason
            if np.max(np.abs(sim.q_arm() - self._look_q[vi])) > 0.02:
                mv = self._move_safe(self._look_q[vi], "look")
                if mv is False:
                    return d_, None, "failed"
                if mv is None:
                    return d_, None, "decision_view_unreachable"
            with self._compute("plan_approach"):
                pl, w = self.planner.plan_approach(fruit, d_, cloud, others, sim.d.qpos.copy(), obstacles=obs)
            return d_, pl, w

        while plan is None and len(self._look_q) < n_max:
            got = self._look_next_view(per)
            if got is False or self.done:
                return False
            if got is None:
                break
            with self._compute("perception"):
                res = per.result()
                fruit, seen = self._match_fruit(res.fruits, coarse)
                if fruit is not None:
                    leaf = res.leaf_points
                    # every other fruit still on its plant is an obstacle of the approach (before v6 the fruits
                    # marked "not this turn" were left out: the jaw pushed a neighbour fruit during an approach)
                    others = merge_fruits(seen + [f for f in self.fruits_scan if f is not coarse and self._fruit_present(f)
                                                  and np.linalg.norm(f.center[:2] - fruit.center[:2]) > 0.10])
                    cloud = np.vstack([leaf, self.scan_leaf]) if len(self.scan_leaf) else leaf
                    obs = obstacle_points(fruit, cloud, self.cfg, others)
            if fruit is None:
                why = "lost_in_look"
                continue
            self._on_look(coarse, fruit, seen)
            d_, plan, why = try_view(len(self._look_q) - 1)       # the view the gripper is in now
            dec = d_ if (d_.ok or dec is None) else dec
            if self.done:
                return False
        if plan is None and fruit is not None and len(self._look_q) > 1:
            # every view again with the fused estimate: most shoulder pixels first, robot side preferred
            def view_key(i):
                sh = shoulder_in_view(per.views[i], fruit)
                toward = np.r_[base[:2] - fruit.center[:2], 0.0]
                vd = fruit.center - per.views[i]["t"]
                facing = float(np.dot(-vd / (np.linalg.norm(vd) + 1e-9), toward / (np.linalg.norm(toward) + 1e-9)))
                return -sh["n_shoulder"] * (1.0 + 0.5 * facing)
            for vi in sorted(range(len(self._look_q)), key=view_key)[:3]:
                d_, plan, why = try_view(vi)
                dec = d_ if (d_.ok or dec is None) else dec
                if self.done:
                    return False
                if plan is not None:
                    break
        self.rec.n_views = len(self._look_q)
        if fruit is None:
            self.rec.plan_reason = "lost_in_look"
            self._event(f"target not found in {len(self._look_q)} close view(s)")
            self._fail("perception_lost")
            return False
        self.decision = dec
        slot, err = self._gt_slot_of(fruit.center)
        self.rec.slot_gt, self.rec.est_error, self.rec.plan_reason = slot, err, why
        self._send_overlay(fruit, dec, plan)
        if plan is None:
            self._fail("plan_" + why)
            return False
        self.plan = plan
        self.target_slot = slot
        self.rec.face, self.rec.clearance = plan.info["face"], plan.info["clearance"]
        self.rec.t_decided = self.now()
        self.leaf_cloud = cloud
        self._plan_ctx = dict(fruit=fruit, cloud=cloud, others=others, decision=dec, obstacles=obs,
                              failed_starts=[], retries=0)
        self._set_obs_points(plan)
        # the gripper stays CLOSED until the tip reaches T2 [USER v5]
        sim.set_gripper(self.cfg["gripper"]["min_opening"])
        self._event(f"target slot {slot}: est err {err * 100:.1f} cm, {self.rec.n_views} view(s), window {dec.method} "
                    f"rank {plan.info['rank']} ({len(dec.candidates)} T2 candidates, shoulder {dec.shoulder.get('ratio', 0):.2f}), "
                    f"clearance {plan.info['clearance'] * 100:.1f} cm, advance {plan.info['advance'] * 100:.1f} cm")
        self.lamp("action")
        self._begin_phase("approach", plan.approach)
        return True

    def _set_obs_points(self, plan):
        # observation obstacles: the obstacle cloud on a coarser grid (speed; 4 nearest points are observed)
        op, orad = plan.obstacles
        if len(op):
            from pinesim.vision.perception import voxel_down
            op = voxel_down(op, float(self.cfg.get("policy", {}).get("obs_voxel", 0.025)))
            orad = np.full(len(op), float(self.cfg["perception"].get("leaf_point_radius", 0.015)) + 0.005)
        self.obs_pts, self.obs_rad = op, orad

    def _send_overlay(self, fruit, dec, plan):
        """Recorder hook: shoulder, window, bounding box, T2 candidates, chosen T1 / T2 (tips) - drawn in the
        wrist-camera view of the Meshcat HTML."""
        if self.overlay_callback is None:
            return
        sh = dec.shoulder if dec is not None else {}
        X = sh.get("X", np.zeros((0, 3))) if sh else np.zeros((0, 3))
        if len(X) > 400:
            X = X[np.random.default_rng(0).choice(len(X), 400, replace=False)]
        t1 = t2 = None
        if plan is not None:
            t1, t2 = plan.line.tip_start, plan.line.tip_target
        elif dec is not None and dec.ok and dec.candidates:
            t1, t2 = dec.candidates[0].t1, dec.candidates[0].t2
        self.overlay_callback(t1=t1, t2=t2, fruit_center=fruit.center, fruit_points=fruit.points,
                              shoulder=X, window=None if dec is None else dec.window_corners,
                              bbox=None if dec is None else dec.bbox_corners,
                              candidates=None if dec is None or not dec.candidates else np.array([c.t2 for c in dec.candidates]),
                              info=None if dec is None else dict(method=dec.method, n_candidates=len(dec.candidates),
                                                                 shoulder_ratio=float(sh.get("ratio", 0.0)) if sh else 0.0,
                                                                 shoulder_px=int(sh.get("n_shoulder", 0)) if sh else 0,
                                                                 clearance=float(plan.info["clearance"]) if plan else None,
                                                                 votes=float(plan.info.get("votes", 0)) if plan else None))

    def _begin_phase(self, name: str, route: Route):
        self._line_best, self._line_best_step = np.inf, self.step_count
        self._prog = [-1, np.inf, self.step_count]      # watchdog: best route index, best distance, step of last progress
        self._residual_off = False
        self._phase_shields = 0
        if self.phase != PHASE_ID[name]:
            self._q_hist = []
        if name == "approach":
            self._bend_allow = 0.0
            # RL control starts from the ACTUAL orientation (the orientation target of the arm command used to carry
            # over from the previous episode -> results depended on which episode ran before)
            self.arm.target_quat = self.sim.tcp_quat().copy()
        self.phase = PHASE_ID[name]
        self.route = route
        route.idx = 0
        self._seg = (self.sim.tcp_pos if name == "approach" else route.start, route.end)
        self._prev_dist = float(np.linalg.norm(route.end - self.sim.tcp_pos))
        self._event(f"phase {name}")

    def _verify_at_t1(self):
        """[USER] The gripper camera detects the fruit again with the gripper in its IDEAL pose: jaw tips at T1, tool
        on the approach line, 20 cm in front of the fruit and outside the leaves. The new estimate corrects T2 and
        the grip point (small shifts only; a large disagreement keeps the plan), so a fruit estimated a little off -
        or pushed by the arm on the way in - is not hit by a jaw during the advance."""
        vc = self.cfg.get("harvest", {}).get("t1_verify", {})
        if not bool(vc.get("enabled", True)) or self.plan is None:
            return
        sim, line = self.sim, self.plan.line
        sim.hold_arm()
        if not self._sim(int(vc.get("settle_steps", 3)), "verify_t1"):
            return
        per = self._perceiver()
        self._capture(per)
        f_old = self.plan.fruit
        with self._compute("perception"):
            res = per.result()
            cand = [f for f in res.fruits if f.n_points >= int(vc.get("min_points", 60))]
            f_new = min(cand, key=lambda f: np.linalg.norm(f.center - f_old.center)) if cand else None
        if f_new is None or np.linalg.norm(f_new.center - f_old.center) > float(vc.get("max_disagreement", 0.06)):
            self._event("T1 check: fruit not confirmed by the gripper camera (plan kept)")
            return
        delta = f_new.center - f_old.center
        a = np.asarray(line.approach, float)
        lat = delta - float(delta @ a) * a                    # across the approach direction (well observed)
        shift = lat + float(vc.get("depth_weight", 0.5)) * float(delta @ a) * a
        n = float(np.linalg.norm(shift))
        self.rec.t1_shift_cm = 100.0 * n
        if n < float(vc.get("min_shift", 0.004)):
            self._event(f"T1 check: estimate confirmed ({n * 1000:.0f} mm)")
            return
        shift *= min(1.0, float(vc.get("max_shift", 0.03)) / n)
        chk = self.arm.checker
        chk.sync_from(sim.d)
        P1, P2, P3 = np.array(line.start, float), np.array(line.target, float) + shift, np.array(line.advance_end, float) + shift
        with self._compute("route_check"):
            wps = chk.waypoints(P1, P2)
            ok, q2, _ = chk.rollout_q(sim.d.qpos.copy(), wps, [line.quat] * len(wps))
            if ok:
                adv = chk.waypoints(P2, P3)
                q2o = q2.copy()
                q2o[sim.finger_qadr] = sim.hinge_for_gap(self.cfg["gripper"]["max_opening"])
                ok, _, _ = chk.rollout_q(q2o, adv, [line.quat] * len(adv))
        if not ok:
            self._event(f"T1 check: shift {n * 100:.1f} cm not reachable (plan kept)")
            return
        import copy as _copy
        fr = _copy.copy(f_old)
        fr.center = f_old.center + shift
        fr.top_z, fr.bottom_z = f_old.top_z + shift[2], f_old.bottom_z + shift[2]
        self.plan.fruit = fr
        if self._plan_ctx is not None:
            self._plan_ctx["fruit"] = fr
        line.target, line.advance_end, line.tip_target = P2, P3, np.array(line.tip_target, float) + shift
        if getattr(line, "grip_end", None) is not None:
            line.grip_end = np.array(line.grip_end, float) + shift
        self.plan.line_route = Route(np.vstack([P1[None, :], wps]), [line.quat] * (len(wps) + 1), "line")
        self._event(f"T1 check: gripper camera moved T2 by {np.linalg.norm(shift) * 100:.1f} cm")

    # ---------------------------------------------------------------- speed governor near fruit
    def _tool_fruit_distance(self, include_target: bool = True) -> float:
        """Distance (m) from the jaws (tip centre, both jaw tips, grip point) to the nearest perceived fruit / crown."""
        hp, hr = self._perceived_fruit_points(include_target)
        if not len(hp):
            return np.inf
        sim = self.sim
        R = sim.tcp_rot()
        half = 0.5 * sim.gripper_gap() + 0.012
        tip = sim.tip_pos
        pts = np.array([tip, tip + R[:, 0] * half, tip - R[:, 0] * half, sim.tcp_pos + R[:, 2] * sim.grip_offset])
        d = np.linalg.norm(pts[:, None, :] - hp[None, :, :], axis=2) - hr[None, :]
        return float(d.min())

    def _governed(self, a: np.ndarray) -> np.ndarray:
        """Translation command limited near perceived fruit [USER: never hit a fruit at speed]: at most `near_speed`
        (m/s) at the fruit, rising linearly to the free speed `free_dist` away. Applies to the reference action and
        to the composed (reference + residual) action; the carried fruit itself is excluded."""
        gv = self.cfg.get("policy", {}).get("fruit_governor", {})
        if not bool(gv.get("enabled", True)):
            return a
        d = self._tool_fruit_distance(include_target=self.phase <= 1)
        self._fruit_dist = d
        if not np.isfinite(d):
            return a
        d0, d1 = float(gv.get("near_dist", 0.02)), float(gv.get("free_dist", 0.25))
        v0, v1 = float(gv.get("near_speed", 0.06)), float(gv.get("free_speed", 2.5))
        k = self._carry_gain()
        if k > 1.0:
            v1 = float(gv.get("carry_free_speed", v1))
        v = v0 + (v1 - v0) * float(np.clip((d - d0) / max(d1 - d0, 1e-6), 0.0, 1.0))
        # a Cartesian action a moves the TARGET a * max_pos_delta ahead of the hand; the joint servos (kp / kv =
        # servo_rate 1/s) then follow at a * max_pos_delta * servo_rate (measured: a = 1 -> 0.5 m/s)
        a_max = v / (self.arm.max_pos_delta * k * float(self.cfg["control"].get("servo_rate", 10.0)))
        n = float(np.linalg.norm(a[:3]))
        if n > a_max:
            a = np.array(a, float)
            a[:3] *= a_max / n
        return a

    def _local_seg(self):
        """Piece of the active route around the current waypoint index. For a straight route (the Stargaze line,
        as in reference Mission 4) this is the whole segment; for curved routes (joint-space approach / transport
        converted to TCP waypoints) the chord start->end is NOT the path, so the cross-track term and observation
        use the local piece instead (the chord version penalised the policy for following the planned route)."""
        r = self.route
        n = len(r.wps)
        if n < 3:
            return self._seg
        i1 = min(r.idx + 3, n - 1)
        i0 = min(r.idx, i1 - 1)
        a, b = r.wps[i0], r.wps[i1]
        if np.linalg.norm(b - a) < 1e-6:
            return self._seg
        return a, b

    # ---------------------------------------------------------------- scripted harvest
    def _jaw_fruit_force(self, slot: int):
        """Contact force (N) of the left / right jaw (whole finger: pad, tip, back) on the target fruit, and the
        side (+1 / -1 along the tool x axis) of each jaw. Real robot: jaw motor current / finger force sensing."""
        sim, m, d, f6 = self.sim, self.sim.m, self.sim.d, np.zeros(6)
        s = sim.field.slots[slot]
        tgt = {s["fruit_geom"], s["neck_geom"]}
        fb = sim.arm_body_ids[-2:]
        out = [0.0, 0.0]
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in tgt or g2 in tgt:
                o = g2 if g1 in tgt else g1
                b = int(m.geom_bodyid[o])
                if b in fb:
                    mujoco.mj_contactForce(m, d, i, f6)
                    out[fb.index(b)] += abs(float(f6[0]))
        x = sim.tcp_rot()[:, 0]
        side = [float(np.sign((d.xpos[b] - sim.tcp_pos) @ x)) or 1.0 for b in fb]
        return out, side

    def _open_and_advance(self) -> str | None:
        """Tip at T2: OPEN the jaws, then move straight on along the line until the fruit axis is between the pads
        [USER v5: buka gripper dan majukan gripper beberapa cm lalu grip nanas]. The advance is slow and force
        guarded [USER: the gripper must not knock the fruit off]: a jaw that touches the fruit stops the motion, the
        hand backs off a few millimetres, shifts toward the touching jaw (centring the fruit in the U) and goes on."""
        sim, hc, slot = self.sim, self.cfg.get("harvest", {}), self.target_slot
        # align: the RL line phase ends within grasp_pos_tol / grasp_orient_deg; the long U-jaws need the planned
        # pose exactly (a 10 deg yaw error swings the jaw tips 3 cm sideways over the advance: measured fruit_hit)
        P2, q2 = np.array(self.plan.line.target, float), self.plan.line.quat
        ms = float(hc.get("align_joint_step", 0.03))
        for k in range(int(hc.get("align_steps", 40))):
            self.arm.servo_to(P2, q2, max_joint_step=ms)
            if not self._sim(1, "align"):
                return self.failure
            if np.linalg.norm(sim.tcp_pos - P2) < 0.004 and quat_angle(sim.tcp_quat(), q2) < np.radians(1.5):
                break
        sim.set_gripper(self.cfg["gripper"]["max_opening"])
        if not self._sim(int(hc.get("open_steps", 15)), "open"):
            return self.failure
        P3, q = np.array(self.plan.line.advance_end, float), self.plan.line.quat
        P3 = P3 + self.plan.line.approach * float(hc.get("advance_margin", 0.0))
        step = float(hc.get("advance_step", 0.004))
        touch = float(hc.get("advance_touch_force", 12.0))
        max_corr = int(hc.get("advance_corrections", 3))
        hit = 1.5 * float(self.lim.get("target_touch_force", 40.0))
        P0 = sim.tcp_pos.copy()
        L = float(np.linalg.norm(P3 - P0))
        u = (P3 - P0) / max(L, 1e-9)
        off = np.zeros(3)                                # lateral correction of the line (jaw touched the fruit)
        s_ref, best, stall, n_corr, k_settle = 0.0, np.inf, 0, 0, 0
        for k in range(int(np.ceil(L / step)) * 3 + 120):
            s_ref = min(L, s_ref + step)
            self.arm.servo_to(P0 + off + u * s_ref, q, max_joint_step=ms)
            if not self._sim(1, "advance"):
                return self.failure
            (fl, fr), side = self._jaw_fruit_force(slot)
            c = sim.classify_contacts(slot)
            if c["fruit_target"] > hit:
                self._event(f"advance: jaw pushes the fruit ({c['fruit_target']:.0f} N)")
                return "fruit_hit"
            n = float(np.linalg.norm(P3 + off - sim.tcp_pos))
            if self._leaf_guard("line_guard_frac"):
                # an open jaw is pushing a leaf toward being torn out: grip from here when the fruit is already
                # between the pads, otherwise back out and take another line [v6: leaf_uprooted in the advance]
                if n < float(hc.get("advance_leaf_grip_dist", 0.04)):
                    self._event(f"advance: leaf root bent {np.degrees(self._root_bend):.0f} deg, gripping {n * 100:.1f} cm early")
                    break
                self._event(f"advance: leaf root bent {np.degrees(self._root_bend):.0f} deg {n * 100:.1f} cm before the grip point")
                return "__leaf_block__"
            if max(fl, fr) > touch and n > 0.02:
                if n_corr >= max_corr:
                    self._event(f"advance: jaw still touches the fruit after {n_corr} corrections, gripping {n * 100:.1f} cm early")
                    break
                n_corr += 1
                sgn = side[0] if fl >= fr else side[1]
                x = sim.tcp_rot()[:, 0]
                off = off + sgn * x * float(hc.get("advance_correct_shift", 0.008))
                s_ref = max(0.0, s_ref - float(hc.get("advance_backoff", 0.012)))
                self._event(f"advance: {'left' if fl >= fr else 'right'} jaw touches the fruit ({max(fl, fr):.0f} N): "
                            f"back off and shift {sgn * float(hc.get('advance_correct_shift', 0.008)) * 1000:+.0f} mm")
                for _ in range(8):                        # back off and re-centre before going on
                    self.arm.servo_to(P0 + off + u * s_ref, q, max_joint_step=ms)
                    if not self._sim(1, "advance"):
                        return self.failure
                best, stall = np.inf, 0
                continue
            if s_ref >= L:
                k_settle += 1
                if n < 0.006 or k_settle > 40:
                    break
            if n < best - 0.001:
                best, stall = n, 0
            else:
                stall += 1
                if stall > 40:                        # blocked (leaves / fruit): grip from here
                    self._event(f"advance stalled {n * 100:.1f} cm before the grip point")
                    break
        # [v6] centre the pads on the fruit's widest section (jaws astride the fruit: move along the fruit axis)
        ge = getattr(self.plan.line, "grip_end", None)
        if ge is not None:
            dv = np.array(ge, float) - np.array(self.plan.line.advance_end, float)
            Lc = float(np.linalg.norm(dv))
            if Lc > 0.003:
                Pc = sim.tcp_pos.copy()
                cs = float(hc.get("centre_step", 0.002))
                for k in range(int(np.ceil(Lc / cs)) + 25):
                    self.arm.servo_to(Pc + dv * min(1.0, (k + 1) * cs / Lc), q, max_joint_step=ms)
                    if not self._sim(1, "advance"):
                        return self.failure
                    c = sim.classify_contacts(slot)
                    if c["fruit_target"] > hit:
                        self._event(f"centring: jaw pushes the fruit ({c['fruit_target']:.0f} N)")
                        return "fruit_hit"
                    if (k + 1) * cs >= Lc and np.linalg.norm(Pc + dv - sim.tcp_pos) < 0.004:
                        break
                self.rec.grip_drop_cm = float(100 * Lc)
        self.rec.advance_cm = float(100 * float(np.linalg.norm(self.plan.line.advance_end - self.plan.line.target)))
        self.rec.advance_corrections = n_corr
        return None

    def _pad_contact_offset(self, slot: int):
        """Where the fruit touches the flat pads, measured across the pad (tool y, m from the pad centre line; the
        pads are 6 cm wide there). A convex fruit touches a flat pad at its widest section if that lies inside the
        pad, otherwise at the pad edge nearest to it - so this is the position of the fruit's widest section
        relative to the pads. Real gripper: tactile pads (the code already uses the pad contact points to find the
        fruit axis for the twist)."""
        sim = self.sim
        pl, pr = sim.pad_contact_points(slot)
        if pl is None or pr is None:
            return None
        y = sim.tcp_rot()[:, 1]
        return float((0.5 * (pl + pr) - sim.tcp_pos) @ y)

    def _regrip_on_widest(self) -> str | None:
        """[v6] Closed-loop grip height. The fruit centre height is only an estimate (its bottom is hidden by leaves):
        pads closed above the widest section hold a taper and the fruit slides down and out when carried; pads closed
        too far below it squeeze the fruit up and out and close on the stalk (both measured). After closing, the pad
        contact position tells where the widest section is; when it is not `grip_equator_offset` (+- tolerance) above
        the pad centre line, the jaws open a little, the hand moves along the pads and closes again."""
        sim, hc, slot = self.sim, self.cfg.get("harvest", {}), self.target_slot
        rg = hc.get("regrip", {})
        if not bool(rg.get("enabled", True)):
            return None
        tol, e_t = float(rg.get("tolerance", 0.008)), float(rg.get("equator_offset", 0.008))
        q = self.plan.line.quat
        fmin = float(hc.get("grasp_min_pad_force", 2.0))

        def shift_grip(shift: float, y) -> str | None:
            """Open a little, move the hand `shift` along the pads, close again."""
            sim.set_gripper(min(self.cfg["gripper"]["max_opening"], sim.gripper_gap() + float(rg.get("release_gap", 0.012))))
            if not self._sim(int(rg.get("release_steps", 10)), "regrip"):
                return self.failure
            P0 = sim.tcp_pos.copy()
            cs = float(hc.get("centre_step", 0.002))
            for k in range(int(np.ceil(abs(shift) / cs)) + 12):
                self.arm.servo_to(P0 + y * shift * min(1.0, (k + 1) * cs / abs(shift)), q, max_joint_step=0.03)
                if not self._sim(1, "regrip"):
                    return self.failure
            sim.hold_arm()
            sim.set_gripper(self.cfg["gripper"]["min_opening"])
            if not self._sim(int(rg.get("close_steps", 25)), "regrip"):
                return self.failure
            c = sim.classify_contacts(slot)
            if not (c["pad_left"] > fmin and c["pad_right"] > fmin):
                return "grasp_miss"
            return None

        prev, total = None, 0.0
        for n in range(int(rg.get("max_tries", 2)) + 1):
            e = self._pad_contact_offset(slot)
            if e is None:
                return None
            y = sim.tcp_rot()[:, 1]
            up = 1.0 if float(y @ np.asarray(self.plan.fruit.axis, float)) >= 0 else -1.0
            err = e - up * e_t                     # > 0: the widest section is further along +y than wanted
            if prev is not None and abs(err) > abs(prev) - float(rg.get("min_gain", 0.003)):
                # the correction must converge: when moving the hand did not bring the contact toward the pad centre
                # line the reading is not the widest section (hinged jaws on a large / tilted fruit touch it near
                # the pad tip) - measured: -1.8 cm -> hand -1.7 cm -> -2.8 cm -> hand -2.8 cm left the fruit centre
                # 3.8 cm off the pads, and it slipped out when the stem snapped. Back to the planned grip height.
                self._event(f"re-grip does not converge ({prev * 100:+.1f} -> {err * 100:+.1f} cm): back to the planned grip height")
                return shift_grip(-total, y) if abs(total) > 1e-4 else None
            if abs(err) <= tol or n >= int(rg.get("max_tries", 2)):
                return None
            shift = float(np.clip(err, -float(rg.get("max_shift", 0.03)), float(rg.get("max_shift", 0.03))))
            self._event(f"re-grip: widest section {e * 100:+.1f} cm from the pad centre line -> hand {shift * 100:+.1f} cm")
            fail = shift_grip(shift, y)
            if fail is not None:
                return fail
            prev, total = err, total + shift
            self.rec.regrips = n + 1
        return None

    def _twist_to(self, pivot, gp, q0, sign: float, th: float, ms: float | None = None, label: str = "twist") -> bool:
        """One control step of the hand toward the twist angle th (rad) about the fruit axis through `pivot`."""
        Rz = quat_axis(self._twist_u, sign * th)
        Rm = np.zeros(9)
        mujoco.mju_quat2Mat(Rm, Rz)
        self.arm.servo_to(pivot + Rm.reshape(3, 3) @ (gp - pivot), quat_mul(Rz, q0), max_joint_step=ms)
        return self._sim(1, label)

    def _grasp_and_twist(self) -> tuple[float, str | None]:
        """Grip, then TWIST the fruit about its axis until the stem snaps. Twisting swings the whole hand around the
        fruit through the leaves, so (version 6):
          * the direction that sweeps fewer perceived leaves and is feasible further is tried first; when leaves stop
            it (leaf load guard) the hand turns back and twists the OTHER way instead of pulling the plant,
          * after the stem snapped the hand turns back to the approach orientation (through the space it has just
            cleared) so that the pull-out runs along the validated, leaf-free line T2 -> T1."""
        sim, fld, rec = self.sim, self.sim.field, self.rec
        slot = self.target_slot
        hc = self.cfg.get("harvest", {})
        fail = self._open_and_advance()
        if fail is not None:
            return 0.0, fail
        sim.hold_arm()
        sim.set_gripper(self.cfg["gripper"]["min_opening"])
        if not self._sim(int(hc.get("close_steps", 40)), "grasp"):
            return 0.0, self.failure
        c = sim.classify_contacts(slot)
        fmin = float(hc.get("grasp_min_pad_force", 2.0))
        if not (c["pad_left"] > fmin and c["pad_right"] > fmin):
            return 0.0, "grasp_miss"
        fail = self._regrip_on_widest()
        if fail is not None:
            return 0.0, fail
        rec.grasped = True
        rec.t_grasped = self.now()
        bonus = self._w("grasp_bonus", 10.0)
        fld.record_rest_load(slot, sim.d)
        gp = sim.tcp_pos
        # fruit axis = midpoint between where the two fingers touch the fruit (finger sensing), not the estimate
        pl, pr = sim.pad_contact_points(slot)
        if pl is None or pr is None:
            pl = np.mean([sim.d.geom_xpos[g] for g in sim.pad_left], axis=0)
            pr = np.mean([sim.d.geom_xpos[g] for g in sim.pad_right], axis=0)
        pivot = 0.5 * (pl + pr)
        self._twist_axis_xy = pivot[:2]
        self._twist_pivot = pivot
        self._twist_u = np.asarray(getattr(self.plan.fruit, "axis", np.array([0.0, 0.0, 1.0])), float)
        q0 = sim.tcp_quat()
        step = np.radians(float(hc.get("twist_step_deg", 3.0)))
        back = np.radians(float(hc.get("untwist_step_deg", 2.4)))
        tmax = np.radians(float(hc.get("twist_max_deg", 120.0)))
        if str(hc.get("method", "twist")) == "pluck":
            broke, fail = self._pluck(pivot, gp, q0)
            if fail is not None:
                return bonus, fail
            if broke:
                return bonus + self._w("detach_bonus", 10.0), None
            self._event("pluck did not take the fruit off: twisting instead")
            self._twist_u = np.asarray(getattr(self.plan.fruit, "axis", np.array([0.0, 0.0, 1.0])), float)        
        with self._compute("twist_plan"):
            options = self._twist_options(gp, q0, tmax, step)
        if not options:
            self._event("twist infeasible (reach / neighbour fruit in the swing) -> pull only")
            rec.twist_deg = 0.0
            if bool(hc.get("pull_up", True)):
                fail = self._pull_up()
                if fail is not None:
                    return bonus, fail
                if rec.detached:
                    bonus += self._w("detach_bonus", 10.0)
            return bonus, None
        guard = float(self.lim.get("twist_guard_frac", self.lim.get("line_guard_frac", 0.85))) * float(self.lim.get("leaf_uproot_rad", 1.15))
        f_abort = float(hc.get("twist_leaf_force_abort", 200.0))
        broke = False
        for n_dir, (sign, th_max) in enumerate(options[:2]):
            th, why = 0.0, "limit"
            while th < th_max:
                th = min(th_max, th + step)
                if not self._twist_to(pivot, gp, q0, sign, th):
                    return bonus, self.failure
                brk, fe, te = fld.check_break(slot, sim.d)
                rec.stem_force, rec.stem_torque = max(rec.stem_force, fe), max(rec.stem_torque, te)
                if brk:
                    fld.detach(slot, sim.d)
                    rec.detached, rec.twist_deg = True, float(np.degrees(th))
                    self._event(f"stem broke at twist {sign * np.degrees(th):+.0f} deg (F {fe:.0f} N, T {te:.2f} N m)")
                    bonus += self._w("detach_bonus", 10.0)
                    broke = True
                    break
                c = sim.classify_contacts(slot)
                if c["pad_left"] < 0.5 and c["pad_right"] < 0.5:
                    return bonus, "grip_slip"
                if self._leaf_guard("twist_guard_frac"):
                    why = f"leaf root bent {np.degrees(self._root_bend):.0f} deg"
                    break
                if c["leaf"] > f_abort:
                    why = f"leaf force {c['leaf']:.0f} N"
                    break
            rec.twist_deg = max(rec.twist_deg, float(np.degrees(th)))
            if not broke:
                self._event(f"twist {sign * np.degrees(th):+.0f} deg stopped ({why}), stem not broken"
                            + (": trying the other direction" if n_dir == 0 and len(options) > 1 else ""))
            # back to the approach orientation: the pull-out / the other direction start from the line pose
            if broke and not bool(hc.get("untwist_after_break", True)):
                break
            if broke:
                for _ in range(int(hc.get("snap_settle_steps", 5))):       # let the snap ring out before moving
                    if not self._twist_to(pivot, gp, q0, sign, th):
                        return bonus, self.failure
                # lift the picked fruit off the stalk first: turned back at the same height its bottom ran into the
                # stalk top, which pushed it up through the pads (measured: fruit climbed 4 cm, pads then closed on
                # the stalk -> plant_hit / fruit dropped)
                lift = self._twist_u * float(hc.get("untwist_lift", 0.02))
                n_l = int(hc.get("untwist_lift_steps", 10))
                for k in range(n_l):
                    if not self._twist_to(pivot + lift * (k + 1) / n_l, gp + lift * (k + 1) / n_l, q0, sign, th):
                        return bonus, self.failure
                pivot, gp = pivot + lift, gp + lift
            th0, ramp = th, np.radians(float(hc.get("untwist_ramp_deg", 25.0)))
            while th > 0.0:
                # smooth start and stop: the fruit hangs between two flat pads and swings about the pad axis when
                # the hand stops abruptly (measured: it rotated in the grip at the end of a 150 deg/s turn-back)
                f = float(np.clip(th / ramp, 0.15, 1.0) * np.clip((th0 - th) / ramp + 0.3, 0.3, 1.0))
                th = max(0.0, th - back * f)
                if not self._twist_to(pivot, gp, q0, sign, th):
                    return bonus, self.failure
                if broke:
                    cf = self._carry_check()
                    if cf:
                        return bonus, cf
                else:
                    brk, fe, te = fld.check_break(slot, sim.d)
                    if brk:                              # (rare) the release swing snaps it
                        fld.detach(slot, sim.d)
                        rec.detached, broke = True, True
                        self._event(f"stem broke while turning back (F {fe:.0f} N, T {te:.2f} N m)")
                        bonus += self._w("detach_bonus", 10.0)
            for _ in range(6):                           # settle on the line pose
                if not self._twist_to(pivot, gp, q0, sign, 0.0):
                    return bonus, self.failure
            if broke:
                break
        if not broke and bool(hc.get("pull_up", True)):
            # the unsuccessful twist moved the fruit in the pads (measured: widest section 2.7 cm off the pad centre
            # line, i.e. held by the pad edge on the taper -> squeezed out while it was carried): grip again on the
            # widest section while the fruit still sits on its stalk
            fail = self._regrip_on_widest()
            if fail is not None:
                return bonus, fail
            fail = self._pull_up()
            if fail is not None:
                return bonus, fail
            if rec.detached:
                bonus += self._w("detach_bonus", 10.0)
        return bonus, None          # not broken yet: the pull-out (retreat) may still break it

    def _pluck_options(self, base, gp, q0, ang) -> list:
        """[(tilt axis (unit, world), feasible angle)] for plucking, best first. The fruit is tilted about a horizontal
        axis through its base (the top of the stalk). Four directions are compared: about the jaw closing axis and
        about the axis square to it, each both ways. Feasible = the same checks as for the twist (IK, contacts,
        clearance to other fruits and to the stalk) plus the crown of the tilting fruit staying clear of the other
        fruits the camera has seen."""
        hc = self.cfg.get("harvest", {})
        pc = hc.get("pluck", {})
        u = self._twist_u
        x = self.sim.tcp_rot()[:, 0]
        x = x - np.dot(x, u) * u
        x = x / (np.linalg.norm(x) + 1e-12)
        op, orad = self._perceived_fruit_points(include_target=False)
        f = self.plan.fruit
        tip0 = base + u * (2.0 * float(getattr(f, "half_height", 0.09)) + float(pc.get("crown_length", 0.15)))
        margin = float(pc.get("fruit_margin", 0.03))
        out = []
        for a in (x, np.cross(u, x)):
            for sign, th_ok, key in self._twist_options(gp, q0, ang, 0.0, u=a, pivot=base, with_key=True):
                if len(op):                              # crown tip of the tilting fruit vs the other fruits
                    th, safe = 0.0, 0.0
                    while th < th_ok:
                        th = min(th_ok, th + np.radians(5.0))
                        Rm = np.zeros(9)
                        mujoco.mju_quat2Mat(Rm, quat_axis(a, sign * th))
                        tip = base + Rm.reshape(3, 3) @ (tip0 - base)
                        if np.min(np.linalg.norm(op - tip[None, :], axis=1) - orad) < margin:
                            break
                        safe = th
                    th_ok = safe
                if th_ok >= np.radians(float(pc.get("min_feasible_deg", 40.0))):
                    out.append((a * sign, th_ok, (min(th_ok, ang),) + tuple(key[1:])))
        out.sort(key=lambda o: o[2], reverse=True)
        return [(o[0], o[1]) for o in out]

    def _pluck(self, grip, gp, q0) -> tuple[bool, str | None]:
        """PLUCK the gripped fruit like a tea shoot [USER]: tilt it about its base (the top of the stalk) by more than
        45 deg while pulling it up along its own axis. The stem kinks and tears off at the plant's own bend angle
        (plant.stem.break_bend_deg_range). Returns (stem broken, failure). Afterwards the hand is back in the grip
        orientation, lifted clear of the stalk when the fruit came off - the same state the twist leaves behind."""
        sim, fld, rec, slot = self.sim, self.sim.field, self.rec, self.target_slot
        hc = self.cfg.get("harvest", {})
        pc = hc.get("pluck", {})
        u = np.asarray(getattr(self.plan.fruit, "axis", np.array([0.0, 0.0, 1.0])), float)
        f = self.plan.fruit
        # base of the fruit = top of the stalk, on the fruit axis below the grip (camera estimate of the fruit length)
        base = grip + u * (float(np.dot(f.center - grip, u)) - float(getattr(f, "half_height", 0.09)))
        ang = np.radians(float(pc.get("max_deg", 60.0)))
        step = np.radians(float(pc.get("step_deg", 1.0)))
        back = np.radians(float(pc.get("back_step_deg", 2.0)))
        lift_tot = float(pc.get("lift", 0.015))
        self._twist_u = u
        with self._compute("twist_plan"):
            options = self._pluck_options(base, gp, q0, ang)
        if not options:
            self._event("pluck infeasible (reach / neighbour fruit / stalk in the swing)")
            return False, None
        f_abort = float(hc.get("twist_leaf_force_abort", 200.0))
        broke = False
        for n_dir, (axis, th_max) in enumerate(options[:2]):
            self._twist_u = axis
            th, why = 0.0, "limit"
            piv, hand = base, gp
            hold = int(pc.get("hold_steps", 10))                     # the stem follows the hand with a short delay
            while th < th_max or hold > 0:
                if th >= th_max:
                    hold -= 1
                th = min(th_max, th + step)
                up = u * (lift_tot * th / ang)                       # pulled up while it tilts
                piv, hand = base + up, gp + up
                if not self._twist_to(piv, hand, q0, 1.0, th, label="pluck"):
                    return broke, self.failure
                brk, fe, te = fld.check_break(slot, sim.d)
                rec.stem_force, rec.stem_torque = max(rec.stem_force, fe), max(rec.stem_torque, te)
                if brk:
                    bend = np.degrees(fld.bend_angle(slot, sim.d))
                    fld.detach(slot, sim.d)
                    rec.detached, rec.pluck_deg, rec.method = True, float(np.degrees(th)), "pluck"
                    self._event(f"stem tore off at tilt {np.degrees(th):.0f} deg (stem bent {bend:.0f} deg, F {fe:.0f} N, T {te:.2f} N m)")
                    broke = True
                    break
                c = sim.classify_contacts(slot)
                if c["pad_left"] < 0.5 and c["pad_right"] < 0.5:
                    return False, "grip_slip"
                if self._leaf_guard("twist_guard_frac"):
                    why = f"leaf root bent {np.degrees(self._root_bend):.0f} deg"
                    break
                if c["leaf"] > f_abort:
                    why = f"leaf force {c['leaf']:.0f} N"
                    break
            rec.pluck_deg = max(rec.pluck_deg, float(np.degrees(th)))
            if not broke:
                self._event(f"pluck tilt {np.degrees(th):.0f} deg stopped ({why}), stem not torn off"
                            + (": trying another direction" if n_dir == 0 and len(options) > 1 else ""))
            else:
                for _ in range(int(hc.get("snap_settle_steps", 5))):
                    if not self._twist_to(piv, hand, q0, 1.0, th, label="pluck"):
                        return broke, self.failure
                lift = u * float(pc.get("clear_lift", 0.06))          # clear of the stalk before turning back
                n_l = int(hc.get("untwist_lift_steps", 10))
                for k in range(n_l):
                    if not self._twist_to(piv + lift * (k + 1) / n_l, hand + lift * (k + 1) / n_l, q0, 1.0, th, label="pluck"):
                        return broke, self.failure
                piv, hand = piv + lift, hand + lift
            th0, ramp = th, np.radians(float(hc.get("untwist_ramp_deg", 25.0)))
            while th > 0.0:                                          # back to the grip orientation, smooth start / stop
                g = float(np.clip(th / ramp, 0.15, 1.0) * np.clip((th0 - th) / ramp + 0.3, 0.3, 1.0))
                th = max(0.0, th - back * g)
                if not broke:
                    up = u * (lift_tot * th / ang)
                    piv, hand = base + up, gp + up
                if not self._twist_to(piv, hand, q0, 1.0, th, label="pluck"):
                    return broke, self.failure
                if broke:
                    cf = self._carry_check()
                    if cf:
                        return broke, cf
                else:
                    brk, fe, te = fld.check_break(slot, sim.d)
                    if brk:
                        fld.detach(slot, sim.d)
                        rec.detached, rec.method, broke = True, "pluck", True
                        self._event(f"stem tore off while tilting back (F {fe:.0f} N, T {te:.2f} N m)")
            for _ in range(6):
                if not self._twist_to(piv, hand, q0, 1.0, 0.0, label="pluck"):
                    return broke, self.failure
            if broke:
                break
        self._twist_u = u
        return broke, None
    
    def _pull_up(self) -> str | None:
        """The twist could not snap the stem (leaves or the arm's reach stopped it): pull the fruit along its own
        axis, away from the stalk - the stalk snaps in tension - slowly and only a few centimetres [v6]. Pulling
        back along the approach line (the old fallback) only bends the stalk sideways and drags the plant."""
        sim, fld, rec, slot = self.sim, self.sim.field, self.rec, self.target_slot
        hc = self.cfg.get("harvest", {})
        u = np.asarray(getattr(self.plan.fruit, "axis", np.array([0.0, 0.0, 1.0])), float)
        dist, step = float(hc.get("pull_up_distance", 0.06)), float(hc.get("pull_up_step", 0.0015))
        P0, q = sim.tcp_pos.copy(), sim.tcp_quat()
        chk = self.arm.checker
        chk.sync_from(sim.d)
        with self._compute("route_check"):
            wps = chk.waypoints(P0, P0 + u * dist)
            ok, _, _ = chk.rollout_q(sim.d.qpos.copy(), wps, [q] * len(wps))
        if not ok:
            self._event("pull along the fruit axis not reachable")
            return None
        self._event("stem not broken by the twist: pulling along the fruit axis")
        for k in range(int(np.ceil(dist / step)) + 30):
            self.arm.servo_to(P0 + u * min(dist, (k + 1) * step), q)
            if not self._sim(1, "pull"):
                return self.failure
            brk, fe, te = fld.check_break(slot, sim.d)
            rec.stem_force, rec.stem_torque = max(rec.stem_force, fe), max(rec.stem_torque, te)
            if brk:
                fld.detach(slot, sim.d)
                rec.detached, rec.method = True, "pull"
                self._event(f"stem broke during the axial pull (F {fe:.0f} N, T {te:.2f} N m)")
                return None
            c = sim.classify_contacts(slot)
            if c["pad_left"] < 0.5 and c["pad_right"] < 0.5:
                return "pull_slip"
        return None

    def _twist_options(self, gp, q0, ang, step, u=None, pivot=None, with_key: bool = False) -> list:
        """[(sign, feasible angle)] of the twist directions, best first. Feasible = IK, no self / environment
        contact along the swing; preference = reaches further (>= 100 deg counts as full), then sweeps fewer
        perceived leaves, then the larger wrist-singularity margin."""
        chk, sim = self.arm.checker, self.sim
        chk.sync_from(sim.d)
        lp = self.leaf_cloud
        if len(lp):
            lp = lp[np.linalg.norm(lp - gp[None, :], axis=1) < 0.5]
        lr = np.full(len(lp), float(self.cfg["perception"].get("leaf_point_radius", 0.015)))
        axis = pivot if pivot is not None else getattr(self, "_twist_pivot", np.r_[self.plan.fruit.center[:2], gp[2]])
        u = u if u is not None else getattr(self, "_twist_u", np.array([0.0, 0.0, 1.0]))
        # other perceived fruits / crowns: the swinging hand must stay clear of them (two neighbour fruits were
        # knocked off by the wrist during the twist in the zig-zag row, measured in Mission 3)
        op, orad = self._perceived_fruit_points(include_target=False)
        o_margin = float(self.cfg.get("harvest", {}).get("twist_fruit_margin", 0.025))
        # the target's own stalk (perceived: below the fruit along its axis): wrist / forearm / housing must not
        # swing into it (measured in Mission 3: wrist_2 pressed the stalk with 578 N during the twist -> plant_hit)
        _, (sp, sr), _ = self.planner.hard_points(self.plan.fruit)
        nonfinger = ~self.arm.finger_mask
        s_margin = float(self.cfg.get("harvest", {}).get("twist_stalk_margin", 0.01))
        if len(sp):
            chk.has_contact(sim.d.qpos.copy())
            c0 = chk.hard_clearance([world_segments(self.arm.caps, chk._data)], sp, sr, nonfinger)
            s_margin = min(s_margin, c0 - 0.005)
        out = []
        for sign in (1.0, -1.0):
            q_full = sim.d.qpos.copy()
            th, th_ok, s_min = 0.0, 0.0, 1.0
            segs = []
            dth = np.radians(float(self.cfg.get("harvest", {}).get("twist_plan_step_deg", 6.0)))
            while th < ang:
                th = min(ang, th + dth)
                Rm = np.zeros(9)
                mujoco.mju_quat2Mat(Rm, quat_axis(u, sign * th))
                p = axis + Rm.reshape(3, 3) @ (gp - axis)
                q, info = self.arm.ik.solve(q_full, p, quat_mul(quat_axis(u, sign * th), q0))
                if info["position_error"] > 0.004 or np.max(np.abs(q - q_full[sim.arm_qadr])) > 0.6:
                    break
                q_full[sim.arm_qadr] = q
                if chk.has_contact(q_full):
                    break
                if len(op) and chk.hard_clearance([world_segments(self.arm.caps, chk._data)], op, orad) < o_margin:
                    break
                if len(sp) and chk.hard_clearance([world_segments(self.arm.caps, chk._data)], sp, sr, nonfinger) < s_margin:
                    break
                th_ok = th
                s_min = min(s_min, abs(np.sin(q[4])))
                if len(lp):
                    segs.append(world_segments(self.arm.caps, chk._data))
            if th_ok < np.radians(float(self.cfg.get("harvest", {}).get("twist_min_feasible_deg", 40.0))):
                continue
            intr = 0.0
            if len(lp) and segs:
                clr = chk.min_clearance_along(segs, lp, lr)
                intr = float(np.sum(np.clip(0.02 - clr, 0.0, None)))
            out.append((sign, th_ok, (min(th_ok, np.radians(100.0)), -intr, s_min)))
        out.sort(key=lambda o: o[2], reverse=True)
        return out if with_key else [(o[0], o[1]) for o in out]

    def _twist_direction(self, gp, q0, ang, step) -> float:
        """Best feasible twist direction (kept for compatibility; see _twist_options)."""
        opt = self._twist_options(gp, q0, ang, step)
        return opt[0][0] if opt else 0.0

    def _release_and_verify(self) -> bool:
        sim, hc = self.sim, self.cfg.get("harvest", {})
        sim.hold_arm()
        sim.set_gripper(self.cfg["gripper"]["max_opening"])
        if not self._sim(int(hc.get("release_steps", 30)), "release", monitor=False):
            return False
        self._count_drop()
        n = int(round(float(self.cfg["basket"]["settle_time"]) / sim.dt))
        if not self._sim(n, "verify", monitor=False):
            return False
        ok = sim.fruit_in_basket(self.target_slot)
        if not ok:                                               # diagnosis: where is the fruit?
            try:
                p = sim.field.fruit_pose(self.target_slot, sim.d)[0]
                out = []
                for k in range(len(sim.baskets)):
                    c, R = sim.basket_frame(k)
                    out.append(np.round(R.T @ (p - c), 3).tolist())
                self._event(f"fruit not in a crate after release: position in the crate frames {out}, "
                            f"tool {np.round(p - np.asarray(sim.tcp_pos), 3).tolist()} from the hand, "
                            f"gripper opening {sim.gripper_gap():.3f}")
                import mujoco as _mj, os as _os
                sl = sim.field.slots[self.target_slot]
                mine = {sl["fruit_geom"], sl["neck_geom"]}
                names = []
                for i in range(sim.d.ncon):
                    cc = sim.d.contact[i]
                    g1, g2 = int(cc.geom1), int(cc.geom2)
                    if g1 in mine or g2 in mine:
                        o = g2 if g1 in mine else g1
                        names.append((_mj.mj_id2name(sim.m, _mj.mjtObj.mjOBJ_GEOM, o),
                                      _mj.mj_id2name(sim.m, _mj.mjtObj.mjOBJ_BODY, int(sim.m.geom_bodyid[o])),
                                      round(float(cc.dist), 4)))
                self._event(f"  fruit contacts: {names}; fruit axis z {sim.d.xmat[sl['fruit_body']].reshape(3, 3)[2, 2]:.2f}")
                if _os.environ.get("PINE_DEBUG_DIR"):
                    r = _mj.Renderer(sim.m, 480, 640)
                    cam = _mj.MjvCamera()
                    for k, (az, el) in enumerate(((90, -20), (0, -35), (180, -60))):
                        cam.lookat[:] = p
                        cam.distance, cam.azimuth, cam.elevation = 0.9, az, el
                        r.update_scene(sim.d, cam)
                        import imageio.v2 as _iio
                        _iio.imwrite(f"{_os.environ['PINE_DEBUG_DIR']}/miss_{self.now():.0f}_{k}.png", r.render())
                    r.close()
            except Exception as e:                               # noqa: BLE001
                self._event(f"fruit not in a crate after release ({e})")
        if ok:
            self.rec.placed = True
            self.rec.t_placed = self.now()
            for f in [self.plan.fruit] + [g for g in self.fruits_scan
                                          if np.linalg.norm(g.center - self.plan.fruit.center) < 0.18]:
                f._picked = True
            base = sim.base_pos
            self._closed_xy.append(self.plan.fruit.center[:2].copy())
            _f = self.plan.fruit
            _ax = np.asarray(getattr(_f, "axis", np.array([0.0, 0.0, 1.0])), float)
            self._stalks.append((_f.center - _ax * getattr(_f, "half_height", 0.5 * (_f.top_z - _f.bottom_z)), _ax))
            self._stalks = self._stalks[-12:]
            self._last_az = float(np.arctan2(self.plan.fruit.center[1] - base[1], self.plan.fruit.center[0] - base[0]))
            sim.field.slots[self.target_slot]["harvested"] = True
            self.n_placed += 1
            self._event(f"fruit {self.target_slot} in basket")
            self.lamp("success")
        return ok

    def _line_leaf_guard(self, force: bool = False) -> bool:
        """A leaf the camera did not see (or that moved) is being pushed toward being torn out while the hand moves
        along the line: back out to T1 and take the next-ranked window line (up to `line_guard_retries` times).
        With no alternative left the line continues (touching leaves never ends the session; only an uprooted leaf
        does). True if it intervened."""
        ctx = getattr(self, "_plan_ctx", None)
        if ctx is None or not (force or self._leaf_guard("line_guard_frac")):
            return False
        if ctx["retries"] >= int(self.lim.get("line_guard_retries", 2)) or ctx.get("exhausted"):
            return False
        ctx["retries"] += 1
        ctx["failed_starts"].append(np.array(self.plan.line.tip_start, float))
        self._event(f"leaf root bent {np.degrees(self._root_bend):.0f} deg on the line: back out and take the next line")
        if not self._back_out() or self.done:
            return True
        with self._compute("plan_approach"):
            plan, why = self.planner.plan_approach(ctx["fruit"], ctx["decision"], ctx["cloud"], ctx["others"],
                                                   self.sim.d.qpos.copy(), exclude_starts=ctx["failed_starts"],
                                                   obstacles=ctx["obstacles"])
        if plan is None and not self.done:
            # [v6] the lines of one window all start within a few centimetres of each other, so "the next line" of
            # the same view is rarely a real alternative: look at the fruit from the next camera view (other side)
            # and decide again from there
            plan, why = self._plan_from_new_view(ctx)
            if self.done:
                return True
        if plan is None:                         # no other line: go on with the current one (leaves bend)
            ctx["exhausted"] = True
            self._event(f"no alternative line ({why}): continue")
            self.lamp("action")
            self._begin_phase("line", self.plan.line_route)
            return True
        self.plan = plan
        self.rec.face, self.rec.clearance = plan.info["face"], plan.info["clearance"]
        self._set_obs_points(plan)
        self._event(f"re-planned: rank {plan.info['rank']}, clearance {plan.info['clearance'] * 100:.1f} cm")
        self._begin_phase("approach", plan.approach)
        return True

    def _plan_from_new_view(self, ctx):
        """Leaf guard: capture the next close view(s) of the target (the view iterator of the look continues with the
        views further around the fruit), refine the estimate and plan an approach from there. Returns (plan, reason)."""
        from pinesim.vision.stargaze_cam import obstacle_points
        sim, per = self.sim, getattr(self, "_look_per", None)
        why = "no_other_view"
        if per is None or not hasattr(self, "_look_iter"):
            return None, why
        self.lamp("observe")
        sim.set_gripper(self.cfg["gripper"]["min_opening"])
        for _ in range(int(self.lim.get("guard_new_views", 2))):
            got = self._look_next_view(per)
            if not got or self.done:
                break
            with self._compute("perception"):
                res = per.result()
                fruit, seen = self._match_fruit(res.fruits, ctx["fruit"])
            if fruit is None:
                why = "lost_in_look"
                continue
            others = merge_fruits(seen + list(ctx["others"]))
            others = [f for f in others if np.linalg.norm(f.center[:2] - fruit.center[:2]) > 0.10]
            cloud = np.vstack([res.leaf_points, self.scan_leaf]) if len(self.scan_leaf) else res.leaf_points
            obs = obstacle_points(fruit, cloud, self.cfg, others)
            with self._compute("decision"):
                d_ = decide(fruit, per.views, obs[0], obs[1], self.cfg, policy=self.t2_policy, rng=self.planner.rng,
                            view_order=[len(self._look_q) - 1])
            if not d_.ok:
                why = "window_" + d_.reason
                continue
            with self._compute("plan_approach"):
                plan, why = self.planner.plan_approach(fruit, d_, cloud, others, sim.d.qpos.copy(), obstacles=obs,
                                                       exclude_starts=ctx["failed_starts"])
            if plan is not None:
                ctx.update(fruit=fruit, cloud=cloud, others=others, decision=d_, obstacles=obs)
                self.decision, self.leaf_cloud = d_, cloud
                self.rec.n_views = len(self._look_q)
                self._send_overlay(fruit, d_, plan)
                self.lamp("action")
                return plan, "ok"
        return None, why

    def _carry_leaf_guard(self) -> bool:
        """[v6] Pulling the picked fruit out / lifting it: a leaf is being dragged toward being torn out (root bend
        beyond the guard). The hand stops, goes back a few centimetres the way it came and the carry is re-planned
        with the OTHER pull-out direction first (straight up instead of along the line, or the reverse) - once per
        fruit. `leaf_uprooted` in the retreat / lift phases was the most frequent failure of the light training
        scenes. True if it intervened."""
        if not self._leaf_guard("carry_guard_frac") or self.carry is None or self.rec is None:
            return False
        if getattr(self.rec, "carry_guards", 0) >= int(self.lim.get("carry_guard_retries", 1)):
            return False
        self.rec.carry_guards = getattr(self.rec, "carry_guards", 0) + 1
        sim = self.sim
        was = self.carry.get("variant", {}).get("retreat", "line")
        self._event(f"leaf root bent {np.degrees(self._root_bend):.0f} deg while pulling out ({was}): stop, other direction")
        sim.hold_arm()                           # stop (no reversal: pushing the fruit back knocked it out of the pads)
        if not self._sim(int(self.lim.get("carry_guard_hold_steps", 5)), "hold"):
            return True
        with self._compute("plan_carry"):
            carry, _ = self._plan_carry(prefer="up" if was == "line" else "line")
        if carry is None or self.done:
            return bool(self.done)
        self.carry = carry
        self.arm.target_quat = sim.tcp_quat()
        self._begin_phase("retreat", carry["retreat"])
        return True

    def _back_out(self) -> bool:
        """Inside the plant (line / retreat phase): go straight back along the Stargaze line to T1 (the leaf-free
        start) before any other motion, instead of sweeping sideways through the leaves. The straight route is
        IK / collision checked first (keeping the hand orientation, or turning back to the approach orientation);
        an unchecked back-out drove the wrist into the vehicle deck (measured, Mission 3)."""
        sim = self.sim
        if self.plan is None or self.phase not in (PHASE_ID["line"], PHASE_ID["retreat"]):
            return True
        from pinesim.vendor.harvest_checker import slerp
        T1 = np.array(self.plan.line.start, float)
        P0, q_now = sim.tcp_pos.copy(), sim.tcp_quat()
        L = float(np.linalg.norm(T1 - P0))
        if L < 0.01:
            return True
        chk = self.arm.checker
        chk.sync_from(sim.d)
        wps = chk.waypoints(P0, T1)
        route = None
        with self._compute("route_check"):
            for q_end in (q_now, np.array(self.plan.line.quat, float)):
                quats = [slerp(q_now, q_end, min(1.0, (i + 1) / max(1, 0.6 * len(wps)))) for i in range(len(wps))]
                ok, _, _ = chk.rollout_q(sim.d.qpos.copy(), wps, quats)
                if ok:
                    route = q_end
                    break
        if route is None:
            self._event("back-out along the line is not collision-free: skipped")
            return True
        u = (T1 - P0) / L
        v = float(self.cfg.get("planning", {}).get("back_out_step", 0.006))
        for k in range(int(self.cfg.get("planning", {}).get("back_out_steps", 80)) + int(L / v)):
            n = float(np.linalg.norm(T1 - sim.tcp_pos))
            if n < 0.01:
                break
            s_ = min(L, (k + 1) * v)
            self.arm.servo_to(P0 + u * s_, slerp(q_now, route, min(1.0, s_ / max(0.6 * L, 1e-6))))
            if not self._sim(1, "back_out"):
                return False
        return True

    def _return_home(self) -> bool:
        """Back to the DEFAULT posture with the jaws closed (end of a session) [USER v5]."""
        sim = self.sim
        if not self._back_out():
            return False
        up = sim.tcp_pos + np.array([0, 0, 0.10])
        q = sim.tcp_quat()
        for _ in range(15):
            self.arm.servo_to(up, q)
            if not self._sim(1, "return", monitor=False):
                return False
        sim.set_gripper(self.cfg["gripper"]["min_opening"])
        target = self.default_q()
        mv = self._move_safe(target, "return")
        if mv is None:                                # no checked path: via the IK work posture
            mv = self._move_safe(self._home_q, "return")
            if mv:
                mv2 = self._move_safe(target, "return")
                if mv2 is None:                       # stay in the work posture (safe, above the plants)
                    self._event("default posture: no collision-free path, staying in the work posture")
                else:
                    mv = mv2
            elif mv is None:                          # nothing checked works: last resort as before
                mv = self._move_joints(self._home_q, "return")
        if mv:
            self._bend_allow = 0.0                    # out of the plant: normal leaf limit again
            if self.rec is not None and not np.isfinite(self.rec.t_end):
                self.rec.t_end = self.now()
            self._t_session = self.now()              # the next fruit's session starts here
        return bool(mv)

    def default_q_near(self) -> np.ndarray:
        """The default posture in the joint representation (+-2 pi per joint) nearest to the current posture: the
        same physical pose, without a full turn of a wrist joint."""
        q, t = self.sim.q_arm(), self.default_q().copy()
        lim = self.sim.arm_range
        for j in range(len(t)):
            k = np.round((q[j] - t[j]) / (2 * np.pi))
            v = t[j] + 2 * np.pi * k
            if lim[j, 0] <= v <= lim[j, 1]:
                t[j] = v
        return t

    def finish_session(self):
        """After the episode ended (evaluation / recording): bring the arm back to the default posture, jaws closed,
        without the safety monitor (nothing is scored any more). Skipped after a collision / numerical failure."""
        if self.failure in ("collision", "sim_unstable", "floor_contact"):
            return
        sim = self.sim
        target = self.default_q_near()
        if np.max(np.abs(sim.q_arm() - target)) < 0.02:
            return
        sim.set_gripper(self.cfg["gripper"]["min_opening"])
        try:
            if self.phase in (PHASE_ID["line"], PHASE_ID["retreat"]) and self.plan is not None:
                T1, q = np.array(self.plan.line.start, float), sim.tcp_quat()
                for _ in range(80):
                    d = T1 - sim.tcp_pos
                    if np.linalg.norm(d) < 0.01:
                        break
                    self.arm.servo_to(sim.tcp_pos + d * min(1.0, 0.02 / np.linalg.norm(d)), q)
                    sim.step(1); self._tick("return")
            for qq in self.arm.joint_path(sim.q_arm(), target, 0.03):
                sim.d.ctrl[sim.arm_act] = qq
                sim.step(1); self._tick("return")
            for _ in range(25):
                sim.step(1); self._tick("return")
        except Exception:
            pass

    # ================================================================== Gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.cam.rng = np.random.default_rng(seed + 7919)
            self.cam._calib_R, self.cam._calib_t = self.cam._sample_calibration()   # per-episode, from the seed
            self.planner.rng = np.random.default_rng(seed + 104729)
        attempts = int(self.task.get("max_scene_attempts", 5)) if self.resample_on_plan_fail else 1
        for k in range(attempts):
            self._reset_state()
            self._scene_reset()
            ok = self._start_episode()
            if ok or k == attempts - 1:
                break
        self.n_resampled = k
        return self._get_obs(), self._get_info()

    def _start_episode(self) -> bool:
        if not self._scan():
            if not self.done:
                self._fail("nothing_detected")
            return False
        return self._acquire_target()

    def _residual_watchdog(self):
        """[v6] The residual policy may only help: when the arm makes no progress along the active route for
        `residual_stall_steps` control steps, or the safety shield had to step in `residual_shield_limit` times in
        this phase, the residual is switched off until the phase ends and the reference controller finishes it alone.
        (Before: a policy that leaned into the shield or dithered next to the route ended the episode with
        `route_blocked` / `timeout` - during training this was the main cause of failed episodes of a policy that was
        otherwise as good as the reference.)"""
        if self._residual_off or self.route is None:
            return
        ps = self.cfg.get("policy", {})
        dist = float(np.linalg.norm(self.route.end - self.sim.tcp_pos))
        if self.route.idx > self._prog[0] or dist < self._prog[1] - 0.005:
            self._prog = [max(self.route.idx, self._prog[0]), min(dist, self._prog[1]), self.step_count]
        stalled = self.step_count - self._prog[2] > int(ps.get("residual_stall_steps", 60))
        if stalled or self._phase_shields >= int(ps.get("residual_shield_limit", 3)):
            self._residual_off = True
            self.residual_off_events = getattr(self, "residual_off_events", 0) + 1
            self._event(f"residual policy off for the rest of '{PHASES[self.phase]}' "
                        f"({'no progress' if stalled else 'safety shield'}): reference controller alone")

    def _carry_gain(self) -> float:
        """Hand speed multiplier: larger than 1 only while the fruit is carried to the crate (transport phase)."""
        return float(self.cfg["policy"].get("carry_pos_gain", 1.0)) if self.phase == PHASE_ID["transport"] else 1.0
    
    def _compose(self, a_rl):
        ref = self._ref_action
        # [v6] the residual acts only where it can matter: on the way to the fruit (approach, line). With the fruit in
        # the hand (retreat, lift, transport) every deviation from the validated route only cost fruit: measured with
        # the trained policies - leaf torn out while pulling out, arm steered next to its own forearm in the
        # transport until the shield blocked the route - never a gain over the reference controller.
        if PHASES[self.phase] not in self.cfg.get("policy", {}).get("residual_phases", PHASES):
            self._gate = 0.0
            return self._governed(np.clip(ref, -1.0, 1.0))
        if self._residual_ban:
            self._gate = 0.0
            return self._governed(np.clip(ref, -1.0, 1.0))
        if np.any(np.abs(np.asarray(a_rl, float)) > 1e-9):      # (reference controller alone: nothing to switch off)
            self._residual_watchdog()
            self._residual_used = True
        if self._residual_off:
            self._gate = 0.0
            return self._governed(np.clip(ref, -1.0, 1.0))
        g = 1.0
        if self.target_gate is not None and self.phase == PHASE_ID["line"]:
            d0, d1 = self.target_gate
            dist = float(np.linalg.norm(self.route.end - self.sim.tcp_pos))
            g = float(np.clip((dist - d0) / max(d1 - d0, 1e-9), 0.0, 1.0))
        self._gate = g
        a = ref + g * self.residual_scale * a_rl
        rp = ref[:3]
        n = float(np.linalg.norm(rp))
        if n > 1e-6 and self.progress_min > 0:
            u = rp / n
            along = float(np.dot(a[:3], u))
            lo = self.progress_min * n
            if along < lo:
                a[:3] += (lo - along) * u
        return self._governed(np.clip(a, -1.0, 1.0))

    def _shield(self, q_cmd, a, tq0):
        """Safety layer between the policy and the servos: a commanded posture that would touch the robot itself,
        the platform / vehicle / basket or the floor (route checker, 1 cm self margin) is replaced by the reference
        action alone, and if that also collides the arm holds its posture for this step. The policy still learns
        from the outcome; the shield only stops avoidable hard collisions (measured in the realistic evaluations:
        wrist vs platform during the approach)."""
        chk = self.arm.checker
        full = self.sim.d.qpos.copy()
        full[self.sim.arm_qadr] = q_cmd
        if not chk.has_contact(full):
            self._shield_hold = 0
            return q_cmd, a
        full[self.sim.arm_qadr] = self.sim.d.ctrl[self.sim.arm_act]
        if chk.has_contact(full):                # already at the margin: do not freeze the arm
            return q_cmd, a
        # [v6] inside the clearance margin (not in contact) and the command does not reduce the clearance: let it
        # move - the arm that started a route 9 mm from the lamp bracket (margin 10 mm) was frozen until the
        # time-out although its route led away from the bracket (measured, Mission 3)
        full[self.sim.arm_qadr] = q_cmd
        d_cmd = chk.margin_distance(full)
        if d_cmd > 0.003:
            full[self.sim.arm_qadr] = self.sim.q_arm()
            if d_cmd >= chk.margin_distance(full) - 1e-4:
                self._shield_hold = 0
                return q_cmd, a
        self.shield_events = getattr(self, "shield_events", 0) + 1
        self._phase_shields = getattr(self, "_phase_shields", 0) + 1
        self._shield_step = True
        self.arm.target_quat = tq0
        ref = np.clip(self._ref_action, -1.0, 1.0)
        q_ref, _ = self.arm.command(ref)
        full[self.sim.arm_qadr] = q_ref
        if not chk.has_contact(full):
            self._shield_hold = 0
            return q_ref, ref
        self.arm.target_quat = tq0                # holding: the orientation target does not advance either
        self._shield_hold = getattr(self, "_shield_hold", 0) + 1
        ps = self.cfg.get("policy", {})
        if self._shield_hold == int(ps.get("shield_backtrack_hold", 5)) and self._residual_used \
                and not self._residual_ban:
            # [v6] the residual policy steered the arm to a posture from which even the reference action runs into
            # the margin (measured with the trained Mission 2 policy: forearm next to the gripper housing 0.9 s
            # after the approach began -> `route_blocked`): go back along the postures the arm came through and
            # finish this fruit with the reference controller alone
            self._residual_ban = True
            if self._backtrack():
                self._shield_hold = 0
                return self.sim.d.ctrl[self.sim.arm_act].copy(), np.zeros(6)
        if self._shield_hold == int(ps.get("shield_replan_hold", 10)):
            # [v6] the reference route itself leads into the safety margin (servo lag / a route that only just
            # cleared): plan a new route from where the arm is instead of waiting for the time-out
            why = chk.contact_reason(full)
            if self._shield_replan():
                self._event(f"safety shield: route blocked ({why}) -> re-planned from the current posture")
                self._shield_hold = 0
                return self.sim.d.ctrl[self.sim.arm_act].copy(), np.zeros(6)
            self._event(f"safety shield: route blocked ({why}), no alternative route")
        if self._shield_hold > int(ps.get("shield_max_hold", 50)):
            self._event(f"safety shield: route blocked (held {self._shield_hold} steps)")
            self._fail("route_blocked")
        return self.sim.d.ctrl[self.sim.arm_act].copy(), np.zeros(6)

    def _backtrack(self) -> bool:
        """Move back through the arm postures of the last policy steps until the arm is clear of the safety margin."""
        sim, chk = self.sim, self.arm.checker
        ps = self.cfg.get("policy", {})
        hist = list(self._q_hist)
        if len(hist) < 3:
            return False
        # all the way back to where this phase began (a posture the reference route starts from): a few steps back
        # were not enough - the wrist had been turned and the reference action ran into the margin again (measured)
        done = 0
        for q in hist[-2::-1][::2] + [hist[0]]:
            sim.d.ctrl[sim.arm_act] = q
            if not self._sim(1, "backtrack"):
                return False
            done += 1
        for _ in range(10):
            if not self._sim(1, "backtrack"):
                return False
        self._q_hist = hist[:1]
        if self.route is not None:
            self.route.idx = 0
        self.arm.target_quat = sim.tcp_quat()
        self._event(f"safety shield: residual policy led into the margin -> went back {done} steps, "
                    f"reference controller alone for this fruit")
        return True

    def _shield_replan(self) -> bool:
        """New route for the active phase from the current posture (approach: next feasible window line incl. the
        current one; transport: joint-space route to the drop posture). False if there is none."""
        sim, ctx = self.sim, getattr(self, "_plan_ctx", None)
        n = getattr(self, "_shield_replans", 0)
        if n >= int(self.cfg.get("policy", {}).get("shield_replans", 2)) or ctx is None:
            return False
        self._shield_replans = n + 1
        name = PHASES[self.phase]
        with self._compute("plan_approach"):
            if name == "approach":
                plan, _ = self.planner.plan_approach(ctx["fruit"], ctx["decision"], ctx["cloud"], ctx["others"],
                                                     sim.d.qpos.copy(), obstacles=ctx["obstacles"],
                                                     exclude_starts=[self.plan.line.tip_start] if n >= 1 else ())
                if plan is None:
                    return False
                self.plan = plan
                self._set_obs_points(plan)
                self._begin_phase("approach", plan.approach)
                return True
            if name in ("lift", "transport") and self.carry is not None:
                tr = self.planner._transport_joint_route(sim.d.qpos.copy(), sim.tcp_pos.copy(), self.carry["D"],
                                                         sim.tcp_quat(), self.plan.fruit)
                if tr is None:
                    return False
                self.carry["transport"] = tr[0]
                self.arm.target_quat = sim.tcp_quat()
                self._begin_phase("transport", tr[0])
                return True
        return False

    def step(self, action):
        a_rl = np.clip(np.asarray(action, float).reshape(6), -1, 1)
        reward = 0.0
        self._rc = {}
        if self.done or self.phase < 0:
            # (a reset whose survey / look / plan failed: the first step returns the failure penalty)
            return self._get_obs(), float(self._charge_failure()), True, False, self._get_info()
        a = self._compose(a_rl)
        if self.phase >= 2:                      # carrying by friction: limit the change of the command per step
            slew = float(self.cfg.get("policy", {}).get("carry_slew", 0.08))
            a = self._prev_action + np.clip(a - self._prev_action, -slew, slew)
        tq0 = np.array(self.arm.target_quat, float).copy()
        self.arm.pos_gain = self._carry_gain()
        q_cmd, ik_info = self.arm.command(a)
        self.arm.pos_gain = 1.0
        self._shield_step = False
        self._q_hist.append(self.sim.q_arm().copy())
        if len(self._q_hist) > 150:
            del self._q_hist[:50]
        q_cmd, a = self._shield(q_cmd, a, tq0)
        self.sim.d.ctrl[self.sim.arm_act] = q_cmd
        self._cur_action = a
        self.sim.step(1)
        self._tick(PHASES[self.phase])
        self._contacts = self.sim.classify_contacts(self.target_slot)
        self._update_leaf_state()
        if (self.phase == PHASE_ID["line"] and self._line_leaf_guard()) or \
                (self.phase in (PHASE_ID["retreat"], PHASE_ID["lift"]) and self._carry_leaf_guard()):
            reward += self._charge_failure()
            obs = self._get_obs()
            info = self._get_info()
            return obs, float(reward), bool(self.done), False, info
        reward += self._phase_reward()
        if self._shield_step:                    # the safety shield had to replace the action: small penalty [v6]
            self._rc["shield"] = self._w("shield_penalty", 0.0)
            reward += self._rc["shield"]
        terminated = self.done
        truncated = False
        if not self.done and self.step_count >= self.max_steps:
            self._fail("timeout")
            reward += self._w("timeout_penalty", -20.0)
            truncated = True
        reward += self._charge_failure()
        self._prev_action = a
        obs = self._get_obs()
        info = self._get_info()
        info["reward_components"] = dict(self._rc)
        info["residual_norm"] = float(np.linalg.norm(a_rl))
        info["residual_gate"] = self._gate
        return obs, float(reward), bool(self.done and not truncated), bool(truncated), info

    # ---------------------------------------------------------------- reward / transitions
    def _charge_failure(self) -> float:
        """[v6] EVERY failure costs the same terminal penalty `fail_penalty`, whether it happened in an RL phase or in
        a scripted move (route blocked by the shield, collision on the way home, target lost, no route for the next
        fruit ...). Before, several failure classes ended the episode without any penalty, so a failed episode could
        return more than a successful one (measured). Class-specific penalties (plant damage, drop) come on top."""
        if self.failure and not self._fail_charged:
            self._fail_charged = True
            self._rc["fail"] = self._w("fail_penalty", 0.0)
            return self._rc["fail"]
        return 0.0

    def _phase_reward(self) -> float:
        sim = self.sim
        tcp = sim.tcp_pos
        tgt = self.route.end
        dist = float(np.linalg.norm(tgt - tcp))
        r_time = self._w("time_penalty", -0.02)
        r_dist = -abs(self._w("distance_penalty", 0.5)) * dist
        r_prog = self._w("progress_weight", 20.0) * (self._prev_dist - dist)
        da = self._cur_action - self._prev_action
        r_smooth = -abs(self._w("smoothness_penalty", 0.05)) * float(np.sum(da ** 2))
        self._prev_dist = dist
        self._rc = {"time": r_time, "distance": r_dist, "progress": r_prog, "smooth": r_smooth}
        r = r_time + r_dist + r_prog + r_smooth
        if self.phase >= 1:                      # deviation from the LOCAL piece of the reference route
            la, lb = self._local_seg()
            dp = perpendicular_distance_to_segment(tcp, la, lb)
            r_ct = -abs(self._w("cross_track_penalty", 150.0)) * min(dp, 0.08) ** 2
            self._rc["cross_track"] = r_ct
            r += r_ct
        r_leaf = -abs(self._w("leaf_energy_penalty", 0.15)) * min(self._leaf_energy / self._w("leaf_energy_ref", 0.1),
                                                                    self._w("leaf_energy_cap", 3.0))
        self._rc["leaf"] = r_leaf
        r += r_leaf
        # ---- safety / damage
        if self.sim.unstable:
            self._fail("sim_unstable")
            return r
        c = self._contacts
        if c.get("self_or_env") or c.get("floor"):
            if c.get("self_or_env"):
                self._event(f"collision: {c.get('env_names', [])[:3]}")
            self._fail("collision" if c.get("self_or_env") else "floor_contact")
            self._rc["contact"] = self._w("contact_penalty", -10.0)
            return r + self._rc["contact"]
        nd = getattr(self, "_new_damage", 0)
        if nd:                                   # leaves bent beyond the damage limit: penalised, not terminal
            self._new_damage = 0
            self._rc["leaf_damage"] = self._w("leaf_damage_penalty", -5.0) * nd
            r += self._rc["leaf_damage"]
        if self._leaf_uprooted():
            return r + self._w("plant_damage_penalty", -50.0)
        if c.get("stalk", 0.0) > float(self.lim.get("stalk_force", 10.0)) or c.get("fruit_other", 0.0) > float(self.lim.get("other_fruit_force", 10.0)):
            self._event(f"plant hit: stalk {c.get('stalk', 0.0):.0f} N, other fruit {c.get('fruit_other', 0.0):.0f} N in "
                        f"'{PHASES[self.phase]}', robot contacts {self._robot_contacts()[:4]}")
            self._fail("plant_hit")
            return r + self._w("plant_damage_penalty", -50.0)
        if self.phase <= 1 and c.get("fruit_target", 0.0) > float(self.lim.get("target_touch_force", 40.0)):
            self._fail("fruit_hit")
            return r + self._w("plant_damage_penalty", -50.0)
        if self.phase <= 1 and not self._stems_intact():
            return r + self._w("plant_damage_penalty", -50.0)
        if self.phase >= 2:
            fail = self._carry_check()
            if fail:
                self._fail(fail)
                return r + self._w("drop_penalty", -30.0)
        # ---- phase transitions
        name = PHASES[self.phase]
        if name == "approach" and dist < float(self.lim.get("t1_threshold", 0.03)):
            self._verify_at_t1()
            if self.done:
                return r
            self._begin_phase("line", self.plan.line_route)
            return r + self._w("phase_bonus", 5.0)
        if name == "line":
            ang = quat_angle(sim.tcp_quat(), self.plan.line.quat)
            ang = min(ang, quat_angle(sim.tcp_quat(), quat_mul(self.plan.line.quat, np.array([0, 0, 0, 1.0]))))
            r_or = -abs(self._w("orientation_penalty", 0.5)) * ang
            self._rc["orientation"] = r_or
            r += r_or
            if dist < float(self.lim.get("grasp_pos_tol", 0.015)) and ang < np.radians(float(self.lim.get("grasp_orient_deg", 12.0))):
                return r + self._after_line()
            # stall: leaves (or the fruit) block the last centimetres -> grasp from there if the fruit is already
            # between the open fingers, otherwise give up early instead of pushing until the time limit
            if dist < self._line_best - 0.003:
                self._line_best, self._line_best_step = dist, self.step_count
            if self.step_count - self._line_best_step > int(self.lim.get("line_stall_steps", 100)):
                if dist < float(self.lim.get("line_stall_grasp_dist", 0.05)) and ang < np.radians(float(self.lim.get("grasp_orient_deg", 12.0))):
                    self._event(f"line stalled {dist * 100:.1f} cm before the grasp point: grasping there")
                    return r + self._after_line()
                self._fail("line_blocked")
                return r + self._w("harvest_fail_penalty", -30.0)
        if name == "retreat" and dist < 0.03:
            self._begin_phase("lift", self.carry["lift"])
            return r + self._w("phase_bonus", 5.0)
        if name == "lift" and dist < 0.03:
            self._begin_phase("transport", self.carry["transport"])
            return r + self._w("phase_bonus", 5.0)
        if name == "transport" and dist < float(self.lim.get("drop_tol", 0.03)):
            return r + self._after_transport()
        # pull-out without a broken stem: the retreat pull may break it
        if name == "retreat" and not self.sim.field.slots[self.target_slot]["detached"]:
            brk, fe, te = self.sim.field.check_break(self.target_slot, self.sim.d)
            self.rec.stem_force, self.rec.stem_torque = max(self.rec.stem_force, fe), max(self.rec.stem_torque, te)
            if brk:
                self.sim.field.detach(self.target_slot, self.sim.d)
                self.rec.detached = True
                self._event(f"stem broke during pull (F {fe:.0f} N, T {te:.2f} N m)")
                r += self._w("detach_bonus", 10.0)
            elif dist < 0.06:
                self._fail("detach_fail")
                return r + self._w("harvest_fail_penalty", -30.0)
        return r

    def _carry_check(self) -> str | None:
        sim = self.sim
        slot = self.target_slot
        p, _ = sim.field.fruit_pose(slot, sim.d)
        lay = sim.field.slots[slot]["layout"]
        # diagnosis [v6]: what the carried fruit touches besides the pads (a drop is logged with its last touch)
        sl = sim.field.slots[slot]
        mine = (sl["fruit_geom"], sl["neck_geom"])
        for i in range(sim.d.ncon):
            cc = sim.d.contact[i]
            g1, g2 = int(cc.geom1), int(cc.geom2)
            if g1 in mine or g2 in mine:
                o = g2 if g1 in mine else g1
                nm = mujoco.mj_id2name(sim.m, mujoco.mjtObj.mjOBJ_GEOM, o) or f"geom{o}"
                if "pad" not in nm and "finger" not in nm:
                    self._carry_touch = (round(self.now(), 2), nm, PHASES[self.phase] if self.phase >= 0 else "")
        if np.linalg.norm(p - sim.tcp_pos) > lay.fruit_r + float(self.lim.get("drop_margin", 0.06)):
            # still on the plant -> the grip slipped while pulling; otherwise the picked fruit fell
            v = float(np.linalg.norm(sim.d.cvel[sim.m.site_bodyid[sim.tcp_site], 3:]))
            self._event(f"fruit left the grip: {np.round(p - sim.tcp_pos, 3).tolist()} from the hand, hand speed "
                        f"{v:.2f} m/s, jaw gap {sim.gripper_gap():.3f}, last touch besides the pads "
                        f"{getattr(self, '_carry_touch', None)}")
            return "pull_slip" if not sim.field.slots[slot]["detached"] else "fruit_dropped"
        if sim.fruit_touches(slot)["floor"]:
            return "fruit_floor"
        return None

    def _after_line(self) -> float:
        """At the grasp pose: scripted grasp + twist, then carry planning."""
        bonus, fail = self._grasp_and_twist()
        if fail == "__leaf_block__":
            # back out to T1 with the jaws closed again and take the next-ranked line; none left -> give this fruit up
            # without tearing the leaf (line_blocked) instead of pushing on
            ctx = self._plan_ctx
            if ctx is not None and ctx["retries"] < int(self.lim.get("line_guard_retries", 2)) and not ctx.get("exhausted"):
                plan0 = self.plan
                self._line_leaf_guard(force=True)
                self.sim.set_gripper(self.cfg["gripper"]["min_opening"])
                if self.done:
                    return bonus
                if self.plan is not plan0 and self.phase == PHASE_ID["approach"]:
                    return bonus
            fail = "line_blocked"
        if fail is not None:
            self._fail(fail)
            return bonus + self._w("harvest_fail_penalty", -30.0)
        with self._compute("plan_carry"):
            carry, why = self._plan_carry()
        if carry is None:
            self._event(f"no carry route: {getattr(self.planner, 'last_carry_reject', {})}")
            if self.sim.field.slots[self.target_slot]["detached"] and self._rescue_carry():
                return bonus + self._after_transport()
            if not self.done:
                self._fail(why)
            return bonus + self._w("harvest_fail_penalty", -30.0)
        self.carry = carry
        self.arm.target_quat = self.sim.tcp_quat()
        self._begin_phase("retreat", carry["retreat"])
        return bonus

    # ------------------------------------------------------------------ where in the crate the fruit goes [v6]
    def _drop_index(self) -> int:
        """Number of fruits already released into the active crate (Missions 1-2: in this episode)."""
        return int(getattr(self, "_drops", 0))

    def _count_drop(self):
        self._drops = int(getattr(self, "_drops", 0)) + 1

    def _drop_candidates(self) -> list:
        """Places on the crate floor (crate frame, m) for the next fruit: `basket.drop_pattern` is walked through
        fruit by fruit so that no fruit is released on top of an earlier one; the next places of the pattern are
        the fallback when no carry route reaches the first one."""
        pat = [list(map(float, o)) for o in self.cfg["basket"].get("drop_pattern", [[0.0, 0.0]])] or [[0.0, 0.0]]
        i = self._drop_index() % len(pat)
        n_alt = int(self.cfg["basket"].get("drop_alternatives", 2))
        out = [pat[(i + k) % len(pat)] for k in range(min(len(pat), 1 + n_alt))]
        return out

    def _plan_carry(self, prefer=None):
        cands = self._drop_candidates()
        self._carry_touch = None
        carry, why = None, "carry_infeasible"
        for k, off in enumerate(cands):
            self.planner.drop_offset = off
            carry, why = self.planner.plan_carry(self.plan, self.sim.tcp_pos, self.sim.d.qpos.copy(), prefer=prefer)
            if carry is not None:
                if k:
                    self._event(f"crate place {cands[0]} not reachable: fruit goes to {off}")
                break
        return carry, why

    def _rescue_carry(self) -> bool:
        """The fruit is already picked but no Cartesian carry route was found: back out along the line, lift, and
        take a checked joint-space path to a drop posture above the basket (scripted) instead of dropping it."""
        sim = self.sim
        self._event("carry route infeasible: scripted rescue to the basket")
        if not self._back_out() or self.done:
            return False
        up, q = sim.tcp_pos + np.array([0, 0, 0.12]), sim.tcp_quat()
        for _ in range(20):
            self.arm.servo_to(up, q)
            if not self._sim(1, "rescue"):
                return False
        D = self.planner.drop_point(self.plan.fruit)
        for qa in self.planner.drop_postures(D, sim.d.qpos.copy(), sim.tcp_quat()):
            mv = self._move_safe(qa, "rescue")
            if mv is None:
                continue
            if not mv or self.done:
                return False
            p, _ = sim.field.fruit_pose(self.target_slot, sim.d)
            if np.linalg.norm(p - sim.tcp_pos) > sim.field.slots[self.target_slot]["layout"].fruit_r + float(self.lim.get("drop_margin", 0.06)):
                self._fail("fruit_dropped")
                return False
            return True
        return False

    def _after_transport(self) -> float:
        if not self._release_and_verify():
            if not self.done:
                self._fail("basket_miss")
            return self._w("drop_penalty", -30.0)
        r = self._w("place_reward", 40.0)
        if self.n_placed >= self._n_targets():
            if self._return_home():
                self._event("all targets placed")
            self.done = True
            return r + self._w("success_bonus", 0.0)
        if not self._return_home():
            return r
        if not self._next_target():
            if not self.done:
                self._fail("next_target_failed")
        return r

    def _n_targets(self) -> int:
        return int(self.task.get("n_targets", self.task["n_plants"]))

    def _next_target(self) -> bool:
        """Mission 2: the remaining detected fruit (scan list) -> look again -> plan."""
        self.plan, self.carry, self.target_slot, self.phase = None, None, None, -1
        return self._acquire_target()

    # ================================================================== observation
    def _get_obs(self) -> np.ndarray:
        sim, S = self.sim, OBS_SLICES
        obs = np.zeros(OBS_DIM)
        if self.phase < 0 or self.route is None:
            return obs.astype(np.float32)
        tcp = sim.tcp_pos
        rot = sim.tcp_rot()
        base = sim.base_pos
        tgt = self.route.end
        obs[S["rel_target"]] = tgt - tcp
        obs[S["q"]] = sim.q_arm()
        obs[S["qd"]] = sim.d.qvel[sim.arm_dadr]
        obs[S["tcp_pos"]] = tcp - base
        obs[S["tcp_rot6"]] = np.concatenate([rot[:, 0], rot[:, 1]])
        obs[S["manip"]] = self.arm.ik.manipulability_from_data(sim.d)
        obs[S["prev_action"]] = self._prev_action
        obs[S["time_left"]] = 1.0 - self.step_count / self.max_steps
        a, b = self._seg
        la, lb = self._local_seg()
        seg = lb - la
        obs[S["phase"]] = float(self.phase)
        obs[S["path_dir"]] = seg / (np.linalg.norm(seg) + 1e-9)
        _, proj = project_point_on_segment(tcp, la, lb)
        obs[S["cross_track_vec"]] = proj - tcp
        t, _ = project_point_on_segment(tcp, a, b)
        obs[S["path_progress"]] = t
        # nearest perceived leaf points (static camera cloud); slot 4 = wrist force
        K = N_NEAREST_OBSTACLES
        rel = np.zeros((K, 3))
        dist = np.full(K, OBS_FAR)
        near = np.zeros(0, int)
        if len(self.obs_pts):
            # only points that can be among the nearest to the gripper / wrist / forearm (exact for those)
            keys = np.vstack([tcp, sim.d.xpos[sim.arm_body_ids[3]], sim.d.xpos[sim.arm_body_ids[2]]])
            dmin = np.min(np.linalg.norm(self.obs_pts[:, None, :] - keys[None, :, :], axis=2), axis=1)
            near = np.where(dmin < float(self.cfg.get("policy", {}).get("obs_radius", 0.6)))[0]
        if len(near):
            pts, rads = self.obs_pts[near], self.obs_rad[near]
            clear, wit = self.arm.clearance(pts, rads)
            self._clear, self._witness = clear, wit
            self._min_clear = float(np.min(clear))
            order = np.argsort(clear)[:K]
            rel[:len(order)] = pts[order] - wit[order]
            dist[:len(order)] = np.clip(clear[order], 0, OBS_FAR)
            obs[S["min_clearance"]] = float(np.clip(self._min_clear, 0, OBS_FAR))
        else:
            self._min_clear = np.inf
            obs[S["min_clearance"]] = OBS_FAR
        fw = self._contacts.get("wrist", np.zeros(3)) if self._contacts else np.zeros(3)
        fs = float(self.cfg.get("policy", {}).get("wrist_force_scale", 20.0))
        rel[3] = np.clip(fw / fs, -5, 5)
        dist[3] = min(float(np.linalg.norm(fw)) / fs, 5.0)
        obs[S["obs_rel"]] = rel.reshape(-1)
        obs[S["obs_dist"]] = dist
        ps = self.cfg.get("policy", {})
        # [v6] the pull-out (retreat) drags the fruit through the leaves: slow; the lift moderate; transport fast
        speed = (float(ps.get("retreat_speed_scale", self.carry_speed)) if self.phase == 2 else
                 float(ps.get("lift_speed_scale", self.carry_speed)) if self.phase == 3 else
                 self.carry_speed if self.phase == 4 else
                 float(ps.get("line_speed_scale", 0.3)) if self.phase == 1 else float(ps.get("approach_speed_scale", 0.4)))
        self._ref_action = self._governed(self.reference.act(tcp, sim.tcp_quat(), self.arm.target_quat, self.route,
                                                             self.arm.max_pos_delta * self._carry_gain(), self.arm.max_rot_delta, speed))
        obs[S["ref_action"]] = self._ref_action
        return obs.astype(np.float32)

    def _get_info(self) -> dict:
        info = dict(phase=int(self.phase), steps_total=int(self.step_count), failure=self.failure,
                    n_placed=int(self.n_placed), n_targets=self._n_targets(),
                    is_success=bool(self.done and not self.failure and self.n_placed >= self._n_targets()),
                    leaf_bend_deg=float(np.degrees(self._leaf_bend)),
                    leaf_root_bend_max_deg=float(np.degrees(self.root_bend_max)), leaf_damage=int(self.leaf_damage),
                    min_clearance=float(self._min_clear) if np.isfinite(self._min_clear) else float(OBS_FAR),
                    shield_events=int(getattr(self, "shield_events", 0)),
                    sim_time_s=float(self.now()), compute_s=float(self.compute_s), fail_phase=str(self.fail_phase),
                    compute_by={k: round(v, 2) for k, v in self.compute_by.items()},
                    residual_off_events=int(getattr(self, "residual_off_events", 0)))
        if self.done:
            info["termination_reason"] = "success" if info["is_success"] else self.failure
            info["targets"] = [vars(r).copy() for r in self.records]
            info["timing"] = dict(self.timing)
        return info

    # ================================================================== helpers for scripted evaluation
    def run_episode(self, policy=None, seed=None, max_rl_steps: int = 5000) -> dict:
        """Run a full episode with `policy(obs) -> a_rl` (None = reference controller alone)."""
        obs, info = self.reset(seed=seed)
        ret, n = 0.0, 0
        while not self.done and n < max_rl_steps:
            a = np.zeros(6) if policy is None else policy(obs)
            obs, r, te, tr, info = self.step(a)
            ret += r
            n += 1
            if te or tr:
                break
        self.finish_session()
        info = self._get_info()
        info["return"] = ret
        info["rl_steps"] = n
        info["events"] = list(self.events)
        return info
