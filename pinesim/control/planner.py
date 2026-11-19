"""
pinesim/control/planner.py
--------------------------
Everything the robot plans from CAMERA estimates (no simulator ground truth):

  scan_views()       wrist-camera survey poses around the robot (find fruit whose position is unknown)
  look_views()       close views of one detected fruit (better fruit centre + leaf cloud for Stargaze)
  plan_approach()    version 5: the ranked (T1, T2) lines of the window decision (pinesim/decision/window.py,
                     diffusion policy) validated in rank order as in the reference Mission 4 `_validate_scenario`.
                     The REFERENCE POINT is the tip of the closed U-jaw (user): the TCP routes are the tip routes
                     shifted back along the tool axis. Routes: approach (current pose -> tip at T1), line (tip T1 -> T2),
                     advance (jaws open, tip moves on until the fruit axis is between the pads). All must be
                     IK-feasible, free of self/floor/platform/basket/vehicle contact, away from wrist singularity,
                     and clear of the PERCEIVED hard plant parts (fruit body, stalk below it) - except the open jaws
                     that straddle the fruit during the advance; leaves are allowed to be touched
  plan_carry()       after the stem broke: retreat (pull out along the line) -> lift -> transport around the base to
                     the basket (polar route, orientation turned with the azimuth so the fruit stays upright)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from pinesim.control.arm import mat2quat, quat2mat, quat_mul, quat_z, tool_quat, unit
from pinesim.control.reference import Route
from pinesim.vendor.harvest_checker import slerp
from pinesim.vendor.pose_utils import quat_angle
from pinesim.vision.stargaze_cam import gripper_spec

RZ_PI = np.array([0.0, 0.0, 0.0, 1.0])


@dataclass
class LinePlan:
    """Approach line of version 5. start / target are TCP positions (what the arm controller tracks); tip_start /
    tip_target are T1 / T2 of the jaw tips (the user's reference point)."""
    start: np.ndarray
    target: np.ndarray
    quat: np.ndarray
    tip_start: np.ndarray
    tip_target: np.ndarray
    approach: np.ndarray
    advance_end: np.ndarray            # TCP after the advance (fruit axis between the pads)
    advance: float                     # tip travel of the advance (m)
    clearance: float
    clear_point: float
    elevation_deg: float
    face: str = "window"
    mode: str = "body"
    votes: float = 0.0
    grip_end: np.ndarray | None = None  # [v6] TCP after centring the pads on the fruit's widest section (along its axis)


@dataclass
class HarvestPlan:
    fruit: object
    line: object                       # vendored ApproachPlan (start=T1, target=T2, quat, face, clearance, ...)
    approach: Route
    line_route: Route
    opening: float
    obstacles: tuple
    hard: tuple
    n_candidates: int = 0
    tried: int = 0
    info: dict = field(default_factory=dict)


class Planner:
    def __init__(self, sim, arm, cfg: dict):
        self.sim, self.arm, self.cfg = sim, arm, cfg
        self.pc = cfg.get("planning", {})
        self.vc = cfg["control"]["validation"]
        self.grip = gripper_spec(cfg)
        self.rng = np.random.default_rng(0)
        self.home_q = None                 # set by the env once the home posture is known

    # ------------------------------------------------------------------ poses
    def base_frame(self):
        """Base position and the robot's 'forward' yaw (vehicle heading, or 0 on the platform)."""
        base = self.sim.base_pos
        yaw = self.sim.vehicle_pose()[1] if self.sim.vehicle_body >= 0 else 0.0
        return base, yaw

    def home_pose(self, az: float | None = None):
        base, yaw = self.base_frame()
        a = yaw if az is None else az
        h = np.asarray(self.cfg["robot"]["home_tcp_local"], float)
        pos = base + np.array([h[0] * np.cos(a) - h[1] * np.sin(a), h[0] * np.sin(a) + h[1] * np.cos(a), h[2]])
        return pos, tool_quat([0, 0, -1], [np.cos(a + np.pi / 2), np.sin(a + np.pi / 2), 0])

    def canonical_seed(self, seed_full, point):
        """Home posture turned toward `point` (azimuth about the base): the 'elbow-up' IK branch used for all views."""
        s = np.array(seed_full, float)
        if self.home_q is None:
            return s
        base, yaw = self.base_frame()
        az = np.arctan2(point[1] - base[1], point[0] - base[0])
        q = self.home_q.copy()
        q[0] = self.home_q[0] + ((az - yaw + np.pi) % (2 * np.pi) - np.pi)
        lo, hi = self.sim.arm_range[0]
        if q[0] < lo + 0.05:                 # keep the pan seed inside the joint range (a view 180 deg behind the
            q[0] += 2 * np.pi                # home azimuth had no IK solution: "nothing detected", measured)
        elif q[0] > hi - 0.05:
            q[0] -= 2 * np.pi
        s[self.sim.arm_qadr] = q
        return s

    def camera_view_q(self, cam_pos, look_dir, seed_full, x_hint=None):
        if x_hint is None:
            h = np.cross([0, 0, 1.0], look_dir)
            x_hint = h if np.linalg.norm(h) > 1e-6 else np.array([1.0, 0, 0])
        p, q = self.arm.tcp_for_camera(cam_pos, look_dir, x_hint)
        canon = self.canonical_seed(seed_full, p)
        best = None
        for s in (canon, np.array(seed_full, float)):
            qa, ok = self.arm.solve_pose(p, q, q_seed_full=s, tries=4, rng=self.rng,
                                         branch_limit=float(self.pc.get("branch_limit", 2.0)),
                                         extra_margin=float(self.pc.get("view_extra_margin", 0.015)))
            if ok:
                cost = float(np.max(np.abs(qa[1:3] - canon[self.sim.arm_qadr][1:3])))
                if best is None or cost < best[1]:
                    best = (qa, cost)
        return (best[0], True) if best is not None else (None, False)

    def scan_views(self, seed_full, azimuths=None):
        """[(azimuth, q_arm)] of survey poses: camera at radius r, height h above the base, looking down/outward."""
        sc = self.pc.get("scan", {})
        base, yaw = self.base_frame()
        r, h, dep = sc.get("radius", 0.45), sc.get("height", 0.35), np.radians(sc.get("depression_deg", 65))
        if azimuths is None:
            azimuths = [yaw + np.radians(a) for a in sc.get("azimuths_deg", [0, 60, 120, 180, 240, 300])]
        out = []
        q_home = seed_full[self.sim.arm_qadr].copy()
        for a in azimuths:
            c = base + np.array([r * np.cos(a), r * np.sin(a), h])
            d = np.array([np.cos(a) * np.cos(dep), np.sin(a) * np.cos(dep), -np.sin(dep)])
            seed = seed_full.copy()
            seed[self.sim.arm_qadr] = q_home
            seed[self.sim.arm_qadr[0]] = q_home[0] + (a - yaw)
            q, ok = self.camera_view_q(c, d, seed, x_hint=[-np.sin(a), np.cos(a), 0.0])
            if ok:
                out.append((float(a), q))
        return out

    def look_views(self, fruit_center, seed_full, want: int | None = None, lateral_pref=None):
        """Close views of one fruit, in order of preference: the robot's side first (azimuth offset 0), then further
        around the fruit. `lateral_pref` (unit vector, world xy): among the offset views, those whose camera lies
        further along this direction come first (Mission 3: the corridor side of the row)."""
        lk = self.pc.get("look", {})
        base, _ = self.base_frame()
        F = np.asarray(fruit_center, float)
        u0 = base[:2] - F[:2]
        u0 /= np.linalg.norm(u0) + 1e-9
        dist = lk.get("distance", 0.70)
        r_lo, r_hi = lk.get("camera_radius_range", [0.32, 1.05])
        want = int(lk.get("n_views", 3)) if want is None else int(want)
        out, used = [], []
        seed = seed_full.copy()
        # candidates in order of preference: robot side first, then further around; steeper views if needed
        offsets = list(lk.get("azimuth_offsets_deg", [0, -35, 35, -70, 70, -110, 110]))
        if lateral_pref is not None:
            lp = np.asarray(lateral_pref, float)[:2]

            def side(off):                       # > 0: the camera of this offset lies on the preferred side
                o = np.radians(off)
                return float(lp @ np.array([u0[0] * np.cos(o) - u0[1] * np.sin(o), u0[0] * np.sin(o) + u0[1] * np.cos(o)])
                             - lp @ u0)
            offsets = sorted(offsets, key=lambda off: (abs(off) > 1e-6, 0 if side(off) > 0 else 1, abs(off)))
        for el_deg in lk.get("elevations_deg", [60, 75]):
            el = np.radians(el_deg)
            for off in offsets:
                if len(out) >= want:
                    return out
                if any(abs(off - u) < 30 for u in used):
                    continue
                o = np.radians(off)
                u = np.array([u0[0] * np.cos(o) - u0[1] * np.sin(o), u0[0] * np.sin(o) + u0[1] * np.cos(o), 0.0])
                c = F + dist * (np.cos(el) * u + np.array([0, 0, np.sin(el)]))
                rc = float(np.hypot(c[0] - base[0], c[1] - base[1]))
                if not (r_lo <= rc <= r_hi):
                    continue
                q, ok = self.camera_view_q(c, F - c, seed)
                if ok:
                    out.append(q)
                    used.append(off)
                    seed = seed.copy()
                    seed[self.sim.arm_qadr] = q
        return out

    # ------------------------------------------------------------------ geometry from perception
    def hard_points(self, fruit, n: int = 6):
        """Perceived rigid parts: fruit axis (radius = estimated radius) and the stalk below it (to the ground)."""
        c = fruit.center
        ax = getattr(fruit, "axis", np.array([0.0, 0.0, 1.0]))
        hh = getattr(fruit, "half_height", 0.5 * (fruit.top_z - fruit.bottom_z))
        hs = np.linspace(-hh + fruit.radius * 0.5, hh - fruit.radius * 0.5, n)
        fp = c[None, :] + hs[:, None] * ax[None, :]
        bot = c - ax * hh
        t = np.linspace(0.0, 1.0, n)[:, None]
        foot = bot - ax * (bot[2] / max(ax[2], 0.5))
        sp = bot[None, :] * (1 - t) + foot[None, :] * t
        sp = sp[sp[:, 2] > 0.02] if np.any(sp[:, 2] > 0.02) else sp
        m = len(sp)
        return (np.vstack([fp, sp]), np.r_[np.full(n, fruit.radius), np.full(m, 0.025)]), (sp, np.full(m, 0.025)), (fp, np.full(n, fruit.radius))

    def reach_filter(self, pts):
        base, _ = self.base_frame()
        R = self.pc.get("start_reach", {})
        rel = pts - base[None, :]
        r = np.linalg.norm(rel[:, :2], axis=1)
        return ((r >= R.get("r_min", 0.25)) & (r <= R.get("r_max", 1.10)) & (rel[:, 2] >= R.get("dz_min", -1.10)) &
                (rel[:, 2] <= R.get("dz_max", 0.40)) & (pts[:, 2] >= R.get("z_min_abs", 0.15)))

    # ------------------------------------------------------------------ approach
    @staticmethod
    def advance_distance(fruit, t2, approach, tip_minus_grip: float) -> float:
        """Tip travel from T2 until the grip point of the open jaws reaches the fruit axis (closest approach of the
        tool line to the axis line) [user: open the gripper at T2, move forward a few cm, grip]."""
        a = np.asarray(approach, float)
        u = getattr(fruit, "axis", np.array([0.0, 0.0, 1.0]))
        w0 = np.asarray(t2, float) - fruit.center
        b = float(a @ u)
        den = 1.0 - b * b
        s_ax = float((b * (w0 @ u) - (w0 @ a)) / den) if den > 1e-6 else float(fruit.radius)
        s_ax = float(np.clip(s_ax, 0.5 * fruit.radius, 2.0 * fruit.radius))
        return s_ax + tip_minus_grip

    def plan_approach(self, fruit, decision, leaf_pts, other_fruits, q_full, x_hint=None, exclude_starts=(),
                      obstacles=None):
        """Validate the ranked window lines (best first) and return the first feasible plan."""
        sim, chk = self.sim, self.arm.checker
        base, _ = self.base_frame()
        x_hint = sim.tcp_rot()[:, 0] if x_hint is None else x_hint
        cands = list(decision.candidates) if decision is not None and decision.ok else []
        if len(exclude_starts):                 # lines that already failed (T1 within 6 cm)
            ex = np.asarray(exclude_starts, float).reshape(-1, 3)
            cands = [c for c in cands if np.min(np.linalg.norm(ex - c.t1[None, :], axis=1)) > 0.06]
        if not cands:
            return None, ("window_no_line" if decision is None or decision.ok else f"window_{decision.reason}")
        (hp, hr), (sp, sr), (fp, fr) = self.hard_points(fruit)
        oth_p, oth_r = [], []
        for of in other_fruits:
            (p_, r_), _, _ = self.hard_points(of)
            oth_p.append(p_); oth_r.append(r_)
        if oth_p:
            oth_p, oth_r = np.vstack(oth_p), np.concatenate(oth_r)
        else:
            oth_p, oth_r = np.zeros((0, 3)), np.zeros(0)
        chk.sync_from(sim.d)
        q0 = np.array(q_full, float)
        q0[sim.finger_qadr] = sim.hinge_for_gap(self.cfg["gripper"]["min_opening"])      # closed while approaching
        tcp0, quat0 = sim.tcp_pos, sim.tcp_quat()
        am, lm = float(self.vc.get("approach_hard_margin", 0.025)), float(self.vc.get("line_hard_margin", 0.008))
        nonfinger = ~self.arm.finger_mask
        lp = np.asarray(leaf_pts, float).reshape(-1, 3)
        if len(lp):
            from pinesim.vision.perception import voxel_down
            near = np.linalg.norm(lp - fruit.center[None, :], axis=1) < 0.9
            lp = voxel_down(lp[near], 0.03)
        lr = np.full(len(lp), float(self.cfg["perception"].get("leaf_point_radius", 0.015)))
        tip_minus_grip = sim.tip_offset - sim.grip_offset
        grip_minus_tcp = sim.grip_offset
        why = {}
        self.last_reject = why
        tried = 0
        hard_all = (np.vstack([hp, oth_p]) if len(oth_p) else hp, np.concatenate([hr, oth_r]) if len(oth_r) else hr)
        for rank, c in enumerate(cands):
            a = c.approach / (np.linalg.norm(c.approach) + 1e-12)
            q_base = tool_quat(a, x_hint)
            adv = self.advance_distance(fruit, c.t2, a, tip_minus_grip)
            for quat in (q_base, quat_mul(q_base, RZ_PI)):
                tried += 1
                P1, P2 = sim.tcp_for_tip(c.t1, quat), sim.tcp_for_tip(c.t2, quat)
                P3 = P2 + a * adv
                ok, q1, segs_a, wps, quats = chk.approach_q(q0, tcp0, P1, quat0, quat, base[:2])
                if not ok:
                    jr = self.joint_route(q0, P1, quat, "approach", hard=hard_all, margin=am)
                    if jr is None:
                        why["approach_ik_or_contact"] = why.get("approach_ik_or_contact", 0) + 1
                        continue
                    route_j, segs_a, q1 = jr
                    wps, quats = route_j.wps[1:], route_j.quats[1:]
                if chk.hard_clearance(segs_a, hp, hr) < am or chk.hard_clearance(segs_a, oth_p, oth_r) < am:
                    why["approach_hits_fruit_or_stalk"] = why.get("approach_hits_fruit_or_stalk", 0) + 1
                    continue
                line = chk.waypoints(P1, P2)
                ok, q2, segs_b = chk.rollout_q(q1, line, [quat] * len(line))
                if not ok:
                    why["line_ik_or_contact"] = why.get("line_ik_or_contact", 0) + 1
                    continue
                if chk.hard_clearance(segs_b, hp, hr, nonfinger) < lm:  # wrist / housing stay off the fruit
                    why["line_arm_hits_fruit"] = why.get("line_arm_hits_fruit", 0) + 1
                    continue
                if chk.hard_clearance(segs_b, fp, 0.8 * fr, self.arm.finger_mask) < 0.0:   # closed jaws end at T2
                    why["line_jaws_hit_fruit"] = why.get("line_jaws_hit_fruit", 0) + 1
                    continue
                if chk.hard_clearance(segs_b, oth_p, oth_r) < lm:
                    why["line_hits_other_fruit"] = why.get("line_hits_other_fruit", 0) + 1
                    continue
                # [v6] grip at / just below the fruit's widest section: T2 lies on the SHOULDER (above the centre), so
                # flat pads closing there hold a taper above the centre of mass - the fruit hung like a pendulum,
                # slipped down and fell while carrying, and slid out of the pads when pulled (measured). With the
                # jaws astride the fruit the hand moves along the fruit axis until the pad centre is
                # grip_height_target (+- grip_height_tol) from the estimated fruit centre: slightly BELOW the centre
                # the widest section sits above the pads, which locks the fruit both against gravity and against
                # the upward pull that snaps the stalk.
                G = P3 + a * grip_minus_tcp
                h3 = float((G - fruit.center) @ fruit.axis)
                tol = float(self.pc.get("grip_height_tol", 0.01))
                tgt = float(self.pc.get("grip_height_target", 0.0))
                drop = h3 - float(np.clip(h3, tgt - tol, tgt + tol))
                P3c = P3 - fruit.axis * drop
                adv_w = chk.waypoints(P2, P3)
                if abs(drop) > 0.003:
                    adv_w = np.vstack([adv_w, chk.waypoints(P3, P3c)])
                q2o = q2.copy()
                q2o[sim.finger_qadr] = sim.hinge_for_gap(self.cfg["gripper"]["max_opening"])
                ok, _, segs_c = chk.rollout_q(q2o, adv_w, [quat] * len(adv_w))
                if not ok:
                    why["advance_ik_or_contact"] = why.get("advance_ik_or_contact", 0) + 1
                    continue
                if chk.hard_clearance(segs_c, hp, hr, nonfinger) < lm:  # wrist / housing must not reach the fruit
                    why["advance_arm_hits_fruit"] = why.get("advance_arm_hits_fruit", 0) + 1
                    continue
                if chk.hard_clearance(segs_c, sp, sr, self.arm.finger_mask) < 0.004:
                    why["advance_jaws_hit_stalk"] = why.get("advance_jaws_hit_stalk", 0) + 1
                    continue
                intr = 0.0
                if len(lp):
                    clr = chk.min_clearance_along(segs_a + segs_b, lp, lr)
                    intr = float(np.sum(np.clip(0.02 - clr, 0.0, None)))
                lplan = LinePlan(start=P1, target=P2, quat=quat, tip_start=c.t1.copy(), tip_target=c.t2.copy(),
                                 approach=a, advance_end=P3, advance=float(adv), clearance=float(c.clear_line),
                                 clear_point=float(c.clear_point), elevation_deg=float(c.elevation_deg), votes=c.votes,
                                 grip_end=P3c)
                obs = obstacles if obstacles is not None else (np.zeros((0, 3)), np.zeros(0))
                plan = HarvestPlan(fruit=fruit, line=lplan, approach=Route(wps, quats, "approach"),
                                   line_route=Route(np.vstack([P1[None, :], line]), [quat] * (len(line) + 1), "line"),
                                   opening=float(self.cfg["gripper"]["max_opening"]), obstacles=obs, hard=(hp, hr),
                                   n_candidates=len(cands), tried=tried,
                                   info=dict(face="window", clearance=float(c.clear_line), rank=rank, votes=float(c.votes),
                                             elevation_deg=float(c.elevation_deg), leaf_intrusion=intr,
                                             advance=float(adv), n_feasible=1))
                return plan, "ok"
        return None, "no_feasible_route"

    # ------------------------------------------------------------------ carry
    def drop_postures(self, D, q_seed_full, qbase):
        """Feasible arm postures with the TCP at the drop point D: hand inclined as when gripping (yaw free) or
        pointing straight down; canonical elbow-up branch turned toward the basket."""
        sim = self.sim
        out = []
        canon = self.canonical_seed(q_seed_full, D)
        Rb = quat2mat(qbase)
        z0 = Rb[:, 2]
        quats = [quat_mul(quat_z(y), qbase) for y in np.radians([0, 90, -90, 180, 45, -45])]
        quats += [tool_quat([0, 0, -1.0], [np.cos(y), np.sin(y), 0.0]) for y in np.radians([0, 90, 180, 270])]
        for q in quats:
            qa, ok = self.arm.solve_pose(D, q, q_seed_full=canon, tries=3, rng=self.rng, branch_limit=2.5)
            if ok:
                out.append(qa)
        return out

    def joint_route(self, q_from_full, pos, quat, name: str, hard=None, margin: float = 0.025, step: float = 0.03):
        """Collision-checked joint-space path to an IK solution of (pos, quat), returned as a Cartesian Route (FK
        samples) for the reference controller, plus the segments (for clearance checks) and the final full state."""
        sim, chk = self.sim, self.arm.checker
        q_from = q_from_full[sim.arm_qadr].copy()
        canon = self.canonical_seed(q_from_full, pos)
        sols = []
        for seed in (np.array(q_from_full, float), canon):
            qa, ok = self.arm.solve_pose(pos, quat, q_seed_full=seed, tries=3, rng=self.rng, branch_limit=2.5)
            if ok:
                sols.append(self._unwrap(qa, q_from))
        sols.sort(key=lambda qa: float(np.max(np.abs(qa - q_from))))
        from pinesim.vendor.robot_geometry import world_segments
        # (via-home chains were tried and removed: the Cartesian controller re-solving IK along such long FK routes
        #  switched branches -> measured self-collision on the vehicle)
        chains = [[qa] for qa in sols]
        for chain in chains:
            qa = chain[-1]
            pts = [q_from] + list(chain)
            path = np.vstack([self.arm.joint_path(a, b, step) for a, b in zip(pts[:-1], pts[1:])])
            full = np.array(q_from_full, float)
            wps, quats, segs, ok = [sim.tcp_pos.copy()], [sim.tcp_quat()], [], True
            for q in path:
                full[sim.arm_qadr] = q
                if chk.has_contact(full):
                    ok = False
                    break
                segs.append(world_segments(self.arm.caps, chk._data))
                p, R = self.arm.ik.site_pose(full)
                qq = mat2quat(R)
                if np.linalg.norm(p - wps[-1]) >= 0.03 or quat_angle(qq, quats[-1]) > np.radians(5.0) or q is path[-1]:
                    wps.append(p)
                    quats.append(qq)
            if not ok:
                continue
            if hard is not None and len(hard[0]) and chk.hard_clearance(segs, hard[0], hard[1]) < margin:
                continue
            return Route(np.array(wps), np.array(quats), name), segs, full
        return None

    def carried_fruit_clear(self, tcp_pts, fruit, margin: float = 0.03) -> bool:
        """Analytic clearance of the carried fruit (sphere of the fruit half-height below the TCP) against the
        basket walls and the robot column / vehicle deck: its lowest point must stay above them where it passes."""
        c, R = self.sim.basket_frame()
        b = self.cfg["basket"]
        il, iw, ih = b["inner_size"]
        fh = max(fruit.top_z - fruit.bottom_z, 2 * fruit.radius)
        reach = 0.5 * fh + fruit.radius
        rim = c[2] + ih
        base, _ = self.base_frame()
        top = base[2] - float(self.cfg["robot"]["base_cylinder_half"])
        ph = self.cfg.get("platform", {}).get("half_size", [0.18, 0.18])
        for p in tcp_pts:
            low = p[2] - reach - margin
            loc = R.T @ (p - c)
            inside = abs(loc[0]) < il / 2 - fruit.radius and abs(loc[1]) < iw / 2 - fruit.radius
            near = abs(loc[0]) < il / 2 + 0.02 + reach and abs(loc[1]) < iw / 2 + 0.02 + reach
            if near and not inside and low < rim:
                return False
            if self.sim.vehicle_body < 0 and abs(p[0] - base[0]) < ph[0] + reach and abs(p[1] - base[1]) < ph[1] + reach \
                    and low < top:
                return False
        return True

    def _transport_joint_route(self, q2_full, E, D, qbase, fruit=None):
        """Joint path (collision-checked) to a posture above the drop point, then a straight vertical descent."""
        sim = self.sim
        q_from = q2_full[sim.arm_qadr].copy()
        pre = D + np.array([0.0, 0.0, float(self.pc.get("carry", {}).get("predrop_height", 0.15))])
        cands = self.drop_postures(pre, q2_full, qbase)
        cands.sort(key=lambda qa: float(np.max(np.abs(self._unwrap(qa, q_from) - q_from))))
        for qa in cands[:5]:
            qa = self._unwrap(qa, q_from)
            if not self.arm.joint_path_ok(q_from, qa, base_full=q2_full):
                continue
            path = self.arm.joint_path(q_from, qa, 0.03)
            full = q2_full.copy()
            wps, quats, allp = [E.copy()], [None], []
            for q in path:
                full[sim.arm_qadr] = q
                p, R = self.arm.ik.site_pose(full)
                allp.append(p)
                qq = mat2quat(R)
                big_turn = quats[-1] is not None and quat_angle(qq, quats[-1]) > np.radians(5.0)
                if np.linalg.norm(p - wps[-1]) >= 0.03 or big_turn or q is path[-1]:
                    wps.append(p)
                    quats.append(qq)
            if fruit is not None and not self.carried_fruit_clear(allp, fruit):
                continue
            quats[0] = quats[1] if len(quats) > 1 else qbase
            q_end_quat = quats[-1]
            # vertical descent pre -> D with the final orientation (IK-checked)
            full[sim.arm_qadr] = qa
            down = self.arm.checker.waypoints(wps[-1], D)
            ok, q_end, _ = self.arm.checker.rollout_q(full, down, [q_end_quat] * len(down))
            if not ok:
                continue
            wps += list(down)
            quats += [q_end_quat] * len(down)
            return Route(np.array(wps), np.array(quats), "transport"), q_end
        return None

    def _unwrap(self, q, ref):
        """Representation of q (+/- 2 pi per joint) closest to ref INSIDE the joint limits (reference IK helper)."""
        return self.arm.ik.unwrap_to_current(np.array(q, float), np.asarray(ref, float))

    def drop_point(self, fruit, offset=None):
        """Release point above the active crate. `offset` (default: self.drop_offset, set by the environment) is the
        place on the crate floor in the crate frame [v6]: every fruit used to be released above the crate CENTRE, so
        from the 5th fruit on it landed upright on top of an upright earlier fruit and stayed there, above the rim
        (measured in Mission 3: `basket_miss` exactly at fruits 5 and 8 of a crate)."""
        c, R = self.sim.basket_frame()
        b = self.cfg["basket"]
        fh = max(fruit.top_z - fruit.bottom_z, 2 * fruit.radius)
        off = getattr(self, "drop_offset", None) if offset is None else offset
        if off is not None:
            c = c + R @ np.array([float(off[0]), float(off[1]), 0.0])
        return c + np.array([0.0, 0.0, b["inner_size"][2] + b["drop_clearance"] + 0.5 * fh + 0.02])

    def plan_carry(self, plan: HarvestPlan, grasp_point, q_full, prefer: str | None = None):
        """Retreat (pull out) -> lift -> transport, tried with several variants until one is IK/collision feasible:
        retreat along the Stargaze line or straight up; lift heights; orientation = post-twist / pre-twist;
        transport with the orientation turned with the azimuth (fruit upright) or kept constant."""
        sim, chk = self.sim, self.arm.checker
        base, _ = self.base_frame()
        cc = self.pc.get("carry", {})
        D = self.drop_point(plan.fruit)
        chk.sync_from(sim.d)
        q_cur = sim.tcp_quat()
        line_dir = unit(plan.line.start - plan.line.target)
        # pull out at least back to T1 (+5 cm): T1 is the leaf-free start of the Stargaze line
        dist = float(cc.get("retreat_distance", 0.20))
        if cc.get("retreat_to_t1", True):
            dist = max(dist, float(np.linalg.norm(plan.line.start - plan.line.target)) + float(cc.get("retreat_beyond_t1", 0.05)))
        retreats = [("line", grasp_point + line_dir * dist), ("up", grasp_point + np.array([0, 0, dist]))]
        if prefer is not None:                   # (re-plan after the leaf guard: the other direction first)
            retreats.sort(key=lambda r_: r_[0] != prefer)
        why = {}
        self.last_carry_reject = why
        for rname, R in retreats:
            for qbase in (q_cur, plan.line.quat):
                r_w = chk.waypoints(grasp_point, R)
                ok, q1, _ = chk.rollout_q(q_full, r_w, [qbase] * len(r_w))
                if not ok:
                    why[f"retreat_{rname}"] = why.get(f"retreat_{rname}", 0) + 1
                    continue
                # lift: straight up to the drop height first (as before); when that is out of reach (far plant /
                # awkward hand orientation: every lift failed -> "carry_infeasible", measured), lower lifts whose
                # joint-space transport climbs the rest [v6]
                lifts = [(lf, True) for lf in cc.get("lift_heights", [0.15, 0.30, 0.45])]
                lifts += [(lf, False) for lf in cc.get("lift_heights_low", [0.20, 0.10, 0.30])]
                seen_E = []
                for lift, clamp in lifts:
                    E = R + np.array([0.0, 0.0, lift])
                    if clamp:
                        E[2] = max(E[2], D[2])
                    if any(abs(E[2] - z) < 0.02 for z in seen_E):
                        continue
                    seen_E.append(float(E[2]))
                    l_w = chk.waypoints(R, E)
                    ok, q2, _ = chk.rollout_q(q1, l_w, [qbase] * len(l_w))
                    if not ok:
                        why[f"lift_{rname}"] = why.get(f"lift_{rname}", 0) + 1
                        continue
                    why["transport"] = why.get("transport", 0) + 1
                    # transport 1: joint-space path to a feasible drop posture above the basket, converted to a
                    # Cartesian route (FK), so the reference controller + residual policy still act in Cartesian space
                    tr = self._transport_joint_route(q2, E, D, qbase, plan.fruit)
                    if tr is not None:
                        t_route, q3 = tr
                        return dict(retreat=Route(np.vstack([grasp_point, r_w]), [qbase] * (len(r_w) + 1), "retreat"),
                                    lift=Route(np.vstack([R, l_w]), [qbase] * (len(l_w) + 1), "lift"),
                                    transport=t_route, R=R, E=E, D=D, q_end=q3,
                                    variant=dict(retreat=rname, lift=lift, transport="joint")), "ok"
                    # transport 2: polar Cartesian route around the base (hand yaw free about the vertical)
                    t_w = chk.polar_waypoints(E, D, base[:2])
                    a0 = np.arctan2(E[1] - base[1], E[0] - base[0])
                    a1 = np.arctan2(D[1] - base[1], D[0] - base[0])
                    dth = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
                    n = len(t_w)
                    for turn in (True, False):
                        for extra in np.radians(cc.get("transport_extra_yaw_deg", [0, 90, -90])):
                            t_q = [quat_mul(quat_z((dth if turn else 0.0) * (k + 1) / n + extra * (k + 1) / n), qbase)
                                   for k in range(n)]
                            ok, q3, _ = chk.rollout_q(q2, t_w, t_q)
                            if not ok:
                                continue
                            return dict(retreat=Route(np.vstack([grasp_point, r_w]), [qbase] * (len(r_w) + 1), "retreat"),
                                        lift=Route(np.vstack([R, l_w]), [qbase] * (len(l_w) + 1), "lift"),
                                        transport=Route(np.vstack([E, t_w]), [qbase] + list(t_q), "transport"),
                                        R=R, E=E, D=D, q_end=q3,
                                        variant=dict(retreat=rname, lift=lift, transport="polar", turn=turn,
                                                     extra_yaw=float(np.degrees(extra)))), "ok"
        return None, "carry_infeasible"
