"""
pinesim/control/arm.py
----------------------
Arm kinematics and validation built on the vendored reference components:

  * IK: damped least squares (pinesim/vendor/ik_solver.py, reference ik/ik_solver.py)
  * capsule model of arm + custom gripper (clearance to point obstacles: leaves, perceived stalk/fruit)
  * RouteChecker = reference HarvestChecker (IK roll-outs with per-waypoint orientation, wrist-singularity and
    joint-jump rejection) whose collision test also covers the platform, basket, vehicle and robot base
  * command(): Cartesian action -> IK -> joint targets with the reference Mission 4 orientation-lead limit and
    uniform joint-step scaling
  * pose helpers: tool orientation from an approach direction, camera-pose -> TCP-pose conversion
"""
from __future__ import annotations

import mujoco
import numpy as np

from pinesim.vendor.harvest_checker import HarvestChecker, slerp
from pinesim.vendor.ik_solver import IKConfig, IKSolver
from pinesim.vendor.pose_utils import integrate_quat, limit_quat_deviation, quat_angle
from pinesim.vendor.robot_geometry import RobotCapsules, obstacle_clearance, world_segments
from pinesim.model.builder import ARM_JOINTS


def unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def quat_z(angle: float) -> np.ndarray:
    return np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])


def mat2quat(R) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def quat2mat(q) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def tool_quat(z_dir, x_hint=(1.0, 0.0, 0.0)) -> np.ndarray:
    """Tool orientation with z = approach direction and x (finger closing axis) horizontal, close to x_hint."""
    z = unit(z_dir)
    if abs(z[2]) > 0.97:
        x = np.asarray(x_hint, float) - np.dot(x_hint, z) * z
        x[2] = 0.0
        if np.linalg.norm(x) < 1e-6:
            x = np.array([1.0, 0.0, 0.0])
    else:
        x = np.cross([0.0, 0.0, 1.0], z)
        if np.dot(x, x_hint) < 0:
            x = -x
    x = unit(x - np.dot(x, z) * z)
    return mat2quat(np.column_stack([x, np.cross(z, x), z]))


def build_arm_capsules(model: mujoco.MjModel, body_ids) -> RobotCapsules:
    """Capsules for every capsule / cylinder / box geom of the arm bodies (a box -> capsule along its longest axis)."""
    out = dict(b=[], p0=[], p1=[], r=[], n=[])
    mat = np.zeros(9)
    bodyset = set(int(b) for b in body_ids)
    for g in range(model.ngeom):
        bid = int(model.geom_bodyid[g])
        if bid not in bodyset or model.geom_contype[g] == 0:
            continue
        t = int(model.geom_type[g])
        mujoco.mju_quat2Mat(mat, model.geom_quat[g])
        R = mat.reshape(3, 3)
        c, sz = model.geom_pos[g].copy(), model.geom_size[g]
        if t in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
            axis, half, r = R[:, 2], float(sz[1]), float(sz[0])
        elif t == int(mujoco.mjtGeom.mjGEOM_BOX):
            k = int(np.argmax(sz))
            axis, half = R[:, k], float(sz[k])
            r = float(np.sqrt(sum(sz[j] ** 2 for j in range(3) if j != k)))
        else:
            continue
        out["b"].append(bid); out["p0"].append(c - axis * half); out["p1"].append(c + axis * half)
        out["r"].append(r); out["n"].append(f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)}:g{g}")
    return RobotCapsules(np.array(out["b"], int), np.array(out["p0"]), np.array(out["p1"]), np.array(out["r"]), out["n"])


class RouteChecker(HarvestChecker):
    """Reference HarvestChecker; contact test = robot vs robot / floor / platform / basket / vehicle / base."""

    def __init__(self, model, ik, caps, robot_geoms, env_geoms, floor_geom, finger_mask, vcfg: dict):
        super().__init__(model, ik, caps, robot_geoms, floor_geom, finger_mask,
                         pos_tol=float(vcfg.get("ik_pos_tol", 3e-3)), rot_tol=float(vcfg.get("ik_rot_tol", 0.04)),
                         waypoint_step=float(vcfg.get("waypoint_step", 0.04)))
        self.env_geoms = np.asarray(env_geoms, bool)
        self.finger_bodies = set()
        self.self_pairs, self.self_margin = [], 0.0
        self.env_pairs, self.env_margin = [], 0.0

    def sync_from(self, data: mujoco.MjData):
        """Copy the non-arm state (vehicle pose, plants) into the scratch data."""
        self._data.qpos[:] = data.qpos
        self._data.mocap_pos[:] = data.mocap_pos
        self._data.mocap_quat[:] = data.mocap_quat

    def has_contact(self, q_full: np.ndarray, extra: float = 0.0) -> bool:
        """`extra` (m) widens the clearance margins (camera view poses keep more room than the routes that start
        from them need)."""
        d = self._data
        d.qpos[:] = q_full
        # kinematics + collision only (no dynamics): the same contacts as mj_forward at 1/3 of the time; this test
        # runs thousands of times per fruit and its wall time counts as the robot's planning time [v6]
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_collision(self.model, d)
        m = self.model
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            r1, r2 = self.robot_geoms[g1], self.robot_geoms[g2]
            if not (r1 or r2):
                continue
            if r1 and r2:
                b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
                if b1 in self.finger_bodies and b2 in self.finger_bodies:
                    continue
                return True
            o = g2 if r1 else g1
            if o == self.floor_geom or self.env_geoms[o]:
                return True
        # self-clearance margin: the servo-driven arm lags the planned joint path, so a path that only just clears
        # (upper arm vs wrist / gripper, forearm vs gripper) collided in the simulation (measured, Mission 3)
        if self.self_pairs:
            ft = self._ft
            for a, b in self.self_pairs:
                if mujoco.mj_geomDistance(m, d, a, b, self.self_margin + extra + 0.01, ft) < self.self_margin + extra:
                    return True
            for a, b in self.env_pairs:
                if mujoco.mj_geomDistance(m, d, a, b, self.env_margin + extra + 0.01, ft) < self.env_margin + extra:
                    return True
        return False

    def margin_distance(self, q_full: np.ndarray) -> float:
        """Smallest distance (m) over the margin-checked geom pairs at q_full (capped at margin + 2 cm); -1 when the
        posture is in real contact (robot / robot, robot / environment, floor)."""
        d, m = self._data, self.model
        d.qpos[:] = q_full
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            r1, r2 = self.robot_geoms[g1], self.robot_geoms[g2]
            if r1 and r2:
                b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
                if b1 in self.finger_bodies and b2 in self.finger_bodies:
                    continue
                return -1.0
            if (r1 or r2) and ((g2 if r1 else g1) == self.floor_geom or self.env_geoms[g2 if r1 else g1]):
                return -1.0
        best = np.inf
        for pairs, mg in ((self.self_pairs, self.self_margin), (self.env_pairs, self.env_margin)):
            for a, b in pairs:
                best = min(best, mujoco.mj_geomDistance(m, d, a, b, mg + 0.02, self._ft))
        return float(best)

    def contact_reason(self, q_full: np.ndarray) -> str:
        """Which test of has_contact fails for q_full (diagnostics of blocked routes)."""
        d, m = self._data, self.model
        d.qpos[:] = q_full
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        nm = lambda g: (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or
                        "body:" + (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "?"))
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            r1, r2 = self.robot_geoms[g1], self.robot_geoms[g2]
            if r1 and r2:
                b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
                if b1 in self.finger_bodies and b2 in self.finger_bodies:
                    continue
                return f"contact {nm(g1)} / {nm(g2)}"
            if (r1 or r2) and ((g2 if r1 else g1) == self.floor_geom or self.env_geoms[g2 if r1 else g1]):
                return f"contact {nm(g1)} / {nm(g2)}"
        for pairs, mg in ((self.self_pairs, self.self_margin), (self.env_pairs, self.env_margin)):
            for a, b in pairs:
                dist = mujoco.mj_geomDistance(m, d, a, b, mg + 0.01, self._ft)
                if dist < mg:
                    return f"margin {nm(a)} / {nm(b)} ({dist * 1000:.0f} mm)"
        return "none"

    def set_self_pairs(self, model, margin: float):
        """Geom pairs (collision geoms) checked with a clearance margin in has_contact."""
        def geoms(names):
            ids = set()
            for nm in names:
                try:
                    ids.add(model.body(nm).id)
                except KeyError:
                    pass
            return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in ids
                    and (model.geom_contype[g] or model.geom_conaffinity[g])]
        tool = ["wrist_1_link", "wrist_2_link", "wrist_3_link", "gripper_housing", "finger_left", "finger_right"]
        ua, fa = geoms(["upper_arm_link"]), geoms(["forearm_link"])
        gr = geoms(tool[3:])
        self.self_pairs = [(a, b) for a in ua for b in geoms(tool)] + [(a, b) for a in fa for b in gr]
        self.self_margin = float(margin)
        self._ft = np.zeros(6)
        # arm vs platform / basket with a clearance margin as well (wrist vs platform collided during an approach
        # that the route check had passed, realistic evaluation); only for small static scenes (the platform)
        envg = [g for g in range(model.ngeom) if self.env_geoms[g] and (model.geom_contype[g] or model.geom_conaffinity[g])]
        arm = geoms(["forearm_link"] + tool)
        self.env_pairs = [(a, b) for a in arm for b in envg] if len(envg) <= 40 else []
        self.env_margin = float(margin)


class Arm:
    def __init__(self, sim, cfg: dict):
        self.sim = sim
        self.cfg = cfg
        c = cfg["control"]
        self.ik = IKSolver(sim.m, "tcp_site", ARM_JOINTS, IKConfig(**c["ik"]))
        self.caps = build_arm_capsules(sim.m, sim.arm_body_ids)
        self.finger_body_ids = set(sim.arm_body_ids[-2:])
        self.finger_mask = np.isin(self.caps.body_ids, list(self.finger_body_ids))
        self.checker = RouteChecker(sim.m, self.ik, self.caps, sim.robot_geom, sim.env_geom, sim.floor_geom,
                                    self.finger_mask, c["validation"])
        self.checker.finger_bodies = self.finger_body_ids | {sim.housing_body}
        self.checker.set_self_pairs(sim.m, float(c["validation"].get("self_margin", 0.01)))
        self.max_pos_delta = float(c["max_pos_delta"])
        self.pos_gain = 1.0
        self.max_rot_delta = float(c["max_rot_delta"])
        self.max_joint_delta = float(c["max_joint_delta"])
        self.orient_lead = np.deg2rad(float(c["max_orient_lead_deg"]))
        self.target_quat = np.array([1.0, 0, 0, 0])

    # ------------------------------------------------------------------ commands
    def command(self, action: np.ndarray) -> tuple[np.ndarray, dict]:
        """Normalised action [-1,1]^6 (dx dy dz in max_pos_delta, drot in max_rot_delta, world frame) -> joint targets.
        Reference Mission 4 `_command`: orientation target at most `orient_lead` ahead of the actual orientation,
        uniform scaling of the joint step."""
        sim = self.sim
        a = np.clip(np.asarray(action, float), -1, 1)
        actual = sim.tcp_quat()
        base_q = limit_quat_deviation(self.target_quat, actual, self.orient_lead)
        quat = integrate_quat(base_q, a[3:] * self.max_rot_delta)
        quat = limit_quat_deviation(quat, actual, self.orient_lead + self.max_rot_delta)
        q_now = sim.q_arm()
        q_t, info = self.ik.solve(sim.d.qpos, sim.tcp_pos + a[:3] * self.max_pos_delta * self.pos_gain, quat)
        dq = q_t - q_now
        big = float(np.max(np.abs(dq)))
        if big > self.max_joint_delta:
            dq *= self.max_joint_delta / big
        self.target_quat = quat
        return q_now + dq, info

    def servo_to(self, pos, quat, max_joint_step: float | None = None):
        """One control step toward an absolute TCP pose (scripted phases)."""
        sim = self.sim
        q_t, info = self.ik.solve(sim.d.qpos, np.asarray(pos, float), np.asarray(quat, float))
        q_now = sim.q_arm()
        dq = q_t - q_now
        lim = self.max_joint_delta if max_joint_step is None else max_joint_step
        big = float(np.max(np.abs(dq)))
        if big > lim:
            dq *= lim / big
        sim.d.ctrl[sim.arm_act] = q_now + dq
        return info

    # ------------------------------------------------------------------ geometry
    def clearance(self, points: np.ndarray, radii: np.ndarray, data=None):
        if len(points) == 0:
            return np.zeros(0), np.zeros((0, 3))
        return obstacle_clearance(self.caps, self.sim.d if data is None else data, points, radii)

    def tcp_for_camera(self, cam_pos, view_dir, x_hint=(1.0, 0.0, 0.0)):
        """TCP pose that puts the wrist camera at cam_pos looking along view_dir (camera axis = tool z)."""
        q = quat_mul(tool_quat(view_dir, x_hint), np.array([0.0, 0.0, 0.0, 1.0]))
        R = quat2mat(q)
        cp = np.asarray(self.cfg["camera"]["pos_in_tool"], float)
        tcp_local = np.array([0.0, 0.0, self.sim.tcp_z])
        return np.asarray(cam_pos, float) + R @ (tcp_local - cp), q

    def solve_pose(self, pos, quat, q_seed_full=None, tries: int = 6, rng=None, branch_limit: float | None = None,
                   extra_margin: float = 0.0):
        """IK from several seeds; returns (q_arm, ok). ok requires small residual, no contact, no wrist singularity."""
        sim = self.sim
        rng = rng or np.random.default_rng(0)
        seeds = [sim.d.qpos.copy() if q_seed_full is None else np.array(q_seed_full, float)]
        for _ in range(tries - 1):
            s = seeds[0].copy()
            s[sim.arm_qadr] = seeds[0][sim.arm_qadr] + rng.uniform(-0.6, 0.6, 6)
            seeds.append(s)
        best = None
        for s in seeds:
            q, info = self.ik.solve(s, np.asarray(pos, float), np.asarray(quat, float))
            if info["position_error"] > 3e-3 or info["rotation_error"] > 0.03:
                continue
            if abs(np.sin(q[4])) < np.sin(np.deg2rad(12.0)):
                continue
            if branch_limit is not None:
                ref = seeds[0][sim.arm_qadr]
                # same IK branch as the reference posture: elbow on the same side, shoulder not flipped over
                if np.sign(q[2]) != np.sign(ref[2]) or abs(q[2]) < 0.15 or abs(q[1] - ref[1]) > branch_limit:
                    continue
            full = s.copy()
            full[sim.arm_qadr] = q
            if self.checker.has_contact(full, extra=extra_margin):
                continue
            cost = float(np.sum((q - seeds[0][sim.arm_qadr]) ** 2))
            if best is None or cost < best[1]:
                best = (q, cost)
        return (self.nearest_equivalent(best[0]), True) if best is not None else (None, False)

    def nearest_equivalent(self, q, q_ref = None, margin: float = 0.10):
        q = np.array(q, float)
        ref = self.sim.q_arm() if q_ref is None else np.asarray(q_ref, float)
        lim = self.sim.arm_range
        for j in (5, ):
            v = q[j] + 2.0*np.pi*np.round((ref[j]-q[j])/(2.0*np.pi))
            if lim[j, 0] +margin <= v <= lim[j, 1] - margin:
                q[j] = v
        return q

    def joint_path(self, q_from, q_to, max_step: float = 0.04) -> np.ndarray:
        n = int(np.ceil(np.max(np.abs(np.asarray(q_to) - np.asarray(q_from))) / max_step)) + 1
        return np.linspace(q_from, q_to, max(n, 2))[1:]

    def joint_path_ok(self, q_from, q_to, base_full=None) -> bool:
        sim = self.sim
        full = (sim.d.qpos.copy() if base_full is None else np.array(base_full, float))
        for q in self.joint_path(q_from, q_to, float(self.cfg.get("planning", {}).get("path_check_step", 0.06))):
            full[sim.arm_qadr] = q
            if self.checker.has_contact(full):
                return False
        return True
