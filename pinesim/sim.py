"""
pinesim/sim.py
--------------
Thin wrapper around one generated MuJoCo scene (platform / platform_detail / xgh_light / xgh_detail / rig): ids,
U-jaw gripper mapping, stepping, contact classification, basket test, indicator lamp. No task logic here.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from pinesim.model.builder import ARM_ACTUATORS, ARM_JOINTS, build
from pinesim.plant import PlantField

ARM_BODIES = ["shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link",
              "gripper_housing", "finger_left", "finger_right"]


class Sim:
    def __init__(self, cfg: dict, mode: str, n_plants: int = 1):
        self.cfg = cfg
        self.mode = mode
        path = build(mode, n_plants, cfg)
        self.model_path = path
        import json as _json
        mp = path.with_name(path.stem + "_meta.json")
        self.meta = _json.loads(mp.read_text()) if mp.exists() else {}
        self.m = m = mujoco.MjModel.from_xml_path(str(path))
        self.d = mujoco.MjData(m)
        self.n_substeps = int(cfg["control"]["n_substeps"])
        self.dt = float(m.opt.timestep) * self.n_substeps
        nid = lambda t, n: mujoco.mj_name2id(m, t, n)
        J, A, B, G, S = (mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR, mujoco.mjtObj.mjOBJ_BODY,
                         mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_SITE)
        self.arm_jid = [nid(J, n) for n in ARM_JOINTS]
        self.arm_qadr = np.array([m.jnt_qposadr[j] for j in self.arm_jid])
        self.arm_dadr = np.array([m.jnt_dofadr[j] for j in self.arm_jid])
        self.arm_act = np.array([nid(A, n) for n in ARM_ACTUATORS])
        self.arm_range = np.array([m.jnt_range[j] for j in self.arm_jid])
        self.finger_qadr = [int(m.jnt_qposadr[nid(J, f"finger_{s}_joint")]) for s in ("left", "right")]
        self.slide_gripper = True                 # version 5: the U-jaw gripper in every mission
        self.finger_act = [nid(A, "finger_left_act"), nid(A, "finger_right_act")]
        self.tcp_site = nid(S, "tcp_site")
        self.tip_site = nid(S, "tip_site")
        self.base_body = nid(B, "base")
        self.housing_body = nid(B, "gripper_housing")
        self.arm_body_ids = [nid(B, n) for n in ARM_BODIES]
        self.floor_geom = nid(G, "floor")
        g = cfg["gripper"]                        # U-jaw parallel gripper: slider 0 = closed, stroke = open
        self.q_open, self.q_closed, self.tcp_z = float(g["stroke"]), 0.0, float(g["tcp_z"])
        self.tip_offset = float(g.get("tip_z", 0.337)) - float(g["tcp_z"])     # tip ahead of the TCP (tool z)
        self.grip_offset = float(g.get("grip_z", 0.262)) - float(g["tcp_z"])
        if mode in ("xgh_detail", "platform_detail"):
            from pinesim.lod import DetailField
            self.field = DetailField(m)
        else:
            self.field = PlantField(m)
        self.field.mute_on_detach = bool(cfg["plant"].get("mute_penetrating_leaves", False))
        root = m.body_rootid
        self.robot_geom = np.zeros(m.ngeom, bool)
        arm_set = set(self.arm_body_ids)
        for g in range(m.ngeom):
            self.robot_geom[g] = int(m.geom_bodyid[g]) in arm_set
        names = [mujoco.mj_id2name(m, G, g) or "" for g in range(m.ngeom)]
        self.geom_names = names
        self.pad_left = {g for g, n in enumerate(names) if n.startswith("finger_left_pad")}
        self.pad_right = {g for g, n in enumerate(names) if n.startswith("finger_right_pad")}
        self.env_geom = np.zeros(m.ngeom, bool)       # static environment the robot must not touch
        for g, n in enumerate(names):
            if n.startswith(("platform", "basket", "chassis", "frame_", "base_geom")):
                self.env_geom[g] = True
        self.basket_geoms = {g for g, n in enumerate(names) if n.startswith("basket")}
        self.basket_body = nid(B, "basket")
        self.basket_site = nid(S, "basket_centre")
        # crates: one ("basket") on the platform / old vehicle, two on the XGH505H ("basket", "basket2")
        prefixes = [b["prefix"] for b in cfg.get("xgh", {}).get("baskets", [])] if mode.startswith("xgh") else ["basket"]
        self.baskets = [dict(prefix=p_, floor=nid(G, f"{p_}_floor"), count=0) for p_ in prefixes]
        self.active_basket = 0
        self.cam_name = "wrist_cam"
        # indicator lamp (Mission 3 vehicle): one geom per state, alpha switched (blinking)
        self.lamp_geoms = {k: nid(G, f"lamp_{k}") for k in ("observe", "action", "success", "fail")}
        self.lamp_geoms = {k: g for k, g in self.lamp_geoms.items() if g >= 0}
        self.lamp_state = "off"
        self.lamp_hz = float(cfg.get("indicator", {}).get("blink_hz", 2.0))
        self._sim_time = 0.0
        # vehicle
        self.vehicle_body = nid(B, "vehicle")
        if self.vehicle_body >= 0:
            jv = nid(J, "vehicle_free")
            self.vehicle_qadr, self.vehicle_dadr = int(m.jnt_qposadr[jv]), int(m.jnt_dofadr[jv])
            self.wheel_jid = [j for j in range(m.njnt) if (mujoco.mj_id2name(m, J, j) or "").startswith("wheel_")]
            self.wheel_qadr = np.array([m.jnt_qposadr[j] for j in self.wheel_jid])
            nw = len(self.wheel_jid) // 2
            self.drive_act = {s: [nid(A, f"drive_{s}{k}") for k in range(nw)] for s in ("L", "R")}
            self.brake_act = {s: [nid(A, f"brake_{s}{k}") for k in range(nw)] for s in ("L", "R")}
            self.wheel_r = float(cfg["xgh"]["wheel_radius"])
        # contact bookkeeping
        self._f6 = np.zeros(6)

    # ------------------------------------------------------------------ basics
    @property
    def tcp_pos(self) -> np.ndarray:
        return self.d.site_xpos[self.tcp_site].copy()

    def tcp_quat(self) -> np.ndarray:
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, self.d.site_xmat[self.tcp_site])
        return q

    def tcp_rot(self) -> np.ndarray:
        return self.d.site_xmat[self.tcp_site].reshape(3, 3).copy()

    @property
    def tip_pos(self) -> np.ndarray:
        """Jaw tips (centre between them): reference point of the approach T1 -> T2."""
        return self.d.site_xpos[self.tip_site].copy()

    def tcp_for_tip(self, tip: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """TCP position that puts the jaw tips at `tip` with the tool orientation `quat` (tool z = approach)."""
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(quat, float))
        return np.asarray(tip, float) - self.tip_offset * R.reshape(3, 3)[:, 2]

    # ------------------------------------------------------------------ indicator lamp
    def set_lamp(self, state: str):
        self.lamp_state = state
        self._update_lamp()

    def _update_lamp(self):
        if not self.lamp_geoms:
            return
        on = (self.d.time * self.lamp_hz) % 1.0 < 0.5           # blinking
        for k, g in self.lamp_geoms.items():
            self.m.geom_rgba[g, 3] = 1.0 if (k == self.lamp_state and on) else 0.0

    @property
    def base_pos(self) -> np.ndarray:
        return self.d.xpos[self.base_body].copy()

    def q_arm(self) -> np.ndarray:
        return self.d.qpos[self.arm_qadr].copy()

    def set_arm(self, q: np.ndarray, ctrl: bool = True):
        self.d.qpos[self.arm_qadr] = q
        self.d.qvel[self.arm_dadr] = 0.0
        if ctrl:
            self.d.ctrl[self.arm_act] = q

    def hold_arm(self):
        self.d.ctrl[self.arm_act] = self.d.qpos[self.arm_qadr]

    _BAD = (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQPOS)

    @property
    def unstable(self) -> bool:
        """True after MuJoCo reported NaN/Inf/huge state (it then resets the data itself, which would otherwise
        look like a teleport + collision). Cleared by mj_resetData at the next scene reset."""
        return any(self.d.warning[int(w)].number > 0 for w in self._BAD)

    def step(self, n_ctrl: int = 1, callback=None):
        for _ in range(n_ctrl):
            for _ in range(self.n_substeps):
                mujoco.mj_step(self.m, self.d)
            self._update_lamp()
            if callback is not None:
                callback()

    # ------------------------------------------------------------------ gripper
    def _gap_of(self, theta: float) -> float:
        return float(self.cfg["gripper"]["closed_gap"]) + 2.0 * float(theta)

    def hinge_for_gap(self, gap: float) -> float:
        """Slider position of the U-jaw gripper for a pad gap."""
        g = self.cfg["gripper"]
        gap = float(np.clip(gap, g["min_opening"], g["max_opening"]))
        return float(np.clip(0.5 * (gap - float(g["closed_gap"])), 0.0, float(g["stroke"])))

    def set_gripper(self, gap: float, instant: bool = False):
        q = self.hinge_for_gap(gap)
        for a in self.finger_act:
            self.d.ctrl[a] = q
        if instant:
            for qa in self.finger_qadr:
                self.d.qpos[qa] = q

    def gripper_gap(self) -> float:
        return float(np.mean([self._gap_of(self.d.qpos[q]) for q in self.finger_qadr]))

    # ------------------------------------------------------------------ contacts
    def _gname(self, g: int) -> str:
        """Geom name, or `body:<body name>` for unnamed geoms (mesh visual/collision geoms)."""
        return self.geom_names[g] or "body:" + (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, int(self.m.geom_bodyid[g])) or str(g))

    def classify_contacts(self, target_slot: int | None = None) -> dict:
        """Forces (N) of robot contacts by class. Leaves are allowed (penalised); the rest are faults."""
        m, d, f6, fld = self.m, self.d, self._f6, self.field
        out = dict(self_or_env=False, floor=False, leaf=0.0, stalk=0.0, fruit_target=0.0, fruit_other=0.0,
                   pad_left=0.0, pad_right=0.0, wrist=np.zeros(3), env_names=[])
        tgt, tgt_tuft = set(), set()
        if target_slot is not None:
            s = fld.slots[target_slot]
            tgt = {s["fruit_geom"], s["neck_geom"]}
            tgt_tuft = set(s["tuft_geoms"])
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            r1, r2 = self.robot_geom[g1], self.robot_geom[g2]
            if not (r1 or r2):
                continue
            if r1 and r2:
                b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
                if {b1, b2} <= {self.housing_body, *self.arm_body_ids[-2:]}:
                    continue      # fingers vs housing at the hinge
                out["self_or_env"] = True
                out["env_names"].append((self._gname(g1), self._gname(g2)))
                continue
            rg, og = (g1, g2) if r1 else (g2, g1)
            if og == self.floor_geom:
                out["floor"] = True
                continue
            if self.env_geom[og]:
                out["self_or_env"] = True
                out["env_names"].append((self._gname(rg), self._gname(og)))
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            fn = abs(float(f6[0]))
            if int(m.geom_bodyid[rg]) in (self.housing_body, *self.arm_body_ids[-2:]):
                fw = c.frame.reshape(3, 3).T @ f6[:3]
                out["wrist"] += fw if rg == g2 else -fw
            if og in tgt:
                out["fruit_target"] += fn
                if rg in self.pad_left:
                    out["pad_left"] += fn
                elif rg in self.pad_right:
                    out["pad_right"] += fn
            elif og in fld.leaf_geom_set or og in tgt_tuft:
                out["leaf"] += fn
            elif og in fld.stalk_geom_set:
                out["stalk"] += fn
            elif og in fld.fruit_geom_set:
                out["fruit_other"] += fn
        return out

    def pad_contact_points(self, slot: int):
        """Mean contact position of the left and right finger pads on the fruit of `slot` (None if no contact).
        A real gripper gets the same information from the finger joint angles + pad geometry (where the fingers
        stopped) or from tactile pads."""
        m, d, s = self.m, self.d, self.field.slots[slot]
        tgt = {s["fruit_geom"], s["neck_geom"]}
        pl, pr = [], []
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in tgt or g2 in tgt:
                o = g2 if g1 in tgt else g1
                if o in self.pad_left:
                    pl.append(c.pos.copy())
                elif o in self.pad_right:
                    pr.append(c.pos.copy())
        return (np.mean(pl, axis=0) if pl else None), (np.mean(pr, axis=0) if pr else None)

    def fruit_touches(self, slot: int) -> dict:
        """What the (free) fruit of `slot` touches: floor / basket / robot / leaves (for judging outcomes)."""
        m, d, s = self.m, self.d, self.field.slots[slot]
        mine = {s["fruit_geom"], s["neck_geom"], *s["tuft_geoms"]}
        out = dict(floor=False, basket=False, robot=False)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in mine or g2 in mine:
                o = g2 if g1 in mine else g1
                if o == self.floor_geom:
                    out["floor"] = True
                elif o in self.basket_geoms:
                    out["basket"] = True
                elif self.robot_geom[o]:
                    out["robot"] = True
        return out

    # ------------------------------------------------------------------ basket
    def basket_frame(self, index: int | None = None):
        """(centre of the basket floor, yaw rotation) in world coordinates (active crate unless `index`)."""
        g = self.baskets[self.active_basket if index is None else index]["floor"]
        c = self.d.geom_xpos[g].copy()
        c[2] += float(self.m.geom_size[g, 2])          # top surface of the floor slab
        R = self.d.geom_xmat[g].reshape(3, 3).copy()
        return c, R

    def fruit_in_basket(self, slot: int) -> bool:
        return self.basket_of(slot) is not None

    def basket_of(self, slot: int):
        """Index of the crate that holds the fruit of `slot` (None if it is in none)."""
        p = self.field.fruit_pose(slot, self.d)[0]
        il, iw, ih = self.cfg["basket"]["inner_size"]
        lay = self.field.slots[slot]["layout"]
        for k in range(len(self.baskets)):
            c, R = self.basket_frame(k)
            loc = R.T @ (p - c)
            if abs(loc[0]) < il / 2 and abs(loc[1]) < iw / 2 and -0.01 < loc[2] < ih + lay.fruit_half + lay.fruit_r:
                return k
        return None

    # ------------------------------------------------------------------ vehicle
    def vehicle_pose(self):
        q = self.d.qpos[self.vehicle_qadr:self.vehicle_qadr + 7]
        w, x, y, z = q[3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return q[:3].copy(), yaw

    def drive(self, v_left: float, v_right: float):
        for s, v in (("L", v_left), ("R", v_right)):
            for a in self.drive_act[s]:
                self.d.ctrl[a] = v / self.wheel_r
            for k, a in enumerate(self.brake_act[s]):
                # brake follows the wheel (no force) while driving
                j = self.m.actuator_trnid[a, 0]
                self.d.ctrl[a] = self.d.qpos[self.m.jnt_qposadr[j]]

    def brake(self):
        for s in ("L", "R"):
            for a in self.drive_act[s]:
                self.d.ctrl[a] = 0.0
            for a in self.brake_act[s]:
                j = self.m.actuator_trnid[a, 0]
                self.d.ctrl[a] = self.d.qpos[self.m.jnt_qposadr[j]]
