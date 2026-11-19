"""
pinesim/lod.py
--------------
Realistic plants and the level of detail of the realistic Mission 3 evaluation.

DetailField  PlantField over the interactive POOL of a realistic scene (platform_detail for Missions 1-2, xgh_detail
             for Mission 3): every slot has the FIXED geometry of one plant kind (per-leaf articulated mesh plant,
             stalk, stem springs, free fruit welded to the stem); `assign` puts a slot onto a plant (position, yaw,
             tilt 0-15 deg) instead of rewriting its geometry.

FieldLOD     Mission 3: the 15 m field has 60 plants. Every plant always exists as a STATIC body (rest-pose plant mesh,
             peduncle, fruit, crown; visual only, no physics). Only THREE plants are interactive at a time [USER v5]:
               * before the first fruit is in the crate : the target plant + the nearest next plant
               * afterwards                            : the target plant + the previously harvested plant + the
                                                         nearest next plant (the one the robot will target next)
             Every other plant is a static mesh.

Swap rules (the risks of this approach and how they are avoided):
  * swaps happen only when the arm is in its default posture or a survey pose (before the close views of a new
    target) - never while the arm is in a plant, never during an RL step; the policy observations never see a swap
  * the static mesh IS the rest pose of the articulated leaves (generated from the same segments, same colours in the
    camera image), leaves and stem start at rest, so a swap is invisible to the camera and in the HTML
  * an assignment that would touch the robot is undone and retried at the next decision (counted: `deferred`)
  * fruit state survives a swap: a fruit still on the plant comes back on the static plant, a picked fruit is not
    drawn on it any more, a fruit lying in a crate is FROZEN there (static crate "fill" geoms with its pose and a
    collision ellipsoid), so recycling the slot never moves a harvested fruit
  * crates hold `basket.capacity` fruits each; when both are full the operator empties them (an "unload" event)
"""
from __future__ import annotations

import mujoco
import numpy as np

from pinesim.plant import PlantField, PlantLayout


def _qyaw(a):
    return np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])


def _qmul(a, b):
    o = np.zeros(4)
    mujoco.mju_mulQuat(o, a, b)
    return o


def _qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _rot(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


class DetailField(PlantField):
    """PlantField over the interactive pool of a realistic scene: slots have a FIXED geometry (one plant kind each);
    `assign` moves a slot onto a plant instead of rewriting its geometry."""

    def __init__(self, model: mujoco.MjModel):
        super().__init__(model)
        m = model
        nid = lambda t, n: mujoco.mj_name2id(m, t, n)
        G = mujoco.mjtObj.mjOBJ_GEOM
        for i, s in enumerate(self.slots):
            s["mocap"] = int(m.body_mocapid[s["plant_body"]])
            s["vis_geoms"] = [g for g in (nid(G, f"fruit{i}_vis"), nid(G, f"fruit{i}_crownvis")) if g >= 0]
            s["slot_geoms"] = [g for g in range(m.ngeom) if self._in_slot(int(m.geom_bodyid[g]), s["plant_body"])]
            s["zb"] = float(m.body_pos[s["stem_body"]][2])
            s["kind"] = -1
            s["plant"] = None
            s["stored"] = False
            s["bodies"] = [b for b in range(m.nbody) if self._in_slot(b, s["plant_body"])] + [s["fruit_body"]]
            s["leaf_bodies"] = np.array([int(m.geom_bodyid[g]) for g in s["leaf_geoms"]], int)
            s["chain_list"] = list(s["chains"]["rosette"]) + list(s["chains"]["block"])
            s["all_root_qadr"] = np.array([[m.jnt_qposadr[j] for j in segs[0]["joints"]] for segs in s["chain_list"]],
                                          dtype=int).reshape(-1, 2)
        self.kind_of = {}
        # parked slots leave the collision broadphase completely (body filter 0/0): halves the step time
        self._body_ct, self._body_ca = m.body_contype.copy(), m.body_conaffinity.copy()

    def _in_slot(self, b, root):
        while b > 0:
            if b == root:
                return True
            b = int(self.m.body_parentid[b])
        return False

    def configure(self, meta: dict, cfg: dict):
        for sd in meta["slots"]:
            self.slots[sd["slot"]]["kind"] = int(sd["kind"])
        rc = cfg["realistic"]
        self.kind_data = []
        for kd in rc["kinds"]:
            D, H, mass = rc["fruit"][str(kd["fruit"])]
            self.kind_data.append(dict(D=float(D), H=float(H), mass=float(mass)))

    def slots_of_kind(self, kind: int):
        return [i for i, s in enumerate(self.slots) if s["kind"] == kind]

    # ------------------------------------------------------------------ slots
    def apply(self, i, lay, data):                    # procedural rewrite does not exist for fixed meshes
        raise NotImplementedError("DetailField slots are assigned, not rewritten")

    def hide(self, i: int, data: mujoco.MjData):
        """Park slot i far below the ground: no contacts, stem released, leaves at rest."""
        self._unmute_slot(i)
        m, s = self.m, self.slots[i]
        s["layout"], s["plant"] = None, None
        data.mocap_pos[s["mocap"]] = [0.0, 0.0, -5.0 - i]
        data.mocap_quat[s["mocap"]] = [1.0, 0.0, 0.0, 0.0]
        data.eq_active[s["eq"]] = 0
        q = s["qadr"]
        data.qpos[q:q + 7] = [0.0, 0.0, -5.0 - i + s["zb"], 1, 0, 0, 0]
        data.qvel[s["dadr"]:s["dadr"] + 6] = 0.0
        data.qpos[s["leaf_qadr"]] = 0.0
        data.qvel[s["leaf_dadr"]] = 0.0
        data.qpos[s["stem_qadr"]] = 0.0
        data.qvel[s["stem_dadr"]] = 0.0
        for g in (s["fruit_geom"], s["neck_geom"]):
            m.geom_contype[g], m.geom_conaffinity[g] = 4, 0
        for g in s["tuft_geoms"]:
            m.geom_contype[g], m.geom_conaffinity[g] = 2, 0
        for g in s["tuft_geoms"] + [s["fruit_geom"], s["neck_geom"]] + s["vis_geoms"]:
            m.geom_rgba[g] = self._leaf_rgba[g]
        s["active"], s["act_qadr"], s["act_k"] = [], np.zeros(0, int), np.zeros(0)
        s["root_qadr"], s["damaged"] = np.zeros((0, 2), int), set()
        s["detached"] = s["harvested"] = s["lost"] = s["stored"] = False
        m.body_contype[s["bodies"]] = 0
        m.body_conaffinity[s["bodies"]] = 0

    def _collide(self, i: int):
        b = self.slots[i]["bodies"]
        self.m.body_contype[b] = self._body_ct[b]
        self.m.body_conaffinity[b] = self._body_ca[b]

    def assign(self, i: int, plant: dict, data: mujoco.MjData):
        """Put slot i onto `plant` (dict: x, y, yaw, tilt, tilt_az, fruit_yaw, break_force, break_torque, index):
        base pose (yaw + tilt), fruit on the stem, everything at rest."""
        from pinesim.model.builder_xgh import plant_quat
        m, s = self.m, self.slots[i]
        self.hide(i, data)
        kd = self.kind_data[s["kind"]]
        D, H = kd["D"], kd["H"]
        ground = np.array([plant["x"], plant["y"], 0.0])
        qp = plant_quat(plant)
        data.mocap_pos[s["mocap"]] = ground
        data.mocap_quat[s["mocap"]] = qp
        q = s["qadr"]
        data.qpos[q:q + 3] = ground + _rot(qp) @ np.array([0.0, 0.0, s["zb"]])
        data.qpos[q + 3:q + 7] = _qmul(qp, _qyaw(plant.get("fruit_yaw", 0.0) - plant.get("yaw", 0.0)))
        lay = PlantLayout(ground=ground, stalk_h=s["zb"], stalk_r=0.016, fruit_r=D / 2, fruit_half=H / 2 - D / 2,
                          neck_r=0.02, neck_half=0.02, fruit_mass=kd["mass"], break_torque=plant["break_torque"],
                          break_force=plant["break_force"], quat=qp, tilt_deg=float(np.degrees(plant.get("tilt", 0.0))))
        s["layout"], s["plant"] = lay, plant.get("index", 0)
        s["act_qadr"] = s["leaf_qadr"].copy()
        s["act_k"] = np.array([m.jnt_stiffness[m.dof_jntid[a]] for a in s["leaf_dadr"]], float)
        s["active"] = [True] if len(s["act_qadr"]) else []
        s["root_qadr"], s["damaged"] = s["all_root_qadr"].copy(), set()
        self._collide(i)
        self.arm_stem(i, data)

    def gate_leaves(self, data: mujoco.MjData, seg_a: np.ndarray, seg_b: np.ndarray, seg_r: np.ndarray,
                    margin: float = 0.20):
        """Collision level of detail inside an interactive plant: only leaf segments within `margin` of the arm
        (capsule model, incl. gripper) take part in the collision broadphase. The arm moves < 2 cm per 20 ms control
        step, so with a 20 cm margin no contact is missed; leaves never touch each other or the vehicle anyway."""
        m = self.m
        act = [s for s in self.slots if s["layout"] is not None]
        if not act:
            return 0
        lb = np.concatenate([s["leaf_bodies"] for s in act])
        lg = np.concatenate([np.asarray(s["leaf_geoms"], int) for s in act])
        c = data.geom_xpos[lg]
        ab = seg_b - seg_a
        ab2 = np.maximum(np.einsum("ck,ck->c", ab, ab), 1e-12)
        ap = c[:, None, :] - seg_a[None, :, :]
        t = np.clip(np.einsum("mck,ck->mc", ap, ab) / ab2[None, :], 0.0, 1.0)
        dist = np.linalg.norm(ap - t[..., None] * ab[None, :, :], axis=2) - seg_r[None, :]
        on = np.min(dist, axis=1) < m.geom_rbound[lg] + margin
        m.body_contype[lb] = np.where(on, self._body_ct[lb], 0)
        return int(on.sum())

    def fruit_pose(self, i: int, data):
        """Fruit CENTRE (the slot fruit body has its origin at the fruit bottom, like the mesh) and orientation."""
        s = self.slots[i]
        kd = self.kind_data[s["kind"]]
        R = data.xmat[s["fruit_body"]].reshape(3, 3)
        return data.xpos[s["fruit_body"]] + R[:, 2] * (kd["H"] / 2), data.xquat[s["fruit_body"]].copy()

    def remove_lost(self, i: int, data: mujoco.MjData):
        super().remove_lost(i, data)
        for g in self.slots[i]["vis_geoms"]:
            self.m.geom_rgba[g, 3] = 0.0

    def store(self, i: int, data: mujoco.MjData):
        """Crate emptied by the operator: the fruit of slot i leaves the scene (counted as harvested)."""
        self.remove_lost(i, data)
        s = self.slots[i]
        s["lost"], s["stored"] = False, True

    def leaf_points(self, data: mujoco.MjData, slots=None, per_seg: int = 3, include_tuft: bool = True):
        geoms = []
        slots = self.active_slots() if slots is None else slots
        for i in slots:
            s = self.slots[i]
            geoms += [g for g in s["leaf_geoms"] if self.m.geom_contype[g]]
            if include_tuft and not s["harvested"]:
                geoms += [g for g in s["tuft_geoms"] if self.m.geom_contype[g]]
        if not geoms:
            return np.zeros((0, 3)), np.zeros(0)
        g = np.array(geoms)
        h = self.m.geom_size[g, 1][:, None]
        rad = self.m.geom_size[g, 0]
        c = data.geom_xpos[g]
        ax = data.geom_xmat[g].reshape(-1, 3, 3)[:, :, 2]
        t = ((np.arange(per_seg) + 0.5) / per_seg)[None, :, None]
        pts = (c - h * ax)[:, None, :] + (2 * h)[:, :, None] * t * ax[:, None, :]
        reff = np.sqrt(rad ** 2 + (h[:, 0] / per_seg) ** 2)
        return pts.reshape(-1, 3), np.repeat(reff, per_seg)


class FieldLOD:
    """Mission 3: streams the 15 m field through the pool of a DetailField with the three-plant rule."""

    def __init__(self, sim, cfg: dict):
        self.sim, self.cfg = sim, cfg
        m = sim.m
        self.field: DetailField = sim.field
        self.layout = sim.meta["layout"]
        lod = cfg["eval_field"]["lod"]
        nid = lambda t, n: mujoco.mj_name2id(m, t, n)
        B, G = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM
        self.static_body = [nid(B, f"splant_{p['index']}") for p in self.layout]
        self.static_fruit = [nid(B, f"splant_{p['index']}_fruit") for p in self.layout]
        self.static_fruit_pos = [m.body_pos[b].copy() for b in self.static_fruit]
        self.cap = int(cfg["basket"]["capacity"])
        self.nkinds = len(cfg["realistic"]["kinds"])
        self.fill = {}                                  # (basket, kind) -> list of (col, vis, crown) geoms
        for bi, b in enumerate(cfg["xgh"]["baskets"]):
            for k in range(self.nkinds):
                self.fill[(bi, k)] = [tuple(nid(G, f'{b["prefix"]}_fill_{k}_{j}_{t}') for t in ("col", "vis", "crown"))
                                      for j in range(self.cap)]
        self.vehicle_body = sim.vehicle_body
        sb = {b: n for n, b in enumerate(self.static_body)} | {b: n for n, b in enumerate(self.static_fruit)}
        self.static_geoms = [[] for _ in self.layout]
        for g in range(m.ngeom):
            n = sb.get(int(m.geom_bodyid[g]))
            if n is not None:
                self.static_geoms[n].append(g)
        # [v6] neighbour beds (static meshes left / right of the harvested bed): drawn inside the render window only
        self.side_layout = list(sim.meta.get("side_layout", []))
        side_b = {}
        for p in self.side_layout:
            for nm in (f"splant_side_{p['index']}", f"splant_side_{p['index']}_fruit"):
                b = nid(B, nm)
                if b >= 0:
                    side_b[b] = p["index"]
        self.side_geoms = [[] for _ in self.side_layout]
        for g in range(m.ngeom):
            n = side_b.get(int(m.geom_bodyid[g]))
            if n is not None:
                self.side_geoms[n].append(g)
        self.render_window = tuple(lod.get("render_window", [-3.0, 7.0]))
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        d = self.sim.d
        for i in range(self.field.n):
            self.field.hide(i, d)
        self.state = [dict(slot=None, fruit="on_plant") for _ in self.layout]   # on_plant | picked | gone
        for n in range(len(self.layout)):
            self._show_static(n, True)
        for key in self.fill:
            for gs in self.fill[key]:
                self._park_fill(gs)
        self.fill_used = {key: 0 for key in self.fill}
        self.swaps = 0
        self.unloads = 0
        self.deferred = 0
        self.stored = 0
        self.max_interactive = 0
        self.prev_plant = None                          # last plant whose fruit went into the crate
        self.log = []

    def _show_static(self, n: int, on: bool):
        m = self.sim.m
        b = self.static_body[n]
        p = self.layout[n]
        m.body_pos[b] = [p["x"], p["y"], 0.0 if on else -30.0]
        fb = self.static_fruit[n]
        pos = self.static_fruit_pos[n].copy()
        if self.state[n]["fruit"] != "on_plant":
            pos[2] = -30.0
        m.body_pos[fb] = pos

    def _park_fill(self, gs):
        m = self.sim.m
        for g in gs:
            m.geom_pos[g] = [0.0, 0.0, -20.0]
        m.geom_contype[gs[0]] = 0
        m.geom_conaffinity[gs[0]] = 0

    def interactive(self):
        return [n for n, st in enumerate(self.state) if st["slot"] is not None]

    # ------------------------------------------------------------------ which plants
    def plant_near(self, xy, max_dist: float = 0.30):
        """Field plant whose FRUIT is nearest to a perceived fruit position (simulation bookkeeping only)."""
        from pinesim.model.builder_xgh import plant_quat
        best, bd = None, max_dist
        for n, p in enumerate(self.layout):
            st = self.state[n]
            if st["fruit"] != "on_plant":
                continue
            fz = np.array([0.0, 0.0, 0.65])
            c = np.array([p["x"], p["y"], 0.0]) + _rot(plant_quat(p)) @ fz
            d = float(np.hypot(*(c[:2] - np.asarray(xy, float)[:2])))
            if d < bd:
                best, bd = n, d
        return best

    def next_plant(self, target: int | None, base_x: float):
        """The plant the robot will target next: nearest plant with its fruit still on, ahead of the target."""
        x_ref = self.layout[target]["x"] if target is not None else base_x
        cands = [n for n, p in enumerate(self.layout) if self.state[n]["fruit"] == "on_plant" and n != target
                 and p["x"] >= x_ref - 0.01]
        if not cands:
            return None
        return min(cands, key=lambda n: (self.layout[n]["x"] - x_ref, abs(self.layout[n]["y"])))

    def wanted(self, target: int | None, base_x: float) -> list:
        """Three-plant rule [USER v5]."""
        out = []
        if target is not None:
            out.append(target)
        nxt = self.next_plant(target, base_x)
        if nxt is not None:
            out.append(nxt)
        if self.prev_plant is not None and self.prev_plant not in out:
            out.append(self.prev_plant)
        return out[:3]

    # ------------------------------------------------------------------ swaps
    def set_interactive(self, plants: list, reason: str = "target") -> dict:
        """Make exactly `plants` interactive (release every other one). Call only with the arm away from the plants
        (default posture / survey pose)."""
        sim, fld, d = self.sim, self.field, self.sim.d
        mujoco.mj_forward(sim.m, d)
        want = [n for n in plants if n is not None]
        released, assigned = [], []
        for n, st in enumerate(self.state):
            if st["slot"] is not None and n not in want:
                self._release(n)
                released.append(n)
        for n in want:
            st = self.state[n]
            if st["slot"] is None and st["fruit"] in ("on_plant", "gone"):
                if self._assign(n):
                    assigned.append(n)
        if assigned:
            for _ in range(5):                                       # stems take the fruit weight
                for _ in range(sim.n_substeps):
                    mujoco.mj_step(sim.m, d)
            for n in assigned:
                fld.record_rest_load(self.state[n]["slot"], d)
        self.max_interactive = max(self.max_interactive, len(self.interactive()))
        ev = dict(reason=reason, released=released, assigned=assigned, interactive=self.interactive())
        self.log.append(ev)
        self.swaps += len(released) + len(assigned)
        return ev

    def set_render_window(self, base_x: float | None):
        """Static plants far from the robot are not drawn by MuJoCo (geom group 5): the cameras never look that
        far, and this machine renders on the CPU. None = draw everything (before a Meshcat export)."""
        m = self.sim.m
        for n, p in enumerate(self.layout):
            on = base_x is None or self.render_window[0] <= p["x"] - base_x <= self.render_window[1]
            m.geom_group[self.static_geoms[n]] = 2 if on else 5
        for n, p in enumerate(self.side_layout):
            on = base_x is None or self.render_window[0] <= p["x"] - base_x <= self.render_window[1]
            if self.side_geoms[n]:
                m.geom_group[self.side_geoms[n]] = 2 if on else 5

    def _free_slot(self, kind: int):
        used = {st["slot"] for st in self.state if st["slot"] is not None}
        for i, s in enumerate(self.field.slots):
            if s["kind"] == kind and i not in used:
                return i
        return None

    def _assign(self, n: int) -> bool:
        p = self.layout[n]
        i = self._free_slot(p["kind"])
        if i is None:
            self.log.append(dict(warning=f"no free slot of kind {p['kind']} for plant {n}"))
            return False
        fld, d, m = self.field, self.sim.d, self.sim.m
        fld.assign(i, p, d)
        if self.state[n]["fruit"] == "gone":          # plant whose fruit was picked / lost earlier
            fld.remove_lost(i, d)
            fld.slots[i]["harvested"] = True
        mujoco.mj_forward(m, d)
        geoms = set(fld.slots[i]["slot_geoms"]) | {fld.slots[i]["fruit_geom"], fld.slots[i]["neck_geom"]}
        for c in d.contact[:d.ncon]:                  # never appear inside the robot
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in geoms and self.sim.robot_geom[g2]) or (g2 in geoms and self.sim.robot_geom[g1]):
                fld.hide(i, d)
                self.deferred += 1
                return False
        self.state[n]["slot"] = i
        self._show_static(n, False)
        return True

    def _release(self, n: int):
        st = self.state[n]
        i = st["slot"]
        fld, d = self.field, self.sim.d
        s = fld.slots[i]
        if s["detached"] or s["harvested"]:
            if not s["lost"] and not s["stored"] and self.sim.basket_of(i) is not None:
                self._freeze_in_crate(i)
            st["fruit"] = "gone"
        fld.hide(i, d)
        st["slot"] = None
        self._show_static(n, True)

    # ------------------------------------------------------------------ crates
    def _freeze_in_crate(self, i: int):
        """Copy the fruit of slot i (lying in a crate) into the crate's static fill geoms (vehicle frame)."""
        sim, m, d, fld = self.sim, self.sim.m, self.sim.d, self.field
        s = fld.slots[i]
        b = sim.basket_of(i)
        k = s["kind"]
        j = self.fill_used[(b, k)]
        if j >= self.cap:
            return
        self.fill_used[(b, k)] = j + 1
        gs = self.fill[(b, k)][j]
        pv, Rv = d.xpos[self.vehicle_body], d.xmat[self.vehicle_body].reshape(3, 3)
        qv = _qconj(d.xquat[self.vehicle_body])
        c, qf = fld.fruit_pose(i, d)
        m.geom_pos[gs[0]] = Rv.T @ (c - pv)
        m.geom_quat[gs[0]] = _qmul(qv, qf)
        m.geom_contype[gs[0]], m.geom_conaffinity[gs[0]] = 1, 5
        for g_fill, g_src in ((gs[1], s["vis_geoms"][0]), (gs[2], s["vis_geoms"][1])):
            m.geom_pos[g_fill] = Rv.T @ (d.geom_xpos[g_src] - pv)
            qs = np.zeros(4)
            mujoco.mju_mat2Quat(qs, d.geom_xmat[g_src])
            m.geom_quat[g_fill] = _qmul(qv, qs)
        fld.store(i, d)
        s["stored"] = True

    def crates_full(self) -> bool:
        return all(b["count"] >= self.cap for b in self.sim.baskets)

    def unload(self):
        """Both crates full: the operator empties them (fill geoms cleared, fruits still lying in them removed)."""
        sim, fld, d = self.sim, self.field, self.sim.d
        for i, s in enumerate(fld.slots):
            if s["layout"] is not None and (s["detached"] or s["harvested"]) and not s["lost"] and not s["stored"] \
                    and sim.basket_of(i) is not None:
                fld.store(i, d)
        for key in self.fill:
            for gs in self.fill[key]:
                self._park_fill(gs)
            self.fill_used[key] = 0
        self.stored += sum(b["count"] for b in sim.baskets)
        for b in sim.baskets:
            b["count"] = 0
            b["drops"] = 0
        self.unloads += 1
        self.log.append(dict(event="unload", n=self.unloads))

    def mark(self, slot: int, fruit_state: str):
        for n, st in enumerate(self.state):
            if st["slot"] == slot:
                st["fruit"] = fruit_state
                return n
        return None

    def plant_of_slot(self, slot: int):
        for n, st in enumerate(self.state):
            if st["slot"] == slot:
                return n
        return None
