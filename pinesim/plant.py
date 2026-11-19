"""
pinesim/plant.py
----------------
Procedural pineapple plants (generator of the reference Missions 4-5, shared/harvest/plant.py @ d46ad6dc, extended)
and the writer that puts a layout into a plant slot of the MuJoCo model.

Per plant, randomised every episode (the user's requirement "komposisi nanas berubah-ubah"; version 5: fruit centre
0.60-0.70 m above the ground, whole plant tilted 0-15 deg in any horizontal direction, green and orange leaves):
  * number of rosette leaves, their length, elevation, bend, width, stiffness,
  * fruit height above the ground, fruit diameter and length, crown neck size, number/length of crown leaves,
  * blocking leaves crossing the fruit from the TOP, FRONT, BACK, LEFT or RIGHT (each side with probability p_block,
    up to max_per_side per side, all sides at once with probability p_all_sides). "front" = side facing the robot
    (or the camera in the pretraining rig). The reference had top/front/left/right; BACK is new here.
  * stem strength (break torque / force) - PROVISIONAL.

Slot model (pinesim/model/builder.py): plant{i} (static; stalk + leaf chains) and fruit{i} (free body: fruit capsule,
crown neck, rigid crown leaves) held by the weld `stem{i}`. The stem breaks when the constraint wrench beyond its
gripped resting load exceeds the plant's thresholds (no teleport, no weld to the gripper).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

SIDES = ("top", "front", "back", "left", "right")


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / (np.linalg.norm(v) + 1e-12)


def _u(rng, r):
    return float(rng.uniform(float(r[0]), float(r[1])))


def quat_from_x_axis(d: np.ndarray) -> np.ndarray:
    x = _unit(d)
    up = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
    y = _unit(np.cross(up, x))
    z = np.cross(x, y)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([x, y, z]).ravel())
    return q


def quat_from_z_axis(d: np.ndarray) -> np.ndarray:
    z = _unit(d)
    up = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = _unit(np.cross(up, z))
    y = np.cross(z, x)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([x, y, z]).ravel())
    return q


def _quat_pitch(angle: float) -> np.ndarray:
    return np.array([np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0])


LEAF_GREEN = (0.23, 0.46, 0.20)
LEAF_GREEN_DARK = (0.18, 0.37, 0.16)
LEAF_ORANGE = (0.74, 0.41, 0.13)


def quat_rotate(q, v) -> np.ndarray:
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, np.asarray(v, float), np.asarray(q, float))
    return out


@dataclass
class LeafSpec:
    kind: str                 # rosette | block
    root: np.ndarray          # plant frame
    direction: np.ndarray
    seg_len: float
    bend: float
    radius: float
    stiffness: float
    damping: float
    side: str = ""
    colours: tuple = ()       # rgb per segment (green / orange; camera classes as the realistic plants)


@dataclass
class PlantLayout:
    ground: np.ndarray
    stalk_h: float
    stalk_r: float
    fruit_r: float
    fruit_half: float
    neck_r: float
    neck_half: float
    fruit_mass: float
    tufts: list = field(default_factory=list)          # (direction, length, radius) relative to the crown base
    leaves: list = field(default_factory=list)
    blocked_sides: tuple = ()
    break_torque: float = 1.0
    break_force: float = 40.0
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))   # plant tilt (wxyz)
    tilt_deg: float = 0.0

    # local (plant frame, before the tilt) heights
    @property
    def local_fruit_center(self) -> np.ndarray:
        return np.array([0.0, 0.0, self.stalk_h + self.fruit_r + self.fruit_half])

    @property
    def axis(self) -> np.ndarray:
        return quat_rotate(self.quat, [0.0, 0.0, 1.0])

    @property
    def fruit_center(self) -> np.ndarray:
        return self.ground + quat_rotate(self.quat, self.local_fruit_center)

    @property
    def fruit_top(self) -> np.ndarray:
        return self.fruit_center + self.axis * (self.fruit_half + self.fruit_r)

    @property
    def fruit_top_z(self) -> float:
        return float(self.fruit_top[2])

    @property
    def neck_local_dz(self) -> float:
        """Neck centre above the fruit centre along the fruit axis."""
        return self.fruit_half + self.fruit_r + self.neck_half + 0.5 * self.neck_r

    @property
    def neck_center(self) -> np.ndarray:
        return self.fruit_center + self.axis * self.neck_local_dz

    @property
    def fruit_height(self) -> float:
        return 2.0 * (self.fruit_half + self.fruit_r)

    def summary(self) -> dict:
        return dict(fruit_diameter=2 * self.fruit_r, fruit_height=self.fruit_height,
                    fruit_center_z=float(self.fruit_center[2]), tilt_deg=float(self.tilt_deg), n_rosette=sum(l.kind == "rosette" for l in self.leaves),
                    n_block=sum(l.kind == "block" for l in self.leaves), n_tuft=len(self.tufts),
                    blocked_sides=list(self.blocked_sides), crown_len_max=max([t[1] for t in self.tufts] or [0.0]))


class PlantSampler:
    def __init__(self, cfg: dict):
        self.cfg = dict(cfg["plant"])
        self.level: dict = {}
        self.pool = self.cfg["pool"]

    def set_level(self, idx: int | None):
        if idx is None:
            self.level = {}
            return
        levels = self.cfg.get("levels", [])
        self.level = dict(levels[int(np.clip(idx, 0, len(levels) - 1))]) if levels else {}

    def _lv(self, key, default):
        return self.level.get(key, default)

    def sample(self, rng: np.random.Generator, ground, front_dir) -> PlantLayout:
        c = self.cfg
        fr = c["fruit"]
        fruit_r = 0.5 * _u(rng, fr["diameter_range"])
        H = _u(rng, fr["height_range"])
        fruit_half = max(0.01, 0.5 * H - fruit_r)
        zc = _u(rng, fr["center_height_range"])
        stalk_h = max(0.10, zc - fruit_half - fruit_r)
        st = c.get("stem", {})
        lay = PlantLayout(
            ground=np.asarray(ground, dtype=float).copy(), stalk_h=stalk_h,
            stalk_r=0.5 * _u(rng, c["stalk"]["diameter_range"]), fruit_r=fruit_r, fruit_half=fruit_half,
            neck_r=0.5 * _u(rng, c["crown"]["neck_diameter_range"]),
            neck_half=0.5 * _u(rng, c["crown"]["neck_length_range"]), fruit_mass=_u(rng, fr["mass_range"]),
            break_torque=_u(rng, st.get("break_torque_range", [1.0, 1.0])),
            break_force=_u(rng, st.get("break_force_range", [40.0, 40.0])))
        stiff = self._lv("stiffness_range", c["leaf"]["stiffness_range"])
        damp = c["leaf"]["damping_range"]
        t0, t1 = np.radians(c.get("tilt_deg_range", [0.0, 0.0]))
        tilt, taz = _u(rng, (t0, t1)), rng.uniform(0, 2 * np.pi)
        sh = np.sin(tilt / 2)
        lay.quat = np.array([np.cos(tilt / 2), sh * np.cos(taz), sh * np.sin(taz), 0.0])
        lay.tilt_deg = float(np.degrees(tilt))
        p_orange = float(c.get("orange_leaf_frac", 0.0))
        top_max = float(c.get("leaf_top_max", np.inf))

        def capped(root, u, L):
            """Leaf (local frame) shortened so that no part rises above leaf_top_max in the WORLD (after the tilt):
            the XGH505H passes over the rows with the arm base 0.91 m above the ground (a block leaf standing 1.02 m
            high was wedged into the arm while driving, measured). Returns (root, L) or None (leaf dropped)."""
            if not np.isfinite(top_max):
                return root, L
            zr = float(quat_rotate(lay.quat, root)[2])
            uz = float(quat_rotate(lay.quat, u)[2])
            zt = zr + uz * L
            if max(zr, zt) <= top_max:
                return root, L
            if uz > 1e-6:                       # tip too high: shorten from the tip
                if zr > top_max - 0.02:
                    return None
                L2 = (top_max - zr) / uz
                return (root, L2) if L2 >= 0.15 else None
            if zt > top_max - 0.02:             # root end too high: shorten from the root
                return None
            sft = (zr - top_max) / max(-uz, 1e-6)
            return (root + sft * u, L - sft) if L - sft >= 0.15 else None
        crng = np.random.default_rng(int(rng.integers(1 << 31)))       # colours: own stream

        def leaf_colours(n_seg):
            u = crng.random()
            if u < 0.35 * p_orange:                                      # whole leaf orange (older leaf)
                return tuple([LEAF_ORANGE] * n_seg)
            if u < p_orange:                                             # orange tip
                return tuple([LEAF_GREEN] * (n_seg - 1) + [LEAF_ORANGE])
            return tuple([LEAF_GREEN if crng.random() < 0.7 else LEAF_GREEN_DARK] * n_seg)

        def leaf_props():
            return dict(radius=0.5 * _u(rng, c["leaf"]["width_range"]), stiffness=_u(rng, stiff), damping=_u(rng, damp))

        # crown leaves (rigid, part of the fruit body)
        tf = c["crown"]
        n_t = int(rng.integers(int(tf["tuft_count_range"][0]), int(tf["tuft_count_range"][1]) + 1))
        n_t = min(n_t, int(self.pool["n_tuft"]))
        az0 = rng.uniform(0, 2 * np.pi)
        for i in range(n_t):
            az = az0 + 2 * np.pi * i / n_t + rng.normal(0, 0.3)
            el = np.deg2rad(_u(rng, tf["tuft_elevation_deg_range"]))
            d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            lay.tufts.append((d, _u(rng, tf["tuft_length_range"]), 0.5 * _u(rng, tf["tuft_width_range"])))

        # rosette leaves
        ro = c["rosette"]
        n_lo, n_hi = self._lv("rosette_count", ro["count_range"])
        n = min(int(rng.integers(int(n_lo), int(n_hi) + 1)), int(self.pool["n_rosette"]))
        seg_r = int(self.pool["rosette_segments"])
        el_rng = self._lv("elevation_deg_range", ro["elevation_deg_range"])
        len_rng = self._lv("length_range", ro["length_range"])
        az0 = rng.uniform(0, 2 * np.pi)
        for i in range(n):
            az = az0 + 2 * np.pi * i / max(n, 1) + rng.normal(0, 0.25)
            rho = _u(rng, ro["root_radius_range"])
            root = np.array([rho * np.cos(az), rho * np.sin(az), _u(rng, ro["root_height_range"])])
            el = np.deg2rad(_u(rng, el_rng))
            d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            cp = capped(root, d, _u(rng, len_rng))
            if cp is None:
                continue
            root, Lr = cp
            lay.leaves.append(LeafSpec("rosette", root, d, Lr / seg_r,
                                       np.deg2rad(_u(rng, ro["bend_deg_range"])), **leaf_props(),
                                       colours=leaf_colours(seg_r)))

        # blocking leaves (top / front / back / left / right)
        bl = c["block"]
        p_side = float(self._lv("p_block", bl.get("p_block", 0.4)))
        max_per = int(self._lv("max_per_side", bl.get("max_per_side", 2)))
        p_all = float(self._lv("p_all_sides", bl.get("p_all_sides", 0.0)))
        seg_b = int(self.pool["block_segments"])
        f = _unit(np.r_[np.asarray(front_dir, float)[:2], 0.0])
        left = np.cross([0.0, 0.0, 1.0], f)
        normals = {"top": np.array([0.0, 0.0, 1.0]), "front": f, "back": -f, "left": left, "right": -left}
        sides = [s for s in bl.get("sides", SIDES) if s in normals]
        all_sides = rng.random() < p_all
        fc_local = lay.local_fruit_center
        blocked = []
        n_block_max = int(self.pool["n_block"])
        for side in sides:
            if not (all_sides or rng.random() < p_side):
                continue
            for _ in range(int(rng.integers(1, max_per + 1))):
                if sum(1 for l in lay.leaves if l.kind == "block") >= n_block_max:
                    break
                nrm = normals[side]
                dist = _u(rng, bl["distance_range"])
                Lb = _u(rng, bl["length_range"])
                if side == "top":
                    C = np.array([fc_local[0], fc_local[1], fc_local[2] + lay.fruit_half + lay.fruit_r + 2 * lay.neck_half + dist])
                    C[:2] += rng.uniform(-0.05, 0.05, 2)
                    ang = rng.uniform(0, 2 * np.pi)
                    u = np.array([np.cos(ang), np.sin(ang), rng.uniform(-0.25, 0.25)])
                else:
                    C = fc_local + nrm * (lay.fruit_r + dist)
                    C[2] += rng.uniform(-0.6, 0.9) * (lay.fruit_half + lay.fruit_r)
                    tang = np.cross([0.0, 0.0, 1.0], nrm)
                    C += tang * rng.uniform(-0.05, 0.05)
                    psi = np.deg2rad(rng.uniform(-60, 60))
                    u = np.cos(psi) * tang + np.sin(psi) * np.array([0.0, 0.0, 1.0])
                u = _unit(u)
                if rng.random() < 0.5:
                    u = -u
                frac = rng.uniform(0.35, 0.65)
                root = C - frac * Lb * u
                if root[2] < 0.03:
                    u = -u
                    root = C - frac * Lb * u
                if root[2] < 0.03:
                    continue
                cp = capped(root, u, Lb)
                if cp is None:
                    continue
                root, Lb = cp
                lay.leaves.append(LeafSpec("block", root, u, Lb / seg_b, np.deg2rad(rng.uniform(-6, 6)), side=side,
                                           **leaf_props(), colours=leaf_colours(seg_b)))
            blocked.append(side)
        lay.blocked_sides = tuple(blocked)
        return lay


def _set_capsule(m, g, radius, half):
    m.geom_size[g, 0] = radius
    m.geom_size[g, 1] = half
    m.geom_rbound[g] = radius + half
    m.geom_aabb[g, :3] = 0.0
    m.geom_aabb[g, 3:] = [radius, radius, radius + half]


class PlantField:
    """All plant slots of a model: write layouts, stem state, ground-truth queries (diagnostics / metrics only)."""

    def __init__(self, model: mujoco.MjModel):
        self.m = m = model
        nid = lambda t, n: mujoco.mj_name2id(m, t, n)
        B, G, J, E = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_EQUALITY
        self.slots = []
        i = 0
        while nid(B, f"plant{i}") >= 0:
            s = dict(plant_body=nid(B, f"plant{i}"), fruit_body=nid(B, f"fruit{i}"), stalk=nid(G, f"stalk{i}"),
                     fruit_geom=nid(G, f"fruit{i}_body"), neck_geom=nid(G, f"fruit{i}_neck"),
                     eq=nid(E, f"stem{i}"), layout=None, detached=False, harvested=False,
                     rest_force=np.zeros(3), rest_torque=np.zeros(3))
            s["stem_body"] = nid(B, f"stem{i}_link")
            sj = [nid(J, f"stem{i}_{n}") for n in ("twist", "bend_x", "bend_y", "pull")]
            s["stem_qadr"] = np.array([m.jnt_qposadr[j] for j in sj], dtype=int)
            s["stem_dadr"] = np.array([m.jnt_dofadr[j] for j in sj], dtype=int)
            s["k_twist"] = float(m.jnt_stiffness[sj[0]])
            s["k_pull"] = float(m.jnt_stiffness[sj[3]])
            s["rest_ext"] = 0.0
            jf = nid(J, f"fruit{i}_free")
            s["qadr"], s["dadr"] = int(m.jnt_qposadr[jf]), int(m.jnt_dofadr[jf])
            s["tuft_geoms"] = []
            k = 0
            while nid(G, f"fruit{i}_tuft{k}") >= 0:
                s["tuft_geoms"].append(nid(G, f"fruit{i}_tuft{k}"))
                k += 1
            chains = {"rosette": [], "block": []}
            for kind, pre in (("rosette", f"p{i}ros"), ("block", f"p{i}blk")):
                c = 0
                while nid(B, f"{pre}{c}_s0") >= 0:
                    segs, sidx = [], 0
                    while nid(B, f"{pre}{c}_s{sidx}") >= 0:
                        segs.append(dict(body=nid(B, f"{pre}{c}_s{sidx}"), geom=nid(G, f"{pre}{c}_g{sidx}"),
                                         joints=[nid(J, f"{pre}{c}_j{sidx}a"), nid(J, f"{pre}{c}_j{sidx}b")]))
                        sidx += 1
                    chains[kind].append(segs)
                    c += 1
            s["chains"] = chains
            s["leaf_geoms"] = [sg["geom"] for k_ in chains for ch in chains[k_] for sg in ch]
            jl = [j for k_ in chains for ch in chains[k_] for sg in ch for j in sg["joints"]]
            s["leaf_qadr"] = np.array([m.jnt_qposadr[j] for j in jl], dtype=int)
            s["leaf_dadr"] = np.array([m.jnt_dofadr[j] for j in jl], dtype=int)
            s["active"] = []
            self.slots.append(s)
            i += 1
        self.n = len(self.slots)
        self.leaf_geom_set = {int(g) for s in self.slots for g in s["leaf_geoms"]} | \
                             {int(g) for s in self.slots for g in s["tuft_geoms"]}
        self.stalk_geom_set = {int(s["stalk"]) for s in self.slots}
        self.fruit_geom_set = {int(g) for s in self.slots for g in (s["fruit_geom"], s["neck_geom"])}
        self._leaf_rgba = m.geom_rgba.copy()
        self.muted = []                          # (slot, [leaf geoms]) muted while a picked fruit leaves them
        self.mute_on_detach = False              # Mission 4 (configs: plant.mute_penetrating_leaves)
        self._leaf_slot = {int(g): i for i, s in enumerate(self.slots) for g in s["leaf_geoms"]}

    # ------------------------------------------------------------------ write
    def _enable(self, g, on: bool, contype: int = 2):
        m = self.m
        if on:
            m.geom_contype[g], m.geom_conaffinity[g] = contype, 0
            m.geom_rgba[g] = self._leaf_rgba[g]
        else:
            m.geom_contype[g] = m.geom_conaffinity[g] = 0
            m.geom_rgba[g, 3] = 0.0
            # unused pool geoms keep their pose from an earlier episode; alpha 0 alone does not keep them out of
            # every render pass, so move them far below the ground (apply() sets the pose again when re-enabled)
            m.geom_pos[g] = [0.0, 0.0, -20.0]

    def hide(self, i: int, data: mujoco.MjData):
        """Remove slot i from the scene (unused plant): below ground, no contacts, invisible."""
        self._unmute_slot(i)
        m, s = self.m, self.slots[i]
        s["layout"] = None
        m.body_pos[s["plant_body"]] = [0.0, 0.0, -5.0 - i]
        for g in s["leaf_geoms"] + s["tuft_geoms"] + [s["stalk"], s["fruit_geom"], s["neck_geom"]]:
            self._enable(g, False)
        data.eq_active[s["eq"]] = 0
        q = s["qadr"]
        data.qpos[q:q + 7] = [0.0, 0.0, -5.0 - i, 1, 0, 0, 0]
        data.qvel[s["dadr"]:s["dadr"] + 6] = 0.0
        data.qpos[s["leaf_qadr"]] = 0.0
        data.qpos[s["stem_qadr"]] = 0.0
        s["active"], s["act_qadr"], s["act_k"] = [], np.zeros(0, int), np.zeros(0)
        s["root_qadr"], s["damaged"] = np.zeros((0, 2), int), set()

    def apply(self, i: int, lay: PlantLayout, data: mujoco.MjData):
        self._unmute_slot(i)
        m, s = self.m, self.slots[i]
        s["layout"], s["detached"], s["harvested"], s["lost"] = lay, False, False, False
        m.body_pos[s["plant_body"]] = lay.ground
        m.body_quat[s["plant_body"]] = lay.quat                       # tilted plant (leaves, stalk, stem link)
        m.body_pos[s["stem_body"]] = [0.0, 0.0, lay.stalk_h]
        _set_capsule(m, s["stalk"], lay.stalk_r, 0.5 * lay.stalk_h)
        m.geom_pos[s["stalk"]] = [0, 0, 0.5 * lay.stalk_h]
        m.geom_contype[s["stalk"]], m.geom_conaffinity[s["stalk"]] = 4, 0
        m.geom_rgba[s["stalk"]] = self._leaf_rgba[s["stalk"]]
        _set_capsule(m, s["fruit_geom"], lay.fruit_r, lay.fruit_half)
        m.geom_pos[s["fruit_geom"]] = [0, 0, 0]
        nz = lay.neck_local_dz
        _set_capsule(m, s["neck_geom"], lay.neck_r, lay.neck_half)
        m.geom_pos[s["neck_geom"]] = [0, 0, nz]
        for g in (s["fruit_geom"], s["neck_geom"]):
            m.geom_contype[g], m.geom_conaffinity[g] = 4, 0
            m.geom_rgba[g] = self._leaf_rgba[g]
        tuft_base = nz + lay.neck_half + lay.neck_r
        for k, g in enumerate(s["tuft_geoms"]):
            if k >= len(lay.tufts):
                self._enable(g, False)
                continue
            d, Lt, rt = lay.tufts[k]
            _set_capsule(m, g, rt, 0.5 * Lt)
            m.geom_pos[g] = np.array([0, 0, tuft_base]) + 0.5 * Lt * _unit(d)
            m.geom_quat[g] = quat_from_z_axis(d)
            self._enable(g, True, 2)
        m.body_mass[s["fruit_body"]] = lay.fruit_mass
        r, h = lay.fruit_r, lay.fruit_half + lay.fruit_r
        m.body_inertia[s["fruit_body"]] = lay.fruit_mass * np.array([(3 * r * r + 4 * h * h) / 12] * 2 + [r * r / 2])
        s["active"] = []
        by_kind = {"rosette": [], "block": []}
        for spec in lay.leaves:
            by_kind[spec.kind].append(spec)
        for kind, chains in s["chains"].items():
            specs = by_kind[kind]
            for ci, segs in enumerate(chains):
                spec = specs[ci] if ci < len(specs) else None
                for si, sg in enumerate(segs):
                    g, b = sg["geom"], sg["body"]
                    if spec is None:
                        self._enable(g, False)
                        continue
                    self._enable(g, True, 2)
                    if spec.colours:
                        m.geom_rgba[g, :3] = spec.colours[min(si, len(spec.colours) - 1)]
                    _set_capsule(m, g, spec.radius, 0.5 * spec.seg_len)
                    m.geom_pos[g] = [0.5 * spec.seg_len, 0, 0]
                    if si == 0:
                        m.body_pos[b] = spec.root
                        m.body_quat[b] = quat_from_x_axis(spec.direction)
                    else:
                        m.body_pos[b] = [spec.seg_len, 0, 0]
                        m.body_quat[b] = _quat_pitch(spec.bend)
                    for j in sg["joints"]:
                        m.jnt_stiffness[j] = spec.stiffness
                        m.dof_damping[m.jnt_dofadr[j]] = spec.damping
                if spec is not None:
                    s["active"].append((segs, spec))
        js = [j for segs, _ in s["active"] for sg in segs for j in sg["joints"]]
        s["act_qadr"] = np.array([m.jnt_qposadr[j] for j in js], dtype=int)
        s["act_k"] = np.array([m.jnt_stiffness[j] for j in js], dtype=float)
        s["root_qadr"] = np.array([[m.jnt_qposadr[j] for j in segs[0]["joints"]] for segs, _ in s["active"]],
                                  dtype=int).reshape(-1, 2)
        s["damaged"] = set()
        data.qpos[s["leaf_qadr"]] = 0.0
        data.qvel[s["leaf_dadr"]] = 0.0
        data.qpos[s["stem_qadr"]] = 0.0
        data.qvel[s["stem_dadr"]] = 0.0
        q = s["qadr"]
        data.qpos[q:q + 3] = lay.fruit_center
        data.qpos[q + 3:q + 7] = lay.quat
        data.qvel[s["dadr"]:s["dadr"] + 6] = 0.0
        self.arm_stem(i, data)

    def arm_stem(self, i: int, data: mujoco.MjData):
        m, s = self.m, self.slots[i]
        mujoco.mj_kinematics(m, data)
        p1, p2 = data.xpos[s["stem_body"]], data.xpos[s["fruit_body"]]
        R1 = data.xmat[s["stem_body"]].reshape(3, 3)
        q1, q2 = data.xquat[s["stem_body"]], data.xquat[s["fruit_body"]]
        q1c = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
        rel = np.zeros(4)
        mujoco.mju_mulQuat(rel, q1c, q2)
        m.eq_data[s["eq"], 0:3] = 0.0
        m.eq_data[s["eq"], 3:6] = R1.T @ (p2 - p1)
        m.eq_data[s["eq"], 6:10] = rel
        data.eq_active[s["eq"]] = 1
        s["detached"] = False

    # ------------------------------------------------------------------ stem
    def stem_load(self, i: int, data: mujoco.MjData):
        """(weld force (3,), stem torque (3,)) - torque = torsion spring k * twist angle about the stalk axis plus the
        bending spring torques (the stem link joints carry the moment, the weld carries the force)."""
        s = self.slots[i]
        if not data.eq_active[s["eq"]]:
            return np.zeros(3), np.zeros(3)
        mask = (data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (data.efc_id == s["eq"])
        f = data.efc_force[mask]
        force = f[:3].copy() if f.shape[0] >= 3 else np.zeros(3)
        th = data.qpos[s["stem_qadr"][:3]]
        k = self.m.jnt_stiffness[[self.m.dof_jntid[a] for a in s["stem_dadr"][:3]]]
        return force, np.array([k[1] * th[1], k[2] * th[2], k[0] * th[0]])

    def twist_torque(self, i: int, data) -> float:
        s = self.slots[i]
        return float(abs(s["k_twist"] * data.qpos[s["stem_qadr"][0]]))

    def record_rest_load(self, i: int, data: mujoco.MjData):
        s = self.slots[i]
        s["rest_force"], s["rest_torque"] = self.stem_load(i, data)
        s["rest_ext"] = float(data.qpos[s["stem_qadr"][3]])

    def pull_force(self, i: int, data) -> float:
        """Tension in the stalk = tension spring k * extension beyond the resting (fruit-weight) sag; N, >= 0."""
        s = self.slots[i]
        return float(max(0.0, s["k_pull"] * (data.qpos[s["stem_qadr"][3]] - s["rest_ext"])))

    def stem_excess(self, i: int, data: mujoco.MjData) -> tuple[float, float]:
        s = self.slots[i]
        f, t = self.stem_load(i, data)
        return float(np.linalg.norm(f - s["rest_force"])), float(np.linalg.norm(t - s["rest_torque"]))

    def check_break(self, i: int, data: mujoco.MjData) -> tuple[bool, float, float]:
        s = self.slots[i]
        if s["detached"] or s["layout"] is None:
            return False, 0.0, 0.0
        # the stalk snaps in TENSION (pull along its axis) or TORSION (twist); lateral pushes only bend it.
        # Both are read from the stem springs (smooth), not from instantaneous constraint forces (contact spikes).
        fe = self.pull_force(i, data)
        te = self.twist_torque(i, data)
        lay = s["layout"]
        return (fe > lay.break_force or te > lay.break_torque), fe, te

    def detach(self, i: int, data: mujoco.MjData):
        m, s = self.m, self.slots[i]
        data.eq_active[s["eq"]] = 0
        for g in (s["fruit_geom"], s["neck_geom"]):
            m.geom_conaffinity[g] = 7
        m.body_conaffinity[s["fruit_body"]] |= 7          # body-level broadphase filter follows the geoms
        # the crown leaves of the picked fruit stop colliding: rigid long capsules on a light body that touch the
        # gripper/wrist make the contact problem stiff (measured: fruit spun up and was ejected); real crown leaves
        # are flexible and simply bend around the hand.
        for g in s["tuft_geoms"]:
            m.geom_contype[g] = 0
            m.geom_conaffinity[g] = 0
        s["detached"] = True
        if self.mute_on_detach:
            self._mute_penetrating(i, data)

    # ------------------------------------------------------------------ leaves inside a picked fruit
    def _mute_penetrating(self, i: int, data: mujoco.MjData):
        """Leaves that already pass THROUGH the fruit when it is picked (dense rosettes, neighbours' leaves) would
        be hit by the now-colliding fruit from the inside -> large contact impulse. Real leaves are thin blades that
        the fruit simply slides past, so those leaves stop colliding until the fruit has left them."""
        m, s = self.m, self.slots[i]
        fg = [s["fruit_geom"], s["neck_geom"]]
        c = data.xpos[s["fruit_body"]]
        ft = np.zeros(6)
        out = []
        for g in self._leaf_slot:
            if m.geom_contype[g] == 0 or np.linalg.norm(data.geom_xpos[g] - c) > m.geom_rbound[g] + 0.25:
                continue
            if min(mujoco.mj_geomDistance(m, data, g, f, 0.02, ft) for f in fg) < 0.004:
                m.geom_contype[g] = 0
                out.append(g)
        if out:
            self.muted.append((i, out))

    def update_muted(self, data: mujoco.MjData):
        """Give the muted leaves their contacts back once the picked fruit is clear of them."""
        if not self.muted:
            return
        m = self.m
        keep = []
        for i, gs in self.muted:
            s = self.slots[i]
            c = data.xpos[s["fruit_body"]]
            rest = [g for g in gs if np.linalg.norm(data.geom_xpos[g] - c) < m.geom_rbound[g] + 0.20]
            for g in gs:
                if g not in rest:
                    m.geom_contype[g] = 2
            if rest:
                keep.append((i, rest))
        self.muted = keep

    def _unmute_slot(self, i: int):
        keep = []
        for j, gs in self.muted:
            if j == i:
                for g in gs:
                    self.m.geom_contype[g] = 2
            else:
                keep.append((j, gs))
        self.muted = keep

    def remove_lost(self, i: int, data: mujoco.MjData):
        """A picked/knocked-off fruit that is neither in the gripper nor in the basket is lost (counted as such).
        It is taken out of the scene: a loose light capsule resting on articulated leaf capsules spun up to
        ~900 rad/s and made the whole simulation unstable (measured, Mission 3)."""
        m, s = self.m, self.slots[i]
        data.eq_active[s["eq"]] = 0
        for g in s["tuft_geoms"] + [s["fruit_geom"], s["neck_geom"]]:
            m.geom_contype[g] = m.geom_conaffinity[g] = 0
            m.geom_rgba[g, 3] = 0.0
        q = s["qadr"]
        data.qpos[q:q + 7] = [0.0, 0.0, -5.0 - i, 1, 0, 0, 0]
        data.qvel[s["dadr"]:s["dadr"] + 6] = 0.0
        s["detached"], s["lost"] = True, True

    # ------------------------------------------------------------------ ground truth (metrics / diagnostics only)
    def fruit_pose(self, i: int, data):
        s = self.slots[i]
        return data.xpos[s["fruit_body"]].copy(), data.xquat[s["fruit_body"]].copy()

    def active_slots(self):
        return [i for i, s in enumerate(self.slots) if s["layout"] is not None]

    def leaf_points(self, data: mujoco.MjData, slots=None, per_seg: int = 3, include_tuft: bool = True):
        geoms, half, rad = [], [], []
        slots = self.active_slots() if slots is None else slots
        for i in slots:
            s = self.slots[i]
            for segs, spec in s["active"]:
                for sg in segs:
                    geoms.append(sg["geom"]); half.append(0.5 * spec.seg_len); rad.append(spec.radius)
            if include_tuft and not s["harvested"]:
                for g in s["tuft_geoms"]:
                    if self.m.geom_contype[g]:
                        geoms.append(g); half.append(float(self.m.geom_size[g, 1])); rad.append(float(self.m.geom_size[g, 0]))
        if not geoms:
            return np.zeros((0, 3)), np.zeros(0)
        g = np.array(geoms)
        h = np.array(half)[:, None]
        c = data.geom_xpos[g]
        ax = data.geom_xmat[g].reshape(-1, 3, 3)[:, :, 2]
        t = ((np.arange(per_seg) + 0.5) / per_seg)[None, :, None]
        pts = (c - h * ax)[:, None, :] + (2 * h)[:, :, None] * t * ax[:, None, :]
        reff = np.sqrt(np.array(rad) ** 2 + (h[:, 0] / per_seg) ** 2)
        return pts.reshape(-1, 3), np.repeat(reff, per_seg)

    def leaf_root_bend(self, i: int, data) -> np.ndarray:
        """Bend (rad) of every leaf at its ROOT hinge pair (where a leaf is torn out of the rosette)."""
        r = self.slots[i].get("root_qadr")
        if r is None or r.size == 0:
            return np.zeros(0)
        q = data.qpos[r]
        return np.hypot(q[:, 0], q[:, 1])

    def leaf_root_bend_max(self, i: int, data) -> float:
        b = self.leaf_root_bend(i, data)
        return float(b.max()) if b.size else 0.0

    def new_leaf_damage(self, i: int, data, limit: float) -> int:
        """Leaves of slot i bent beyond `limit` (any hinge) for the first time: counted once per leaf."""
        s = self.slots[i]
        adr = s.get("act_qadr")
        if adr is None or adr.size == 0:
            return 0
        if s.get("chain_list") is not None:                  # realistic slot: every chain is active
            sizes = [2 * len(segs) for segs in s["chain_list"]]
        else:                                                 # light plant: active chains only
            sizes = [2 * len(segs) for segs, _ in s["active"]]
        q = np.abs(data.qpos[adr])
        n_new, k = 0, 0
        for li, n in enumerate(sizes):
            if li not in s["damaged"] and np.any(q[k:k + n] > limit):
                s["damaged"].add(li)
                n_new += 1
            k += n
        return n_new

    def leaf_bend_max(self, i: int, data) -> float:
        adr = self.slots[i].get("act_qadr")
        return float(np.max(np.abs(data.qpos[adr]))) if adr is not None and adr.size else 0.0

    def leaf_energy(self, i: int, data) -> float:
        s = self.slots[i]
        adr = s.get("act_qadr")
        if adr is None or adr.size == 0:
            return 0.0
        q = data.qpos[adr]
        return float(0.5 * np.sum(s["act_k"] * q * q))
