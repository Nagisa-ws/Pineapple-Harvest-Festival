"""
Realistic assets (pretraining rig, realistic evaluation of Missions 1-3) -> assets_xgh/

    python scripts/make_xgh_assets.py --src /path/to/mesh_pineapple      # folder of the user's mesh_pineapple.zip

Writes (MuJoCo uses STL/OBJ, Meshcat uses glTF with external .bin buffers - Drake de-duplicates those in the
standalone HTML, so the 5 plant kinds are stored once even when 100 plants show them):
  vehicle/      XGH505H + basket STL (user's CAD), *.gltf (Meshcat, one colour)
  ur10e/        UR10e OBJ (MuJoCo) + decimated glTF (Meshcat), Servo-v1 (gripper box)
  pineapple/    fruit_k (visual), fruit_k_col (convex hull), crown_k (visual): STL + glTF (vertex colours)
  plants/       PER-LEAF plants: for plant kind p, every leaf is cut into 2-3 segments along its spine;
                leaves/p{p}_l{L}_s{S}.stl/.gltf (segment mesh in its segment frame) + p{p}.json (segment frames,
                collision capsules, stiffness) + p{p}_static.stl/.gltf (all leaves in the rest pose = exactly what
                the articulated version looks like before it is touched: no visual jump at a level-of-detail swap)
  ground/       ground.obj + soil.png (textured raised bed + track furrows, ONE mesh, visual only) + ground.gltf
The plant leaves are regenerated with the same random sequence as make_pineapple_meshes.py (same plants), at a
lower resolution (18 x 5 instead of 44 x 7 vertices per leaf side).

Colours (user's request): bed green, frame silver, tracks black, arm white, gripper servo silver, jaws silver + black
pads, leaves green with orange leaves / orange tips, crown dark green, fruit yellow. The camera classifies colours
(pinesim/vision/perception.py): fruit = yellow (OpenCV hue 21-34), leaves = green (36-90) or orange (4-18); colour
gradients between green and orange pass through a desaturated khaki so that no leaf pixel ever looks like fruit.
Meshes are NOT decimated any more for the arm, the vehicle and the crates (quadric decimation produced flipped /
missing triangles: the see-through patches on the arm in the Mission 4 HTML).
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets_xgh"
GOLDEN = math.radians(137.50776)

# plant kinds = make_pineapple_meshes.py PLANTS (same seeds -> same plants)
PLANTS = [
    dict(name="plant_1", n=40, L=(0.50, 0.85), w=0.050, el=(82, 56), tip=(66, 18), droop=1.4, jitter=6, stem_h=0.18, seed=31),
    dict(name="plant_2", n=50, L=(0.50, 0.85), w=0.055, el=(72, 30), tip=(38, -18), droop=1.8, jitter=8, stem_h=0.20, seed=32),
    dict(name="plant_3", n=28, L=(0.60, 0.95), w=0.048, el=(78, 46), tip=(56, 2), droop=1.6, jitter=12, stem_h=0.16, seed=33),
    dict(name="plant_4", n=70, L=(0.40, 0.75), w=0.045, el=(82, 42), tip=(60, 4), droop=1.5, jitter=7, stem_h=0.22, seed=34),
    dict(name="plant_5", n=45, L=(0.55, 0.90), w=0.060, el=(72, 46), tip=(16, -45), droop=2.4, jitter=8, stem_h=0.18, seed=35),
]
N_ROWS, N_W = 18, 5            # leaf resolution (rows along the spine, vertices across)
MAX_CHORD_DEV = 0.03           # m: a segment is split until its spine stays within this of its chord
MAX_SEG = 3


# ============================================================================= leaf (spine + surface)
LEAF_PALETTES = {      # (s along the leaf, rgb); hue checked against the camera classes (see module docstring)
    "green": [(0.0, (0.70, 0.72, 0.55)), (0.12, (0.20, 0.45, 0.20)), (1.0, (0.22, 0.40, 0.17))],
    "green2": [(0.0, (0.70, 0.72, 0.55)), (0.12, (0.25, 0.50, 0.22)), (1.0, (0.18, 0.36, 0.15))],
    "orange_tip": [(0.0, (0.70, 0.72, 0.55)), (0.12, (0.21, 0.45, 0.20)), (0.52, (0.24, 0.44, 0.19)),
                   (0.66, (0.46, 0.42, 0.33)), (0.80, (0.80, 0.44, 0.13)), (1.0, (0.70, 0.35, 0.10))],
    "orange": [(0.0, (0.70, 0.62, 0.45)), (0.20, (0.78, 0.46, 0.15)), (1.0, (0.62, 0.32, 0.10))],
}


def leaf_colours(s, palette: str):
    pts = LEAF_PALETTES[palette]
    xs = [p[0] for p in pts]
    return np.stack([np.interp(s, xs, [p[1][c] for p in pts]) for c in range(3)], 1)


def leaf_surface(L, w_max, *, theta0, theta1, bend_p=1.6, twist=0.0, fold=25.0, thick=0.0015, sheath=0.6, w_peak=0.18,
                 taper_start=0.55, side_curl=0.0, palette="green"):
    """make_pineapple_meshes.leaf_mesh as arrays: spine P (S,3), top/bottom surfaces (S,W,3), colours (S,3), width."""
    s = np.linspace(0.0, 1.0, N_ROWS)
    th = np.radians(theta0 + (theta1 - theta0) * s ** bend_p)
    ph = np.radians(side_curl) * s ** 2
    ds = L / (N_ROWS - 1)
    T = np.stack([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), np.sin(th)], 1)
    P = np.vstack([[0, 0, 0], np.cumsum(T[:-1] * ds, 0)])
    w = np.where(s < w_peak, sheath + (1 - sheath) * (s / w_peak) ** 0.7, 1.0)
    w = np.where(s > taper_start, w * np.clip((1 - s) / (1 - taper_start), 0, 1) ** 0.85, w) * w_max
    up = np.array([0, 0, 1.0])
    B = np.cross(up, T)
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
    tw = np.radians(twist) * s
    N0 = np.cross(T, B)
    Bt = B * np.cos(tw)[:, None] + N0 * np.sin(tw)[:, None]
    Nn = np.cross(T, Bt)
    u = np.linspace(-1, 1, N_W)
    half = 0.5 * w[:, None] * u[None, :]
    lift = np.abs(half) * math.tan(math.radians(fold))
    top = P[:, None, :] + half[..., None] * Bt[:, None, :] + lift[..., None] * Nn[:, None, :]
    bot = top - thick * Nn[:, None, :] * (1 - 0.6 * s)[:, None, None]
    col = leaf_colours(s, palette)
    return P, top, bot, np.clip(col, 0, 1), w


def surface_mesh(top, bot, col, i0, i1, cap_base: bool):
    """Closed-ish thin solid of rows i0..i1 (inclusive) of a leaf surface -> (V, F, C)."""
    tp, bt = top[i0:i1 + 1], bot[i0:i1 + 1]
    S, W = tp.shape[:2]
    V = np.vstack([tp.reshape(-1, 3), bt.reshape(-1, 3)])
    it = np.arange(S * W).reshape(S, W)
    ib = it + S * W
    F = []
    for i in range(S - 1):
        for j in range(W - 1):
            a, b, c, d = it[i, j], it[i + 1, j], it[i + 1, j + 1], it[i, j + 1]
            F += [[a, b, c], [a, c, d]]
            a, b, c, d = ib[i, j], ib[i + 1, j], ib[i + 1, j + 1], ib[i, j + 1]
            F += [[a, c, b], [a, d, c]]
        for j in (0, W - 1):
            a, b, c, d = it[i, j], it[i + 1, j], ib[i + 1, j], ib[i, j]
            F += [[a, c, b], [a, d, c]] if j == 0 else [[a, b, c], [a, c, d]]
    if cap_base:
        for j in range(W - 1):
            a, b, c, d = it[0, j], it[0, j + 1], ib[0, j + 1], ib[0, j]
            F += [[a, b, c], [a, c, d]]
    c_rows = col[i0:i1 + 1]
    C = np.vstack([np.repeat(c_rows, W, 0), np.clip(np.repeat(c_rows, W, 0) * 0.85 + 0.12, 0, 1)])
    return V, np.array(F), C


def frame_x(d):
    """Rotation with x along d and z as close to world up as possible (columns x, y, z)."""
    x = d / (np.linalg.norm(d) + 1e-12)
    up = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
    y = np.cross(up, x)
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


def split_rows(P):
    """Row indices where the spine is cut so that every segment stays within MAX_CHORD_DEV of its chord."""
    cuts = [0, len(P) - 1]
    for _ in range(MAX_SEG - 1):
        worst, where = 0.0, None
        for a, b in zip(cuts[:-1], cuts[1:]):
            ch = P[b] - P[a]
            n = np.linalg.norm(ch) + 1e-12
            rel = P[a:b + 1] - P[a]
            dev = np.linalg.norm(rel - np.outer(rel @ ch / n, ch / n), axis=1)
            k = int(np.argmax(dev))
            if dev[k] > worst and 1 <= k <= (b - a) - 1:
                worst, where = dev[k], a + k
        if worst < MAX_CHORD_DEV or where is None:
            break
        cuts = sorted(cuts + [where])
    if len(cuts) == 2:                                  # at least two segments: a leaf bends at its base and mid
        cuts = [0, int(0.45 * (len(P) - 1)), len(P) - 1]
    return cuts


def mesh_obj(V, F, C=None):
    m = trimesh.Trimesh(V, F, process=False)
    if C is not None:
        m.visual.vertex_colors = np.hstack([(np.clip(C, 0, 1) * 255).astype(np.uint8), np.full((len(C), 1), 255, np.uint8)])
    return m


def write_gltf(mesh_or_scene, path: Path):
    """glTF + external .bin (Drake Meshcat de-duplicates external buffers by content in the standalone HTML)."""
    m = mesh_or_scene.copy()
    # glTF is +Y up; Drake rotates glTF files to +Z up when loading them, so store Z-up data as Y-up
    m.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))
    sc = m if isinstance(m, trimesh.Scene) else trimesh.Scene(m)
    files = trimesh.exchange.gltf.export_gltf(sc, embed_buffers=False)
    stem = path.stem
    ren = {}
    for k in files:
        if k.endswith(".bin"):
            ren[k] = f"{stem}_{k}"
    for k, v in files.items():
        if k == "model.gltf":
            js = json.loads(v.decode())
            for b in js.get("buffers", []):
                if b.get("uri") in ren:
                    b["uri"] = ren[b["uri"]]
            for im in js.get("images", []):
                if im.get("uri") in ren:
                    im["uri"] = ren[im["uri"]]
            path.write_text(json.dumps(js))
        else:
            (path.parent / ren.get(k, k)).write_bytes(v)


CROWN_SIZE = {1: (0.132, 0.140), 2: (0.152, 0.127), 3: (0.098, 0.163), 4: (0.123, 0.137), 5: (0.130, 0.120)}  # width, height


def keepout_radius(z, keep):
    """Radius the leaves must keep from the plant axis at height z: the fruit (+1.5 cm) on its stalk and the crown.
    Real inner leaves stand AROUND the peduncle and the fruit; the procedural rosette put some of them through the
    fruit volume, which a detached fruit would hit (contact impulse) - so they are bent outward here."""
    zb, zt, rf, ch, cr = keep["zb"], keep["zt"], keep["rf"], keep["crown_h"], keep["crown_r"]
    r = np.zeros_like(z)
    t = np.clip((z - (zb - 0.08)) / 0.08, 0, 1)
    r = np.where(z < zb, rf * t * t * (3 - 2 * t), r)
    r = np.where((z >= zb) & (z <= zt), rf, r)
    tc = np.clip((z - zt) / max(ch, 1e-6), 0, 1)
    r = np.where((z > zt) & (z <= zt + ch), cr * (0.9 - 0.4 * tc), r)
    return r


def deform(X, keep, yaw):
    if keep is None:
        return X
    X = X.copy()
    r = np.hypot(X[..., 0], X[..., 1])
    R = keepout_radius(X[..., 2], keep)
    inside = r < R
    if np.any(inside):
        az = np.where(r > 1e-6, np.arctan2(X[..., 1], X[..., 0]), yaw)
        X[..., 0] = np.where(inside, R * np.cos(az), X[..., 0])
        X[..., 1] = np.where(inside, R * np.sin(az), X[..., 1])
    return X


# MuJoCo cannot draw vertex colours: every leaf segment gets the colour of its CLASS (same for the static rest-pose
# meshes and the articulated segments, so a level-of-detail swap does not change the camera image); Meshcat uses the
# vertex colours of the glTF files.
LEAF_CLASSES = {"green": (0.23, 0.46, 0.20), "green_dark": (0.18, 0.37, 0.16), "orange": (0.74, 0.41, 0.13)}


def leaf_class(rgb) -> str:
    rgb = np.asarray(rgb, float)
    hue_orange = rgb[0] > rgb[1] * 1.15
    if hue_orange:
        return "orange"
    return "green" if rgb[1] > 0.40 else "green_dark"


def plant_kind(spec, out_dir: Path, keep: dict | None = None) -> dict:
    """Per-leaf articulated description of one plant kind + its static rest-pose mesh. `keep` = fruit/crown of the
    pineapple combination that grows on this plant kind (leaves are kept out of its volume)."""
    rng = np.random.default_rng(spec["seed"])
    crng = np.random.default_rng(spec["seed"] + 1000)
    n, sh = spec["n"], spec["stem_h"]
    leaves, static_parts = [], []
    ldir = out_dir / "leaves"
    ldir.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        a = k / (n - 1)
        L = (spec["L"][1] - (spec["L"][1] - spec["L"][0]) * a ** 1.8) * rng.uniform(0.85, 1.1)
        el0 = spec["el"][1] + (spec["el"][0] - spec["el"][1]) * a + rng.normal(0, spec["jitter"] * 0.5)
        el1 = spec["tip"][1] + (spec["tip"][0] - spec["tip"][1]) * a ** 0.7 + rng.normal(0, spec["jitter"])
        w_leaf = spec["w"] * rng.uniform(0.85, 1.12)
        twist = rng.normal(0, 25)
        fold = rng.uniform(22, 35)
        side_curl = rng.normal(0, 10)
        tip_red = 0.25 * rng.uniform(0, 1) * (a < 0.4)          # kept: same random sequence as the original plants
        # colour: outer (older) leaves are more often orange or orange-tipped, inner ones green (own rng: the leaf
        # geometry sequence above is unchanged)
        u = crng.random()
        palette = ("orange" if u < 0.22 * (1 - a) else "orange_tip" if u < 0.22 * (1 - a) + 0.30
                   else ("green" if k % 3 else "green2"))
        P, top, bot, col, wprof = leaf_surface(
            L, w_leaf, theta0=el0, theta1=el1, bend_p=spec["droop"], twist=twist, fold=fold, thick=0.002, sheath=0.85,
            w_peak=0.10, taper_start=0.62, side_curl=side_curl, palette=palette)
        z0 = 0.03 + (sh - 0.03) * a ** 0.9
        r0 = 0.045 * (1 - 0.7 * a)
        yaw = math.degrees(k * GOLDEN) + rng.normal(0, 4)
        ya = math.radians(yaw)
        Rz = np.array([[math.cos(ya), -math.sin(ya), 0], [math.sin(ya), math.cos(ya), 0], [0, 0, 1]])
        off = np.array([r0 * math.cos(ya), r0 * math.sin(ya), z0])

        def to_plant(X):
            Y = X @ Rz.T + off
            low = Y[..., 2] < 0.004                       # leaves that would go below the ground lie on it
            Y[low, 2] = 0.004 + (Y[low, 2] - 0.004) * 0.02
            return deform(Y, keep, ya)
        Pw, topw, botw = to_plant(P), to_plant(top), to_plant(bot)
        cuts = split_rows(Pw)
        segs = []
        for si, (i0, i1) in enumerate(zip(cuts[:-1], cuts[1:])):
            p0, p1 = Pw[i0], Pw[i1]
            R = frame_x(p1 - p0)
            V, F, C = surface_mesh(topw, botw, col, i0, i1, cap_base=(si == 0))
            cls = leaf_class(np.mean(col[min(i0 + 1, i1):i1 + 1], axis=0))
            static_parts.append((cls, mesh_obj(V, F, C)))
            Vl = (V - p0) @ R                              # segment frame
            name = f"p{spec['name'][-1]}_l{k:02d}_s{si}"
            m = mesh_obj(Vl, F, C)
            m.export(ldir / f"{name}.stl")
            write_gltf(m, ldir / f"{name}.gltf")
            wmid = float(np.mean(wprof[i0:i1 + 1]))
            segs.append(dict(mesh=name, p0=p0.tolist(), p1=p1.tolist(), R=R.tolist(), length=float(np.linalg.norm(p1 - p0)),
                             radius=float(np.clip(0.35 * wmid, 0.008, 0.02)), cls=cls))
        leaves.append(dict(index=k, length=float(L), width=float(w_leaf), segments=segs))
    stem = trimesh.creation.cylinder(radius=1.0, height=sh + 0.02, sections=24)
    v = stem.vertices.copy()
    t = (v[:, 2] + (sh + 0.02) / 2) / (sh + 0.02)
    v[:, :2] *= (0.055 + (0.035 - 0.055) * t)[:, None]
    v[:, 2] += (sh + 0.02) / 2
    stem.vertices = v
    stem.visual.vertex_colors = np.array([92, 107, 61, 255], np.uint8)     # green-brown (camera: plant, not fruit)
    pk = spec['name'][-1]
    stem.export(out_dir / f"p{pk}_stem.stl")
    write_gltf(stem, out_dir / f"p{pk}_stem.gltf")
    static = trimesh.util.concatenate([m for _, m in static_parts] + [stem])
    static.export(out_dir / f"p{pk}_static.stl")
    write_gltf(static, out_dir / f"p{pk}_static.gltf")            # Meshcat: one mesh with vertex colours
    for f in out_dir.glob(f"p{pk}_static_*.stl"):
        f.unlink()
    classes = {}
    for cls in LEAF_CLASSES:                                         # MuJoCo: one rest-pose mesh per colour class
        parts = [m for c, m in static_parts if c == cls]
        if parts:
            trimesh.util.concatenate(parts).export(out_dir / f"p{pk}_static_{cls}.stl")
            classes[cls] = len(parts)
    cols = np.vstack([np.asarray(m.visual.vertex_colors)[:, :3] for _, m in static_parts]) / 255.0
    info = dict(name=spec["name"], kind=int(spec["name"][-1]), stem_h=sh, n_leaves=n, keepout=keep,
                static_classes=classes, class_rgb=LEAF_CLASSES,
                n_segments=sum(len(l["segments"]) for l in leaves), mean_color=np.clip(cols.mean(0) * 1.05, 0, 1).tolist(),
                height=float(static.bounds[1, 2]), spread=float(np.max(np.ptp(static.bounds[:, :2], axis=0))),
                triangles_static=int(len(static.faces)), leaves=leaves)
    (out_dir / f"p{spec['name'][-1]}.json").write_text(json.dumps(info))
    return info


# ============================================================================= ground
def soil_texture(path: Path, size=1024, seed=5):
    """Tileable soil texture: brown base, multi-scale noise, darker clods, few light pebbles (saturation < 0.4 so the
    camera's fruit colour class never fires on soil)."""
    from PIL import Image
    rng = np.random.default_rng(seed)

    def tile_noise(cells):
        g = rng.random((cells, cells))
        g = np.pad(g, ((0, 1), (0, 1)), mode="wrap")
        x = np.linspace(0, cells, size, endpoint=False)
        xi = x.astype(int)
        xf = x - xi
        xf = xf * xf * (3 - 2 * xf)
        a = g[np.ix_(xi, xi)] * (1 - xf)[None, :] + g[np.ix_(xi, xi + 1)] * xf[None, :]
        b = g[np.ix_(xi + 1, xi)] * (1 - xf)[None, :] + g[np.ix_(xi + 1, xi + 1)] * xf[None, :]
        return a * (1 - xf)[:, None] + b * xf[:, None]
    n = sum(tile_noise(c) / (i + 1) for i, c in enumerate([4, 8, 16, 32, 64, 128]))
    n = (n - n.min()) / (n.max() - n.min())
    base = np.array([0.40, 0.31, 0.22])
    dark = np.array([0.24, 0.18, 0.13])
    img = dark[None, None, :] + (base - dark)[None, None, :] * n[..., None]
    clod = tile_noise(96) > 0.80
    img[clod] *= 0.82
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(260):                                   # pebbles
        cx, cy, r = rng.integers(0, size), rng.integers(0, size), rng.uniform(1.5, 4.5)
        d2 = ((xx - cx + size // 2) % size - size // 2) ** 2 + ((yy - cy + size // 2) % size - size // 2) ** 2
        m = d2 < r * r
        img[m] = np.array([0.55, 0.50, 0.44]) * rng.uniform(0.8, 1.05)
    img += rng.normal(0, 0.015, img.shape)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)


def ground_height(x, y, rng_field):
    """Raised bed under the two plant rows (|y| < ~0.4), track furrows at |y| 0.45-0.65 (height <= 0: the tracks
    stand on the physics plane z = 0 and must not look sunk in), outer ridges, small undulation and clods."""
    ay = np.abs(y)
    bed = 0.022 * (1 / (1 + np.exp((ay - 0.38) / 0.03)))
    furrow = -0.012 * np.exp(-((ay - 0.55) / 0.07) ** 2)
    ridge = 0.03 * np.exp(-((ay - 0.95) / 0.12) ** 2) + 0.012 * np.clip(ay - 1.1, 0, 2)
    und = rng_field(x, y)
    h = bed + furrow + ridge + und * (1 - 0.85 * np.exp(-((ay - 0.55) / 0.12) ** 2))
    in_lane = (ay > 0.45) & (ay < 0.65)
    return np.where(in_lane, np.minimum(h, -0.002), h)


def make_ground(out_dir: Path, x0=-3.0, x1=19.0, y0=-3.0, y1=3.0, dx=0.08, dy=0.04, tex_m=1.2):
    out_dir.mkdir(parents=True, exist_ok=True)
    soil_texture(out_dir / "soil.png")
    rng = np.random.default_rng(11)
    waves = [(rng.uniform(0.3, 1.4), rng.uniform(0.3, 1.4), rng.uniform(0, 6.3), rng.uniform(0.002, 0.006)) for _ in range(14)]

    def und(x, y):
        h = np.zeros_like(x)
        for fx, fy, ph, amp in waves:
            h += amp * np.sin(2 * np.pi * (fx * x / 1.7 + fy * y / 1.3) + ph)
        return h + rng.normal(0, 0.0015, x.shape)
    xs = np.arange(x0, x1 + 1e-9, dx)
    ys = np.arange(y0, y1 + 1e-9, dy)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = ground_height(X, Y, und)
    V = np.stack([X, Y, Z], -1).reshape(-1, 3)
    UV = np.stack([X / tex_m, Y / tex_m], -1).reshape(-1, 2)
    nx, ny = len(xs), len(ys)
    idx = np.arange(nx * ny).reshape(nx, ny)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    F = np.vstack([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    with open(out_dir / "ground.obj", "w") as f:
        f.write("# ground: raised bed + track furrows, generated by scripts/make_xgh_assets.py (visual only)\n")
        f.write("mtllib ground.mtl\nusemtl soil\n")
        for v in V:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for t in UV:
            f.write(f"vt {t[0]:.4f} {t[1]:.4f}\n")
        for t in F + 1:
            f.write(f"f {t[0]}/{t[0]} {t[1]}/{t[1]} {t[2]}/{t[2]}\n")
    (out_dir / "ground.mtl").write_text("newmtl soil\nKa 1 1 1\nKd 1 1 1\nKs 0.02 0.02 0.02\nNs 4\nmap_Kd soil.png\n")
    from PIL import Image
    m = trimesh.Trimesh(V, F, process=False,
                        visual=trimesh.visual.TextureVisuals(uv=UV, image=Image.open(out_dir / "soil.png")))
    write_gltf(m, out_dir / "ground.gltf")
    return dict(vertices=len(V), triangles=len(F), z_range=[float(Z.min()), float(Z.max())], x=[x0, x1], y=[y0, y1])


# ============================================================================= copies / conversions
def clean(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """Drop zero-area triangles (the source crowns have ~2 % of them) and unused vertices."""
    m = m.copy()
    keep = m.area_faces > 1e-12
    if not keep.all():
        m.update_faces(keep)
        m.remove_unreferenced_vertices()
    return m


def decimated(m: trimesh.Trimesh, faces: int) -> trimesh.Trimesh:
    """Quadric decimation that keeps the vertex colours (trimesh drops them: crowns came out white in Meshcat)."""
    m = clean(m)
    if len(m.faces) <= faces:
        return m
    d = m.simplify_quadric_decimation(face_count=faces)
    vc = getattr(m.visual, "vertex_colors", None)
    if m.visual.kind == "vertex" and vc is not None and len(vc) == len(m.vertices):
        from scipy.spatial import cKDTree
        _, idx = cKDTree(m.vertices).query(d.vertices)
        d.visual = trimesh.visual.ColorVisuals(d, vertex_colors=np.asarray(vc)[idx])
    return clean(d)                  # the decimation itself leaves a few zero-area triangles (crowns: 5-8)


def coloured(m: trimesh.Trimesh, rgb) -> trimesh.Trimesh:
    m = m.copy()
    m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=np.tile(np.r_[np.asarray(rgb) * 255, 255].astype(np.uint8), (len(m.vertices), 1)))
    return m


# arm white (UR blue caps and dark rubber rings kept, as in the user's picture), gripper servo silver
UR_COLOURS = dict(black=(0.10, 0.10, 0.11), jointgray=(0.80, 0.81, 0.83), linkgray=(0.96, 0.96, 0.95),
                  urblue=(0.49, 0.678, 0.8))
UR_MATERIAL = dict(base_0="black", base_1="jointgray", shoulder_0="urblue", shoulder_1="black", shoulder_2="jointgray",
                   upperarm_0="black", upperarm_1="jointgray", upperarm_2="urblue", upperarm_3="linkgray",
                   forearm_0="urblue", forearm_1="black", forearm_2="jointgray", forearm_3="linkgray",
                   wrist1_0="black", wrist1_1="urblue", wrist1_2="jointgray", wrist2_0="black", wrist2_1="urblue",
                   wrist2_2="jointgray", wrist3="linkgray")
SILVER = (0.74, 0.75, 0.77)
VEHICLE_COLOURS = dict(track_guide_teeth_left=(0.07, 0.07, 0.07), track_guide_teeth_right=(0.07, 0.07, 0.07),
                       track_wheels_right=SILVER, bed=(0.22, 0.50, 0.20), **{"basket-v1": (0.16, 0.36, 0.72)})
FRUIT_DARK, FRUIT_LIGHT = np.array([0.52, 0.40, 0.07]), np.array([0.98, 0.86, 0.22])     # yellow (hue 46-52 deg)
CROWN_DARK, CROWN_LIGHT = np.array([0.05, 0.22, 0.07]), np.array([0.17, 0.40, 0.15])     # dark green


def recolour(m: trimesh.Trimesh, dark, light) -> trimesh.Trimesh:
    """Keep the texture pattern (luminance of the original vertex colours), replace the hue."""
    m = m.copy()
    c = np.asarray(m.visual.vertex_colors)[:, :3].astype(float) / 255.0
    lum = c @ np.array([0.30, 0.59, 0.11])
    t = np.clip((lum - np.percentile(lum, 2)) / max(np.percentile(lum, 98) - np.percentile(lum, 2), 1e-6), 0, 1)
    rgb = dark[None, :] + (light - dark)[None, :] * t[:, None]
    m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=np.hstack([(rgb * 255).astype(np.uint8),
                                                                      np.full((len(rgb), 1), 255, np.uint8)]))
    return m


def check_mesh(m: trimesh.Trimesh, name: str) -> dict:
    """Integrity report: degenerate faces, flipped normals relative to the outward direction, NaNs."""
    area = m.area_faces
    out = dict(name=name, faces=int(len(m.faces)), degenerate=int(np.sum(area < 1e-12)),
               nan=bool(np.isnan(m.vertices).any()))
    try:
        c = m.vertices.mean(0)
        fc = m.triangles_center - c
        out["outward_frac"] = float(np.mean(np.einsum("ij,ij->i", fc, m.face_normals) > 0))
    except Exception:
        out["outward_frac"] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="unzipped mesh_pineapple folder (pineapple_field/, xgh505h_ur10e/, ...)")
    ap.add_argument("--skip-plants", action="store_true")
    a = ap.parse_args()
    src = Path(a.src)
    OUT.mkdir(exist_ok=True)
    summary, checks = {}, []
    # vehicle (user's CAD, mm): original triangles (no decimation)
    vdir = OUT / "vehicle"
    vdir.mkdir(exist_ok=True)
    for name, rgb in VEHICLE_COLOURS.items():
        f = src / "xgh505h_ur10e" / "meshes" / f"{name}.stl"
        shutil.copy(f, vdir / f.name)
        m = trimesh.load(f, force="mesh")
        m.apply_scale(0.001)
        checks.append(check_mesh(m, f"vehicle/{name}"))
        write_gltf(coloured(m, rgb), vdir / f"{name}.gltf")
    # UR10e + gripper servo: original meshes for MuJoCo AND Meshcat
    udir = OUT / "ur10e"
    udir.mkdir(exist_ok=True)
    for f in sorted(udir.glob("*_lo.obj")):
        f.unlink()
    for f in sorted((src / "xgh505h_ur10e" / "assets").glob("*")):
        shutil.copy(f, udir / f.name)
        if f.suffix == ".obj":
            m = clean(trimesh.load(f, force="mesh"))
            checks.append(check_mesh(m, f"ur10e/{f.stem}"))
            write_gltf(coloured(m, UR_COLOURS[UR_MATERIAL.get(f.stem, "linkgray")]), udir / f"{f.stem}.gltf")
    m = trimesh.load(udir / "Servo-v1.stl", force="mesh")
    m.apply_scale(0.001)
    checks.append(check_mesh(m, "ur10e/Servo-v1"))
    write_gltf(coloured(m, SILVER), udir / "Servo-v1.gltf")
    # pineapple fruit (yellow) / crown (dark green)
    pdir = OUT / "pineapple"
    pdir.mkdir(exist_ok=True)
    fsrc = src / "pineapple_field"
    for kind, faces in (("fruit", 9000), ("crown", 12000)):
        for i in range(1, 6):
            # MuJoCo visual mesh (camera images): decimated - CPU rendering (llvmpipe); fruit/crown are smooth
            # closed shapes, the decimation is checked below
            g0 = trimesh.load(fsrc / "assets" / f"{kind}_{i}.stl", force="mesh")
            g0d = decimated(g0, 2500 if kind == "fruit" else 3000)
            checks.append(check_mesh(g0d, f"pineapple/{kind}_{i}.stl"))
            g0d.export(pdir / f"{kind}_{i}.stl")
            if kind == "fruit":
                shutil.copy(fsrc / "assets" / f"fruit_{i}_col.stl", pdir / f"fruit_{i}_col.stl")
            g = trimesh.load(fsrc / kind / f"{kind}_{i}.glb", force="mesh")
            g = recolour(g, FRUIT_DARK, FRUIT_LIGHT) if kind == "fruit" else recolour(g, CROWN_DARK, CROWN_LIGHT)
            gd = decimated(g, faces)
            checks.append(check_mesh(gd, f"pineapple/{kind}_{i}.gltf"))
            write_gltf(gd, pdir / f"{kind}_{i}.gltf")
            c = np.asarray(g.visual.vertex_colors)[:, :3].mean(0) / 255.0
            top = float(g.vertices[:, 2].max())
            summary[f"{kind}_{i}"] = dict(mean_color=np.clip(c * 1.05, 0, 1).tolist(), top=top)
    # plants: leaves kept out of a COLUMN that holds any fruit / crown combination at any fruit height of the
    # configured range (fruit centre 0.60-0.70 m) - the pretraining rig combines all 5 x 5 x 5 meshes
    if not a.skip_plants:
        import sys
        sys.path.insert(0, str(ROOT))
        from pinesim.config import load_config
        cfg = load_config("mission3")
        ef = cfg["realistic"]
        zr = cfg["plant"]["fruit"]["center_height_range"]
        Dmax = max(v[0] for v in ef["fruit"].values())
        Hmax = max(v[1] for v in ef["fruit"].values())
        cw = max(v[0] for v in CROWN_SIZE.values())
        chh = max(v[1] for v in CROWN_SIZE.values())
        keep = dict(zb=float(zr[0]) - Hmax / 2, zt=float(zr[1]) + Hmax / 2, rf=Dmax / 2 + 0.015, crown_h=chh,
                    crown_r=cw / 2)
        for spec in PLANTS:
            info = plant_kind(spec, OUT / "plants", keep)
            summary[spec["name"]] = {k: v for k, v in info.items() if k != "leaves"}
            print(spec["name"], {k: v for k, v in summary[spec["name"]].items() if k != "mean_color"})
    summary["ground"] = make_ground(OUT / "ground")
    summary["mesh_checks"] = checks
    bad = [c for c in checks if c["degenerate"] > 0.02 * c["faces"] or c["nan"]]
    print("mesh checks:", len(checks), "meshes,", len(bad), "with problems", bad[:3])
    old = json.loads((OUT / "summary.json").read_text()) if (OUT / "summary.json").exists() else {}
    old.update(summary)
    (OUT / "summary.json").write_text(json.dumps(old, indent=1))
    print("ground", summary["ground"])


if __name__ == "__main__":
    main()
