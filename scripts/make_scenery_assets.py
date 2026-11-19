"""Decoration assets of the Mission 3 HTML (Meshcat only - the simulation and the robot cameras never see them):

  assets_xgh/scenery/dome.gltf        dome with a 360 degree panorama photo on its inner surface
  assets_xgh/scenery/soil_disc.gltf   soil widened to the dome radius (texture and material of the field soil)
  assets_xgh/scenery/grass_ring.gltf  grass that fades in towards the rim (colour of the photo below its horizon)
  assets_xgh/scenery/rock.gltf        one rock shape  (placed many times with different sizes / headings)
  assets_xgh/scenery/tree.gltf        one tree shape  (placed many times with different sizes / headings)
  assets_xgh/scenery/hut.gltf         one field hut (several colours)
  assets_xgh/scenery/panorama.jpg     the tone-mapped panorama that is on the dome (for reference)

  python scripts/make_scenery_assets.py --panorama /path/to/panorama_4k.hdr        # .hdr / .exr / .jpg / .png, 2:1
  python scripts/make_scenery_assets.py --panorama foto.jpg --radius 60 --exposure 1.3
  python scripts/make_scenery_assets.py --only-meshes                              # keep the dome, rebuild rock/tree/hut

The dome radius is stored in assets_xgh/scenery/scenery.json; pinesim/viz/scenery.py reads it. Where the objects are
placed is decided there (`scenery:` block of configs/common.yaml), not here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinesim.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "assets_xgh" / "scenery"


# ----------------------------------------------------------------------------- glTF (external buffers, like the
# other assets: Drake Meshcat stores repeated buffers once in the standalone HTML)
def write_gltf(path: Path, V, F, C=None, UV=None, image: bytes | None = None, mime="image/jpeg", material=None):
    """Z-up vertices V (n,3), triangles F (m,3), optional vertex colours C (n,3 in 0..1, LINEAR multipliers of the
    base colour), optional UV (n,2, origin top-left) + texture image. No normals are written: the viewer then
    shades every triangle flat, which is the low-poly look wanted here."""
    V = np.asarray(V, np.float32)
    Vy = np.stack([V[:, 0], V[:, 2], -V[:, 1]], axis=1).astype(np.float32)      # glTF is +Y up (Drake turns it back)
    F = np.asarray(F, np.uint32).reshape(-1)
    bufs, views, acc = [], [], []

    def add(data: bytes, name: str):
        fn = f"{path.stem}_{name}.bin"
        (path.parent / fn).write_bytes(data)
        bufs.append(dict(uri=fn, byteLength=len(data)))
        views.append(dict(buffer=len(bufs) - 1, byteOffset=0, byteLength=len(data)))
        return len(views) - 1

    acc.append(dict(bufferView=add(F.tobytes(), "idx"), componentType=5125, type="SCALAR", count=int(F.size),
                    max=[int(F.max())], min=[int(F.min())]))
    acc.append(dict(bufferView=add(Vy.tobytes(), "pos"), componentType=5126, type="VEC3", count=len(Vy),
                    max=Vy.max(0).tolist(), min=Vy.min(0).tolist()))
    attr = dict(POSITION=1)
    if C is not None:
        C = np.maximum(np.asarray(C, np.float32), 0)
        acc.append(dict(bufferView=add(C.tobytes(), "col"), componentType=5126, type="VEC3" if C.shape[1] == 3 else "VEC4",
                        count=len(C)))
        attr["COLOR_0"] = len(acc) - 1
    if UV is not None:
        UV = np.asarray(UV, np.float32)
        acc.append(dict(bufferView=add(UV.tobytes(), "uv"), componentType=5126, type="VEC2", count=len(UV),
                        max=UV.max(0).tolist(), min=UV.min(0).tolist()))
        attr["TEXCOORD_0"] = len(acc) - 1
    mat = dict(pbrMetallicRoughness=dict(baseColorFactor=[1, 1, 1, 1], metallicFactor=0.0, roughnessFactor=0.95),
               doubleSided=True)
    js = dict(asset=dict(version="2.0", generator="pinesim make_scenery_assets"), scene=0, scenes=[dict(nodes=[0])],
              nodes=[dict(name=path.stem, mesh=0)], accessors=acc, bufferViews=views, buffers=bufs)
    if image is not None:
        js["images"] = [dict(bufferView=add(image, "img"), mimeType=mime)]
        js["samplers"] = [dict(magFilter=9729, minFilter=9987, wrapS=10497, wrapT=10497)]
        js["textures"] = [dict(source=0, sampler=0)]
        mat["pbrMetallicRoughness"]["baseColorTexture"] = dict(index=0)
    if material:
        for k, v in material.items():
            if k == "pbrMetallicRoughness":
                mat[k].update(v)
                mat[k] = {a: b for a, b in mat[k].items() if b is not None}      # None = leave the glTF default
            else:
                mat[k] = v
        if "extensions" in material:
            js["extensionsUsed"] = list(material["extensions"])
    js["materials"] = [mat]
    js["meshes"] = [dict(name=path.stem, primitives=[dict(attributes=attr, indices=0, mode=4, material=0)])]
    path.write_text(json.dumps(js))


# ----------------------------------------------------------------------------- panorama
def load_panorama(path: Path) -> np.ndarray:
    """RGB float image (linear for .hdr / .exr, 0..1 sRGB for .jpg / .png)."""
    ext = path.suffix.lower()
    try:
        import cv2
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise RuntimeError("cv2 could not read the file")
        im = im[:, :, :3][:, :, ::-1].astype(np.float32)
    except ImportError:
        try:
            import imageio.v3 as iio
            im = np.asarray(iio.imread(str(path)), np.float32)[:, :, :3]
        except Exception as e:                               # noqa: BLE001
            raise SystemExit(f"cannot read {path}: {e}\n   pip install opencv-python   (reads .hdr / .exr)") from e
    if ext not in (".hdr", ".exr"):
        im = im / (255.0 if im.max() > 1.5 else 1.0)
    return im


def tonemap(im: np.ndarray, exposure: float) -> np.ndarray:
    """Linear HDR -> 8 bit sRGB: exposure from the log-average luminance, Reinhard curve, gamma 2.2."""
    lum = im.mean(axis=2)
    key = 0.18 / max(1e-6, float(np.exp(np.mean(np.log(lum + 1e-4)))))
    y = im * key * 2.2 * exposure
    y = y / (1.0 + y)
    return (np.clip(y, 0, 1) ** (1 / 2.2) * 255.0 + 0.5).astype(np.uint8)


def make_dome(pano: Path, radius: float, exposure: float, width: int, quality: int) -> dict:
    from PIL import Image
    im = load_panorama(pano)
    h, w = im.shape[:2]
    if abs(w / h - 2.0) > 0.05:
        print(f"   WARNING: {pano.name} is {w} x {h}; a 360 degree (equirectangular) panorama is 2 : 1")
    rgb = tonemap(im, exposure) if pano.suffix.lower() in (".hdr", ".exr") else \
        (np.clip(im * exposure, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    pil = Image.fromarray(rgb)
    if w > width:
        pil = pil.resize((width, width // 2), Image.LANCZOS)
    jpg = OUT / "panorama.jpg"
    pil.save(jpg, quality=quality, optimize=True)
    # sphere seen from inside, from 4 degrees below the horizon (hidden by the soil disc) up to the zenith
    n_az, n_el = 72, 26
    el = np.radians(np.linspace(-4.0, 90.0, n_el))
    u = np.linspace(0.0, 1.0, n_az + 1)
    V, UV = [], []
    for e in el:
        for uu in u:
            az = -2 * np.pi * uu                             # the picture runs to the right when the viewer turns right
            V.append([radius * np.cos(e) * np.cos(az), radius * np.cos(e) * np.sin(az), radius * np.sin(e)])
            UV.append([uu, 0.5 - e / np.pi])
    F = []
    for i in range(n_el - 1):
        for j in range(n_az):
            a, b, c, d = i * (n_az + 1) + j, i * (n_az + 1) + j + 1, (i + 1) * (n_az + 1) + j, (i + 1) * (n_az + 1) + j + 1
            F += [[a, b, d], [a, d, c]]
    Va = np.array(V)
    f0 = F[len(F) // 2]
    if np.cross(Va[f0[1]] - Va[f0[0]], Va[f0[2]] - Va[f0[0]]) @ Va[f0[0]] > 0:      # front faces must look at the centre
        F = [[a, c, b] for a, b, c in F]
    # the photo is its own light: unlit material (the lamps of the viewer do not shade it); single sided = drawn
    # from inside only, so a viewer who zooms out of the dome still sees the field
    write_gltf(OUT / "dome.gltf", V, F, UV=UV, image=jpg.read_bytes(),
               material=dict(extensions={"KHR_materials_unlit": {}}, doubleSided=False))
    # colour of the photo just below the horizon: the rim of the soil disc is tinted towards it
    a = np.asarray(pil, np.float32) / 255.0
    band = a[int(0.53 * a.shape[0]):int(0.60 * a.shape[0])]
    return dict(panorama=pano.name, radius=radius, exposure=exposure, size=[pil.width, pil.height],
                jpg_mb=round(jpg.stat().st_size / 1e6, 2), rim_rgb=np.round(np.median(band.reshape(-1, 3), axis=0), 3).tolist())


def _disc(r_in: float, r_out: float, n_rings: int, n: int = 72):
    """Flat ring (r_in = 0: disc) as rings of n vertices; returns V, F, radius of every vertex."""
    rs = np.linspace(r_in, r_out, n_rings) if r_in > 0 else np.linspace(3.0, r_out, n_rings - 1)
    V, F = [], []
    start = 0
    if r_in <= 0:
        V.append([0.0, 0.0, 0.0]); start = 1
    for r in rs:
        V += [[r * np.cos(2 * np.pi * k / n), r * np.sin(2 * np.pi * k / n), 0.0] for k in range(n)]
    if start:
        F += [[0, 1 + k, 1 + (k + 1) % n] for k in range(n)]
    for i in range(len(rs) - 1):
        o0, o1 = start + i * n, start + (i + 1) * n
        for k in range(n):
            a, b, c, d = o0 + k, o0 + (k + 1) % n, o1 + k, o1 + (k + 1) % n
            F += [[a, c, d], [a, d, b]]
    V = np.array(V)
    return V, F, np.linalg.norm(V[:, :2], axis=1)


def grass_texture(rim_rgb, size=512, seed=3) -> bytes:
    """Tileable grass-like noise whose mean colour is the colour of the photo just below its horizon."""
    import io
    from PIL import Image
    rng = np.random.default_rng(seed)
    acc = np.zeros((size, size))
    for cells, w in ((4, 0.45), (16, 0.3), (64, 0.25)):
        g = rng.random((cells, cells))
        k = size // cells
        g = np.kron(g, np.ones((k, k)))
        for ax in (0, 1):                                    # wrap-around box blur keeps the texture tileable
            g = sum(np.roll(g, sh, axis=ax) for sh in range(-(k // 2), k // 2 + 1)) / (2 * (k // 2) + 1)
        acc += w * (g - g.mean()) / (g.std() + 1e-9)
    acc += 0.35 * rng.normal(size=(size, size))
    t = np.clip(0.5 + 0.22 * acc, 0, 1)[:, :, None]
    c = np.asarray(rim_rgb, float)
    dark, light = c * np.array([0.62, 0.70, 0.55]), np.clip(c * np.array([1.22, 1.20, 1.05]), 0, 1)
    img = (np.clip(dark * (1 - t) + light * t, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def fix_ground_material() -> dict:
    """The field soil mesh of the HTML (assets_xgh/ground/ground.gltf, written by trimesh) has no `metallicFactor`;
    the glTF default is 1.0 = metal, which the viewer draws almost black. As a small dark rectangle that went
    unnoticed, as a 40 m disc it does not: both get metallicFactor 0 (soil-brown). Only the HTML look changes -
    MuJoCo and the robot cameras use ground.obj + soil.png. Idempotent."""
    g = PROJECT_ROOT / "assets_xgh" / "ground" / "ground.gltf"
    js = json.loads(g.read_text())
    pbr = js["materials"][0].setdefault("pbrMetallicRoughness", {})
    if pbr.get("metallicFactor") != 0.0:
        pbr["metallicFactor"] = 0.0
        g.write_text(json.dumps(js))
        print("   ground.gltf: metallicFactor 0 (soil was drawn as dark metal in the HTML)")
    return pbr


def make_soil_disc(radius: float, rim_rgb, tile_m: float = 1.2, blend_from: float = 0.30, blend_to: float = 0.80):
    """soil_disc: flat soil of the dome radius with the texture AND the material of the field soil mesh
    (assets_xgh/ground/ground.gltf), so the widened soil looks the same as the field.
    grass_ring: a transparent ring 1 cm above it that fades in from `blend_from` x radius and is solid grass from
    `blend_to` x radius to the rim - unlit, in the colour of the photo, so soil and picture meet without an edge."""
    soil = PROJECT_ROOT / "assets_xgh" / "ground" / "soil.png"
    gmat = fix_ground_material()
    V, F, r = _disc(0.0, radius * 1.004, 14)
    # under the grass ring the soil sinks away (0.6 m at the rim): two surfaces 1 cm apart flicker when they are
    # looked at from far away (depth resolution of the viewer)
    V[:, 2] = -0.6 * np.clip((r / radius - blend_from) / max(1e-6, blend_to - blend_from), 0, 1)
    write_gltf(OUT / "soil_disc.gltf", V, F, UV=np.stack([V[:, 0] / tile_m, -V[:, 1] / tile_m], axis=1),
               image=soil.read_bytes(), mime="image/png",
               material=dict(doubleSided=False, pbrMetallicRoughness=dict(
                   baseColorFactor=gmat.get("baseColorFactor", [0.4, 0.4, 0.4, 1.0]),
                   metallicFactor=gmat.get("metallicFactor"), roughnessFactor=gmat.get("roughnessFactor", 0.9))))
    V, F, r = _disc(radius * blend_from, radius * 1.004, 16)
    t = np.clip((r / radius - blend_from) / max(1e-6, blend_to - blend_from), 0, 1)
    alpha = t * t * (3 - 2 * t)
    V[:, 2] = 0.01
    write_gltf(OUT / "grass_ring.gltf", V, F, C=np.concatenate([np.ones((len(V), 3)), alpha[:, None]], axis=1),
               UV=np.stack([V[:, 0] / 2.5, -V[:, 1] / 2.5], axis=1), image=grass_texture(rim_rgb),
               material=dict(doubleSided=False, alphaMode="BLEND", extensions={"KHR_materials_unlit": {}}))


# ----------------------------------------------------------------------------- low-poly meshes (vertex colours)
def icosphere(sub: int):
    t = (1 + 5 ** 0.5) / 2
    V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    V /= np.linalg.norm(V, axis=1)[:, None]
    F = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
         [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    V = V.tolist()
    for _ in range(sub):
        cache, F2 = {}, []

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                m = (np.array(V[a]) + np.array(V[b])) / 2
                V.append((m / np.linalg.norm(m)).tolist())
                cache[k] = len(V) - 1
            return cache[k]
        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            F2 += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        F = F2
    return np.array(V), np.array(F)


class Soup:
    """Triangle soup: every triangle has its own three vertices, so every triangle can have its own colour."""

    def __init__(self):
        self.V, self.C = [], []

    def tris(self, V, F, colour, jitter=0.0, rng=None, shade=None):
        V = np.asarray(V, float)
        for f in np.asarray(F):
            c = np.asarray(colour, float)
            if jitter and rng is not None:
                c = c * (1.0 + rng.uniform(-jitter, jitter))
            if shade is not None:
                c = c * shade(V[f].mean(0))
            for i in f:
                self.V.append(V[i]); self.C.append(np.clip(c, 0, 1))

    def box(self, lo, hi, colour, **kw):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        P = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        F = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4],
             [1, 5, 7], [1, 7, 3]]
        self.tris(P, F, colour, **kw)

    def quad(self, a, b, c, d, colour, **kw):
        self.tris(np.array([a, b, c, d], float), [[0, 1, 2], [0, 2, 3]], colour, **kw)

    def write(self, path: Path):
        V = np.array(self.V)
        write_gltf(path, V, np.arange(len(V)).reshape(-1, 3), C=np.array(self.C) ** 2.2)   # sRGB look -> linear factors
        return dict(triangles=len(V) // 3, size=np.round(V.max(0) - V.min(0), 2).tolist())


def lump(rng, sub, amp, freq=1.6):
    """Unit icosphere pushed in and out by a few smooth random waves."""
    V, F = icosphere(sub)
    d = np.ones(len(V))
    for _ in range(5):
        k = rng.normal(size=3) * freq
        d += amp * 0.45 * np.sin(V @ k + rng.uniform(0, 6.28))
    return V * d[:, None], F


def make_rock() -> dict:
    rng = np.random.default_rng(11)
    V, F = lump(rng, 2, 0.22)
    V = V * np.array([0.55, 0.42, 0.34])                      # ~1.1 m long at scale 1
    V[:, 2] = np.maximum(V[:, 2] + 0.16, -0.02)               # sits in the soil, flat underside
    s = Soup()
    light = np.array([0.35, -0.25, 0.9]) / np.linalg.norm([0.35, -0.25, 0.9])

    def shade(c):                                             # darker towards the ground, a little moss on top
        return 0.72 + 0.5 * np.clip(c[2] / 0.5, 0, 1)
    for f in F:
        P = V[f]
        nrm = np.cross(P[1] - P[0], P[2] - P[0]); nrm = nrm / (np.linalg.norm(nrm) + 1e-9)
        base = np.array([0.52, 0.50, 0.47]) * (0.92 + 0.16 * rng.random())
        if nrm[2] > 0.75 and rng.random() < 0.45:
            base = 0.6 * base + 0.4 * np.array([0.36, 0.44, 0.24])        # moss
        s.tris(P, [[0, 1, 2]], base * (0.8 + 0.25 * max(0.0, float(nrm @ light))), shade=shade)
    return s.write(OUT / "rock.gltf")


def make_tree() -> dict:
    rng = np.random.default_rng(23)
    s = Soup()
    # trunk: tapered, slightly bent 8-gon, bark colours per face
    n, hs = 8, np.linspace(0.0, 2.9, 6)
    rings = []
    for h in hs:
        r = 0.21 - 0.035 * h + (0.09 if h == 0 else 0.0)
        cx, cy = 0.05 * np.sin(h * 0.9), 0.04 * np.sin(h * 1.3 + 1.0)
        rings.append([[cx + r * np.cos(2 * np.pi * k / n), cy + r * np.sin(2 * np.pi * k / n), h] for k in range(n)])
    for i in range(len(rings) - 1):
        for k in range(n):
            a, b, c, d = rings[i][k], rings[i][(k + 1) % n], rings[i + 1][(k + 1) % n], rings[i + 1][k]
            s.quad(a, b, c, d, np.array([0.36, 0.26, 0.17]) * (0.8 + 0.4 * rng.random()))
    # two short branches
    for az, h0, ln in ((0.6, 2.0, 1.2), (3.6, 2.3, 1.0)):
        d = np.array([np.cos(az) * 0.75, np.sin(az) * 0.75, 0.66])
        p0, p1 = np.array([0.0, 0.0, h0]), np.array([0.0, 0.0, h0]) + d * ln
        side = np.cross(d, [0, 0, 1]); side /= np.linalg.norm(side); upv = np.cross(side, d)
        for k in range(5):
            a0, a1 = 2 * np.pi * k / 5, 2 * np.pi * (k + 1) / 5
            o0, o1 = side * np.cos(a0) + upv * np.sin(a0), side * np.cos(a1) + upv * np.sin(a1)
            s.quad(p0 + 0.07 * o0, p0 + 0.07 * o1, p1 + 0.035 * o1, p1 + 0.035 * o0, np.array([0.34, 0.25, 0.16]) * (0.85 + 0.3 * rng.random()))
    # canopy: overlapping lumps, lighter on top, darker underneath
    greens = [np.array([0.17, 0.30, 0.12]), np.array([0.22, 0.36, 0.15]), np.array([0.14, 0.25, 0.11]), np.array([0.27, 0.40, 0.17])]
    blobs = [((0.0, 0.0, 4.3), 1.55), ((1.0, 0.3, 3.7), 1.15), ((-0.9, 0.5, 3.8), 1.2), ((0.2, -1.0, 3.6), 1.1),
             ((-0.3, 0.9, 4.9), 0.95), ((0.6, -0.2, 5.2), 0.9), ((-0.8, -0.6, 4.6), 1.0)]
    for j, (c, r) in enumerate(blobs):
        V, F = lump(rng, 1, 0.28, 1.2)
        V = V * r * np.array([1.0, 1.0, 0.8]) + np.array(c)
        g = greens[j % len(greens)]
        s.tris(V, F, g, jitter=0.16, rng=rng, shade=lambda q: 0.62 + 0.55 * np.clip((q[2] - 2.6) / 3.2, 0, 1))
    return s.write(OUT / "tree.gltf")


def make_hut() -> dict:
    """Field hut on stilts (gubuk): timber posts, plank floor, woven half walls, thatched gable roof, ladder."""
    rng = np.random.default_rng(31)
    s = Soup()
    L, W, zf, zw, ze, zr = 3.0, 2.4, 0.75, 1.55, 2.35, 3.35       # length x, width y, floor / wall top / eave / ridge
    wood_dark, wood, bamboo = np.array([0.30, 0.21, 0.13]), np.array([0.47, 0.33, 0.20]), np.array([0.74, 0.62, 0.36])
    thatch = [np.array([0.66, 0.54, 0.30]), np.array([0.58, 0.46, 0.25]), np.array([0.72, 0.60, 0.36]), np.array([0.50, 0.40, 0.22])]
    # posts
    for x in (-L / 2 + 0.1, L / 2 - 0.1):
        for y in (-W / 2 + 0.1, W / 2 - 0.1):
            s.box([x - 0.07, y - 0.07, 0.0], [x + 0.07, y + 0.07, ze], wood_dark, jitter=0.12, rng=rng)
    # floor beams and planks (every plank its own brown)
    for y in (-W / 2 + 0.1, W / 2 - 0.1):
        s.box([-L / 2, y - 0.05, zf - 0.16], [L / 2, y + 0.05, zf - 0.04], wood_dark, jitter=0.1, rng=rng)
    npl = 12
    for i in range(npl):
        x0 = -L / 2 + i * L / npl
        s.box([x0 + 0.008, -W / 2, zf - 0.04], [x0 + L / npl - 0.008, W / 2, zf], wood * (0.8 + 0.4 * rng.random()))
    # woven half walls on three sides (front, x = +L/2 side towards +y, stays open): strips in two tones
    def wall(p0, p1, z0, z1, nstrip=10, nrow=4):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        for r in range(nrow):
            for k in range(nstrip):
                a, b = p0 + (p1 - p0) * k / nstrip, p0 + (p1 - p0) * (k + 1) / nstrip
                za, zb = z0 + (z1 - z0) * r / nrow, z0 + (z1 - z0) * (r + 1) / nrow
                c = bamboo * (1.0 if (r + k) % 2 == 0 else 0.78) * (0.92 + 0.16 * rng.random())
                s.quad([a[0], a[1], za], [b[0], b[1], za], [b[0], b[1], zb], [a[0], a[1], zb], c)
    wall([-L / 2, -W / 2], [L / 2, -W / 2], zf, zw, 12)          # back
    wall([-L / 2, -W / 2], [-L / 2, W / 2], zf, zw, 9)           # left
    wall([L / 2, -W / 2], [L / 2, W / 2], zf, zw, 9)             # right
    wall([-L / 2, W / 2], [-L / 2 + 0.9, W / 2], zf, zw, 4)      # front, beside the entrance
    for (a, b) in (([-L / 2, -W / 2], [L / 2, -W / 2]), ([-L / 2, -W / 2], [-L / 2, W / 2]), ([L / 2, -W / 2], [L / 2, W / 2])):
        lo = [min(a[0], b[0]) - 0.03, min(a[1], b[1]) - 0.03, zw - 0.02]
        hi = [max(a[0], b[0]) + 0.03, max(a[1], b[1]) + 0.03, zw + 0.04]
        s.box(lo, hi, wood_dark, jitter=0.1, rng=rng)            # top rail
    # roof: gable along x, overhanging, thatch in layered strips of different straw tones
    ov, nrow, ncol = 0.45, 6, 9
    for sgn in (+1, -1):
        for r in range(nrow):
            t0, t1 = r / nrow, (r + 1) / nrow
            y0, y1 = sgn * (W / 2 + ov) * (1 - t0), sgn * (W / 2 + ov) * (1 - t1)
            z0 = ze - 0.25 + (zr - ze + 0.25) * t0 - 0.03
            z1 = ze - 0.25 + (zr - ze + 0.25) * t1 + 0.03
            for k in range(ncol):
                x0, x1 = -L / 2 - ov + k * (L + 2 * ov) / ncol, -L / 2 - ov + (k + 1) * (L + 2 * ov) / ncol
                c = thatch[int(rng.integers(len(thatch)))] * (0.9 + 0.2 * rng.random())
                s.quad([x0, y0, z0], [x1, y0, z0], [x1, y1, z1], [x0, y1, z1], c)
    s.box([-L / 2 - ov, -0.06, zr - 0.02], [L / 2 + ov, 0.06, zr + 0.07], thatch[3] * 0.85)      # ridge
    for x in (-L / 2 + 0.02, L / 2 - 0.02):                      # gable triangles (dark planks)
        s.tris(np.array([[x, -W / 2, ze], [x, W / 2, ze], [x, 0.0, zr - 0.05]]), [[0, 1, 2]], wood * 0.8)
    # ladder at the entrance
    for x in (0.2, 0.75):
        s.box([x - 0.03, W / 2 + 0.02, 0.0], [x + 0.03, W / 2 + 0.42, 0.06], wood_dark)
        s.quad([x - 0.03, W / 2 + 0.40, 0.0], [x + 0.03, W / 2 + 0.40, 0.0], [x + 0.03, W / 2 + 0.02, zf], [x - 0.03, W / 2 + 0.02, zf], wood_dark * 1.1)
    for i in range(3):
        t = (i + 1) / 4
        y, z = W / 2 + 0.40 - 0.38 * t, zf * t
        s.box([0.2, y - 0.03, z - 0.02], [0.75, y + 0.03, z + 0.02], wood * (0.85 + 0.3 * rng.random()))
    return s.write(OUT / "hut.gltf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panorama", default=None, help="360 degree photo (2 : 1), .hdr / .exr / .jpg / .png")
    ap.add_argument("--radius", type=float, default=40.0, help="dome and soil radius (m); the HTML viewer draws up to 100 m")
    ap.add_argument("--exposure", type=float, default=1.0, help="> 1 brighter, < 1 darker")
    ap.add_argument("--width", type=int, default=4096, help="picture width stored in the HTML (4096 ~ 3 MB)")
    ap.add_argument("--quality", type=int, default=88)
    ap.add_argument("--only-meshes", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    info_p = OUT / "scenery.json"
    info = json.loads(info_p.read_text()) if info_p.exists() else {}
    if not a.only_meshes:
        if not a.panorama:
            raise SystemExit("--panorama <file> is needed (or --only-meshes)")
        info.update(make_dome(Path(a.panorama), a.radius, a.exposure, a.width, a.quality))
        make_soil_disc(a.radius, info["rim_rgb"])
    info["rock"], info["tree"], info["hut"] = make_rock(), make_tree(), make_hut()
    info_p.write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
