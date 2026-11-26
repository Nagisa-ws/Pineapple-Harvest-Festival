"""Add vertex normals to glTF files that have none.

Why: a mesh without normals cannot RECEIVE shadows in the Meshcat HTML (its shadow coordinates become NaN), so the
ground, the soil disc and the vehicle bed need normals before the sun of `scenery.shadows` shows any shadow. A mesh
without normals still CASTS shadows, so the other files can stay as they are.

    python scripts/add_gltf_normals.py                 # the default receivers (ground, soil disc, vehicle bed)
    python scripts/add_gltf_normals.py a.gltf b.gltf   # these files
Each file is changed in place (one extra `<name>_nrm<k>.bin` next to it); files that already have normals are skipped.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ["assets_xgh/ground/ground.gltf", "assets_xgh/scenery/soil_disc.gltf", "assets_xgh/vehicle/bed.gltf"]
DTYPE = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def read(g, folder, acc_i):
    acc = g["accessors"][acc_i]
    bv = g["bufferViews"][acc["bufferView"]]
    raw = (folder / g["buffers"][bv["buffer"]]["uri"]).read_bytes()
    dt, nc = np.dtype(DTYPE[acc["componentType"]]), NCOMP[acc["type"]]
    off = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride", 0) or dt.itemsize * nc
    if stride == dt.itemsize * nc:
        return np.frombuffer(raw, dt, acc["count"] * nc, off).reshape(acc["count"], nc)
    return np.stack([np.frombuffer(raw, dt, nc, off + i * stride) for i in range(acc["count"])])


def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])          # length = 2 x area: area weighting
    N = np.zeros_like(V, dtype=np.float64)
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    ln = np.linalg.norm(N, axis=1, keepdims=True)
    N = np.where(ln > 1e-20, N / np.maximum(ln, 1e-20), np.array([0.0, 0.0, 1.0]))
    return N.astype(np.float32)


def add_normals(path: Path) -> int:
    g = json.loads(path.read_text())
    n_new = 0
    for mesh in g.get("meshes", []):
        for prim in mesh["primitives"]:
            if "NORMAL" in prim["attributes"] or prim.get("mode", 4) != 4:
                continue
            V = read(g, path.parent, prim["attributes"]["POSITION"]).astype(np.float64)
            F = (read(g, path.parent, prim["indices"]).astype(np.int64).reshape(-1, 3) if "indices" in prim
                 else np.arange(len(V)).reshape(-1, 3))
            N = vertex_normals(V, F)
            uri = f"{path.stem}_nrm{n_new}.bin"
            (path.parent / uri).write_bytes(N.tobytes())
            g["buffers"].append({"uri": uri, "byteLength": N.nbytes})
            g["bufferViews"].append({"buffer": len(g["buffers"]) - 1, "byteOffset": 0, "byteLength": N.nbytes, "target": 34962})
            g["accessors"].append({"bufferView": len(g["bufferViews"]) - 1, "componentType": 5126, "count": len(N), "type": "VEC3",
                                   "min": N.min(0).tolist(), "max": N.max(0).tolist()})
            prim["attributes"]["NORMAL"] = len(g["accessors"]) - 1
            n_new += 1
    if n_new:
        path.write_text(json.dumps(g))
    return n_new


if __name__ == "__main__":
    files = [Path(a) for a in sys.argv[1:]] or [ROOT / f for f in DEFAULT]
    for f in files:
        if not f.exists():
            print(f"{f}: not found")
            continue
        k = add_normals(f)
        print(f"{f}: {'normals added to %d mesh part(s)' % k if k else 'already has normals'}")
