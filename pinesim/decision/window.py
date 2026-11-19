"""
pinesim/decision/window.py
--------------------------
From one camera view of the target fruit to ranked approach lines (T1, T2) - the user's version 5 concept:

  shoulder   fruit pixels between the fruit centre and the crown base (perception.shoulder_in_view; leaf-covered
             fruit parts are leaf pixels and never count)
  window     image bounding box of the shoulder
  T2         SEVERAL candidates inside the window: the window is split into a grid; every cell that holds shoulder
             pixels gives one candidate (median 3D point of those pixels = a point on the fruit surface)
  T1         for each T2: the start of a straight line toward the fruit axis (radial in the horizontal plane, several
             elevations); only lines whose T1 projects INSIDE the window (+ margin) of this view are allowed - such
             a line runs along camera rays that reached the shoulder, i.e. through space the camera saw free
  line       the safest of those lines = largest minimum distance to the leaf points seen by the camera (Stargaze)
  decision   which T2 is best: the diffusion policy (pinesim/decision/diffusion.py) trained in pretraining on the TRUE
             leaf geometry; fallback = heuristic (camera clearance only)

Features (input of the diffusion policy), per grid cell (G x G):
  0 shoulder occupancy        1 camera clearance of T2 itself (leaf points, /0.15 m)
  2 camera line clearance     3 height of the cell's T2 on the shoulder (0 = centre, 1 = crown base)
  4 leaf-pixel occupancy      5 valid candidate (1/0)
plus globals: window width / height (px / 100), shoulder ratio, fruit radius (/0.1), view elevation (rad).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pinesim.vision.perception import FruitEstimate, shoulder_in_view

N_FEAT = 6
N_GLOBAL = 5


@dataclass
class Candidate:
    cell: tuple                 # (row, col) in the window grid
    uv: np.ndarray              # normalised cell centre in [-1, 1]^2 (u right, v down)
    t2: np.ndarray              # world: on the fruit surface (shoulder)
    t1: np.ndarray              # world: start of the line (T1 projects inside the window)
    approach: np.ndarray        # unit vector T1 -> T2 (tool z while approaching)
    elevation_deg: float        # elevation of the outward direction T2 -> T1
    clear_line: float           # camera leaf clearance of the segment T1-T2 (m)
    clear_point: float          # camera leaf clearance of T2 itself (m)
    score: float
    votes: float = 0.0
    t1_uv: tuple = (0.0, 0.0)


@dataclass
class WindowDecision:
    ok: bool
    reason: str = ""
    view_index: int = -1
    shoulder: dict = field(default_factory=dict)
    window: tuple | None = None
    bbox: tuple | None = None
    candidates: list = field(default_factory=list)       # ranked (best first)
    features: np.ndarray | None = None
    globals: np.ndarray | None = None
    method: str = ""
    window_corners: np.ndarray | None = None             # 3D corners of the window at the shoulder depth (overlay)
    bbox_corners: np.ndarray | None = None               # 3D corners of the visible-fruit bounding box (overlay)


def _project(view, X):
    R, t, intr = view["R"], view["t"], view["intr"]
    Xc = (np.atleast_2d(X) - t[None, :]) @ R
    z = np.maximum(Xc[:, 2], 1e-6)
    return intr.fx * Xc[:, 0] / z + intr.cx, intr.fy * Xc[:, 1] / z + intr.cy, Xc[:, 2]


def _backproject(view, u, v, z):
    R, t, intr = view["R"], view["t"], view["intr"]
    x = (np.asarray(u) - intr.cx) / intr.fx * z
    y = (np.asarray(v) - intr.cy) / intr.fy * z
    return np.stack([x, y, np.broadcast_to(z, np.shape(x))], -1) @ R.T + t


def segment_clearance(pts, rad, A, B):
    """min_k |p_k - segment(A_j, B_j)| - r_k for every segment j: (n_seg,)"""
    if len(pts) == 0:
        return np.full(len(A), 1.0)
    AB = B - A
    L2 = np.maximum(np.einsum("jk,jk->j", AB, AB), 1e-12)
    AP = pts[:, None, :] - A[None, :, :]
    t = np.clip(np.einsum("ijk,jk->ij", AP, AB) / L2[None, :], 0.0, 1.0)
    d = np.linalg.norm(AP - t[..., None] * AB[None, :, :], axis=2) - rad[:, None]
    return d.min(axis=0)


def outward_dirs(t2, fruit: FruitEstimate, elevations_deg):
    """Outward unit vectors from T2 (horizontal radial direction of the fruit axis at T2, lifted by each elevation)."""
    rel = t2 - fruit.center
    h = float(rel @ fruit.axis)
    rad = rel - h * fruit.axis
    rad[2] = 0.0
    n = np.linalg.norm(rad)
    rad = rad / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
    e = np.radians(np.asarray(elevations_deg, float))
    return np.cos(e)[:, None] * rad[None, :] + np.sin(e)[:, None] * np.array([0.0, 0.0, 1.0])[None, :]


def build_candidates(fruit: FruitEstimate, view: dict, leaf_pts: np.ndarray, leaf_rad: np.ndarray, cfg: dict,
                     shoulder: dict | None = None):
    """Candidates + feature grid for one view. Returns (candidates (unranked), features, globals, shoulder)."""
    sg = cfg["stargaze"]
    G = int(sg.get("grid", 8))
    L = float(sg.get("line_length", 0.20))
    elev = list(sg.get("elevations_deg", [-10, 0, 10, 20, 30, 40, 50, 60]))
    marg = float(sg.get("window_margin_px", 6))
    min_cell = int(sg.get("min_cell_pixels", 3))
    w_pt = float(sg.get("t2_point_weight", 0.5))
    sh = shoulder if shoulder is not None else shoulder_in_view(view, fruit)
    feats = np.zeros((N_FEAT, G, G), np.float32)
    glob = np.zeros(N_GLOBAL, np.float32)
    if sh.get("window") is None or sh["n_shoulder"] < int(sg.get("min_shoulder_pixels", 20)):
        return [], feats, glob, sh
    u0, v0, u1, v1 = sh["window"]
    w, h = max(u1 - u0, 1.0), max(v1 - v0, 1.0)
    cu = np.clip(((sh["us"] + 0.5 - u0) / w * G).astype(int), 0, G - 1)
    cv = np.clip(((sh["vs"] + 0.5 - v0) / h * G).astype(int), 0, G - 1)
    lm = view["leaf_mask"]
    H_img, W_img = lm.shape
    cands = []
    cand_cells, t2s = [], []
    for r in range(G):
        for c in range(G):
            sel = (cv == r) & (cu == c)
            n = int(sel.sum())
            # pixels of this cell
            ua, ub = int(np.floor(u0 + c * w / G)), int(np.ceil(u0 + (c + 1) * w / G))
            va, vb = int(np.floor(v0 + r * h / G)), int(np.ceil(v0 + (r + 1) * h / G))
            area = max((ub - ua) * (vb - va), 1)
            feats[0, r, c] = min(1.0, n / area)
            if vb > va and ub > ua:
                feats[4, r, c] = float(lm[max(va, 0):min(vb, H_img), max(ua, 0):min(ub, W_img)].mean()) if \
                    (min(vb, H_img) > max(va, 0) and min(ub, W_img) > max(ua, 0)) else 0.0
            if n < min_cell:
                continue
            X = sh["X"][sel]
            t2 = np.median(X, axis=0)
            hh = float((t2 - fruit.center) @ fruit.axis)
            feats[3, r, c] = float(np.clip(hh / max(fruit.half_height, 1e-3), 0, 1))
            cand_cells.append((r, c))
            t2s.append(t2)
    if not t2s:
        return [], feats, glob, sh
    t2s = np.array(t2s)
    # T2 point clearance (camera leaf points)
    if len(leaf_pts):
        dp = np.linalg.norm(leaf_pts[None, :, :] - t2s[:, None, :], axis=2) - leaf_rad[None, :]
        clr_pt = dp.min(axis=1)
    else:
        clr_pt = np.full(len(t2s), 0.15)
    for k, ((r, c), t2) in enumerate(zip(cand_cells, t2s)):
        outs = outward_dirs(t2, fruit, elev)
        T1 = t2[None, :] + L * outs
        uu, vv, zz = _project(view, T1)
        inside = (uu >= u0 - marg) & (uu <= u1 + marg) & (vv >= v0 - marg) & (vv <= v1 + marg) & (zz > 0.05)
        if not inside.any():
            continue
        idx = np.where(inside)[0]
        clr = segment_clearance(leaf_pts, leaf_rad, T1[idx], np.repeat(t2[None, :], len(idx), 0))
        j = int(np.argmax(clr))
        e_i = idx[j]
        uv = np.array([(c + 0.5) / G * 2 - 1, (r + 0.5) / G * 2 - 1])
        cl, cp = float(clr[j]), float(clr_pt[k])
        feats[1, r, c] = float(np.clip(cp / 0.15, -1, 1))
        feats[2, r, c] = float(np.clip(cl / 0.15, -1, 1))
        feats[5, r, c] = 1.0
        cands.append(Candidate(cell=(r, c), uv=uv, t2=t2, t1=T1[e_i], approach=-outs[e_i],
                               elevation_deg=float(elev[e_i]), clear_line=cl, clear_point=cp,
                               score=cl + w_pt * cp, t1_uv=(float(uu[e_i]), float(vv[e_i]))))
    R = view["R"]
    fwd = R[:, 2]
    glob[:] = [w / 100.0, h / 100.0, sh["ratio"], fruit.radius / 0.1, float(np.arcsin(np.clip(-fwd[2], -1, 1)))]
    return cands, feats, glob, sh


def window_corners_3d(view, sh):
    """3D corners of the window at the median shoulder depth (overlay: looks like the 2D box from that camera)."""
    if sh.get("window") is None or len(sh["X"]) == 0:
        return None
    _, _, z = _project(view, sh["X"])
    zm = float(np.median(z))
    u0, v0, u1, v1 = sh["window"]
    return _backproject(view, np.array([u0, u1, u1, u0]), np.array([v0, v0, v1, v1]), zm)


def bbox_corners_3d(view, sh, fruit: FruitEstimate):
    if sh.get("bbox") is None:
        return None
    u0, v0, u1, v1 = sh["bbox"]
    _, _, zc = _project(view, fruit.center[None, :])
    z = float(zc[0]) - fruit.radius
    return _backproject(view, np.array([u0, u1, u1, u0]), np.array([v0, v0, v1, v1]), z)


def decide(fruit: FruitEstimate, perceiver_views: list, leaf_pts: np.ndarray, leaf_rad: np.ndarray, cfg: dict,
           policy=None, rng=None, view_order=None) -> WindowDecision:
    """Pick the decision view (largest shoulder), build candidates, rank them (diffusion policy / heuristic)."""
    if not perceiver_views:
        return WindowDecision(False, "no_view")
    shs = [shoulder_in_view(v, fruit) for v in perceiver_views]
    order = view_order if view_order is not None else sorted(range(len(shs)), key=lambda i: -shs[i]["n_shoulder"])
    method = str(cfg["stargaze"].get("decision", "diffusion"))
    last = WindowDecision(False, "no_shoulder")
    for vi in order:
        sh = shs[vi]
        if sh.get("window") is None:
            continue
        cands, feats, glob, sh = build_candidates(fruit, perceiver_views[vi], leaf_pts, leaf_rad, cfg, sh)
        if not cands:
            last = WindowDecision(False, "no_t1_in_window", vi, sh, sh.get("window"), sh.get("bbox"))
            continue
        used = method
        if method == "diffusion" and policy is not None:
            ranked = policy.rank(feats, glob, cands, rng=rng)
        elif method == "random":
            rng = rng or np.random.default_rng(0)
            ranked = list(np.array(cands, dtype=object)[rng.permutation(len(cands))])
        else:
            used = "heuristic"
            ranked = sorted(cands, key=lambda c: -c.score)
        top_k = int(cfg["stargaze"].get("top_k", 8))
        view = perceiver_views[vi]
        return WindowDecision(True, "ok", vi, sh, sh["window"], sh.get("bbox"), ranked[:top_k], feats, glob, used,
                              window_corners_3d(view, sh), bbox_corners_3d(view, sh, fruit))
    return last


# ----------------------------------------------------------------------------- ground-truth labels (pretraining)
def true_scores(cands: list, gt_pts: np.ndarray, gt_rad: np.ndarray, cfg: dict, fruit_center=None, fruit_axis=None):
    """TRUE clearance of each candidate's best line (T1 restricted to the window as in build_candidates) against the
    simulator's leaf geometry (pretraining labels only)."""
    w_pt = float(cfg["stargaze"].get("t2_point_weight", 0.5))
    out = []
    for c in cands:
        cl = float(segment_clearance(gt_pts, gt_rad, c.t1[None, :], c.t2[None, :])[0])
        cp = float((np.linalg.norm(gt_pts - c.t2[None, :], axis=1) - gt_rad).min()) if len(gt_pts) else 0.15
        out.append(cl + w_pt * cp)
    return np.array(out)
