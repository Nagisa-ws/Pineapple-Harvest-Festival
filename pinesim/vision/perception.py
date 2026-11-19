"""
pinesim/vision/perception.py
----------------------------
Camera-only scene understanding used by the Stargaze planner (pretraining rig and Missions 1-3).

Nothing here reads the simulator state: input = RGB-D frames + calibrated camera pose.

  1. colour classification (HSV) -> fruit pixels (yellow), leaf pixels (green leaves, crown, stalk AND orange leaves
     / orange leaf tips; version 5)
  2. back-projection with the calibrated intrinsics/extrinsics -> world point clouds (several views are fused)
  3. fruit points -> voxel down-sampling -> single-link clustering (one cluster per visible fruit)
  4. per cluster: circle fit of the visible surface in height bands -> fruit AXIS (the plants stand 0-15 deg off
     vertical; the fitted tilt is bounded and shrunk toward vertical when the bands are short), radius bounded to the
     configured range, top along the axis (crown base), centre = top - half height, confidence
  5. leaf points -> voxel grid (the obstacle "constellation" of Stargaze)
  6. per view, for the target fruit: the SHOULDER (fruit pixels between the fruit centre and the crown base; parts
     of the fruit covered by leaves are leaf pixels, so they never count) and the WINDOW (its image bounding box)

All long loops call `cancel()` (if given) so the pretraining rig can abort the computation as soon as the user
moves the camera.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


class Cancelled(Exception):
    pass


def _check(cancel):
    if cancel is not None and cancel():
        raise Cancelled()


@dataclass
class FruitEstimate:
    center: np.ndarray            # world (m)
    radius: float
    top_z: float
    bottom_z: float
    n_points: int
    arc_deg: float                # horizontal angular coverage of the visible surface around the centre
    confidence: float
    points: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 3)))
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    half_height: float = 0.09

    @property
    def top(self) -> np.ndarray:              # crown base (top of the fruit body) along the axis
        return self.center + self.axis * self.half_height

    @property
    def neck_center(self) -> np.ndarray:      # nominal crown neck just above the visible top
        return self.top + self.axis * 0.025

    @property
    def tilt_deg(self) -> float:
        return float(np.degrees(np.arccos(np.clip(self.axis[2], -1, 1))))


@dataclass
class PerceptionResult:
    fruits: list
    leaf_points: np.ndarray       # (M, 3) voxel centres
    n_views: int
    fruit_pixels: int
    leaf_pixels: int
    view_positions: list


def voxel_down(points: np.ndarray, voxel: float) -> np.ndarray:
    """Voxel-grid down-sampling (mean of the points of each occupied voxel). [v6] the voxel index is packed into one
    integer per point, so a 1-D unique replaces the row-wise unique (lexicographic sort of 3 columns): ~5x faster."""
    if len(points) == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    keys -= keys.min(axis=0)
    span = keys.max(axis=0) + 1
    flat = keys[:, 0] + span[0] * (keys[:, 1] + span[1] * keys[:, 2])
    _, inv, counts = np.unique(flat, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    out = np.empty((counts.size, 3))
    for k in range(3):
        out[:, k] = np.bincount(inv, weights=points[:, k], minlength=counts.size)
    return out / counts[:, None]


def cluster_points(points: np.ndarray, radius: float, min_points: int, cancel=None) -> list:
    """Single-link clustering (connected components of the radius graph). [v6] components by scipy's C routine
    (the Python union-find loop over all point pairs was the largest part of the robot's computing time)."""
    if len(points) == 0:
        return []
    tree = cKDTree(points)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    _check(cancel)
    n = len(points)
    if len(pairs) == 0:
        return []
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    g = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    counts = np.bincount(lab)
    out = []
    for r in np.where(counts >= min_points)[0]:
        out.append(points[lab == r])
    return out


def _circle_center(xy, r_range, c0, r0):
    def resid(p):
        return np.linalg.norm(xy - p[:2], axis=1) - p[2]
    try:
        sol = least_squares(resid, x0=[c0[0], c0[1], r0], bounds=([-np.inf, -np.inf, r_range[0]], [np.inf, np.inf, r_range[1]]),
                            loss="soft_l1", f_scale=0.005, max_nfev=60)
        return sol.x[:2], float(sol.x[2])
    except Exception:
        return np.asarray(c0, float), float(r0)


def fit_fruit(pts: np.ndarray, view_positions: list, r_range, height_ratio: float = 1.3, height_range=(0.14, 0.22),
              max_tilt_deg: float = 20.0) -> tuple:
    """Centre, radius, arc coverage (deg), axis, half height of a fruit from its visible surface points.

    The axis is fitted from circle centres of 3 height bands (radius fixed to the overall fit); its tilt is bounded to
    max_tilt_deg and shrunk toward vertical in proportion to the evidence (short visible extent / few points)."""
    z_lo, z_hi = np.percentile(pts[:, 2], [3, 97])
    band = (pts[:, 2] > z_lo + 0.3 * (z_hi - z_lo)) & (pts[:, 2] < z_hi - 0.3 * (z_hi - z_lo))
    xy = pts[band, :2] if band.sum() >= 12 else pts[:, :2]
    cam_xy = np.mean([np.asarray(p)[:2] for p in view_positions], axis=0)
    m = xy.mean(axis=0)
    away = m - cam_xy
    away /= np.linalg.norm(away) + 1e-9
    r0 = float(np.mean(r_range))
    c, r = _circle_center(xy, r_range, m + away * r0 * 0.6, r0)
    # axis: cylinder least squares on the middle 80 % of the visible height (all views fused): centre line through
    # (cx, cy, zc_ref) with direction (tx, ty, 1), radius bounded to the cultivar range; tilt bounded
    zc_ref = 0.5 * (z_lo + z_hi)
    axis = np.array([0.0, 0.0, 1.0])
    ext = z_hi - z_lo
    mid = (pts[:, 2] > z_lo + 0.1 * ext) & (pts[:, 2] < z_hi - 0.1 * ext)
    P = pts[mid] if mid.sum() >= 40 else pts
    if max_tilt_deg > 0 and len(P) >= 40 and ext > 0.05:
        tmax = float(np.tan(np.radians(max_tilt_deg)))

        w_prior = np.sqrt(len(P)) * 0.002 / np.tan(np.radians(10.0))   # weak evidence -> stays near vertical

        def cres(p):
            a = np.array([p[2], p[3], 1.0])
            a /= np.linalg.norm(a)
            d = P - np.array([p[0], p[1], zc_ref])
            pr = d @ a
            return np.r_[np.linalg.norm(d - pr[:, None] * a[None, :], axis=1) - p[4], w_prior * p[2], w_prior * p[3]]
        try:
            sol = least_squares(cres, x0=[c[0], c[1], 0.0, 0.0, r],
                                bounds=([-np.inf, -np.inf, -tmax, -tmax, r_range[0]], [np.inf, np.inf, tmax, tmax, r_range[1]]),
                                loss="soft_l1", f_scale=0.004, max_nfev=80)
            cx, cy, tx, ty, rr = sol.x
            # weak evidence (narrow arc) -> keep part of the vertical prior
            t = np.array([tx, ty])
            if np.linalg.norm(t) > tmax:                  # bound the combined tilt
                t *= tmax / np.linalg.norm(t)
            a = np.array([t[0], t[1], 1.0])
            axis = a / np.linalg.norm(a)
            c, r = np.array([cx, cy]), float(rr)
        except Exception:
            pass
    ang = np.degrees(np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0]))
    a = np.sort(ang)
    gaps = np.diff(np.r_[a, a[0] + 360.0])
    arc = float(360.0 - gaps.max()) if len(a) > 2 else 0.0
    # top along the axis (crown is excluded by colour, the top is well visible from above); the bottom is hidden
    base = np.array([c[0], c[1], zc_ref])
    h = (pts - base[None, :]) @ axis
    h_lo, h_hi = np.percentile(h, [3, 97])
    H = float(np.clip(max(h_hi - h_lo, height_ratio * 2.0 * r), *height_range))
    center = base + axis * (h_hi - 0.5 * H)
    return center, float(r), arc, axis, 0.5 * H


def fit_vertical_fruit(pts, view_positions, r_range, height_ratio=1.3, height_range=(0.14, 0.22)):
    """Backward-compatible wrapper: (centre, radius, arc, top z, bottom z)."""
    c, r, arc, axis, hh = fit_fruit(pts, view_positions, r_range, height_ratio, height_range, max_tilt_deg=0.0)
    return c, r, arc, float(c[2] + hh), float(c[2] - hh)


class Perceiver:
    """Accumulates views, then estimates fruits and the leaf cloud."""

    def __init__(self, cfg: dict, self_filter=None):
        p = cfg["perception"]
        self.p = p
        self.self_filter = self_filter          # callable(points) -> bool mask of points on the robot itself
        self.reset()

    def reset(self):
        self._fruit_pts, self._leaf_pts, self._views = [], [], []
        self.views = []                    # per view: pose, intrinsics, fruit pixels + world points, leaf mask
        self._npx_f = self._npx_l = 0

    def masks(self, rgb: np.ndarray):
        p = self.p
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        fruit = cv2.inRange(hsv, tuple(p["fruit_hsv_lo"]), tuple(p["fruit_hsv_hi"])) > 0
        leaf = cv2.inRange(hsv, tuple(p["leaf_hsv_lo"]), tuple(p["leaf_hsv_hi"])) > 0
        if "leaf2_hsv_lo" in p:                                    # orange leaves / leaf tips
            leaf |= cv2.inRange(hsv, tuple(p["leaf2_hsv_lo"]), tuple(p["leaf2_hsv_hi"])) > 0
        k = np.ones((3, 3), np.uint8)
        fruit = cv2.morphologyEx(fruit.astype(np.uint8), cv2.MORPH_OPEN, k) > 0
        # isolated noise pixels on the soil: keep only leaf components of a few pixels or more
        n, lab, st, _ = cv2.connectedComponentsWithStats(leaf.astype(np.uint8), connectivity=8)
        keep = np.zeros(n, bool)
        keep[1:] = st[1:, cv2.CC_STAT_AREA] >= int(p.get("min_leaf_component_px", 8))
        leaf = keep[lab]
        return fruit, leaf & ~fruit

    def add_view(self, frame: dict, cancel=None):
        depth, R, t, intr = frame["depth"], frame["R"], frame["t"], frame["intr"]
        fruit, leaf = self.masks(frame["rgb"])
        _check(cancel)
        valid = (depth > 0.02) & (depth < float(self.p["max_range"]))
        rec = dict(R=np.asarray(R, float), t=np.asarray(t, float), intr=intr, leaf_mask=leaf & valid,
                   us=np.zeros(0, int), vs=np.zeros(0, int), X=np.zeros((0, 3)))
        for name, mask, store in (("fruit", fruit & valid, self._fruit_pts), ("leaf", leaf & valid, self._leaf_pts)):
            vs, us = np.nonzero(mask)
            if len(vs) == 0:
                continue
            z = depth[vs, us]
            x = (us + 0.5 - intr.cx) / intr.fx * z
            y = (vs + 0.5 - intr.cy) / intr.fy * z
            Xw = np.stack([x, y, z], axis=1) @ R.T + t
            if self.self_filter is not None:                 # the robot's own (green) vehicle bed is not a leaf
                keep = ~self.self_filter(Xw)
                Xw, us, vs = Xw[keep], us[keep], vs[keep]
                if name == "leaf":
                    m2 = np.zeros_like(mask)
                    m2[vs, us] = True
                    rec["leaf_mask"] = m2
            store.append(Xw)
            if name == "fruit":
                rec["us"], rec["vs"], rec["X"] = us, vs, Xw
        self.views.append(rec)
        self._npx_f += int((fruit & valid).sum())
        self._npx_l += int((leaf & valid).sum())
        self._views.append(np.asarray(t, float).copy())
        _check(cancel)

    def result(self, cancel=None) -> PerceptionResult:
        p = self.p
        fp = np.vstack(self._fruit_pts) if self._fruit_pts else np.zeros((0, 3))
        lp = np.vstack(self._leaf_pts) if self._leaf_pts else np.zeros((0, 3))
        fp = fp[fp[:, 2] > 0.02]
        lp = lp[lp[:, 2] > float(p.get("leaf_min_height", 0.04))]   # soil noise pixels sit at z ~ 0
        fp = voxel_down(fp, 0.006)
        _check(cancel)
        clusters = cluster_points(fp, float(p["cluster_radius"]), int(p["min_cluster_points"]), cancel)
        fruits = []
        for c in clusters:
            _check(cancel)
            center, r, arc, axis, hh = fit_fruit(c, self._views, p["fruit_radius_range"],
                                                 float(p.get("height_ratio", 1.3)),
                                                 tuple(p.get("fruit_height_range", (0.14, 0.22))),
                                                 float(p.get("max_tilt_deg", 20.0)))
            conf = float(np.clip(len(c) / 300.0, 0, 1) * 0.6 + np.clip(arc / 180.0, 0, 1) * 0.4)
            top, bot = center + axis * hh, center - axis * hh
            zr = p.get("fruit_center_z_range")
            if zr is not None and not (float(zr[0]) <= center[2] <= float(zr[1])):
                continue                               # not at fruit height: leaf tips, soil, crate
            fruits.append(FruitEstimate(center, r, float(top[2]), float(bot[2]), len(c), arc, conf, c, axis, hh))
        leaf = voxel_down(lp, float(p["voxel"]))
        _check(cancel)
        return PerceptionResult(fruits, leaf, len(self._views), self._npx_f, self._npx_l, list(self._views))


def merge_fruits(fruits: list, dist: float = 0.08) -> list:
    """Merge estimates of the same fruit (e.g. from separate scans), keeping the most confident."""
    out = []
    for f in sorted(fruits, key=lambda e: -e.confidence):
        if all(np.linalg.norm(f.center[:2] - g.center[:2]) > dist for g in out):
            out.append(f)
    return out


def shoulder_in_view(view: dict, fruit: FruitEstimate, margin: float = 0.02, crown_gap: float = 0.004) -> dict:
    """SHOULDER of `fruit` in one camera view: fruit pixels whose 3D point lies on this fruit between its centre and
    the crown base (axial coordinate 0 .. half height). Leaf-covered parts of the fruit are leaf pixels and never
    counted. Returns pixel coordinates, points, the WINDOW (image bounding box of the shoulder), and the shoulder
    area ratio (shoulder pixels / all visible pixels of this fruit)."""
    X, us, vs = view["X"], view["us"], view["vs"]
    out = dict(n_visible=0, n_shoulder=0, ratio=0.0, window=None, us=np.zeros(0, int), vs=np.zeros(0, int),
               X=np.zeros((0, 3)), view=view)
    if len(X) == 0:
        return out
    rel = X - fruit.center[None, :]
    h = rel @ fruit.axis
    radial = np.linalg.norm(rel - h[:, None] * fruit.axis[None, :], axis=1)
    on = (radial < fruit.radius + margin) & (h > -fruit.half_height - margin) & (h < fruit.half_height + margin)
    sh = on & (h >= 0.0) & (h <= fruit.half_height - crown_gap)
    out["n_visible"], out["n_shoulder"] = int(on.sum()), int(sh.sum())
    out["ratio"] = float(sh.sum() / max(on.sum(), 1))
    if sh.sum() == 0:
        return out
    u, v = us[sh], vs[sh]
    out["us"], out["vs"], out["X"] = u, v, X[sh]
    out["h"] = h[sh]
    if len(u) >= 8:
        u0, u1 = np.percentile(u, [2, 98])
        v0, v1 = np.percentile(v, [2, 98])
    else:
        u0, u1, v0, v1 = u.min(), u.max(), v.min(), v.max()
    out["window"] = (float(u0), float(v0), float(u1) + 1.0, float(v1) + 1.0)
    # visible-fruit bounding box (image) for the overlay
    ua, va = us[on], vs[on]
    out["bbox"] = (float(ua.min()), float(va.min()), float(ua.max()) + 1.0, float(va.max()) + 1.0)
    return out
