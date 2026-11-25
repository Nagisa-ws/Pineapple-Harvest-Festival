"""Realistic Mission 3 evaluation (version 5): XGH505H + UR10e + U-jaw gripper harvesting the 15 m field of the
user's mesh pineapples (2 x 30 plants, spacing 0.50 m, zig-zag 0.25 m, tilted 0-15 deg); only THREE plants are
interactive at a time (target + next, then target + previous + next). Training used the light scene; this script
never trains.

  python scripts/eval_xgh.py --onnx models/mission3_v1.onnx                     # full 15 m, recording + HTML
  python scripts/eval_xgh.py --plants-per-row 4 --tag quick                     # first ~2 m only (quick check)
  python scripts/eval_xgh.py --viewer                                           # live MuJoCo viewer (desktop)
  python scripts/eval_xgh.py --recording outputs/eval_xgh/full.npz --html-only  # re-export HTML from a recording

Outputs (outputs/eval_xgh/<tag>/):
  result.json          per-fruit outcomes, failures, crate unloads, LOD statistics, timing
  recording.npz        compact recording (only moved bodies are stored)
  html/part_XX.html    standalone Drake Meshcat pages (open in a browser, press play in "Animations"); camera panel
                       bottom-left: Bebas / Ikuti kendaraan / Kamera pojok / Kamera pergelangan (keys 1-4)
  corner_camera.mp4    corner camera (+ wrist camera inset) during the harvesting phases

Live viewer (--viewer, needs a display): MuJoCo's own viewer; press C to cycle the camera
(bebas -> ikuti kendaraan -> kamera pojok -> kamera pergelangan), or pick it in the viewer's Camera menu.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from pinesim.config import HARVEST_METHODS, PROJECT_ROOT, choose_harvest_method
from pinesim.envs.xgh_env import XghEvalEnv
from pinesim.rl.policy import load_policy
from pinesim.viz.xgh_recorder import CornerVideo, XghRecorder, export_html
from pinesim.timing import fruit_times, summarize

CAMERAS = ["free", "follow", "corner_cam", "wrist_cam"]


class LiveViewer:
    """MuJoCo passive viewer with a camera switch (key C)."""

    def __init__(self, env, every: int = 2):
        import mujoco.viewer
        self.env, self.every, self.k, self.mode = env, every, 0, 0
        self.v = mujoco.viewer.launch_passive(env.sim.m, env.sim.d, key_callback=self._key)

    def _key(self, code):
        if code in (ord("C"), ord("c")):
            self.mode = (self.mode + 1) % len(CAMERAS)
            name = CAMERAS[self.mode]
            with self.v.lock():
                if name == "free":
                    self.v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                else:
                    self.v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    self.v.cam.fixedcamid = self.env.sim.m.camera(name).id
            print("camera:", name)

    def __call__(self, label):
        self.k += 1
        if self.k % self.every == 0 and self.v.is_running():
            self.v.sync()


def export_parts(env_model, rec_path: Path, out_dir: Path, meters_per_part: float, max_frames: int, info: dict,
                 x_window: float = 2.0, scenery=None):
    z = np.load(rec_path, allow_pickle=False)
    vx = z["vehicle_x"]
    n = len(vx)
    parts, start = [], 0
    while start < n:
        end = start
        while end < n and vx[end] - vx[start] < meters_per_part and end - start < max_frames:
            end += 1
        end = max(end, start + 1)
        parts.append((start, end))
        start = end
    if len(parts) > 1:                       # a short tail (the last few steps) goes into the previous part
        a0, b0 = parts[-1]
        if vx[b0 - 1] - vx[a0] < 0.5 and b0 - a0 < 1000:
            parts[-2] = (parts[-2][0], b0)
            parts.pop()
    out = []
    for j, (a, b) in enumerate(parts):
        lines = [f"<b>Misi 3 (XGH505H) - evaluasi realistis</b> bagian {j + 1}/{len(parts)}",
                 f"kendaraan x {vx[a]:.2f} -> {vx[b - 1]:.2f} m; hanya 3 tanaman interaktif sekaligus",
                 f"buah di keranjang (total run): {info.get('n_placed')} / {info.get('n_targets')}",
                 (f"waktu per buah: observasi -> keranjang rata-rata {info['timing']['observe_to_basket_s_mean']:.1f} s, "
                  f"seluruh run / buah {info['timing']['s_per_fruit']:.1f} s (gerak + komputasi)"
                  if info.get("timing") and info["timing"].get("observe_to_basket_s_mean") is not None else "waktu per buah: -"),
                 f"baris tetangga kiri/kanan: mesh diam, jarak tepi ladang {info.get('side_edge_gap_m', '-')} m",
                 f"policy: {info.get('policy', '-')}",
                 "lampu (pojok kiri depan): kuning = observasi, biru = aksi, hijau = sukses/standby, merah = gagal"]
        r = export_html(env_model, rec_path, out_dir / f"part_{j + 1:02d}.html", (a, b), x_window=x_window, info_lines=lines,
                        scenery=scenery)
        out.append(dict(part=j + 1, frame_range=[int(a), int(b)], x=[float(vx[a]), float(vx[b - 1])], **r))
        print(f"  part {j + 1}: frames {a}-{b}, x {vx[a]:.2f}-{vx[b - 1]:.2f} m, {r['size_mb']} MB")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--vecnorm", default=None)
    ap.add_argument("--seed", type=int, default=990000)
    ap.add_argument("--plants-per-row", type=int, default=None, help="shorter field for quick checks (default 30 = 15 m)")
    ap.add_argument("--section", type=int, nargs=2, default=None, metavar=("FIRST", "N"),
                    help="evaluate plants FIRST..FIRST+N-1 (sorted along the row) of the 15 m field in this run "
                         "(the field is the same; the vehicle starts in front of the section)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--every", type=int, default=5, help="record every N control steps (20 ms each)")
    ap.add_argument("--video", action="store_true", help="also write the corner-camera MP4 (slow: CPU rendering)")
    ap.add_argument("--video-every", type=int, default=25, help="corner-camera video frame every N control steps "
                    "(CPU rendering here takes ~1 s per frame)")
    ap.add_argument("--no-html", action="store_true")
    ap.add_argument("--meters-per-part", type=float, default=3.0)
    ap.add_argument("--max-frames-per-part", type=int, default=6000)
    ap.add_argument("--viewer", action="store_true", help="live MuJoCo viewer (needs a display)")
    ap.add_argument("--recording", default=None)
    ap.add_argument("--html-only", action="store_true")
    ap.add_argument("--max-rl-steps", type=int, default=400000)
    ap.add_argument("--out-root", default="outputs/eval_xgh")
    ap.add_argument("--field-seed", type=int, default=None, help="layout seed of the evaluation field (plant kinds, "
                    "yaw, tilt, stem strength; default eval_field.seed = 2026): another seed = another, unseen row")
    ap.add_argument("--x-window", type=float, default=2.0, help="HTML: static plants this far (m) behind / ahead of "
                    "the part's vehicle range are included")
    ap.add_argument("--side-beds", type=int, default=None, help="neighbour beds on EACH side of the harvested bed "
                    "(static meshes; default field.side_rows.beds_per_side)")
    ap.add_argument("--no-scenery", action="store_true", help="HTML without dome / widened soil / rocks / trees / hut")
    ap.add_argument("--edge-gap", type=float, default=None, help="distance (m) between the edge of the harvested bed "
                    "and the neighbour beds (default: field.side_rows.edge_gap in configs/common.yaml; 0 = no neighbour beds)")
    ap.add_argument("--harvest", choices=HARVEST_METHODS, default=None,
                    help="how the fruit is taken off the stalk (asked on the terminal when omitted; default harvest.method)")    
    a = ap.parse_args()
    over = {}
    if not a.html_only:
        harvest = choose_harvest_method(a.harvest, "mission3")
        over["harvest"] = {"method": harvest}
        print(f"harvest method: {harvest}")
    if a.plants_per_row:
        over["eval_field"] = {"plants_per_row": int(a.plants_per_row)}
    if a.section:
        over.setdefault("eval_field", {})["section"] = [int(a.section[0]), int(a.section[1])]
    tag = a.tag or ("full" if not (a.plants_per_row or a.section) else
                    f"section_{a.section[0]:03d}" if a.section else f"rows{a.plants_per_row}")
    if a.field_seed is not None:
        over.setdefault("eval_field", {})["seed"] = int(a.field_seed)
    if a.edge_gap is not None:
        over.setdefault("field", {})["side_rows"] = {"enabled": a.edge_gap > 0, "edge_gap": float(a.edge_gap)}
    if a.side_beds is not None:
        over.setdefault("field", {}).setdefault("side_rows", {}).update(
            {"enabled": a.side_beds > 0, "beds_per_side": max(1, int(a.side_beds))})
    out = PROJECT_ROOT / a.out_root / tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    env = XghEvalEnv("mission3", seed=a.seed, overrides=over or None)
    # decoration of the HTML (dome, widened soil, rocks, trees, hut): Meshcat only, the simulation does not contain it
    scenery = None if a.no_scenery else dict(env.cfg.get("scenery") or {})
    print(f"scene: {env.sim.meta['n_plants']} plants, pool {env.sim.field.n} slots ({env.sim.meta['pool_per_kind']} per kind), "
          f"nv {env.sim.m.nv}, built in {time.time() - t0:.1f} s")
    if a.html_only:
        info = json.loads((out / "result.json").read_text()) if (out / "result.json").exists() else {}
        export_parts(env.sim.m, Path(a.recording or out / "recording.npz"), out / "html", a.meters_per_part,
                     a.max_frames_per_part, info, a.x_window, scenery)
        return
    pol, desc = load_policy(onnx=a.onnx, model=a.model, vecnorm=a.vecnorm)
    rec = XghRecorder(env.sim, every=a.every)
    vid = None if not a.video else CornerVideo(env.sim, out / "corner_camera.mp4", every=a.video_every,
                                               size=(480, 360), fps=4)
    live = LiveViewer(env) if a.viewer else None

    tprog = [time.time(), 0]

    def frame(label):
        rec(label)
        if rec.k % 1000 == 0:
            now = time.time()
            print(f"[{rec.k} steps | sim {rec.k * env.sim.dt:.0f} s | wall {now - t0:.0f} s | "
                  f"{(now - tprog[0]) / max(rec.k - tprog[1], 1) * 1e3:.0f} ms/step] x={env.sim.vehicle_pose()[0][0]:.2f} m, "
                  f"fruits {env.n_placed}, phase '{label}', interactive {len(env.lod.interactive())}", flush=True)
            tprog[0], tprog[1] = now, rec.k
        if vid is not None:
            vid.text = f"buah {env.n_placed}"
            vid(label)
        if live is not None:
            live(label)
    env.frame_callback = frame
    env.overlay_callback = rec.overlay
    t1 = time.time()
    info = env.run_episode(pol, seed=a.seed, max_rl_steps=a.max_rl_steps)
    wall = time.time() - t1
    if vid is not None:
        vid.close()
    tg = info.get("targets", [])
    res = dict(policy=desc, harvest_method=harvest, seed=a.seed, field_seed=int(env.cfg["eval_field"]["seed"]),
               n_plants=env.sim.meta["n_plants"], n_placed=int(info["n_placed"]),
               n_targets=int(info["n_targets"]), failure=info.get("failure"), termination=info.get("termination_reason"),
               stops=info.get("stops"), vehicle_travel_m=info.get("vehicle_travel"), drift_max_m=info.get("vehicle_drift_max"),
               crate_counts=info.get("crate_counts"), unloads=info.get("unloads"), stored_by_operator=int(env.lod.stored),
               lod_swaps=info.get("lod_swaps"), lod_deferred=info.get("lod_deferred"),
               lod_max_interactive=info.get("lod_max_interactive"),
               skipped=dict(Counter(info.get("skipped", []))), damage_events=info.get("damage_events"),
               leaf_damage=info.get("leaf_damage"),
               attempts=len(tg), grasped=sum(bool(t.get("grasped")) for t in tg), detached=sum(bool(t.get("detached")) for t in tg),
               est_error_cm_median=float(np.nanmedian([t["est_error"] for t in tg]) * 100) if tg else None,
               control_steps=int(info["steps_total"]), rl_steps=int(info["rl_steps"]), sim_time_s=info["steps_total"] * env.sim.dt,
               compute_s=float(info.get("compute_s", 0.0)), fail_phase=info.get("fail_phase", ""),
               timing=summarize([dict(targets=tg, sim_time_s=info["steps_total"] * env.sim.dt,
                                      compute_s=float(info.get("compute_s", 0.0)), n_placed=int(info["n_placed"]))]),
               observe_to_basket_s=[round(x[0], 1) for x in fruit_times(tg)],
               side_beds_per_side=int(env.cfg["field"].get("side_rows", {}).get("beds_per_side", 1)),
               side_edge_gap_m=float(env.cfg["field"].get("side_rows", {}).get("edge_gap", 0.0))
               if env.cfg["field"].get("side_rows", {}).get("enabled") else None,
               side_plants=len(env.sim.meta.get("side_layout", [])), targets=tg,
               wall_s=round(wall, 1), video_frames=None if vid is None else vid.n, events=[f"{k}: {e}" for k, e in info["events"]],
               lod_log=env.lod.log[-200:])
    rec.save(out / "recording.npz", res)
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k not in ("events", "lod_log")}, indent=1, default=str))
    env.lod.set_render_window(None)
    if not a.no_html:
        res["html_parts"] = export_parts(env.sim.m, out / "recording.npz", out / "html", a.meters_per_part,
                                         a.max_frames_per_part, res, a.x_window, scenery)
        (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    print("->", out)


if __name__ == "__main__":
    main()
