"""Realistic evaluation of Missions 1-2 (mode platform_detail: the user's mesh pineapples, per-leaf articulated
leaves, tilted 0-15 deg, fruit 0.60-0.70 m, ground mesh, robot meshes) + Meshcat HTML recordings.

  python scripts/eval_realistic.py --mission mission1 --onnx models/mission1_v1.onnx --seeds 900000 10 --record 2
  python scripts/eval_realistic.py --mission mission2 --onnx models/mission2_v1.onnx --seeds 910000 6 --record 1

Outputs: outputs/eval_realistic/<mission>_<tag>/result.json, rec_<seed>.npz, <mission>_seed<seed>.html
(camera modes: free orbit / top view / GRIPPER CAMERA with bounding box, window, shoulder, T2 candidates, T1, T2).
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

import numpy as np

from pinesim.config import PROJECT_ROOT
from pinesim.envs import make_env
from pinesim.rl.policy import load_policy
from pinesim.viz.xgh_recorder import XghRecorder, export_html
from pinesim.timing import fruit_times, summarize


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mission", default="mission1", choices=["mission1", "mission2"])
    ap.add_argument("--seeds", nargs=2, type=int, default=[900000, 10], metavar=("FIRST", "N"))
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--record", type=int, default=1, help="record (and export HTML of) the first N episodes")
    ap.add_argument("--every", type=int, default=4, help="record every N control steps")
    ap.add_argument("--max-rl-steps", type=int, default=8000)
    ap.add_argument("--out-root", default="outputs/eval_realistic")
    ap.add_argument("--record-seeds", type=int, nargs="*", default=None, help="record exactly these seeds")
    ap.add_argument("--record-failures", action="store_true", help="re-run and record every failed seed afterwards")
    ap.add_argument("--max-failure-html", type=int, default=3)
    ap.add_argument("--keep-recordings", action="store_true", help="keep rec_<seed>.npz next to the HTML")
    a = ap.parse_args()
    policy, desc = load_policy(onnx=a.onnx)
    tag = a.tag or ("reference" if policy is None else Path(a.onnx).stem)
    out = PROJECT_ROOT / a.out_root / f"{a.mission}_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(a.mission, training=False, seed=a.seeds[0], overrides={"task": {"mode": "platform_detail"}})
    rows, htmls, recorded = [], [], set()

    def export(rec, row, seed, suffix=""):
        rp = out / f"rec_{seed}.npz"
        rec.save(rp, result=row)
        lines = [f"<b>{a.mission.replace('mission', 'Misi ')} - evaluasi realistis</b> (seed {seed})",
                 f"buah di keranjang: {row['n_placed']} / {row['n_targets']}"
                 + ("" if row["success"] else f" &nbsp; gagal: {row['failure']}"),
                 "waktu observasi -> keranjang per buah: " + (", ".join(f"{x:.1f} s" for x in row["observe_to_basket_s"]) or "-")
                 + f" (gerak {row['sim_time_s']:.1f} s + komputasi {row['compute_s']:.1f} s per episode)",
                 f"policy: {desc}",
                 "lampu: kuning = observasi, biru = aksi, hijau = sukses/standby, merah = gagal"]
        hp = out / f"{a.mission}_seed{seed}{suffix}.html"
        st = export_html(env.sim.m, rp, hp, info_lines=lines)
        if not a.keep_recordings:
            rp.unlink(missing_ok=True)
        print("   html ->", hp, st, flush=True)
        return str(hp)

    t_all = time.time()
    for k in range(a.seeds[1]):
        seed = a.seeds[0] + k
        rec = None
        if (k < a.record) if a.record_seeds is None else (seed in a.record_seeds):
            rec = XghRecorder(env.sim, every=a.every)
            env.frame_callback, env.overlay_callback = rec, rec.overlay
        else:
            env.frame_callback = env.overlay_callback = None
        t0 = time.time()
        info = env.run_episode(policy, seed=seed, max_rl_steps=a.max_rl_steps)
        row = dict(seed=seed, success=bool(info["is_success"]), failure=info.get("failure", ""),
                   n_placed=info.get("n_placed", 0), n_targets=info.get("n_targets", 1), steps=info["steps_total"],
                   rl_steps=info["rl_steps"], ret=round(float(info["return"]), 2), wall_s=round(time.time() - t0, 1),
                   leaf_damage=info.get("leaf_damage", 0), leaf_root_bend_max_deg=round(info.get("leaf_root_bend_max_deg", 0.0), 1),
                   shield_events=info.get("shield_events", 0), fail_phase=info.get("fail_phase", ""),
                   sim_time_s=round(float(info.get("sim_time_s", info["steps_total"] * env.sim.dt)), 2),
                   compute_s=round(float(info.get("compute_s", 0.0)), 2),
                   plants=info.get("plants", []), targets=info.get("targets", []), events=info.get("events", [])[-30:])
        rows.append(row)
        ft = fruit_times(row["targets"])
        row["observe_to_basket_s"] = [round(x[0], 1) for x in ft]
        print(f"seed {seed}: success={row['success']} placed={row['n_placed']}/{row['n_targets']} fail={row['failure']}"
              f"{'@' + row['fail_phase'] if row['failure'] else ''} motion={row['sim_time_s']:.1f}s compute={row['compute_s']:.1f}s "
              f"observe->basket={row['observe_to_basket_s']} leaf_damage={row['leaf_damage']} (wall {row['wall_s']} s)", flush=True)
        if rec is not None:
            recorded.add(seed)
            htmls.append(export(rec, row, seed))
    # failed episodes that were not recorded: run them again with the recorder (episodes are deterministic per seed)
    if a.record_failures:
        for r in [r for r in rows if not r["success"] and r["seed"] not in recorded][:a.max_failure_html]:
            rec = XghRecorder(env.sim, every=a.every)
            env.frame_callback, env.overlay_callback = rec, rec.overlay
            info = env.run_episode(policy, seed=r["seed"], max_rl_steps=a.max_rl_steps)
            if info.get("failure", "") != r["failure"]:
                print(f"   seed {r['seed']}: re-run ended with '{info.get('failure', '')}' instead of '{r['failure']}'")
            htmls.append(export(rec, r, r["seed"], suffix="_gagal"))
    n = len(rows)
    k = sum(r["success"] for r in rows)
    fr = sum(r["n_placed"] for r in rows)
    ft = sum(r["n_targets"] for r in rows)
    est = [t["est_error"] for r in rows for t in r["targets"] if t.get("est_error") == t.get("est_error")]
    summ = dict(mission=a.mission, scene="platform_detail (realistic)", policy=desc, n=n, success=k, success_rate=k / max(n, 1),
                fruits_placed=fr, fruits_total=ft, fruit_rate=fr / max(ft, 1),
                failures=dict(Counter(r["failure"] for r in rows if not r["success"])),
                leaf_damage_total=int(sum(r["leaf_damage"] for r in rows)),
                est_error_cm_median=float(np.median(est) * 100) if est else None,
                timing=summarize(rows), wall_s=round(time.time() - t_all, 1), seeds=a.seeds, html=htmls)
    (out / "result.json").write_text(json.dumps(dict(summary=summ, episodes=rows), indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
