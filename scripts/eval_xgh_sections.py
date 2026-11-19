"""Run the realistic 15 m evaluation section by section (resumable) and merge the results.

  python scripts/eval_xgh_sections.py run --onnx models/mission3_v1.onnx --sections 0 10 20 30     # one lane
  python scripts/eval_xgh_sections.py merge --tag-prefix final                                      # summary

The field (60 plants, sorted along the row) is split into consecutive sections of --size plants. Each section is a
separate run of scripts/eval_xgh.py on the SAME detailed scene, with the vehicle starting 0.83 m (base) in front of
the section's first plant; a section whose result.json exists is skipped (resume after an interruption).
Why sections: on a 2-core machine without GPU the realistic scene runs slower than real time; sections can run in
parallel and survive interruptions.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "eval_xgh"
sys.path.insert(0, str(ROOT))


def run(a):
    for k0 in a.sections:
        tag = f"{a.tag_prefix}_s{k0:03d}"
        if (OUT / tag / "result.json").exists() and not a.force:
            print("skip (done):", tag, flush=True)
            continue
        cmd = [sys.executable, str(ROOT / "scripts" / "eval_xgh.py"), "--section", str(k0), str(a.size), "--tag", tag,
               "--seed", str(a.seed + k0), "--every", str(a.every), "--out-root", str(OUT.relative_to(ROOT))]
        if a.edge_gap is not None:
            cmd += ["--edge-gap", str(a.edge_gap)]
        if a.no_html:
            cmd.append("--no-html")
        if a.onnx:
            cmd += ["--onnx", a.onnx]
        if a.video:
            cmd.append("--video")
        print("run:", " ".join(cmd), flush=True)
        with open(OUT / f"{tag}.log", "w") as log:
            (OUT / tag).mkdir(parents=True, exist_ok=True)
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        print(tag, "exit", r.returncode, flush=True)


def merge(a):
    rows = []
    for d in sorted(OUT.glob(f"{a.tag_prefix}_s*")):
        f = d / "result.json"
        if f.exists():
            rows.append((d.name, json.loads(f.read_text())))
    if not rows:
        print("no sections found")
        return
    tot = Counter()
    skipped = Counter()
    for name, r in rows:
        for k in ("n_placed", "n_targets", "attempts", "grasped", "detached", "control_steps", "rl_steps",
                  "damage_events", "leaf_damage", "lod_swaps", "lod_deferred", "unloads"):
            tot[k] += int(r.get(k) or 0)
        tot["wall_s"] += float(r.get("wall_s") or 0)
        tot["sim_time_s"] += float(r.get("sim_time_s") or 0)
        tot["vehicle_travel_m"] += float(r.get("vehicle_travel_m") or 0)
        skipped.update(r.get("skipped", {}))
    from pinesim.timing import summarize
    timing = summarize([dict(targets=r.get("targets", []), sim_time_s=r.get("sim_time_s", 0.0),
                             compute_s=r.get("compute_s", 0.0), n_placed=r.get("n_placed", 0)) for _, r in rows])
    summary = dict(sections=[dict(tag=n, plants=r["n_targets"], placed=r["n_placed"], failure=r.get("failure"),
                                  skipped=r.get("skipped"), sim_time_s=r.get("sim_time_s"), compute_s=r.get("compute_s"),
                                  s_per_fruit=(r.get("timing") or {}).get("s_per_fruit"), wall_s=r.get("wall_s")) for n, r in rows],
                   totals=dict(tot), fruit_rate=tot["n_placed"] / max(tot["n_targets"], 1), skipped=dict(skipped),
                   row_complete=all(r.get("termination") in ("success", "row_finished_incomplete") and not
                                    (r.get("failure") in ("collision", "floor_contact", "timeout", "sim_unstable", "too_many_stops"))
                                    for _, r in rows),
                   timing=timing, policy=rows[0][1].get("policy"))
    (OUT / f"{a.tag_prefix}_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    vids = [OUT / n / "corner_camera.mp4" for n, _ in rows if (OUT / n / "corner_camera.mp4").exists()]
    if vids and not a.no_video:
        lst = OUT / f"{a.tag_prefix}_videos.txt"
        lst.write_text("".join(f"file '{v}'\n" for v in vids))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                        str(OUT / f"{a.tag_prefix}_corner_camera.mp4")])
        print("video ->", OUT / f"{a.tag_prefix}_corner_camera.mp4")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "merge"])
    ap.add_argument("--sections", type=int, nargs="*", default=list(range(0, 60, 10)))
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--seed", type=int, default=990000)
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--tag-prefix", default="final")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--edge-gap", type=float, default=None)
    ap.add_argument("--no-html", action="store_true")
    ap.add_argument("--out-root", default=None)
    a = ap.parse_args()
    global OUT
    if a.out_root:
        OUT = ROOT / a.out_root
    OUT.mkdir(parents=True, exist_ok=True)
    run(a) if a.cmd == "run" else merge(a)


if __name__ == "__main__":
    main()
