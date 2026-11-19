"""Merge evaluation JSON files (episodes concatenated, summary recomputed)."""
import argparse, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_mission import wilson

ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--out", required=True)
a = ap.parse_args()
eps, base = [], None
for f in a.files:
    d = json.load(open(f)); eps += d["episodes"]; base = base or d["summary"]
n = len(eps); k = sum(e["success"] for e in eps)
fr = sum(e["n_placed"] for e in eps); ft = sum(e["n_targets"] for e in eps)
est = [t["est_error"] for e in eps for t in e["targets"] if t.get("est_error") == t.get("est_error")]
s = dict(base); s.update(n=n, success=k, success_rate=k / max(n, 1), ci95=wilson(k, n), fruits_placed=fr, fruits_total=ft,
                        fruit_rate=fr / max(ft, 1), failures=dict(Counter(e["failure"] for e in eps if not e["success"])),
                        est_error_cm=dict(mean=float(np.mean(est) * 100) if est else None, median=float(np.median(est) * 100) if est else None),
                        mean_steps=float(np.mean([e["steps"] for e in eps])), seeds=[min(e["seed"] for e in eps), n])
Path(a.out).write_text(json.dumps(dict(summary=s, episodes=sorted(eps, key=lambda e: e["seed"])), indent=1, default=str))
print(json.dumps({k_: s[k_] for k_ in ("n", "success", "success_rate", "ci95", "fruit_rate", "failures", "est_error_cm")}, default=str))
