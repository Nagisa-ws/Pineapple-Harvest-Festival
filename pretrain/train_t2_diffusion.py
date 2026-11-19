"""Train the T2 diffusion policy on the realistic-rig dataset and compare it with the heuristic and random choices.

  python pretrain/train_t2_diffusion.py --data outputs/pretrain/t2_data.npz

Writes models/t2_diffusion.pt and outputs/pretrain/t2_diffusion_eval.json (held-out views: TRUE score of the
chosen T2 - line clearance + 0.5 x point clearance to the simulator's leaves - for diffusion / heuristic / random /
oracle).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from pinesim.config import PROJECT_ROOT, load_config
from pinesim.decision.diffusion import T2DiffusionPolicy
from pinesim.decision.window import Candidate


def load(paths):
    parts = [np.load(p) for p in paths]
    keys = ["feats", "glob", "uv", "cam_score", "true_score", "n", "meta"]
    return {k: np.concatenate([z[k] for z in parts]) for k in keys}


def targets(d, rng, rel: float = 0.10):
    """One target (u, v) per sample: a random candidate among the good ones (true score within 10 % of the best)."""
    N = len(d["n"])
    out = np.zeros((N, 2), np.float32)
    for i in range(N):
        n = d["n"][i]
        ts = d["true_score"][i, :n]
        good = np.where(ts >= ts.max() - max(0.005, rel * abs(ts.max())))[0]
        j = rng.choice(good)
        out[i] = d["uv"][i, j]
    return out


def evaluate(policy, d, idx, K=24):
    res = dict(diffusion=[], heuristic=[], random=[], oracle=[], top1_hit=[])
    rng = np.random.default_rng(0)
    for i in idx:
        n = d["n"][i]
        ts, cs, uv = d["true_score"][i, :n], d["cam_score"][i, :n], d["uv"][i, :n]
        cands = [Candidate(cell=(0, 0), uv=uv[k], t2=np.zeros(3), t1=np.zeros(3), approach=np.zeros(3), elevation_deg=0.0,
                           clear_line=0.0, clear_point=0.0, score=float(cs[k])) for k in range(n)]
        ranked = policy.rank(d["feats"][i], d["glob"][i], cands, K=K, rng=rng)
        k_d = next(k for k in range(n) if cands[k] is ranked[0])
        res["diffusion"].append(ts[k_d])
        res["heuristic"].append(ts[int(np.argmax(cs))])
        res["random"].append(ts[rng.integers(n)])
        res["oracle"].append(ts.max())
        res["top1_hit"].append(float(ts[k_d] >= ts.max() - max(0.005, 0.1 * abs(ts.max()))))
    return {k: float(np.mean(v)) for k, v in res.items()} | {"n": len(idx)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=[str(PROJECT_ROOT / "outputs/pretrain/t2_data_0.npz"),
                                                   str(PROJECT_ROOT / "outputs/pretrain/t2_data_1.npz")])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "models/t2_diffusion.pt"))
    a = ap.parse_args()
    cfg = load_config("pretrain")
    dc = cfg["diffusion"]
    torch.manual_seed(0)
    torch.set_num_threads(2)
    d = load(a.data)
    N = len(d["n"])
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    n_val = max(200, N // 10)
    val, tr = perm[:n_val], perm[n_val:]
    pol = T2DiffusionPolicy(G=int(cfg["stargaze"]["grid"]), steps=int(dc["steps"]), sample_steps=int(dc["sample_steps"]),
                            hidden=int(dc["hidden"]))
    opt = torch.optim.AdamW(pol.model.parameters(), lr=float(dc["lr"]), weight_decay=1e-4)
    F = torch.tensor(d["feats"], dtype=torch.float32)
    Gl = torch.tensor(d["glob"], dtype=torch.float32)
    epochs = int(a.epochs or dc["epochs"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    B = int(dc["batch"])
    log = []
    t0 = time.time()
    for ep in range(epochs):
        X = torch.tensor(targets(d, rng), dtype=torch.float32)
        X += torch.randn_like(X) * (0.5 / int(cfg["stargaze"]["grid"]))       # within-cell jitter
        pol.model.train()
        tot = 0.0
        sh = rng.permutation(tr)
        for b0 in range(0, len(sh), B):
            ib = torch.tensor(sh[b0:b0 + B])
            loss = pol.loss(X[ib], F[ib], Gl[ib])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(ib)
        sched.step()
        if ep % 50 == 0 or ep == epochs - 1:
            pol.model.eval()
            ev = evaluate(pol, d, val[:300], K=16)
            log.append(dict(epoch=ep, loss=tot / len(tr), **ev))
            print(f"epoch {ep}: loss {tot / len(tr):.4f} | val true score: diffusion {ev['diffusion']:.4f} heuristic "
                  f"{ev['heuristic']:.4f} random {ev['random']:.4f} oracle {ev['oracle']:.4f} top1-good {ev['top1_hit']:.2f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    pol.model.eval()
    ev = evaluate(pol, d, val, K=24)
    meta = dict(samples=24, n_train=int(len(tr)), n_val=int(len(val)), epochs=epochs, eval=ev, log=log,
                data=[str(p) for p in a.data], wall_s=time.time() - t0)
    pol.save(a.out, meta)
    out = PROJECT_ROOT / "outputs" / "pretrain" / "t2_diffusion_eval.json"
    out.write_text(json.dumps(meta, indent=1))
    print("final held-out:", json.dumps(ev), "->", a.out)


if __name__ == "__main__":
    main()
