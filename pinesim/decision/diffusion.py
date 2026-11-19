"""
pinesim/decision/diffusion.py
-----------------------------
Diffusion policy that chooses the best T2 on the fruit shoulder (version 5 decision maker).

Action  x = normalised position of T2 inside the window, (u, v) in [-1, 1]^2.
Context the window feature grid of pinesim/decision/window.py (6 x G x G: shoulder occupancy, camera clearance of T2,
        camera line clearance, height on the shoulder, leaf-pixel occupancy, valid candidate) + 5 global features.
Model   DDPM (Ho et al. 2020) with an epsilon-network conditioned on the context (small CNN encoder + MLP), as in
        Diffusion Policy (Chi et al. 2023) for a single action; DDIM sampling at inference.
Train   pretrain/train_t2_diffusion.py on views of the realistic rig: the target is the candidate with the best TRUE
        line clearance (simulator leaf geometry); every candidate within 90 % of the best is a valid mode, so the
        policy learns the multi-modal distribution of good T2 positions from what the camera sees.
Decide  K samples -> each snapped to the nearest valid candidate cell -> votes. The cells with at least 12 % of the
        votes are the policy's PROPOSAL set (the modes of the distribution); inside it the line with the largest camera
        clearance comes first, then the remaining cells by votes. The arm planner validates them in that order (IK,
        collisions) and takes the first feasible. Held-out rig views (TRUE leaf clearance score, see
        outputs/pretrain/t2_diffusion_eval.json): proposal 0.105, votes only 0.104, camera heuristic 0.105, random
        0.072, oracle 0.108.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except Exception:          # pragma: no cover
    HAVE_TORCH = False

from pinesim.decision.window import N_FEAT, N_GLOBAL


def _betas(T: int):
    s = 0.008                                          # cosine schedule (Nichol & Dhariwal 2021)
    t = np.linspace(0, T, T + 1) / T
    f = np.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    a = f / f[0]
    return np.clip(1 - a[1:] / a[:-1], 1e-5, 0.5)


if HAVE_TORCH:
    class EpsNet(nn.Module):
        def __init__(self, G: int = 8, hidden: int = 256, t_dim: int = 32):
            super().__init__()
            self.G, self.t_dim = G, t_dim
            self.enc = nn.Sequential(nn.Conv2d(N_FEAT, 32, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(32, 16, 1), nn.ReLU())
            self.ctx = nn.Sequential(nn.Linear(16 * G * G + N_GLOBAL, hidden), nn.ReLU(), nn.Linear(hidden, 128))
            self.net = nn.Sequential(nn.Linear(2 + t_dim + 128, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2))

        def context(self, feats, glob):
            e = self.enc(feats).flatten(1)
            return self.ctx(torch.cat([e, glob], 1))

        def temb(self, t):
            half = self.t_dim // 2
            f = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device) / half)
            a = t.float()[:, None] * f[None, :]
            return torch.cat([torch.sin(a), torch.cos(a)], 1)

        def forward(self, x, t, c):
            return self.net(torch.cat([x, self.temb(t), c], 1))


class T2DiffusionPolicy:
    def __init__(self, G: int = 8, steps: int = 50, sample_steps: int = 10, hidden: int = 256):
        assert HAVE_TORCH, "torch is required for the diffusion policy"
        self.G, self.T, self.S = G, int(steps), int(sample_steps)
        self.model = EpsNet(G, hidden)
        b = _betas(self.T)
        self.alpha_bar = np.cumprod(1.0 - b)
        self.meta = {}

    # ------------------------------------------------------------------ io
    def save(self, path, meta=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state=self.model.state_dict(), G=self.G, T=self.T, S=self.S, meta=meta or self.meta), path)

    @classmethod
    def load(cls, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        p = cls(int(ck["G"]), int(ck["T"]), int(ck["S"]))
        p.model.load_state_dict(ck["state"])
        p.model.eval()
        p.meta = ck.get("meta", {})
        return p

    # ------------------------------------------------------------------ training
    def loss(self, x0, feats, glob):
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,))
        ab = torch.tensor(self.alpha_bar, dtype=torch.float32)[t][:, None]
        eps = torch.randn_like(x0)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        c = self.model.context(feats, glob)
        return ((self.model(xt, t, c) - eps) ** 2).mean()

    # ------------------------------------------------------------------ sampling
    @torch.no_grad() if HAVE_TORCH else (lambda f: f)
    def sample(self, feats: np.ndarray, glob: np.ndarray, K: int = 24, seed: int = 0) -> np.ndarray:
        """K samples of (u, v) for one context (DDIM, deterministic given the seed)."""
        g = torch.Generator().manual_seed(int(seed))
        f = torch.tensor(feats, dtype=torch.float32)[None].repeat(K, 1, 1, 1)
        gl = torch.tensor(glob, dtype=torch.float32)[None].repeat(K, 1)
        c = self.model.context(f, gl)
        x = torch.randn((K, 2), generator=g)
        ts = np.linspace(self.T - 1, 0, self.S).round().astype(int)
        for i, t in enumerate(ts):
            ab = float(self.alpha_bar[t])
            ab_prev = float(self.alpha_bar[ts[i + 1]]) if i + 1 < len(ts) else 1.0
            eps = self.model(x, torch.full((K,), int(t)), c)
            x0 = (x - math.sqrt(1 - ab) * eps) / math.sqrt(ab)
            x0 = x0.clamp(-1.2, 1.2)
            x = math.sqrt(ab_prev) * x0 + math.sqrt(1 - ab_prev) * eps
        return x.numpy()

    def rank(self, feats, glob, cands: list, K: int | None = None, rng=None, mode: str = "proposal") -> list:
        """Candidates ranked by the policy: 'proposal' = voted cells (>= 12 % of K) by camera clearance, then the rest
        by votes; 'votes' = by votes (ties: camera score)."""
        if not cands:
            return []
        K = K or int(self.meta.get("samples", 24))
        seed = int(rng.integers(1 << 30)) if rng is not None else 0
        xs = self.sample(feats, glob, K, seed)
        uv = np.array([c.uv for c in cands])
        d = np.linalg.norm(xs[:, None, :] - uv[None, :, :], axis=2)
        nearest = np.argmin(d, axis=1)
        votes = np.bincount(nearest, minlength=len(cands)).astype(float)
        for c, v in zip(cands, votes):
            c.votes = float(v)
        if mode == "votes":
            order = sorted(range(len(cands)), key=lambda i: (-votes[i], -cands[i].score))
        else:
            thr = max(2.0, 0.12 * K)
            prop = [i for i in range(len(cands)) if votes[i] >= thr] or [int(np.argmax(votes))]
            rest = [i for i in range(len(cands)) if i not in prop]
            order = sorted(prop, key=lambda i: -cands[i].score) + sorted(rest, key=lambda i: (-votes[i], -cands[i].score))
        return [cands[i] for i in order]


_CACHE = {}


def load_policy(cfg: dict):
    """The trained policy of configs (stargaze.diffusion_model), or None (-> heuristic) if it is not there."""
    from pinesim.config import PROJECT_ROOT
    if not HAVE_TORCH or str(cfg["stargaze"].get("decision", "diffusion")) != "diffusion":
        return None
    p = Path(cfg["stargaze"].get("diffusion_model", "models/t2_diffusion.pt"))
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return None
    key = str(p)
    if key not in _CACHE:
        torch.set_num_threads(1)
        _CACHE[key] = T2DiffusionPolicy.load(p)
    return _CACHE[key]
