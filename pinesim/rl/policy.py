"""Residual policies for evaluation: ONNX (normalisation baked in) or SB3 checkpoint + VecNormalize statistics."""
from __future__ import annotations

import pickle

import numpy as np


def load_policy(onnx: str | None = None, model: str | None = None, vecnorm: str | None = None):
    """Return (policy(obs) -> a_rl, description) or (None, 'reference') when nothing is given."""
    if onnx:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name

        def pol(obs):
            return sess.run(None, {name: np.asarray(obs, np.float32)[None]})[0][0]
        return pol, f"onnx:{onnx}"
    if model:
        from stable_baselines3 import SAC
        m = SAC.load(model, device="cpu")
        mean, var, clip, eps = None, None, 10.0, 1e-8
        if vecnorm:
            with open(vecnorm, "rb") as f:
                vn = pickle.load(f)
            mean, var, clip, eps = vn.obs_rms.mean, vn.obs_rms.var, vn.clip_obs, vn.epsilon

        def pol(obs):
            o = np.asarray(obs, np.float64)
            if mean is not None:
                o = np.clip((o - mean) / np.sqrt(var + eps), -clip, clip)
            a, _ = m.predict(o.astype(np.float32), deterministic=True)
            return a
        return pol, f"sb3:{model}"
    return None, "reference"
