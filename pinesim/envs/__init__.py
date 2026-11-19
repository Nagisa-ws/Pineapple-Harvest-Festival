"""Environment factory: make_env('mission1' | 'mission2' | 'mission3', ...).

  mission1 / mission2   HarvestEnv (light, training)      | PlatformDetailEnv (overrides task.mode = platform_detail)
  mission3              XghHarvestEnv (light, training)   | XghEvalEnv (overrides task.mode = xgh_detail)
"""
from __future__ import annotations


def make_env(mission: str = "mission1", training: bool = True, seed: int | None = None, overrides: dict | None = None,
             level: int | None = None):
    mode = (overrides or {}).get("task", {}).get("mode")
    if mission.startswith("mission3"):
        if mode == "xgh_detail":
            from pinesim.envs.xgh_env import XghEvalEnv
            return XghEvalEnv(mission, training=training, seed=seed, overrides=dict(overrides), level=level)
        from pinesim.envs.xgh_env import XghHarvestEnv
        return XghHarvestEnv(mission, training=training, seed=seed, overrides=overrides, level=level)
    if mode == "platform_detail":
        from pinesim.envs.platform_detail import PlatformDetailEnv
        return PlatformDetailEnv(mission, training=training, seed=seed, overrides=dict(overrides), level=level)
    from pinesim.envs.harvest_env import HarvestEnv
    return HarvestEnv(mission, training=training, seed=seed, overrides=overrides, level=level)
