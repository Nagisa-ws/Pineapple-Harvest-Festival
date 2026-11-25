"""Configuration loading: configs/common.yaml + a mission file (deep-merged, mission keys win)."""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_yaml(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(mission_file: str | Path | None = None, overrides: dict | None = None) -> dict:
    """common.yaml, then the mission file (name like 'mission1' or a path), then `overrides`."""
    cfg = load_yaml(CONFIG_DIR / "common.yaml")
    if mission_file:
        p = Path(mission_file)
        if not p.suffix:
            p = CONFIG_DIR / f"{mission_file}.yaml"
        elif not p.is_absolute() and not p.exists():
            p = CONFIG_DIR / p
        mc = load_yaml(p)
        parent = mc.pop("inherit", None)
        if parent:                                   # e.g. mission2.yaml: "inherit: mission1"
            cfg = load_config(parent)
        cfg = deep_merge(cfg, mc)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def get(cfg: dict, dotted: str, default=None):
    cur = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

HARVEST_METHODS = ("twist", "pluck")

def choose_harvest_method(arg: str | None, mission: str = "mission1") -> str:
    """How the fruit is taken off the stalk in an evaluation run, in this order: the --harvest option; the environment
    variable PINESIM_HARVEST (for shell scripts that run unattended); a question on the terminal (only when a person
    runs the script: tests and pipes get no question); harvest.method of the mission's configuration."""
    default = str(load_config(mission).get("harvest", {}).get("method", "twist"))
    if arg:
        return arg
    env = os.environ.get("PINESIM_HARVEST", "").strip().lower()
    if env in HARVEST_METHODS:
        return env
    if sys.stdin is not None and sys.stdin.isatty() and sys.stdout.isatty():
        ans = input(f"Harvest method?  [1] twist (twist, then lift)   [2] pluck (tilt > 45 deg while pulling up)   "
                    f"Enter = {default}: ").strip().lower()
        return {"1": "twist", "t": "twist", "twist": "twist", "2": "pluck", "p": "pluck", "pluck": "pluck"}.get(ans, default)
    return default