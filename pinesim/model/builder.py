"""
pinesim/model/builder.py
------------------------
Generates every MuJoCo scene of the curriculum from configs/common.yaml (single source of truth).

Modes (version 5)
  rig              pretraining: the user's REALISTIC pineapples (static meshes, one combination shown at a time) + a
                   camera on a pole that orbits it (two mocap bodies). No robot (pinesim/model/builder_xgh.py).
  rig_light        old procedural rig (capsule plant), kept for quick tests.
  platform         Missions 1-2 TRAINING: UR10e + U-jaw gripper on a column whose top is `robot.base_mount_height`
                   (1.15 m) above the ground, a basket on its own stand (moved by the env to 180 deg from the plants),
                   N procedural flexible plants (primitive shapes).
  platform_detail  Missions 1-2 EVALUATION: the same robot / column / basket collision geometry + robot meshes, the
                   user's per-leaf articulated mesh pineapples (5 kinds x N slots), ground mesh.
  xgh_light        Mission 3 TRAINING (ATEX XGH505H crawler), xgh_detail Mission 3 EVALUATION (15 m field, level of
                   detail) - see pinesim/model/builder_xgh.py.

Plants (all modes): plant{i} body (stalk + articulated rosette / blocking leaf chains, static, moved by the env) and
fruit{i} free body (fruit capsule + crown neck + rigid crown leaves) welded (`stem{i}`) to a stem link that sits on top
of the stalk with a torsion spring and two bending springs; the weld is released when the stem breaks.
Every size, direction and stiffness is rewritten per episode by pinesim/plant.py; the XML holds placeholders.

Contact classes (contype / conaffinity)
  floor                    1 / 1
  robot, platform, basket, vehicle   1 / 7
  hidden wheels            8 / 1      (floor only)
  leaves (incl. crown)     2 / 0      (touched by the robot / vehicle / detached fruit only)
  stalk, attached fruit    4 / 0
  detached fruit           4 / 7      (set by the env)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from pinesim.config import PROJECT_ROOT, load_config

OUT_DIR = PROJECT_ROOT / "outputs" / "models"

UR10E_DEFAULTS = """
    <default class="ur10e">
      <material specular="0.5" shininess="0.25"/>
      <joint axis="0 1 0" range="-6.28319 6.28319" armature="0.1"/>
      <position ctrlrange="-6.2831 6.2831"/>
      <general biastype="affine" ctrlrange="-6.2831 6.2831" gainprm="5000" biasprm="0 -5000 -500"/>
      <default class="size4">
        <joint damping="10"/>
        <general forcerange="-330 330"/>
      </default>
      <default class="size3">
        <joint damping="5"/>
        <general forcerange="-150 150"/>
        <default class="size3_limited">
          <joint range="-3.1415 3.1415"/>
          <general ctrlrange="-3.1415 3.1415"/>
        </default>
      </default>
      <default class="size2">
        <joint damping="2"/>
        <general forcerange="-56 56"/>
      </default>
      <default class="link_geom">
        <geom type="capsule" group="1" contype="1" conaffinity="7" rgba="0.6 0.6 0.65 1"/>
        <default class="eef_geom">
          <geom type="cylinder"/>
        </default>
      </default>
      <site size="0.001" rgba="0.5 0.5 0.5 0.3" group="4"/>
    </default>
"""

ARM_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
ARM_ACTUATORS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]


def _f(x: float) -> str:
    return f"{float(x):.6g}"


def _vec(v) -> str:
    return " ".join(_f(x) for x in v)


# ----------------------------------------------------------------------------- pieces
def leaf_chain_xml(prefix: str, n_seg: int, indent: str, rgba: str = "0.23 0.46 0.20 1", seg_len: float = 0.15) -> str:
    """Articulated capsule chain along local +x (two spring hinges per joint) [REF]."""
    out, close = [], []
    for s in range(n_seg):
        pos = "0 0 0" if s == 0 else f"{seg_len} 0 0"
        ind = indent + "  " * s
        out.append(f'{ind}<body name="{prefix}_s{s}" pos="{pos}" gravcomp="1">')
        out.append(f'{ind}  <inertial mass="0.02" pos="{seg_len / 2} 0 0" diaginertia="0.00002 0.00004 0.00004"/>')
        out.append(f'{ind}  <joint name="{prefix}_j{s}a" type="hinge" axis="0 1 0" class="leafjoint"/>')
        out.append(f'{ind}  <joint name="{prefix}_j{s}b" type="hinge" axis="0 0 1" class="leafjoint"/>')
        out.append(f'{ind}  <geom name="{prefix}_g{s}" class="leaf" size="0.012 {seg_len / 2}" pos="{seg_len / 2} 0 0" '
                   f'quat="0.70710678 0 0.70710678 0" rgba="{rgba}"/>')
        close.append(f"{ind}</body>")
    return "\n".join(out + close[::-1])


def plants_xml(cfg: dict, n_plants: int, positions: list | None = None) -> tuple[str, str]:
    pool = cfg["plant"]["pool"]
    out, eq = [], []
    for i in range(n_plants):
        x, y = (positions[i] if positions else (1.0 + 0.5 * i, 0.0))
        out.append(f'    <body name="plant{i}" pos="{_f(x)} {_f(y)} 0">')
        out.append(f'      <geom name="stalk{i}" class="hardplant" type="capsule" size="0.02 0.15" pos="0 0 0.15" material="stalk"/>')
        # stem link at the top of the stalk: torsion hinge (twist spring) + two bending hinges; the fruit is welded to it
        st = cfg["plant"].get("stem", {})
        out.append(f'      <body name="stem{i}_link" pos="0 0 0.3" gravcomp="1">')
        out.append('        <inertial mass="0.1" pos="0 0 0" diaginertia="0.0005 0.0005 0.0005"/>')
        out.append(f'        <joint name="stem{i}_twist" type="hinge" axis="0 0 1" stiffness="{_f(st.get("torsion_stiffness", 1.0))}" damping="{_f(st.get("torsion_damping", 0.3))}" armature="0.01" limited="false"/>')
        out.append(f'        <joint name="stem{i}_bend_x" type="hinge" axis="1 0 0" stiffness="{_f(st.get("bending_stiffness", 40.0))}" damping="{_f(st.get("bending_damping", 2.0))}" armature="0.01" limited="false"/>')
        out.append(f'        <joint name="stem{i}_bend_y" type="hinge" axis="0 1 0" stiffness="{_f(st.get("bending_stiffness", 40.0))}" damping="{_f(st.get("bending_damping", 2.0))}" armature="0.01" limited="false"/>')
        out.append(f'        <joint name="stem{i}_pull" type="slide" axis="0 0 1" stiffness="{_f(st.get("tension_stiffness", 4000.0))}" damping="{_f(st.get("tension_damping", 40.0))}" armature="0.05" limited="false"/>')
        out.append('      </body>')
        for k in range(int(pool["n_rosette"])):
            out.append(leaf_chain_xml(f"p{i}ros{k}", int(pool["rosette_segments"]), "      "))
        for k in range(int(pool["n_block"])):
            out.append(leaf_chain_xml(f"p{i}blk{k}", int(pool["block_segments"]), "      "))
        out.append('    </body>')
        out.append(f'    <body name="fruit{i}" pos="{_f(x)} {_f(y)} 0.45">')
        out.append(f'      <freejoint name="fruit{i}_free"/>')
        out.append('      <inertial mass="1.5" pos="0 0 0" diaginertia="0.006 0.006 0.003"/>')
        out.append(f'      <geom name="fruit{i}_body" class="fruitgeom" type="capsule" size="0.055 0.04" material="fruit"/>')
        out.append(f'      <geom name="fruit{i}_neck" class="fruitgeom" type="capsule" size="0.02 0.015" pos="0 0 0.11" material="neck"/>')
        out.append(f'      <site name="fruit{i}_centre" pos="0 0 0" size="0.008" rgba="1 1 0 0.6" group="4"/>')
        for k in range(int(pool["n_tuft"])):
            out.append(f'      <geom name="fruit{i}_tuft{k}" class="leaf" size="0.009 0.07" pos="0 0 0.2" material="crown" solref="0.03 1" solimp="0.8 0.9 0.005 0.5 2" friction="0.3 0.01 0.001"/>')
        out.append('    </body>')
        eq.append(f'    <weld name="stem{i}" body1="stem{i}_link" body2="fruit{i}" active="true" solref="0.01 1"/>')
    return "\n".join(out), "\n".join(eq)


def header(model_name: str, extra_defaults: str = "", extra_assets: str = "", timestep: float = 0.001) -> str:
    return f"""<mujoco model="{model_name}">
  <!-- GENERATED by pinesim/model/builder.py from configs/common.yaml. Do not edit by hand. -->
  <compiler angle="radian" autolimits="true" meshdir="assets"/>
  <option timestep="{timestep}" integrator="implicitfast" cone="elliptic" impratio="10">
    <!-- midphase disabled: plant geometry is rewritten at runtime (reference finding) -->
    <flag midphase="disable"/>
  </option>
  <size memory="96M"/>
  <default>
{UR10E_DEFAULTS}
    <default class="envstatic">
      <geom contype="1" conaffinity="7" group="1" friction="0.9 0.005 0.0001"/>
    </default>
    <default class="envnoleaf">
      <!-- platform / basket: robot, floor, hard plant parts and detached fruit, but not leaves -->
      <geom contype="1" conaffinity="5" group="1" friction="0.9 0.005 0.0001"/>
    </default>
    <default class="wheel">
      <geom contype="8" conaffinity="1" group="3" rgba="0.3 0.3 0.3 0.25" condim="3"/>
    </default>
    <default class="leaf">
      <geom type="capsule" contype="2" conaffinity="0" condim="3" friction="0.4 0.01 0.001"
            solref="0.01 1" solimp="0.9 0.95 0.002 0.5 2" group="2" material="leaf"/>
    </default>
    <default class="leafjoint">
      <joint stiffness="3" damping="0.08" armature="0.0005" limited="false"/>
    </default>
    <default class="hardplant">
      <geom contype="4" conaffinity="0" condim="3" friction="0.8 0.02 0.001" group="2"/>
    </default>
    <default class="fruitgeom">
      <geom contype="4" conaffinity="0" condim="4" friction="1.0 0.005 0.0001" solref="0.008 1" group="2"/>
    </default>
    <default class="vis">
      <geom contype="0" conaffinity="0" group="1"/>
    </default>
{extra_defaults}
  </default>
  <visual>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="150" elevation="-20" offwidth="640" offheight="480"/>
    <map znear="0.01" zfar="20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.45 0.6 0.8" rgb2="0.1 0.1 0.15" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.36 0.29 0.22" rgb2="0.31 0.25 0.19"
      markrgb="0.45 0.38 0.3" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.02"/>
    <material name="fruit" rgba="0.92 0.78 0.16 1"/>
    <material name="neck" rgba="0.30 0.42 0.18 1"/>
    <material name="crown" rgba="0.10 0.32 0.11 1"/>
    <material name="leaf" rgba="0.23 0.46 0.20 1"/>
    <material name="stalk" rgba="0.36 0.42 0.24 1"/>
    <material name="steel" rgba="0.55 0.57 0.6 1"/>
    <material name="basket" rgba="0.16 0.36 0.72 1"/>
    <material class="ur10e" name="black" rgba="0.033 0.033 0.033 1"/>
    <material class="ur10e" name="jointgray" rgba="0.278 0.278 0.278 1"/>
    <material class="ur10e" name="linkgray" rgba="0.82 0.82 0.82 1"/>
    <material class="ur10e" name="urblue" rgba="0.49 0.678 0.8 1"/>
{extra_assets}
  </asset>
"""


def basket_geoms(cfg: dict, cx: float, cy: float, floor_z: float, indent: str, prefix: str = "basket") -> list:
    """Open box. The floor is a thick slab below floor_z (a thin floor let a falling fruit tunnel through it:
    measured 8 mm penetration of a 5 mm half-thickness floor -> unstable contact)."""
    b = cfg["basket"]
    il, iw, ih = b["inner_size"]
    t = b["wall_thickness"]
    ft = float(b.get("floor_thickness", 0.04))
    L = [f'{indent}<site name="{prefix}_centre" pos="{_f(cx)} {_f(cy)} {_f(floor_z + ih / 2)}" size="0.01" rgba="1 0 0 0.3" group="4"/>',
         f'{indent}<geom name="{prefix}_floor" class="envnoleaf" type="box" material="basket" size="{_f(il / 2 + t)} {_f(iw / 2 + t)} {_f(ft / 2)}" pos="{_f(cx)} {_f(cy)} {_f(floor_z - ft / 2)}"/>',
         f'{indent}<geom name="{prefix}_wall_px" class="envnoleaf" type="box" material="basket" size="{_f(t / 2)} {_f(iw / 2 + t)} {_f(ih / 2)}" pos="{_f(cx + il / 2 + t / 2)} {_f(cy)} {_f(floor_z + ih / 2)}"/>',
         f'{indent}<geom name="{prefix}_wall_nx" class="envnoleaf" type="box" material="basket" size="{_f(t / 2)} {_f(iw / 2 + t)} {_f(ih / 2)}" pos="{_f(cx - il / 2 - t / 2)} {_f(cy)} {_f(floor_z + ih / 2)}"/>',
         f'{indent}<geom name="{prefix}_wall_py" class="envnoleaf" type="box" material="basket" size="{_f(il / 2)} {_f(t / 2)} {_f(ih / 2)}" pos="{_f(cx)} {_f(cy + iw / 2 + t / 2)} {_f(floor_z + ih / 2)}"/>',
         f'{indent}<geom name="{prefix}_wall_ny" class="envnoleaf" type="box" material="basket" size="{_f(il / 2)} {_f(t / 2)} {_f(ih / 2)}" pos="{_f(cx)} {_f(cy - iw / 2 - t / 2)} {_f(floor_z + ih / 2)}"/>']
    return L


SENSORS = ['    <force name="wrist_force" site="ft_site"/>', '    <torque name="wrist_torque" site="ft_site"/>']


# ----------------------------------------------------------------------------- scenes
def build_rig_pole(cfg: dict, lamp: bool = False) -> str:
    """Pole on a circular track + camera head (mocap bodies) of the pretraining rig (+ status lamp on the head)."""
    lamp_xml_s = ""
    if lamp:
        from pinesim.model.builder_xgh import lamp_xml
        lamp_xml_s = chr(10).join(lamp_xml(cfg, [0.0, 0.0, 0.13], 0.022, "      "))
    cam = cfg["camera"]
    ring = []
    for k in range(48):
        a0, a1 = 2 * math.pi * k / 48, 2 * math.pi * (k + 0.6) / 48
        ring.append(f'    <geom name="ring{k}" class="vis" type="capsule" fromto="{_f(math.cos(a0))} {_f(math.sin(a0))} 0.03 {_f(math.cos(a1))} {_f(math.sin(a1))} 0.03" size="0.006" rgba="0.8 0.8 0.85 0.6"/>')
    return chr(10).join(ring) + f"""
    <body name="rig_pole" mocap="true" pos="1 0 0">
      <geom name="rig_pole_geom" class="vis" type="cylinder" size="0.02 0.5" pos="0 0 0.5" material="steel"/>
      <geom name="rig_pole_foot" class="vis" type="cylinder" size="0.08 0.01" pos="0 0 0.01" material="steel"/>
    </body>
    <body name="rig_head" mocap="true" pos="1 0 1">
      <geom name="rig_head_geom" class="vis" type="box" size="0.03 0.025 0.04" pos="0 0 0.045" rgba="0.1 0.1 0.1 1"/>
      <camera name="rig_cam" pos="0 0 0" xyaxes="1 0 0 0 1 0" fovy="{_f(cam['fov_vertical_deg'])}"/>
      <site name="rig_cam_site" pos="0 0 0" size="0.01" rgba="0 0 1 1"/>
{lamp_xml_s}
    </body>"""


def build_rig_light(cfg: dict) -> tuple[str, dict]:
    """Old pretraining rig: one procedural pineapple at the origin, pole + camera head."""
    plants, eq = plants_xml(cfg, 1, [(0.0, 0.0)])
    xml = header("stargaze_rig", timestep=cfg["control"].get("timestep", 0.001)) + f"""
  <worldbody>
    <light name="sun" pos="1 -1 4" dir="-0.2 0.3 -1" directional="true" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2" castshadow="false"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" contype="1" conaffinity="1"/>
{build_rig_pole(cfg)}
{plants}
  </worldbody>
  <equality>
{eq}
  </equality>
</mujoco>
"""
    return xml, dict(mode="rig_light", n_plants=1)


def field_positions(cfg: dict) -> list:
    f = cfg["field"]
    out = []
    n = int(f["plants_per_row"])
    for k in range(n):        # left row (y = +offset) first, right row staggered by row_stagger
        out.append((f["first_plant_x"] + k * f["plant_spacing"], +f["row_offset_y"]))
        out.append((f["first_plant_x"] + f["row_stagger"] + k * f["plant_spacing"], -f["row_offset_y"]))
    return out


def build(mode: str, n_plants: int = 1, cfg: dict | None = None, out_dir: Path = OUT_DIR) -> Path:
    """Write outputs/models/<mode>[_n].xml (+ meta json) and return the XML path."""
    cfg = cfg or load_config()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from pinesim.model import builder_xgh as bx
    if mode == "rig":
        xml, meta = bx.build_rig_real(cfg)
        name = "rig"
    elif mode == "rig_light":
        xml, meta = build_rig_light(cfg)
        name = "rig_light"
    elif mode in ("platform", "platform_detail"):
        xml, meta = bx.build_platform_u(cfg, n_plants, detail=(mode == "platform_detail"))
        name = f"{mode}_{n_plants}"
    elif mode in ("xgh_light", "xgh_detail"):
        xml, meta = bx.build_xgh(cfg, detail=(mode == "xgh_detail"))
        name = f"{mode}_{meta['n_slots']}"
    else:
        raise ValueError(mode)
    path = out_dir / f"{name}.xml"
    import os
    tmp = path.with_name(f".{name}.{os.getpid()}.xml")          # atomic: parallel runs may build the same scene
    tmp.write_text(xml, encoding="utf-8")
    os.replace(tmp, path)
    mp = out_dir / f".{name}_meta.{os.getpid()}.json"
    mp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    os.replace(mp, out_dir / f"{name}_meta.json")
    return path


if __name__ == "__main__":
    import mujoco
    for mission, mode, n in (("pretrain", "rig", 1), ("mission1", "platform", 1), ("mission1", "platform_detail", 1),
                             ("mission2", "platform", 2), ("mission3", "xgh_light", 0), ("mission3", "xgh_detail", 0)):
        p = build(mode, n, load_config(mission))
        m = mujoco.MjModel.from_xml_path(str(p))
        print(f"{p.name}: nq={m.nq} nv={m.nv} nbody={m.nbody} ngeom={m.ngeom} nu={m.nu} neq={m.neq} ncam={m.ncam}")
