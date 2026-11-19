#!/usr/bin/env bash
# Full v5 chain: T2 diffusion pretraining -> M1 (light) -> M2 (light) -> M3 (XGH light), weight transfer, ONNX export.
# usage: scripts/run_curriculum.sh [m1_steps] [m2_steps] [m3_steps]
# Steps already done are skipped (existing dataset / model / checkpoint); delete them to redo a step.
set -e
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl
M1=${1:-30000}; M2=${2:-20000}; M3=${3:-15000}
log() { echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p outputs/logs outputs/pretrain models

[ -d assets_xgh ] || { log "realistic assets"; python3 scripts/make_xgh_assets.py --src "${MESH_SRC:?set MESH_SRC=/path/to/mesh_pineapple}"; }
if [ ! -f models/t2_diffusion.pt ]; then
  log "T2 dataset (2 processes)"
  python3 pretrain/make_t2_dataset.py --scenes 450 --seed0 0 --out outputs/pretrain/t2_data_0.npz > outputs/logs/t2_data_0.log 2>&1 &
  python3 pretrain/make_t2_dataset.py --scenes 450 --seed0 1 --out outputs/pretrain/t2_data_1.npz > outputs/logs/t2_data_1.log 2>&1 &
  wait
  log "T2 diffusion training"
  python3 pretrain/train_t2_diffusion.py > outputs/logs/t2_train.log 2>&1
fi
train() {
  if [ -f checkpoints/$1/final_model.zip ]; then log "$1: checkpoint exists, skipped"; return; fi
  log "$1 training ($2 steps)"
  python3 scripts/train_mission.py --mission $1 --timesteps $2 ${3:+--n-envs $3} > outputs/logs/train_$1.log 2>&1
}
train mission1 $M1 2
train mission2 $M2 2
train mission3 $M3 1
log "done; evaluation: scripts/run_final_eval.sh"
