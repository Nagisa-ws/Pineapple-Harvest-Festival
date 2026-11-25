#!/usr/bin/env bash
# Version 6 final evaluation: realistic scenes, TEST seeds (not the validation seeds used to select the checkpoint),
# ONNX policy and - for comparison - the reference controller alone, Meshcat HTML of a few episodes, full row of
# Mission 3. usage: scripts/run_final_eval.sh [m1_episodes] [m2_episodes]
set -e
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl
export PINESIM_HARVEST=${PINESIM_HARVEST:-twist}     # no question on the terminal in this unattended run
N1=${1:-30}; N2=${2:-20}
log() { echo "[$(date +%H:%M:%S)] $*"; }
log "Mission 1 realistic: ONNX policy"
python3 scripts/eval_realistic.py --mission mission1 --onnx models/mission1_v1.onnx --seeds 940000 $N1 --record 2 --tag onnx --record-failures
log "Mission 1 realistic: reference controller alone"
python3 scripts/eval_realistic.py --mission mission1 --seeds 940000 $N1 --record 0 --tag referensi
log "Mission 2 realistic: ONNX policy"
python3 scripts/eval_realistic.py --mission mission2 --onnx models/mission2_v1.onnx --seeds 950000 $N2 --record 2 --tag onnx --record-failures
log "Mission 2 realistic: reference controller alone"
python3 scripts/eval_realistic.py --mission mission2 --seeds 950000 $N2 --record 0 --tag referensi
log "Mission 3 realistic: full row (2 x 30 plants, 15 m), reference controller, HTML"
python3 scripts/eval_xgh.py --tag ref_baris_penuh --every 6 --meters-per-part 8 --max-frames-per-part 8000
log "Mission 3 realistic: first 20 plants, ONNX policy"
python3 scripts/eval_xgh.py --onnx models/mission3_v1.onnx --section 0 20 --tag onnx_seksi_000_020 --no-html
log "Mission 3 realistic: another field layout (seed 2027, 2 x 10 plants), reference controller"
python3 scripts/eval_xgh.py --field-seed 2027 --plants-per-row 10 --tag ref_ladang_lain_seed2027 --no-html
log "final eval done: outputs/eval_realistic/*/result.json, outputs/eval_xgh/*/result.json"
