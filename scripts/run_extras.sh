#!/usr/bin/env bash
# Extra measurements after the curriculum: stricter leaf-damage limit (0.6 rad = 34 deg), Meshcat recordings, tests.
set -e
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl
log() { echo "[$(date +%H:%M:%S)] $*"; }
M1=$(ls -t models/mission1_v*.onnx | head -1); M2=$(ls -t models/mission2_v*.onnx | head -1); M3=$(ls -t models/mission3_v*.onnx | head -1)
[ -f outputs/eval/mission1_reference_leaf06.json ] || { log "M1 reference, leaf limit 0.6 rad"; scripts/eval_parallel.sh mission1 900000 15 reference_leaf06 --set limits.leaf_bend_rad=0.6; }
[ -f outputs/eval/mission1_trained_leaf06.json ] || { log "M1 policy, leaf limit 0.6 rad"; scripts/eval_parallel.sh mission1 900000 15 trained_leaf06 --onnx $M1 --set limits.leaf_bend_rad=0.6; }
log "Meshcat recordings"
python3 scripts/visualize_meshcat.py --mission mission1 --seed ${M1_SEED:-900005} --onnx $M1 --every 4 --html outputs/meshcat/mission1_seed${M1_SEED:-900005}.html > outputs/viz_m1.log 2>&1 &
python3 scripts/visualize_meshcat.py --mission mission2 --seed ${M2_SEED:-910010} --onnx $M2 --every 5 --html outputs/meshcat/mission2_seed${M2_SEED:-910010}.html > outputs/viz_m2.log 2>&1
wait
python3 scripts/visualize_meshcat.py --mission mission3 --seed ${M3_SEED:-930003} --onnx $M3 --every 12 --html outputs/meshcat/mission3_seed${M3_SEED:-930003}.html > outputs/viz_m3.log 2>&1
log "tests"
python3 -m pytest -q tests/test_core.py > outputs/test_core.log 2>&1 || true
python3 -m pytest -q tests/test_missions_slow.py > outputs/test_slow.log 2>&1 || true
tail -n 3 outputs/test_core.log outputs/test_slow.log
python3 scripts/make_results.py > /dev/null
log "extras done"
