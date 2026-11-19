#!/usr/bin/env bash
# usage: scripts/eval_parallel.sh <mission> <first_seed> <n_per_proc> <tag> [extra args]
# runs two evaluation processes (two CPU cores) and merges their JSON summaries
set -e
M=$1; S=$2; N=$3; T=$4; shift 4
export MUJOCO_GL=egl
python3 scripts/eval_mission.py --mission $M --seeds $S $N --tag ${T}_a "$@" > outputs/eval_${M}_${T}_a.log 2>&1 &
python3 scripts/eval_mission.py --mission $M --seeds $((S+N)) $N --tag ${T}_b "$@" > outputs/eval_${M}_${T}_b.log 2>&1 &
wait
python3 scripts/merge_eval.py outputs/eval/${M}_${T}_a.json outputs/eval/${M}_${T}_b.json --out outputs/eval/${M}_${T}.json
