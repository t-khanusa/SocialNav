#!/bin/bash
# Crowd simulator v2 (raw occupancy map, Helbing-Molnar potential + body repulsion + hard
# bodies, Weidmann speeds): run_v2.sh <seed> <cuda_device> <condition>...
# Writes outputs/crowd_eval_v2/seed<seed>/<condition>/... and logs next to it.
cd /project/khanhnt/SocialNav
seed=$1; gpu=$2; shift 2
mkdir -p outputs/crowd_eval_v2/seed$seed
for cond in "$@"; do
  echo "=== seed $seed $cond $(date) ==="
  CROWD_CONDITION=$cond CROWD_SEED=$seed CUDA_VISIBLE_DEVICES=$gpu \
  PYTHONPATH=/project/khanhnt/SocialNav:/project/khanhnt/SocialNav/ABot-Navigation \
  .venv/bin/python -m abotn_evaluator.point_goal.runner \
    --agent-module utils.abotn_agent:SocialNavPointGoalAgent --agent-config configs/abotn_socialnav.yaml \
    --evaluator-module abotn_evaluator.crowd.evaluator:CrowdPointGoalEvaluator \
    --data-dir data/ABotN-PointBench/Outdoor/annotations --map-dir data/ABotN-PointBench/Outdoor/occmaps \
    --render-url http://127.0.0.1:7036/render_gs --output-dir outputs/crowd_eval_v2/seed$seed/$cond --mode outdoor \
    > outputs/crowd_eval_v2/seed$seed/$cond.log 2>&1
  echo "=== done seed $seed $cond $(date) exit=$? ==="
done
