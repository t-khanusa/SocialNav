#!/bin/bash
# original -> normal -> rush, 15 tasks each, same policy, same render server (crowd sim v1).
cd /project/khanhnt/SocialNav
for cond in original normal rush; do
  echo "=== $cond $(date) ==="
  CROWD_CONDITION=$cond CROWD_SEED=0 CUDA_VISIBLE_DEVICES=2 \
  PYTHONPATH=/project/khanhnt/SocialNav:/project/khanhnt/SocialNav/ABot-Navigation \
  .venv/bin/python -m abotn_evaluator.point_goal.runner \
    --agent-module utils.abotn_agent:SocialNavPointGoalAgent --agent-config configs/abotn_socialnav.yaml \
    --evaluator-module abotn_evaluator.crowd.evaluator_v1:CrowdPointGoalEvaluator \
    --data-dir data/ABotN-PointBench/Outdoor/annotations --map-dir data/ABotN-PointBench/Outdoor/occmaps \
    --render-url http://127.0.0.1:7036/render_gs --output-dir outputs/crowd_eval/$cond --mode outdoor \
    > outputs/crowd_eval/$cond.log 2>&1
  echo "=== done $cond $(date) exit=$? ==="
done
