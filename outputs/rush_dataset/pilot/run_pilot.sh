#!/bin/bash
# Pilot of the rush-hour dataset: held-out test scenes first (for the baseline eval), then 3 train scenes.
set -e
cd /project/khanhnt/SocialNav/ABot-Navigation
PY=/project/khanhnt/SocialNav/.venv/bin/python
OUT=/project/khanhnt/SocialNav/outputs/rush_dataset/pilot
$PY -m abotn_evaluator.crowd.datagen --out $OUT --split test --episodes 8 2>&1 | grep --line-buffered "^\[datagen\]\|Error\|Traceback"
$PY -m abotn_evaluator.crowd.export --root $OUT --split test > $OUT/export_test.log 2>&1
echo "[pilot] test split done"
$PY -m abotn_evaluator.crowd.datagen --out $OUT --split train --scenes 20260316153248,cross1,park1 --episodes 12 2>&1 | grep --line-buffered "^\[datagen\]\|Error\|Traceback"
$PY -m abotn_evaluator.crowd.export --root $OUT --split train > $OUT/export_train.log 2>&1
echo "[pilot] train split done"
