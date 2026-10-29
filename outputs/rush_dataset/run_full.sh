#!/bin/bash
# Full rush-hour dataset: one render server and one datagen worker per GPU, the scenes of a split
# divided between the workers (a server holds one scene at a time, datagen goes scene by scene).
#   bash run_full.sh <out> <split> <episodes per scene> <gpu,gpu,...> [first episode]
# e.g. bash run_full.sh outputs/rush_dataset/v1 train 160 0,1,2      (~50k samples, ~3 h on 3 GPUs)
#      bash run_full.sh outputs/rush_dataset/v1 test 24 0,1,2
# Re-running resumes: finished episodes (episode.json present) are skipped.
set -e
OUT=$(realpath -m "$1"); SPLIT=$2; EPISODES=$3; IFS=, read -ra GPUS <<< "$4"; FIRST=${5:-0}
ROOT=/project/khanhnt/SocialNav
PY=$ROOT/.venv/bin/python
N=${#GPUS[@]}
mkdir -p "$OUT/logs"

servers=()
cleanup() { kill "${servers[@]}" 2>/dev/null || true; }
trap cleanup EXIT
for i in "${!GPUS[@]}"; do
  port=$((7040 + i))
  (cd $ROOT/ABot-Navigation/render_server && CUDA_VISIBLE_DEVICES=${GPUS[$i]} exec $PY render_sim_dynamic.py \
      --port $port --gpus 0 --max_scenes_per_gpu 1 --render_scale 1.0 \
      --scenes_root $ROOT/data/ABotN-PointBench/Outdoor/occmaps --sprite_dir $ROOT/data/pedestrian_sprites \
      > "$OUT/logs/render_$port.log" 2>&1) &
  servers+=($!)
done
for i in "${!GPUS[@]}"; do
  until curl -s -o /dev/null "http://127.0.0.1:$((7040 + i))/"; do sleep 2; done
done
echo "[full] $N render servers up"

cd $ROOT/ABot-Navigation
workers=()
for i in "${!GPUS[@]}"; do
  $PY -m abotn_evaluator.crowd.datagen --out "$OUT" --split $SPLIT --episodes $EPISODES --first-episode $FIRST \
      --worker $i --num-workers $N --render-url "http://127.0.0.1:$((7040 + i))/render_gs" \
      > "$OUT/logs/datagen_${SPLIT}_$i.log" 2>&1 &
  workers+=($!)
done
status=0
for w in "${workers[@]}"; do wait $w || status=1; done
[ $status = 0 ] || { echo "[full] a datagen worker failed, see $OUT/logs"; exit 1; }
echo "[full] $SPLIT episodes done"

$PY -m abotn_evaluator.crowd.export --root "$OUT" --split $SPLIT --absolute-paths > "$OUT/logs/export_$SPLIT.log" 2>&1
$PY -m abotn_evaluator.crowd.qc --root "$OUT" --split $SPLIT --videos 4 > "$OUT/logs/qc_$SPLIT.log" 2>&1
echo "[full] $SPLIT exported: $OUT/sft_$SPLIT.jsonl, QC: $OUT/qc_$SPLIT.json"
