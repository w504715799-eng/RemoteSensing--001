#!/usr/bin/env bash
# 04_baseline_eval.sh — 用 O2-RTDETR 官方权重做评测, 产出基线指标(不训练)
# 用法: bash scripts/04_baseline_eval.sh [r18|r34|r50] [val|test]
#   默认: r18 val —— DOTA-v1.0 val 切分(有公开标注)上评估官方权重
#   说明: 论文报告 DOTA test AP50 来自官方评测服务器(无公开 test 标注);
#         本地统一用 val 切分作为前后对比协议, 保证内部一致性。
set -euo pipefail

REPRO=/root/rivermind-data/repro
AI4RS=$REPRO/ai4rs
WEIGHTS=$REPRO/weights
RUN=${1:-r18}
SPLIT=${2:-val}

case "$RUN" in
  r18) DIR=o2_rtdetr_r18vd_2xb4_72e_dota ;;
  r34) DIR=o2_rtdetr_r34vd_2xb4_72e_dota ;;
  r50) DIR=o2_rtdetr_r50vd_2xb4_72e_dota ;;
  *) echo "unknown: $RUN"; exit 1;;
esac
CKPT="$WEIGHTS/$DIR-epoch_72.pth"
[ -f "$CKPT" ] || { echo "权重不存在: $CKPT, 先运行 02"; exit 1; }

CONFIG="projects/rotated_rtdetr/configs/$DIR.py"
[ -f "$CONFIG" ] || CONFIG=$(ls projects/rotated_rtdetr/configs/${DIR}*.py | head -1)

OUT_DIR=$REPRO/experiments/baseline_${RUN}_${SPLIT}
mkdir -p "$OUT_DIR"
cd "$AI4RS"

echo "==> 评测 $RUN 官方权重 @ $SPLIT"
echo "config: $CONFIG"
echo "ckpt  : $CKPT"

PYTHONPATH=. python tools/test.py "$CONFIG" "$CKPT" \
  --out "$OUT_DIR/preds.pkl" \
  2>&1 | tee "$OUT_DIR/eval.log"

echo "==> 提取指标"
grep -E "AP50|mAP|map" "$OUT_DIR/eval.log" | tail -20 || true
echo "结果目录: $OUT_DIR"
tail -30 "$OUT_DIR/eval.log"