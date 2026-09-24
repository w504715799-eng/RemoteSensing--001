#!/usr/bin/env bash
# 05_train_ablation.sh — 单卡训练模板(自定义创新配置, 供消融实验使用)
# 用法: bash scripts/05_train_ablation.sh <config.py> <run_name>
# 注意: 单卡 RTX 3090 24GB; 若 config 为多卡(如 2xb4), 需先改成单卡 batch/流程:
#   - 训练: python tools/train.py <config>
#   - 数据根: data/split_ss_dota (trainval/val)
set -euo pipefail

REPRO=/data/repro
AI4RS=$REPRO/ai4rs
CFG=${1:?config 路径必填}
RUN=${2:?run_name 必填}

OUT=$REPRO/experiments/$RUN
mkdir -p "$OUT"
cd "$AI4RS"

cp "$CFG" "$OUT/config_$(basename "$CFG")"   # 留档, 防止后续误改

echo "==> 训练 $RUN (config=$CFG)"
PYTHONPATH=. python tools/train.py "$CFG" \
  --work-dir "$OUT" 2>&1 | tee "$OUT/train.log"

echo "==> 评测(按 04 的 val 协议)"
PYTHONPATH=. python tools/test.py "$CFG" \
  "$OUT/latest.pth" \
  --out "$OUT/preds.pkl" 2>&1 | tee "$OUT/eval.log"
grep -E "AP50|mAP" "$OUT/eval.log" | tail -10 || true
echo "DONE: $OUT"