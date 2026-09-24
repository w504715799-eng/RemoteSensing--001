#!/usr/bin/env bash
# 04_baseline_eval.sh — 用 O2-RTDETR 官方权重评测 DOTA **val 切分**(不训练, sanity check)
# 用法: bash scripts/04_baseline_eval.sh [r18|r34|r50]
#
# 说明:
#   - O2-RTDETR 官方配置用 mmrotate 的 DOTADataset + DOTAMetric, 直接读切分后的
#     txt annfiles(不需要 COCO json); 官方 val_dataloader 指向 trainval, 我们
#     覆盖为 val/ 切分(本地公平协议: train 训练 / val 评测)
#   - 官方权重在 val 上的数字只作"框架跑通+sanity check"; 正式对比基线 = 本地重训
#   - 依赖: 03_prep_dota.sh val 已执行(生成 split_ss_dota/val/{images,annfiles})
set -euo pipefail

REPRO=/root/rivermind-data/repro
AI4RS=$REPRO/ai4rs
WEIGHTS=$REPRO/weights
RUN=${1:-r18}

[ -d "$AI4RS/data" ] || ln -s "$REPRO/data" "$AI4RS/data"   # data_root='data/split_ss_dota/' 可解析

case "$RUN" in
  r18) DIR=o2_rtdetr_r18vd_2xb4_72e_dota ;;
  r34) DIR=o2_rtdetr_r34vd_2xb4_72e_dota ;;
  r50) DIR=o2_rtdetr_r50vd_2xb4_72e_dota ;;
  *) echo "unknown: $RUN"; exit 1;;
esac
CKPT="$WEIGHTS/$DIR-epoch_72.pth"
[ -f "$CKPT" ] || { echo "权重不存在: $CKPT, 先运行 02"; ls "$WEIGHTS"; exit 1; }

OUT_DIR=$REPRO/experiments/baseline_${RUN}_val
mkdir -p "$OUT_DIR"
cd "$AI4RS"

# 1) 生成 val 评测覆盖配置(继承官方配置, 只替换数据集根与切分)
EVAL_CFG="$OUT_DIR/o2_${RUN}_eval_val.py"
cat > "$EVAL_CFG" <<PY
from mmengine.config import read_base
with read_base():
    from .o2_${RUN}_2xb4_72e_dota import *
# 覆盖 val/test 数据源为 val 切分(官方是 trainval 或 test 提交)
val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val/annfiles/',
        data_prefix=dict(img_path='val/images/'),
        test_mode=True,
        pipeline=val_pipeline))
val_evaluator = dict(type='DOTAMetric', metric='mAP')
test_dataloader = val_dataloader
test_evaluator = val_evaluator
PY

echo "==> 评测 $RUN 官方权重 @ DOTA val"
echo "config: $EVAL_CFG"
echo "ckpt  : $CKPT"
PYTHONPATH=. python tools/test.py "$EVAL_CFG" "$CKPT" \
  --out "$OUT_DIR/preds.pkl" \
  2>&1 | tee "$OUT_DIR/eval.log"

echo "==> 指标(AP50/mAP)"
grep -iE "mAP|AP50|^.*mAP" "$OUT_DIR/eval.log" | tail -10 || true
echo "结果目录: $OUT_DIR"