#!/usr/bin/env bash
# 02_fetch_baseline_weights.sh — 从 ModelScope 下载 O2-RTDETR 官方权重（基准，不训练）
# 用法: bash scripts/02_fetch_baseline_weights.sh [variant]
#   variant: r18 (默认) | r34 | r50   —— 对应 DOTA-v1.0 单尺度官方权重
set -euo pipefail

REPRO=/root/rivermind-data/repro
WEIGHTS=$REPRO/weights
mkdir -p "$WEIGHTS"

# ModelScope 仓库 model id 与文件路径（root 为 wokaikaixinxin/ai4rs 的 master 分支）
# 参考: https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr
MODEL=modelscope.cn/models/wokaikaixinxin/ai4rs

case "${1:-r18}" in
  r18) DIR=o2_rtdetr_r18vd_2xb4_72e_dota ;;
  r34) DIR=o2_rtdetr_r34vd_2xb4_72e_dota ;;
  r50) DIR=o2_rtdetr_r50vd_2xb4_72e_dota ;;
  *) echo "unknown variant: $1"; exit 1;;
esac

URL="https://$MODEL/resolve/master/o2_rtdetr/$DIR/epoch_72.pth"
OUT="$WEIGHTS/$DIR-epoch_72.pth"

echo "==> 下载 $URL"
if [ -f "$OUT" ]; then echo "已存在: $OUT ($(du -h "$OUT" | cut -f1))"; else
  curl -fL --retry 3 -o "$OUT" "$URL" || {
    echo "直链下载失败(可能需要登录/modelscope CLI), 尝试 modelscope CLI ..."
    pip install -q modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
    modelscope download --model wokaikaixinxin/ai4rs \
      --include "o2_rtdetr/$DIR/epoch_72.pth" --local_dir "$WEIGHTS/modelscope" || exit 1
    OUT="$WEIGHTS/modelscope/o2_rtdetr/$DIR/epoch_72.pth"
  }
fi
ls -l "$OUT"
echo "DONE. 下一步: bash scripts/03_prep_dota.sh"