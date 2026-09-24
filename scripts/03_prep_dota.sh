#!/usr/bin/env bash
# 03_prep_dota.sh — DOTA-v1.0 数据流式准备(磁盘受限版):
#   下载 -> 解压 -> 切 1024x1024(jpg) -> 删除解压原始图与压缩包, 只保留补丁
# 用法: bash scripts/03_prep_dota.sh [val|train|all]
#   val   (默认): 只处理 val (第2轮基线评测需要)
#   train: 只处理 train (训练用)
#   all  : 两者都处理
#
# 数据源: HF isaaccorley/dota (CC-BY-NC-4.0, 研究用途), 经 hf-mirror.com 可达(服务器实测)
# 协议:   本地训练=train，评测=val（无标注泄漏）；官方 test 无公开标注, 论文数字走官方评测服务器
# 磁盘:   流式处理, 任意时刻峰值 < 25G (rivermind 可用 31G)
set -euo pipefail

REPRO=/root/rivermind-data/repro
AI4RS=$REPRO/ai4rs
DATA=$REPRO/data
RAW=$DATA/DOTA            # 临时: 解压后的原始图(处理完即删)
SPLIT=$DATA/split_ss_dota # 持久: jpg 补丁 + annfiles
HFD=https://hf-mirror.com/datasets/isaaccorley/dota/resolve/main
EXT=.jpg                  # img_split.py 支持 ./jpg (官方默认 png, 用 jpg 省 ~60% 磁盘)

MODE=${1:-val}
cd "$AI4RS"

split_subset () {  # $1=子集名(train|val) $2=tar列表
  local subset=$1; shift
  local SL=$SPLIT/$subset
  if [ -d "$SL/images" ] && [ -n "$(ls -A "$SL/images" 2>/dev/null)" ]; then
    echo "==> $subset 已切分完成, 跳过"; return
  fi
  echo "==> 下载 $subset 图像/标注包"
  for f in "$@"; do
    [ -f "$DATA/$f" ] || curl -fL --retry 3 -C - -o "$DATA/$f" "$HFD/$f"
  done
  echo "==> 解压 $subset 到 $RAW/$subset (临时)"
  local D=$RAW/$subset
  mkdir -p "$D/images" "$D/labelTxt"
  for f in "$@"; do
    case "$f" in
      *_images_*) tar -xzf "$DATA/$f" -C "$D/images";;
      *_annotations_*) tar -xzf "$DATA/$f" -C "$D/labelTxt";;
    esac
  done
  echo "    images=$(ls "$D/images" | wc -l) labels=$(ls "$D/labelTxt" | wc -l)"

  echo "==> 切分 $subset -> $SL (1024x1024, gap 200, jpg)"
  cat > "$DATA/split_${subset}.json" <<JSON
{
  "nproc": 32,
  "img_dirs": ["$RAW/$subset/images/"],
  "ann_dirs": ["$RAW/$subset/labelTxt/"],
  "sizes": [1024],
  "gaps": [200],
  "rates": [1.0],
  "img_rate_thr": 0.6,
  "iof_thr": 0.7,
  "no_padding": false,
  "padding_value": [104, 116, 124],
  "save_dir": "$SL/",
  "save_ext": "$EXT"
}
JSON
  python tools/data/dota/split/img_split.py --base-json "$DATA/split_${subset}.json"

  echo "==> 回收空间: 删除 $subset 原始图与压缩包(只留补丁)"
  rm -rf "$D"
  for f in "$@"; do rm -f "$DATA/$f"; done
  echo "    $subset 补丁: $(ls "$SL/images" | wc -l) 张"
  df -h "$REPRO" | tail -1
}

case "$MODE" in
  val)
    split_subset val \
      dotav1.0_images_val.tar.gz \
      dotav1.0_annotations_val.tar.gz
    ;;
  train)
    split_subset train \
      dotav1.0_images_train.tar.gz \
      dotav1.0_annotations_train.tar.gz
    ;;
  all)
    split_subset val \
      dotav1.0_images_val.tar.gz \
      dotav1.0_annotations_val.tar.gz
    split_subset train \
      dotav1.0_images_train.tar.gz \
      dotav1.0_annotations_train.tar.gz
    ;;
  *) echo "unknown mode: $MODE"; exit 1;;
esac

echo "==> 生成 COCO 标注 json"
for s in "$MODE"; do
  [ "$s" = "all" ] && for s2 in train val; do
    [ -d "$SPLIT/$s2/images" ] && python tools/data/dota/dota2coco.py \
      "$SPLIT/$s2" "$SPLIT/$s2.json"
  done && continue
  [ -d "$SPLIT/$s/images" ] && python tools/data/dota/dota2coco.py \
    "$SPLIT/$s" "$SPLIT/$s.json"
done

echo "DONE. 持久占用:"; du -sh "$SPLIT" 2>/dev/null || du -sh "$SPLIT"/* 2>/dev/null