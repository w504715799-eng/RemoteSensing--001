#!/usr/bin/env bash
# 03_prep_dota.sh — DOTA-v1.0: 下载(hf-mirror) -> 整理目录 -> 切 1024x1024 -> COCO 转换
# 用法: bash scripts/03_prep_dota.sh
# 数据源: HF 数据集 isaaccorley/dota (CC-BY-NC-4.0, 研究用途), 经 hf-mirror.com 可达
set -euo pipefail

REPRO=/data/repro
AI4RS=$REPRO/ai4rs
DATA=$REPRO/data
RAW=$DATA/DOTA
SPLIT=$DATA/split_ss_dota
HFD=https://hf-mirror.com/datasets/isaaccorley/dota/resolve/main

cd "$AI4RS"
mkdir -p "$RAW/train" "$RAW/val" "$SPLIT"

echo "==> [1/4] 下载 DOTA-v1.0 (train/val 图像+标注, 共约 13.7GB)"
for f in \
  dotav1.0_images_train.tar.gz \
  dotav1.0_images_val.tar.gz \
  dotav1.0_annotations_train.tar.gz \
  dotav1.0_annotations_val.tar.gz; do
  if [ ! -f "$DATA/$f" ]; then
    echo "  下载 $f ..."
    curl -fL --retry 3 -C - -o "$DATA/$f" "$HFD/$f"
  fi
done

echo "==> [2/4] 解压并整理为 data/DOTA/{train,val}/{images,labelTxt}"
# tar 包内部结构: images/<name>.png 与 labelTxt/<name>.txt
for f in "$DATA"/dotav1.0_images_*.tar.gz; do
  case "$f" in
    *train*) D="$RAW/train";;
    *val*)   D="$RAW/val";;
  esac
  if [ ! -d "$D/images" ]; then
    echo "  解压 $f -> $D/images"; mkdir -p "$D/images"; tar -xzf "$f" -C "$D/images"
  fi
done
for f in "$DATA"/dotav1.0_annotations_*.tar.gz; do
  case "$f" in
    *train*) D="$RAW/train";;
    *val*)   D="$RAW/val";;
  esac
  if [ ! -d "$D/labelTxt" ]; then
    echo "  解压 $f -> $D/labelTxt"; mkdir -p "$D/labelTxt"; tar -xzf "$f" -C "$D/labelTxt"
  fi
done
echo "  图像数量: train=$(ls "$RAW/train/images" | wc -l) val=$(ls "$RAW/val/images" | wc -l)"
echo "  标注数量: train=$(ls "$RAW/train/labelTxt" | wc -l) val=$(ls "$RAW/val/labelTxt" | wc -l)"

echo "==> [3/4] 切分 1024x1024/overlap200 (trainval 用于训练, val 用于本地评测)"
if [ ! -d "$SPLIT/trainval/images" ]; then
  python tools/data/dota/split/img_split.py --base-json \
    tools/data/dota/split/split_configs/ss_trainval.json
fi
if [ ! -d "$SPLIT/val/images" ]; then
  # 自定义 val 切分配置(与 ss_trainval 同参数, 只切 val 并在本地评测)
  cat > "$DATA/ss_val.json" <<'JSON'
{
  "nproc": 10,
  "img_dirs": ["data/DOTA/val/images/"],
  "ann_dirs": ["data/DOTA/val/labelTxt/"],
  "sizes": [1024],
  "gaps": [200],
  "rates": [1.0],
  "img_rate_thr": 0.6,
  "iof_thr": 0.7,
  "no_padding": false,
  "padding_value": [104, 116, 124],
  "save_dir": "data/split_ss_dota/val/",
  "save_ext": ".png"
}
JSON
  python tools/data/dota/split/img_split.py --base-json "$DATA/ss_val.json"
fi

echo "==> [4/4] DOTA txt -> COCO json"
python tools/data/dota/dota2coco.py data/split_ss_dota/trainval data/split_ss_dota/trainval.json
python tools/data/dota/dota2coco.py data/split_ss_dota/val data/split_ss_dota/val.json

echo "DONE."
du -sh "$SPLIT"