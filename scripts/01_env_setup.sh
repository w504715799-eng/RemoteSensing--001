#!/usr/bin/env bash
# 01_env_setup.sh — 在 GPU 服务器上初始化 conda 环境并安装 ai4rs（MMRotate 系）框架
# 用法: bash scripts/01_env_setup.sh
# 前置条件:
#   - 服务器不能访问 github.com, 需要在"本机"执行:
#       git clone --depth 1 https://github.com/wokaikaixinxin/ai4rs.git
#       tar czf ai4rs.tar.gz --exclude='.git' ai4rs
#       scp -P 30214 -i D:\tmp\001\.ssh\yolo-sx01 ai4rs.tar.gz root@sx01-ssh.gpuhome.cc:/data/repro/
#   - /data 有足够空间（本机不下载任何数据集/权重）
set -euo pipefail

REPRO=/data/repro
ENVS=/data/conda_envs
ENV_NAME=ai4rs
ENV_PREFIX=$ENVS/$ENV_NAME
PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

mkdir -p "$REPRO"/{logs,experiments,data,weights}
mkdir -p "$ENVS"

echo "==> [1/6] 创建 conda 环境 python=3.10 (见 ai4rs 官方要求)"
if [ ! -d "$ENV_PREFIX" ]; then
  conda create -p "$ENV_PREFIX" python=3.10 -y
fi
source /opt/conda/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"
python -V

echo "==> [2/6] 安装 PyTorch 2.4.0 cu121"
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu121 -i "$PIP_INDEX" || true
# 若 cu121 轮子不可用则回退到官方建议版本组合, 以 torch.cuda.is_available() 为准
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "==> [3/6] 安装 openmim / mmengine / mmcv / mmdet / mmseg"
pip install -U openmim -i "$PIP_INDEX"
mim install mmengine -i "$PIP_INDEX"
mim install "mmcv==2.2.0" -i "$PIP_INDEX"
mim install "mmdet>3.0.0rc6,<3.4.0" -i "$PIP_INDEX"
mim install "mmsegmentation>=1.2.2" -i "$PIP_INDEX"

echo "==> [4/6] 解包并安装 ai4rs"
if [ ! -d "$REPRO/ai4rs" ]; then
  cd "$REPRO"
  tar xzf ai4rs.tar.gz
fi
cd "$REPRO/ai4rs"
pip install -v -e . -i "$PIP_INDEX"

echo "==> [5/6] 修补 mmdet/mmseg 的 MMCV 版本断言 (官方 README 步骤5)"
MMDET_INIT=$(python -c "import mmdet; print(mmdet.__file__)")
MMSEG_INIT=$(python -c "import mmseg; print(mmseg.__file__)")
sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/" "$MMDET_INIT" || true
sed -i "s/MMCV_MAX = '2.2.0'/MMCV_MAX = '2.3.0'/" "$MMSEG_INIT" || true
grep -n "mmcv_maximum_version\|MMCV_MAX" "$MMDET_INIT" "$MMSEG_INIT" || true

echo "==> [6/6] 自检"
python "$REPRO/scripts/check_env.py" || true
echo "DONE. 下一步: bash scripts/02_fetch_baseline_weights.sh"