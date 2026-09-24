#!/usr/bin/env python
"""check_env.py — ai4rs 环境自检(在 01_env_setup.sh 结尾调用)"""
import importlib
import sys
import torch

print(f"python      : {sys.version.split()[0]}")
print(f"cuda avail  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu         : {torch.cuda.get_device_name(0)} / {torch.cuda.get_device_properties(0).total_memory // 2**20} MiB")

for pkg in ("mmcv", "mmengine", "mmdet", "mmseg", "mmrotate", "rtdetr", "o2"):
    try:
        m = importlib.import_module(pkg)
        print(f"{pkg:<10}: {getattr(m, '__version__', '?')} @ {getattr(m, '__file__', '?')}")
    except ImportError as e:  # noqa
        print(f"{pkg:<10}: MISSING ({e})")

# onnx / 其它可选依赖(不阻塞)
for pkg in ("numpy", "cv2", "yaml", "tqdm"):
    try:
        m = importlib.import_module(pkg)
        print(f"{pkg:<10}: {getattr(m, '__version__', 'ok')}")
    except ImportError:
        print(f"{pkg:<10}: MISSING")

print("OK" if torch.cuda.is_available() else "WARNING: CUDA unavailable")