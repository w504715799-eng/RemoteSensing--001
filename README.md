# RemoteSensing--001

遥感/无人机影像 **重叠目标识别**（dense / overlapping / oriented targets）复现与应用创新仓库。

目标：复现一篇满足以下筛选条件的基准论文 → 用其公开权重作为基线（不训练）→ 通过**应用创新**（把多个论文的优点集中到一起）取得积极增益 → 冲刺中科院 SCI 二区及以上。

## 筛选条件（论文选择）

| 条件 | 要求 | 本仓库选择 |
| --- | --- | --- |
| 发表时间 | 2026 年 | ✅ IEEE TGRS 2026（DOI: 10.1109/TGRS.2026.3671683） |
| 分区 | 中科院二区及以上 | ✅ IEEE TGRS（遥感顶刊，中科院一/二区） |
| 公开代码 | 有 | ✅ [wokaikaixinxin/ai4rs](https://github.com/wokaikaixinxin/ai4rs)（MMRotate 系），项目页 [O2-RT-DETR](https://github.com/wokaikaixinxin/O2-RT-DETR) |
| 公开数据集验证 | 有 | ✅ DOTA-v1.0 / DOTA-v1.5 / DIOR-R / FAIR1M-v1.0 / DroneVehicle / RSAR |
| 显存 | 约 24GB | ✅ RT-DETR 系实时检测器，R18/34/50vd @ 1024×1024，单卡 24GB 可训可测 |
| 现有权重作基准 | 有 | ✅ ModelScope 官方权重（epoch_72.pth 等，见 [scripts/02_fetch_baseline_weights.sh](scripts/02_fetch_baseline_weights.sh)） |

**基准论文：O2-RT-DETR — "Real-Time Oriented Object Detection Transformer in Remote Sensing Images"（IEEE TGRS 2026）**
三大贡献：① Angle Distribution Refinement（角度分布细化回归）；② Chamfer Distance Cost（顶点集匹配代价）；③ Oriented Contrastive Denoising（朝向对比去噪，训练稳定化）。面向遥感任意朝向、密集重叠目标（船只、飞机、车辆、建筑物）。

**创新来源（组合优点，不一刀切地重做）：**

| 来源论文 | 优点 | 拟引入组件 |
| --- | --- | --- |
| [FSDETR (IJCNN 2026)](https://github.com/YT3DVision/FSDETR) | 频域-空间特征金字塔（FSFPN）+ 形变注意力缓解密集遮挡（DA-AIFI），小目标/遮挡下游明显增益 | 频域增强 FPN 模块（插入 O2-RTDETR neck） |
| [FAA: Fourier Angle Align (CVPR 2026)](https://github.com/wokaikaixinxin/ai4rs)（同仓库） | 傅里叶角度对齐，频域角度表征 | 角度表征增强（与 Angle Distribution Refinement 协同） |
| [DEIM / O2-DEIM (同一仓库)](https://github.com/wokaikaixinxin/ai4rs) | 解耦一对一/多对多匹配，免二分匹配冷启动，收敛快、精度高 | 匹配策略替换/对比实验 |
| PKINet / LSKNet（同仓库 backbone 库） | 遥感骨干网络，多尺度感受野更好 | backbone 消融 |

创新路线一句话：**O2-RTDETR 的朝向建模 + FSDETR 的频域空间增强 + （可选）DEIM 匹配/FAA 角度对齐 → 面向密集重叠/小目标的实时朝向检测器**，在 DOTA-v1.0 / DOTA-v1.5 / DIOR-R 上验证积极增益。

## 仓库结构

```
RemoteSensing--001/
├── README.md                  # 本文件
├── docs/
│   ├── paper_survey.md        # 论文调研与筛选证据
│   ├── experiment_plan.md     # 复现 + 创新实验计划（含实验矩阵）
│   └── progress_log.md        # 进度日志（每轮记录）
├── scripts/                   # 云端复现脚本（在 GPU 服务器上执行）
│   ├── 01_env_setup.sh        # conda 环境 + ai4rs 框架安装
│   ├── 02_fetch_baseline_weights.sh  # 从 ModelScope 下载官方权重（基线）
│   ├── 03_prep_dota.sh        # DOTA 数据下载/切分/COCO 转换
│   ├── 04_baseline_eval.sh    # 用官方权重评测，产出基线指标
│   ├── 05_train_ablation.sh   # （后续）单卡训练脚本模板
│   └── check_env.py           # 环境自检
├── experiments/               # 实验输出（日志、指标汇总，均为文本）
└── .gitignore
```

## 云端环境（GPU 服务器）

- `ssh root@sx01-ssh.gpuhome.cc -p 30214 -i D:\tmp\001\.ssh\yolo-sx01`
- 1× RTX 3090 24GB / 88 核 / 251GB RAM；`/data` 剩余约 355GB（数据与虚拟环境都放 `/data`）
- 服务器**无法访问 github.com**（pypi / modelscope / hf-mirror / tsinghua / baidu 可达）→ 代码通过本机打包 scp 上传（见 [docs/experiment_plan.md](docs/experiment_plan.md)），权重走 ModelScope，数据走 Baidu/HF 镜像
- 本机**不下载数据集与模型权重**，全部在服务器上完成

## 快速开始（服务器）

```bash
# 1) 环境
bash /data/repro/scripts/01_env_setup.sh 2>&1 | tee /data/repro/logs/env_setup.log
# 2) 基线权重
bash /data/repro/scripts/02_fetch_baseline_weights.sh
# 3) 数据
bash /data/repro/scripts/03_prep_dota.sh
# 4) 基线评测
bash /data/repro/scripts/04_baseline_eval.sh
```

## 里程碑

- [x] 论文调研与基准选定（第 1 轮）
- [ ] 服务器环境 + 框架安装（第 2 轮）
- [ ] 官方权重基线评测（DOTA-v1.0 AP50 复现 ≈ 77.31@R18 / 78.45@R50）
- [ ] 创新模块设计 + 消融实验
- [ ] 论文实验补齐（DOTA-v1.5 / DIOR-R）+ 写作