# 论文调研与筛选（第 1 轮，2026-09-24）

研究对象：**遥感影像 / 无人机影像的重叠目标识别**（密集、遮挡、任意朝向的船只/飞机/车辆/建筑物）。

筛选条件：2026 年发表 · 中科院二区及以上 · 有公开代码 · 公开数据集验证 · 适合约 24GB 显存 ·（基准用论文已有权重，不训练）。

## 候选论文汇总与筛选举证

### 1. O2-RT-DETR（✅ 选定为基准）

- **论文**：Ding Zeyu 等, *Real-Time Oriented Object Detection Transformer in Remote Sensing Images*, **IEEE TGRS 2026**, vol.64, DOI: 10.1109/TGRS.2026.3671683. arXiv: [2603.15497](https://arxiv.org/abs/2603.15497), IEEE: [11424629](https://ieeexplore.ieee.org/document/11424629)
- **代码**：[wokaikaixinxin/ai4rs](https://github.com/wokaikaixinxin/ai4rs)（MMRotate 系，`projects/rotated_rtdetr/`）+ 项目页 [O2-RT-DETR](https://github.com/wokaikaixinxin/O2-RT-DETR)
- **验证数据集**（全部公开）：DOTA-v1.0、DOTA-v1.5、DIOR-R、FAIR1M-v1.0、DroneVehicle（RGB/IR）、RSAR
- **官方权重**：ModelScope `wokaikaixinxin/ai4rs`（每数据集 R18/34/50vd 的 epoch_72.pth 等）
- **显存**：RT-DETR 系（R18vd@1024² 约 14-30M 参数量级；R50vd@1024²），单卡 24GB 可训可测（官方 2×4 batch，单卡降至 2-4 batch 即可）
- **达标**：2026 ✓ / TGRS 中科院一/二区 ✓ / 代码 ✓ / 数据集 ✓ / 24GB ✓ / 权重即用 ✓
- **为什么适合"重叠目标"**：显式建模物体任意朝向（角度分布细化回归），用 Chamfer 距离做顶点集匹配，直接针对遥感密集重叠目标的旋转框检测；DOTA 本身就是密集小目标（船只、车辆）互相重叠的代表性基准。
- **复现要点**：
  - 训练：`bash tools/dist_train.sh projects/rotated_rtdetr/configs/o2_rtdetr_r18vd_2xb4_72e_dota.py 2`（单卡用 `python tools/train.py`）
  - 评测：`bash tools/dist_test.sh .../o2_rtdetr_r18vd_2xb4_72e_dota.py <ckpt> 2`
  - 权重：`https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r18vd_2xb4_72e_dota/epoch_72.pth`
  - DOTA 需要：原图切 1024×1024（overlap 200）→ `tools/data/dota/dota2coco.py` 生成 COCO 标注
  - 报告指标（单尺度 DOTA-v1.0）：R18vd 77.31 / R34vd 78.13 / R50vd 78.45（AP50）

### 2. FSDETR（✅ 选为创新来源 #1：频域-空间增强）

- **论文**：Huang Jianchao 等, *FSDETR: Frequency-Spatial Feature Enhancement for Small Object Detection*, **IJCNN 2026**, arXiv: [2604.14884](https://arxiv.org/abs/2604.14884)
- **代码**：[YT3DVision/FSDETR](https://github.com/YT3DVision/FSDETR)（基于 RT-DETR 基线）
- **数据集**：VisDrone 2019（APS 13.9%）、TinyPerson（AP50 tiny 48.95%），14.7M 参数
- **可借鉴优点**：① Spatial Hierarchical Attention Block（SHAB，局部细节+全局依赖）；② Deformable Attention 的 DA-AIFI 缓解密集场景遮挡；③ **FSFPN 频域-空间特征金字塔**（频域滤波+空间边缘提取，CFSB 交叉域模块）——对密集小目标细节保持是明确增益点，与 O2-RTDETR 的朝向建模互补。
- 注：IJCNN 为会议，故不作基准，只取其模块优点做应用创新。

### 3. FAA: Fourier Angle Align（✅ 备选创新来源 #2：频域角度对齐）

- 同属 ai4rs 仓库（CVPR 2026，`projects/FAA/`），傅里叶域角度对齐——天然可和 O2-RTDETR 的角度分布细化协同，且在同一代码库内，改动最小。

### 4. DEIM / O2-DEIM（✅ 备选创新来源 #3：匹配策略）

- 同一仓库 `projects/rtdetr`（O2-DEIM 多尺度 DOTA-v1.0 AP50 79.49 @ R18）。解耦一对一/多对多分配，免二分匹配冷启动 → 训练更稳收敛更快。可做匹配策略消融。

### 5. 未选/待考察

| 论文 | 未选原因 |
| --- | --- |
| D2M-DETR（IEEE 2026, 11664569） | 检索未见公开代码 |
| DeCenter: Density-Center（IEEE 2026, 11430630） | 检索未见公开代码 |
| CVIU 2026 轻量 UAV 小目标框架（S1077314226001025） | 未见公开代码 |
| TAM-TR（ISPRS 2025） | 多模态文本引导，任务偏离"重叠目标识别"，且非 2026 |
| RT-DETR（CVPR 2024） | 非朝向检测，且非 2026（作为 O2-RTDETR 的底层基线隐含其内） |

## 结论

基准 = **O2-RT-DETR (TGRS 2026)**；创新 = 其朝向建模 + FSDETR 频域空间增强（+ 可选 FAA/DEIM 对照）。全部可在单一代码库（ai4rs）与单一数据管线（DOTA 系）内完成，避免"局部反复优化"陷阱，直接按实验矩阵推进。