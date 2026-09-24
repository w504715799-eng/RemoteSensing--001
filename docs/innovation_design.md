# 应用创新设计：FS-O2-RTDETR（组合优点，定向增益）

> 原则（避免局部陷阱）：不重新发明轮子。把两篇论文的**已验证优点**组合到一个框架里，
> 用固定实验矩阵一次验证，看到积极增益即可支撑论文，不为单点指标反复调参。

## 1. 组合来源

| 论文 | 已验证优点 | 本设计取用 |
| --- | --- | --- |
| O2-RT-DETR（TGRS 2026，基准） | ① Angle Distribution Refinement（角度分布细化回归）② Chamfer 顶点集匹配代价 ③ Oriented Contrastive Denoising | 保留全部（作为主干），只动 neck/encoder 的增强 |
| FSDETR（IJCNN 2026） | ④ CFSB：**跨域频域-空间块**（频域滤波 + 空间边缘提取，保持细粒度细节）⑤ SHAB：空间层级注意力（局部细节+全局依赖）⑥ DA-AIFI：形变注意力聚集信息区域（缓解密集遮挡） | ④ 为核心引入（FS-FPN）；⑤⑥ 为可选对照 |

## 2. 目标与假设

- **任务**：遥感/无人机影像**密集重叠/任意朝向小目标**的实时检测（DOTA 系：船只、车辆、飞机互相重叠、朝向任意）。
- **假设**：O2-RTDETR 把"朝向"建模好了（角度分支），但对**密集小目标的细粒度特征保持**（频域细节、遮挡下的注意力）仍有空间；FSDETR 的频域-空间增强正是为小目标/遮挡设计，两者**互补不冲突** → 组合应带来正向增益。
- **论文叙事**：实时朝向检测 + 频域-空间特征增强（首个将频域特征金字塔引入朝向检测 transformer 的工作），消融验证各模块增益。

## 3. 方案：FS-O2-RTDETR

```
输入
 └─ ResNet-vd backbone (R18/34/50)
     └─ [不变] ChannelMapper → 3 层 C=256 特征
         └─ [改] FSRTDETRFPN  ← 替代 RTDETRFPN（encoder.fpn_cfg）
             ├─ 顶层 AIFI（自注意力，可加 DA-AIFI 形变采样→ 可选模块 M2）
             ├─ top-down: reduce(1x1) + upsample + FSCSPLayer   ← 模块 M1
             ├─ bottom-up: downsample(3x3 s2) + FSCSPLayer      ← 模块 M1
             └─ YOLO-style 输出 3 层 → [不变] RotatedRTDETRHead
                                        （角度细化 + Chamfer 匹配 + 朝向对比去噪）
```

### 模块 M1：FSCSPLayer（CSP + CFSB 频域-空间分支）—— 核心创新

改造 `RTDETRFPN.csp_block`（原为 mmdet `CSPLayer`）：每个 CSP 块的主分支后并联一个
频域-空间增强分支，输出相加（残差），几乎不增加参数量（轻量 CFSB，通道数不变）。

```python
# 草图（落地时按 ai4rs 的 mmengine 注册风格实现，ConvModule/norm 复用 mmdet）
class CFSB(nn.Module):              # Cross-domain Frequency-Spatial Block
    def __init__(self, channels, hidden=channels // 2):
        super().__init__()
        # 空间支路：local 细节 + 边缘
        self.edge = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, groups=channels // 8),
            nn.Conv2d(hidden, channels, 1))
        # 频域支路：FFT 域可学习门控（rfft2 -> 1x1 conv on magnitude -> irfft2）
        self.freq_gate = nn.Conv2d(channels // 2 + 1, channels // 2 + 1, 1)  # 频域通道门控
        self.fuse = nn.Conv2d(channels, channels, 1)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        # --- 空间支路 ---
        s = self.edge(x)
        # --- 频域支路（高频细节保持）---
        B, C, H, W = x.shape
        Xf = torch.fft.rfft2(x, norm='ortho')               # [B,C,H,W//2+1]
        m = torch.abs(Xf).view(B, C, H, W // 2 + 1)
        g = self.freq_gate(m)                               # 可学习频段门控
        Xf2 = Xf * (1.0 + g.unsqueeze(-1))                  # 幅度加权
        f = torch.fft.irfft2(Xf2, s=(H, W), norm='ortho')
        return self.fuse(self.norm(x + s + f))

class FSCSPLayer(mmdet CSPLayer 子类):  # csp_block = FSCSPLayer
    # 在 CSPLayer 原输出上并联 CFSB 分支：
    #   out = csp_out + cfsb(csp_out)
```

> 落地注意：rfft2 在 1024×1024 上开销可接受（R18 总 FLOPs 占比 <3% 目标）；
> 若训练不稳则退化为"高帽残差"等效表示：`x + (x - GaussianBlur(x))`（同为频域高通，更稳）。

### 模块 M2（可选）：DA-AIFI —— encoder 自注意力改形变采样

参考 FSDETR `DA_AIFI`：AIFI（仅顶层自注意力）中加入可变形偏移采样，聚焦物体区域、
缓解密集遮挡下的注意力分散。实现可用 mmcv `DeformableAttention`，插入 `encoder.layer_cfg`。

### 模块 M3（可选对照）：角度/匹配增强（同仓库现成）

- FAA（CVPR 2026，同仓库 `projects/FAA/`）：傅里叶角度对齐 → 角度表征频域化，与 M1 可形成"频域双通道"叙事；做**对照**而非默认上。
- DEIM（同仓库 `projects/rtdetr`）：解耦一对一/多对多匹配 → 训练稳定对照。

## 4. 实现清单（在 ai4rs 代码库内）

```
ai4rs/projects/rotated_rtdetr/rotated_rtdetr/
├── fs_csp_layer.py        # 新增：FSCSPLayer + CFSB（M1）
├── rtdetr_layers.py       # 修改：RTDETRFPN.csp_block 可通过子类替换
└── (可选) fs_da_aifi.py    # 新增：DA-AIFI（M2）

ai4rs/projects/rotated_rtdetr/configs/
├── o2_rtdetr_r18vd_2xb4_72e_dota_fs.py        # E1: encoder.fpn_cfg 换 FSRTDETRFPN
├── o2_rtdetr_r18vd_2xb4_72e_dota_fs_aifi.py   # E2: +DA-AIFI
└── o2_rtdetr_r18vd_2xb4_72e_dota_fs_deim.py   # E3(可选): +DEIM 匹配
```

训练配置变更点（相对官方 `o2_rtdetr_r18vd_2xb4_72e_dota.py`）：
1. `encoder.fpn_cfg`：`type=RTDETRFPN` → `type=FSRTDETRFPN, csp_block=dict(type=FSCSPLayer)`（或直接新类）
2. 数据集：`_base_` 覆盖为 `dota_train_val.py`（train 切分训练 / val 切分评测），`data_root='data/split_ss_dota/'`，单卡 `batch_size=2`、`num_workers=4`
3. 预训练 backbone 权重沿用官方（ModelScope resnet18vd_pretrained）

## 5. 实验矩阵与成功判据（详见 experiment_plan.md §3）

| 实验 | 内容 | 对比 |
| --- | --- | --- |
| E0' | O2-RTDETR-R18 重训基线（train→val） | 公平基线 |
| E1 | + FSCSPLayer（M1） | vs E0' —— **主创新判定** |
| E2 | + DA-AIFI（M2） | vs E1 |
| E3 | + FAA 或 DEIM（M3，可选） | vs E1/E2 |
| E4 | 最优组合 → R34/R50 | 规模 |

- 判据：E1 相对 E0' 的 **val AP50 ≥ +0.5**（同协议），并在 DOTA-v1.5 / DIOR-R 上 ≥3 数据集正向 → 论文成立。
- 训练协议固定：72e、1024×1024、单卡 bs2、train 训练/val 评测，不做重复微调。

## 6. 风险与对策

| 风险 | 对策 |
| --- | --- |
| rfft2 训练不稳定 | 退化为 `x + (x - blur(x))` 高通残差（数学等价）；或仅高频支路参与梯度（detach 低频） |
| 参数量/FLOPs 超标 | CFSB 用分组卷积 + 单层门控，目标 <3% FLOPs 增加 |
| 与角度分支干扰 | M1 只作用 neck 特征（不动 head/匹配/去噪），消融确认无负向 |
| GPU 被旧项目占用 | 已按用户决定：CPU 部分（环境/数据/代码）先行，GPU 实验等空闲后批量执行 |