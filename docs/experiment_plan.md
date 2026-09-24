# 实验计划（复现 + 应用创新）

目标导向：先拿到**不训练的官方权重基线**，再叠加组合创新模块，见到积极增益后再扩展数据集与消融。**避免陷入某个指标反复调参的局部陷阱**——用固定实验矩阵一次跑完，统一记录。

## 0. 服务器事实（2026-09-24 探测）

- 连接：`ssh root@sx01-ssh.gpuhome.cc -p 30214 -i D:\tmp\001\.ssh\yolo-sx01`
- 硬件：1× RTX 3090 24GB、88 核、251GB RAM
- 磁盘：`/` 剩余约 12GB（小！）；`/data` 剩余约 355GB（所有大文件放这里）
- 网络：✅ pypi.org / modelscope.cn / hf-mirror.com / pypi.tuna.tsinghua.edu.cn / pan.baidu.com；❌ github.com / drive.google.com / kaggle.com
- 系统：Python 3.12.13（/opt/conda）、torch 2.12.1+cu130（base 环境）、nvcc 13.0、git 2.34
- 目录约定（全新开始，不碰 /workspace 旧代码）：
  - `/data/repro` —— 工作根（代码 + 脚本 + 日志 + 权重）
  - `/data/repro/ai4rs` —— 框架源码（本机打包 scp 上传）
  - `/data/repro/data` —— 数据集（DOTA 原始 + 切分）
  - `/data/conda_envs/ai4rs` —— 独立 conda 环境（py3.10，按官方要求）
  - `/data/repro/experiments` —— 训练/评测输出

## 1. 环境安装（脚本 01）

按 [ai4rs 官方 README](https://github.com/wokaikaixinxin/ai4rs) 的依赖组合（Python 3.10 + PyTorch 2.4.0/cu121 + MMCV 2.2.0 + `mmdet>=3.0.0rc6,<3.4.0` + mmseg），需要两台机器配合：

1. 本机：`git clone --depth 1 https://github.com/wokaikaixinxin/ai4rs` → `tar` 打包 → `scp -P 30214` 上传到 `/data/repro/`
2. 服务器：`bash scripts/01_env_setup.sh`（建 conda 环境 → pip 装 torch/mmcv/mmdet（清华源）→ 解包 ai4rs → `pip install -e .` → 修补 mmdet/mmseg `__init__.py` 的 MMCV 版本断言，如官方文档所述）

> 因服务器访问不到 github.com，ai4rs 源码走本机中转；之后我们的改动在 `/data/repro/ai4rs` 内进行，并把**改动补丁**同步回本仓库 `patches/`（保持 GitHub 仓库为代码与实验保存仓库）。

## 2. 基线复现（不训练，直接用官方权重）（脚本 02 + 04）

| 步骤 | 操作 | 预期 |
| --- | --- | --- |
| 权重 | `scripts/02_fetch_baseline_weights.sh`：ModelScope 下载 `o2_rtdetr_r18vd_2xb4_72e_dota/epoch_72.pth`（及 R50 备用） | ~1 个权重文件（数百 MB） |
| 数据 | `scripts/03_prep_dota.sh`：DOTA-v1.0 原始图 → 切 1024×1024/overlap 200 → `dota2coco.py` | `split_ss_dota/{trainval,test}` + json |
| 评测 | `scripts/04_baseline_eval.sh`：`python tools/test.py .../o2_rtdetr_r18vd_2xb4_72e_dota.py <ckpt>`（单卡） | **AP50 ≈ 77.31**（若与官方一致 → 复现成功，基线确立） |

- 若 DOTA 官方下载源不可达：备选 Baidu 网盘官方链接（服务器可达 baidu）或 HF 镜像（hf-mirror 可达）；下载后校验文件数/大小。
- 权重 URL 需要验证 ModelScope 是否要求登录（若 401/403，改用 `modelscope` CLI 下载并记录登录方式；必要时在后续轮次处理）。

## 3. 应用创新（组合优点 → 积极增益）

**核心想法：把"朝向建模"（O2-RTDETR）与"频域-空间增强"（FSDETR）组合，形成面向密集重叠小目标的实时朝向检测器。**

创新点（按风险从低到高排序，全部在同代码库内实现）：

1. **FS-FPN（频域空间特征金字塔）**：把 FSDETR 的 FSFPN/CFSB 移植到 O2-RTDETR 的 neck（RT-DETR 的 AIFI + PAN 结构），重点提升密集场景小目标细节。→ 最容易与角度分布细化叠加的正增益点。
2. **对照项 A——FAA（傅里叶角度对齐，同仓库 CVPR 2026）**：角度表征频域化，配合 Angle Distribution Refinement，验证角度分支增益。
3. **对照项 B——DEIM 匹配**：替换二分匹配为解耦一对一/多对多，验证训练稳定与精度。
4. **Backbone 消融**：R18vd → R34vd → R50vd →（可选 PKINet/LSKNet），选出性价比最优组合上报。

### 实验矩阵（一次排好，统一跑，避免局部反复）

> 训练协议固定（72e、单卡 bs=2~4、1024×1024、ss），只改变量。训练用 `python tools/train.py <cfg>`（单卡适配 config 的 `batch_size/num_gpus` 字段即可）。

| # | 配置 | 目的 | 状态 |
| --- | --- | --- | --- |
| E0 | O2-RTDETR-R18（官方权重，不训练） | 复现基线 | 待跑 |
| E1 | O2-RTDETR-R18 + FS-FPN（训练） | 主创新 vs 复现值 | 待跑 |
| E2 | E1 + FAA 角度对齐 | 组合增益（角度+频域） | 待跑 |
| E3 | E1 + DEIM 匹配 | 组合增益（匹配+频域） | 待跑（可选） |
| E4 | 最优组合 R34/R50 backbone | 规模提升 | 待跑（可选） |

评测协议：DOTA-v1.0 **测试集** AP50（官方基线 77.31@R18），随后推广 DOTA-v1.5（70.83@R18）与 DIOR-R（67.00@R18）。

### 积极增益判据

- 主创新（E1）相对 E0 官方权重的测试集 AP50 提升 ≥ +0.5（且同训练协议下与官方数值可比）；若提升不显著，则以 E2/E3 组合寻找增益，不做单点无限调参。
- 三个以上通用性试验（DOTA-v1.0 / 1.5 / DIOR-R）都有正增益 → 论文结论成立。

## 4. 实验管理（防局部陷阱）

- 每次训练/评测用 `experiments/<run_name>/` 存 config 副本 + 日志 + 指标 JSON，`docs/progress_log.md` 记一行结论。
- 指标统一由脚本抽取（`--out` + 解析 AP50），不手工抄。
- 一个配置只跑一次（除非官方数值无法复现或明显 bug），不再对同一配置反复微参。

## 5. 论文写作对照（最后阶段）

- 目标期刊：IEEE TGRS / IEEE TNNLS / Pattern Recognition / IEEE TCSVT（中科院二区及以上）
- 贡献叙事：面向密集重叠/朝向目标的实时检测 → 频域-空间增强 + 朝向建模组合（结合我们验证的组合增益）+ 消融与可视化。