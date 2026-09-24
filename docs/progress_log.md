# 进度日志

## R1 · 2026-09-24 —— 调研与仓库搭建（本轮）

**做了什么**
1. 打通服务器：`ssh root@sx01-ssh.gpuhome.cc -p 30214 -i D:\tmp\001\.ssh\yolo-sx01`
   - 硬件 1×RTX 3090 24GB / 88 核 / 251GB RAM
   - **磁盘勘误**：`/`（overlay）仅 12G；`/data/tini` 实为只读文件挂载（不可用）；唯一可写大分区 `/root/rivermind-data`（49G，31G 可用）→ 一切大文件放 `/root/rivermind-data/repro`，流式处理（峰值 <25G）
   - 网络：github/gdrive/kaggle **不通**；pypi / modelscope / hf-mirror / tsinghua / baidu **通**
   - 已确认 /workspace 有旧代码（RemoteSensing 等）→ 按要求**不使用**，全新目录 `/root/rivermind-data/repro`
2. 论文调研与基线选定：
   - ✅ 基准 = **O2-RT-DETR（IEEE TGRS 2026）** —— 朝向检测、DOTA 系、公开代码（ai4rs）+ ModelScope 官方权重（实测匿名可下，HTTP 200）、24GB 可跑
   - ✅ 创新来源 = **FSDETR（IJCNN 2026）** 频域-空间 FPN（FSFPN/CFSB）+ DA-AIFI 遮挡缓解；备选 FAA（CVPR 2026）/ DEIM（同仓库）
   - 淘汰：D2M-DETR / DeCenter / CVIU 轻量框架（未见公开代码）
   - 证据见 [paper_survey.md](paper_survey.md)
3. DOTA-v1.0 数据源验证：HF `isaaccorley/dota`（CC-BY-NC-4.0），hf-mirror 服务器实测可下载
4. 编写云端复现脚本（scripts/01~05 + check_env.py），按实测磁盘约束改为流式 + jpg 切分；评测协议定为 **train 训练 / val 评测**（test 无公开标注）
5. GitHub 仓库初始化并推送：
   - 本机 git push 依赖 msys sh，被沙箱命名管道限制阻断 → 以 `danger-full-access` 重试成功；已用现有 SSH 公钥作为**写入 deploy key**（`dsh-push-20260924`）并持久化 `core.sshCommand`（正斜杠密钥路径）
   - 仓库：https://github.com/w504715799-eng/RemoteSensing--001（master 分支）

**关键结论（防局部陷阱）**
- 先拿官方权重基线（不训练）→ 再排实验矩阵一次性跑 → 组合增益判据（E1 vs E0' 同协议 +0.5 AP50）
- 评测协议：本地统一 val 切分；正式对比基线为本地重训 O2-RTDETR（E0'），官方权重仅做 sanity check

**下一步（R2）**
1. 本机 clone ai4rs → scp 到服务器 `/root/rivermind-data/repro/`
2. 服务器 `01_env_setup.sh`（conda py3.10 + torch2.4/cu121 + mmcv2.2 + ai4rs）
3. `02_fetch_baseline_weights.sh` + `03_prep_dota.sh val`（hf-mirror 下载 ≈3.8GB，流式切 jpg）
4. `04_baseline_eval.sh r18 val` → 记录官方权重 @ val sanity check 指标
5. 之后进入实验矩阵：E0'（重训基线）→ E1（FS-FPN）→ E2/E3 组合

## 链接速查
- 论文: https://ieeexplore.ieee.org/document/11424629 · arXiv 2603.15497
- 代码: https://github.com/wokaikaixinxin/ai4rs · https://github.com/wokaikaixinxin/O2-RT-DETR
- 权重: https://modelscope.cn/models/wokaikaixinxin/ai4rs
- 创新源: https://github.com/YT3DVision/FSDETR（IJCNN 2026, arXiv 2604.14884）