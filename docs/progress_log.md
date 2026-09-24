# 进度日志

## R1 · 2026-09-24 —— 调研与仓库搭建（本轮）

**做了什么**
1. 打通服务器：`ssh root@sx01-ssh.gpuhome.cc -p 30214 -i D:\tmp\001\.ssh\yolo-sx01`
   - 硬件 1×RTX 3090 24GB / 88 核 / 251GB RAM；`/` 仅剩 12GB，`/data` 剩 355GB
   - 网络：github/gdrive/kaggle **不通**；pypi / modelscope / hf-mirror / tsinghua / baidu **通**
   - 已确认 /workspace 有旧代码（RemoteSensing 等）→ 按要求**不使用**，全新目录 `/data/repro`
2. 论文调研与基线选定：
   - ✅ 基准 = **O2-RT-DETR（IEEE TGRS 2026）** —— 朝向检测、DOTA 系、公开代码（ai4rs）+ ModelScope 官方权重、24GB 可跑
   - ✅ 创新来源 = **FSDETR（IJCNN 2026）** 频域-空间 FPN（FSFPN/CFSB）+ DA-AIFI 遮挡缓解；备选 FAA（CVPR 2026）/ DEIM（同仓库）
   - 淘汰：D2M-DETR / DeCenter / CVIU 轻量框架（未见公开代码）
   - 证据见 [paper_survey.md](paper_survey.md)
3. DOTA-v1.0 数据源验证：HF `isaaccorley/dota`（CC-BY-NC-4.0），hf-mirror 从服务器可下载（实测 README 200）；含 train/val 图像+标注 tar
4. 编写云端复现脚本（scripts/01~05 + check_env.py）与计划文档（experiment_plan.md）
5. GitHub 仓库 [w504715799-eng/RemoteSensing--001](https://github.com/w504715799-eng/RemoteSensing--001) 初始化并推送

**关键结论（防局部陷阱）**
- 先拿官方权重基线（不训练）→ 再排实验矩阵一次性跑 → 组合增益判据 (+0.5 AP50 @ 官方可比协议)
- 评测协议：DOTA test 无公开标注（官方走评测服务器），本地统一用 **val 切分**做前后对比

**下一步（R2）**
1. 本机 clone ai4rs → scp 到服务器 `/data/repro/`
2. 服务器 `01_env_setup.sh`（conda py3.10 + torch2.4/cu121 + mmcv2.2 + ai4rs）
3. `02_fetch_baseline_weights.sh`（验证 ModelScope 直链 / modelscope CLI）
4. `03_prep_dota.sh`（≈13.7GB 下载 + 切分 + COCO）
5. `04_baseline_eval.sh` → 记录 R18/R50 val 基线 AP50

## 链接速查
- 论文: https://ieeexplore.ieee.org/document/11424629 · arXiv 2603.15497
- 代码: https://github.com/wokaikaixinxin/ai4rs · https://github.com/wokaikaixinxin/O2-RT-DETR
- 权重: https://modelscope.cn/models/wokaikaixinxin/ai4rs
- 创新源: https://github.com/YT3DVision/FSDETR（IJCNN 2026, arXiv 2604.14884）