# 04 需求（Requirements）

## 数据（需手动下载，容器网络受限）
| 数据 | 用途 | 状态 |
|---|---|---|
| CUB-200-2011（~1.1GB） | 主表 | 待下载；需 images.txt、image_class_labels.txt、train_test_split.txt、parts/part_locs.txt |
| RIVAL10（~GB 级） | 机理分析 | 待下载；需整物体/属性 mask；转换脚本待写 |
| PartImageNet++ / OccludedPASCAL3D+ | 泛化（可选） | 暂缓 |
| ImageNet 预训练权重 | 骨干 | 首次运行自动下载，HPC 计算节点可能无外网 → 在登录节点预下载 |

## 计算
| 阶段 | 资源【推测】 |
|---|---|
| R18 原型（CUB/RIVAL10） | 1×GPU ≥12GB，约 数小时/种子 |
| R50 主表 | 1×GPU ≥24GB，224/bs64；ERF 双反传时显存约 ×1.5–2 |
| 基线（ICNN/ProtoPNet/PIP-Net 等） | 各 1×GPU，数小时–1 天 |

## Imperial HPC（GPU 前需完成）
- [ ] College 账号；导师在 Group Management 把你的 user ID 加入 Delegated Access（授课型 MSc 可能自动开通）。
- [ ] 校外使用 Unified Access（Zscaler）。
- [ ] 登录节点测试 SSH；确认调度器（PBS/Slurm）、GPU 队列名、gpu_type、最长 walltime、配额。
- [ ] 建 conda/venv，装依赖（`hpc/setup_env.sh`）；在登录节点预下载预训练权重。
- [ ] 数据放 RDS/scratch，而非 home（容量与 IO）。
- [ ] 用 `hpc/job_train.pbs` 提交 1 次 CPU/短 GPU 冒烟作业。

## 待向老师确认
1. “filter 提升”指学出的部件滤波器还是手工 Gabor/匹配滤波先验？
2. 部件是否允许强监督？
3. 是否同意 CUB 为第二主数据集？
4. 能否接受“精度持平、鲁棒/可解释提升”的帕累托叙事？
5. 能否用 DINO/SAM 3 伪掩码扩展到 PartImageNet++？

## 软件/代码需求清单（2026-11-17b）
- [x] 优化器覆盖修复 + 回归测试（含变异检验）
- [x] 对照臂拆分：gap_resnet / bottleneck（容量匹配）/ cutout / multitask / fate_learned（LCMV 初始化）/ fate
- [x] 签名 EMA 覆盖门控、LCMV 假设诊断、三臂共享区域基项、小部件消失率量化
- [x] 成对干预评估器（同面积非目标对照、未见填充、他部件连带/非连带分列、天然不可见单列、匹配 vs 非匹配通道抑制、val 阈值）
- [x] 合成数据：固定调色板、冗余构造、换色单列
- [x] 冻结验收单 docs/07；早期负结果存档 results/early_diagnostics/
- [ ] **RIVAL10 转换脚本**（阻塞：数据 + 官方标注格式）
- [ ] **CUB 数据**（阻塞：需有网机器下载）
- [ ] 外部基线 ICNN / CompositionalNets / MPM（原创性判定前必须）
- [ ] HPC 调度器/队列核实
- [ ] 三臂签名、ERF 正则、R50、30 作业矩阵（延后至 stage‑1 判定后）
