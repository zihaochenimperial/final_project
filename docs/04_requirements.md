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

## 软件/代码需求清单
- [x] LCMV 读出、损失、指标、冒烟训练
- [x] CUB 加载（关键点→部件掩码）、通用部件数据集接口
- [x] 遮挡协议（灰/噪声/纹理/块）、评估脚本
- [x] HPC 作业模板（单作业 + 数组作业）、环境脚本
- [x] 统一协议（docs/05）、manifest、collect/check_data
- [x] 对照方法：plain / cutout（同遮挡量）/ multitask（同部件监督）/ fate_learned（LCMV 消融），均在同一骨干与调参预算下
- [x] 三臂签名估计（region / contour / edge，见 `lcmv.estimate_signatures`；操作性定义为本仓库的约定，待与老师确认其对应“仅区域/区域+轮廓/区域+通用边缘”的口径）
- [x] 实验矩阵 `scripts/experiments.py`（10 组 × 3 种子 = 30 作业）+ 预检 `scripts/preflight.py`
- [x] 种子聚合与 bootstrap 置信区间（`scripts/aggregate.py`）
- [ ] RIVAL10 转换脚本（需数据；其官方标注格式尚未核对）
- [ ] 外部基线：Interpretable CNN、ProtoPNet/PIP-Net、CompositionalNets、Concept Whitening、MPM（需引入官方实现并对齐骨干/预算）
- [ ] 真实 CUB 数据上的超参调试（lr/λ 目前为初始值）
- [ ] Imperial HPC 调度器/队列核实后更新 `hpc/*.pbs`
