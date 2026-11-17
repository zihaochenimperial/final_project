# 07 冻结的最小验收单（训练前）

冻结日期：2026-11-17。**本单中的阈值在任何真实数据训练之前设定；修改必须追加带日期的修订，不得事后改。**
正式训练与 HPC 提交不在当前授权范围；本单只定义“什么证据算通过”。

## A. 数据验收
| 项 | 要求 | 状态 |
|---|---|---|
| RIVAL10（首个机制检验） | 真实局部掩码；仅用 11 个真正局部属性作部件（7 个整物体属性排除）；转换为 `FolderParts` 格式（`<root>/{train,val,test}/index.json`，掩码 npy (K,H,W)）；记录每部件“可见比例”与面积分布 | **阻塞**：容器无法下载；官方标注格式未核对，转换脚本未写 |
| CUB-200-2011（跨数据验证） | 官方关键点**可见性标志**决定部件是否存在；圆盘掩码为**邻域遮挡**（`CubParts.NEIGHBOURHOOD_MASKS=True`），不作轮廓主张；train/val 分层划分（val=10%） | **阻塞**：数据未下载（`scripts/download_data.sh` 在有网机器运行，URL 未验证） |
| 划分与阈值 | 阈值只在 train/val 定义；test 只出结果 | 已实现（`thresholds.json`，测试覆盖） |
| 合成数据 | 固定调色板；train/val/test 只换样本 seed；换色 `test_shift` 单列；冗余构造用于遮挡分类诊断，旧“缺部件”构造仅作对照 | 已实现（测试覆盖） |

## B. 对照验收（同骨干、同预算、同调参量）
| 臂 | 结构 | 目标 | 部件监督 | 遮挡 | 读出初始化 |
|---|---|---|---|---|---|
| `gap_resnet` | 标准 stem..layer4+GAP+FC | CE | – | – | – |
| `bottleneck` | stem..layer3→1×1(K)→证据→FC，容量匹配 | CE | – | – | 默认 |
| `cutout` | 同 bottleneck | CE（干净+遮挡） | – | 同 FATE 的部件遮挡量 | 默认 |
| `multitask` | 同 bottleneck | CE + 部件掩码 BCE | 有 | – | 默认 |
| `fate_learned` | 同 bottleneck，+ 同一标定层 | CE + 沉默/稳定 | 掩码用于遮挡与初始化 | 有 | **LCMV 首批闭式解** |
| `fate`（LCMV） | 同上，但读出 W 闭式更新 | CE + 沉默/稳定 | 同上 | 有 | LCMV |
- LCMV vs learned：同初始化策略、骨干、监督/遮挡、分类目标、预算；唯一差别是 W 由闭式求得还是梯度学习（`test_learned_init_from_lcmv…`、`test_controls_are_distinct…`）。
- **同一读出族内消融**沉默/稳定损失：`fate_learned_no_sil/no_stab`，`fate_lcmv_no_sil/no_stab`。
- 沉默损失只对**真正实施了遮挡且部件可见**的样本计算；空部件/小部件不更新签名 EMA（有效覆盖 `ema_cov`、更新次数 `n_upd`、约束残差由 `diagnostics()` 记录）。
- 优化器覆盖断言：`make_optimizer` 断言每个可训练参数恰在一个组内；回归测试对每个臂验证读出权重在一步训练后确实改变（变异检验：回退旧写法，该测试失败）。
- 三臂签名（contour/edge）延后，且圆盘数据上不做轮廓主张；共享区域基项有测试（γ=0 退回区域项）。

## C. 指标验收（见 docs/02 修订 2026-11-17b）
pre-gate 成对连续分数；干净响应覆盖；他部件扰动（未连带 / 连带分列）；干净与遮挡 Top‑1；同面积非目标对照；未见填充（texture）；部件重叠率；天然不可见单列；匹配 vs 非匹配通道抑制；LCMV 诊断（rank、cond、约束残差、有效覆盖）。

## D. 首个 R18 机制检验：RIVAL10，通过/失败条件（预先约定）
设置：ResNet‑18，224，3 个种子（0,1,2），6 个臂 + 4 个消融（见 `scripts/experiments.py` STAGE1），`lambda_erf=0`，**不做 R50、不做 ERF 正则、不做三臂签名、不做 30 作业矩阵**。
统计：以种子为配对单位的 bootstrap 95% CI（3 种子区间宽，如实报告；不得因区间宽而改口径）。所有判断在 **test 上的 texture 未见填充**、宏平均（仅覆盖达标的部件）。

**有效性前置（不满足 → 检验无效，既不算通过也不算失败）**
- V1 覆盖：≥ 8/11 个部件的 `coverage` ≥ 0.6。
- V2 LCMV 假设：`rank_C = K+1`，`constraint_residual < 1e-3`，`min_sig_updates > 0`，`cond_G` 有限并记录。
- V3 数据：每个部件在 test 中可见样本 ≥ 50。

**通过（以下全部成立才算首个机制检验通过）**
- P1 选择性：`fate`（LCMV）的 `silence_selectivity`（目标遮挡沉默率 − 同面积对照沉默率）≥ 0.20，CI 排除 0。
- P2 对照占优：P1 指标比 `bottleneck`、`cutout`、`multitask`、`fate_learned` 都高 ≥ 0.10。
- P3 其余部件保持：`other_unconnected` ≤ 0.25，且不比最优对照差 > 0.05。
- P4 分类依赖：`suppress_matched_drop − suppress_nonmatched_drop` ≥ 0.05，CI 排除 0。
- P5 代价：干净 Top‑1 ≥ `bottleneck` − 1.0 个点。
**LCMV 增量判定**：若 `fate` 与 `fate_learned` 在 P1/P3/P4 上差异的 CI 都包含 0 → **LCMV 闭式贡献不成立**（贡献降为擦除损失与协议）。
**失败（任一成立）**：P1 或 P4 不成立；选择性被同面积对照解释（`silence_control` 与 `silence` 无差）；`multitask` 或 `cutout` 达到 `fate` 的 P1/P3/P4；texture 上 `silence_selectivity` 相对 gray 下降 > 50%（填充捷径）。
**不得**用阈值调整（只在 train/val 定义）、换填充、事后改部件子集来挽救失败。

## E. 当前阻塞（需要真实数据 / 外部条件）
1. RIVAL10 数据与转换脚本（官方标注格式待核对）。
2. CUB 数据下载。
3. 外部基线 ICNN / CompositionalNets / MPM / ProtoPNet 官方实现（不在首个检验内，但在“原创性判定”前必须有）。
4. Imperial HPC 调度器/队列/gpu_type 核实。
5. 以上之前，不运行真实训练，不提交 HPC 作业。

## F. 审计记录
见 `results/early_diagnostics/`（早期负结果原样保留与固定设置复核）、`results/small_part_vanish.md`、docs/06。
