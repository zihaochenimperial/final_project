# Rerun of early diagnostics at HEAD (fixed settings: 64px, loading 1e-2, amax pooling, lr 3e-4, 20 epochs, train n=512; eval n=128 per split, seed 0). Not results; 1 seed.

## mode=redundant seeds=[0]  (test and colour-shift test; thresholds from val)

| arm | seed | split | clean_acc | acc_target/texture | acc_control/texture | selectivity/texture | silence/texture | other_unconnected/texture | coverage | suppress_matched_drop | suppress_nonmatched_drop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gap_resnet | 0 | test | 1.000 | 0.828 | 0.875 | - | - | - | 1.000 | - | - |
| gap_resnet | 0 | test_shift | 0.000 | 0.092 | 0.061 | - | - | - | 1.000 | - | - |
| bottleneck | 0 | test | 0.750 | 0.742 | 0.723 | -3.234 | 0.000 | 0.619 | 0.467 | 0.090 | 0.090 |
| bottleneck | 0 | test_shift | 0.000 | 0.039 | 0.023 | -3.805 | 0.007 | 0.361 | 0.477 | -0.143 | -0.143 |
| cutout | 0 | test | 0.750 | 0.750 | 0.748 | -0.881 | 0.042 | 0.777 | 0.480 | 0.090 | 0.090 |
| cutout | 0 | test_shift | 0.000 | 0.008 | 0.004 | 0.560 | 0.000 | 0.241 | 0.473 | -0.143 | -0.143 |
| multitask | 0 | test | 0.750 | 0.670 | 0.709 | 0.705 | 0.074 | 1.556 | 0.584 | 0.092 | 0.092 |
| multitask | 0 | test_shift | 0.000 | 0.062 | 0.055 | 1.865 | 0.363 | 0.524 | 0.564 | -0.143 | -0.143 |
| fate_learned | 0 | test | 0.258 | 0.258 | 0.258 | -3044.566 | 0.000 | 0.216 | 0.258 | 0.129 | 0.129 |
| fate_learned | 0 | test_shift | 0.258 | 0.258 | 0.258 | -2898.441 | 0.000 | 0.225 | 0.250 | 0.000 | 0.000 |
| fate_lcmv | 0 | test | 0.258 | 0.258 | 0.258 | 105.556 | 0.000 | 506.303 | 0.473 | 0.041 | 0.041 |
| fate_lcmv | 0 | test_shift | 0.258 | 0.258 | 0.258 | 81.581 | 0.000 | 3.204 | 0.984 | -0.027 | -0.027 |
|  | 0 | lcmv_diag | {'rank_C': 4, 'need_rank': 5, 'cond_R': 5372.79150390625, 'cond_G': 3306544384.0, 'constraint_residual': 2.176687717437744, 'w_norm_max': 458.0193786621094, 'min_ema_coverage': 84.49996948242188, 'min_sig_updates': 320.0} |

## mode=missing seeds=[0]  (test and colour-shift test; thresholds from val)

| arm | seed | split | clean_acc | acc_target/texture | acc_control/texture | selectivity/texture | silence/texture | other_unconnected/texture | coverage | suppress_matched_drop | suppress_nonmatched_drop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gap_resnet | 0 | test | 1.000 | 0.940 | 0.842 | - | - | - | 1.000 | - | - |
| gap_resnet | 0 | test_shift | 1.000 | 0.929 | 0.638 | - | - | - | 1.000 | - | - |
| bottleneck | 0 | test | 0.750 | 0.749 | 0.687 | -2.430 | 0.000 | 0.510 | 0.519 | 0.060 | 0.104 |
| bottleneck | 0 | test_shift | 0.750 | 0.741 | 0.580 | -1.306 | 0.023 | 3.778 | 0.575 | 0.241 | 0.231 |
| cutout | 0 | test | 0.750 | 0.749 | 0.725 | -3.261 | 0.000 | 1.393 | 0.478 | 0.155 | 0.174 |
| cutout | 0 | test_shift | 0.750 | 0.749 | 0.597 | -1.569 | 0.086 | 0.227 | 0.676 | 0.241 | 0.231 |
| multitask | 0 | test | 0.750 | 0.749 | 0.621 | 2.119 | 0.300 | 1.504 | 0.609 | 0.155 | 0.174 |
| multitask | 0 | test_shift | 0.750 | 0.678 | 0.426 | -1.028 | 0.013 | 0.827 | 0.201 | 0.155 | 0.174 |
| fate_learned | 0 | test | 0.750 | 0.610 | 0.657 | 9.778 | 0.329 | 0.468 | 0.481 | 0.155 | 0.174 |
| fate_learned | 0 | test_shift | 0.570 | 0.558 | 0.415 | 0.262 | 0.161 | 0.801 | 0.316 | 0.095 | 0.070 |
| fate_lcmv | 0 | test | 0.570 | 0.261 | 0.265 | 0.381 | 0.560 | 0.704 | 0.811 | 0.190 | 0.130 |
| fate_lcmv | 0 | test_shift | 0.258 | 0.258 | 0.258 | -0.116 | 0.090 | 1.173 | 0.222 | 0.000 | 0.000 |
|  | 0 | lcmv_diag | {'rank_C': 5, 'need_rank': 5, 'cond_R': 10335.7275390625, 'cond_G': 947.4264526367188, 'constraint_residual': 1.538551259727683e-06, 'w_norm_max': 2.0190117359161377, 'min_ema_coverage': 62.29397964477539, 'min_sig_updates': 320.0} |

## mode=distinct seeds=[0]  (test and colour-shift test; thresholds from val)

| arm | seed | split | clean_acc | acc_target/texture | acc_control/texture | silence/texture | silence_control/texture | silence_selectivity/texture | other_unconnected/texture | coverage | suppress_matched_drop | suppress_nonmatched_drop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gap_resnet | 0 | test | 1.000 | 0.309 | 0.344 | - | - | - | - | 1.000 | - | - |
| gap_resnet | 0 | test_shift | 0.688 | 0.266 | 0.312 | - | - | - | - | 1.000 | - | - |
| bottleneck | 0 | test | 0.750 | 0.328 | 0.398 | 0.065 | 0.166 | -0.101 | 0.704 | 0.375 | 0.168 | 0.168 |
| bottleneck | 0 | test_shift | 0.750 | 0.203 | 0.215 | 0.140 | 0.175 | -0.035 | 24.641 | 0.262 | 0.168 | 0.168 |
| cutout | 0 | test | 0.750 | 0.709 | 0.723 | 0.115 | 0.029 | 0.086 | 1.266 | 0.334 | 0.168 | 0.168 |
| cutout | 0 | test_shift | 0.750 | 0.672 | 0.664 | 0.000 | 0.139 | -0.139 | 0.414 | 0.252 | 0.168 | 0.168 |
| multitask | 0 | test | 0.750 | 0.383 | 0.422 | 0.556 | 0.233 | 0.323 | 6.549 | 0.596 | 0.168 | 0.168 |
| multitask | 0 | test_shift | 0.258 | 0.238 | 0.242 | 0.500 | 0.167 | 0.333 | 3.241 | 0.205 | -0.014 | -0.014 |
| fate_learned | 0 | test | 0.258 | 0.258 | 0.275 | 0.000 | 0.000 | 0.000 | 0.157 | 0.258 | 0.129 | 0.129 |
| fate_learned | 0 | test_shift | 0.258 | 0.277 | 0.266 | 0.012 | 0.012 | 0.000 | 2.116 | 1.000 | -0.027 | -0.027 |
| fate_lcmv | 0 | test | 0.258 | 0.258 | 0.258 | 0.125 | 0.000 | 0.125 | 4.093 | 0.078 | 0.000 | 0.000 |
| fate_lcmv | 0 | test_shift | 0.258 | 0.258 | 0.258 | 0.000 | 0.000 | 0.000 | 0.196 | 1.000 | -0.027 | -0.027 |
|  | 0 | lcmv_diag | {'rank_C': 4, 'need_rank': 5, 'cond_R': 5031.2783203125, 'cond_G': 231383728.0, 'constraint_residual': 1.8260481357574463, 'w_norm_max': 336.528564453125, 'min_ema_coverage': 84.49996948242188, 'min_sig_updates': 320.0} |

## single-factor follow-ups (distinct mode, seed 0, test split)

| run | clean | acc_target/tex | silence/tex | silence_ctrl/tex | silence_sel/tex | lcmv_diag |
|---|---|---|---|---|---|---|
| fate_learned | readout_init=default | 0.750 | 0.736 | 0.161 | 0.226 | -0.065 | - |
| fate_lcmv | loading=1.0 | 0.258 | 0.258 | 0.543 | 0.235 | 0.307 | {'rank_C': 4, 'cond_G': 120013168.0, 'constraint_residual': 1.0, 'w_norm_max': 0.138} |
| fate_learned | init=lcmv, loading=1.0 | 0.570 | 0.498 | 0.394 | 0.000 | 0.394 | - |

## 复核结论与撤回（人工记录，非自动生成）
设置：64px、loading 1e-2、amax 池化、lr 3e-4、20 epoch；训练 n=512，评估每个 split n=128；**1 个种子**；CPU。均不是实验结果。
1. 先前“0 沉默率 / 0.53 vs 0.74”在修复后**不能复现为同一现象**：各构造下 FATE 臂的结果取决于数据构造与 LCMV 诊断（见下），不能归因于方法本身。
2. `distinct`/`redundant` 构造：两个 FATE 臂都塌到约 0.258（≈随机），`fate_learned` 仅因用病态 LCMV 解初始化而塌：
   单因素检验——只把 `fate_learned` 初始化改为 default → 干净 0.750（恢复）。（1 种子）
3. `loading` 1e-2→1.0 **不能**修复：`rank_C` 仍 4/5、约束残差 1.0（秩由签名矩阵决定，与协方差正则无关）。
4. 我曾推测“部件永远同时出现 → 签名共线”。**被证伪**：冻结随机初始化骨干上，三种构造的 rank_C 都是 5/5、残差 < 1e-3（G0 全部通过）。秩亏出现在**训练过程中**：
   逐 epoch 监控（distinct，fate）：ep0 rank 5/5 cond_G 1.9e4 resid 3e-3；ep1 cond_G 1e6 resid 1e-2；ep2 起 resid=1；ep4 起 rank 4/5；验证精度全程 0.234。
   机制未知，不下结论。
5. 旧 `missing` 构造下 LCMV 诊断健康（rank 5/5，residual 1.5e-6，cond_G 947），LCMV 臂干净 0.570、silence/texture 0.560、coverage 0.811，
   但该构造下遮挡后分类歧义，且仅 1 种子/128 张，只能说明“诊断健康时管线可训练”，不是优于对照的证据。
6. 同面积对照：`bottleneck/cutout` 在 distinct 构造上的 silence_selectivity ≈ 0 或为负，`multitask` ≈ +0.32（1 种子）；这些是对照臂的基线水平，不是 FATE 的证据。
7. 色彩换色集（test_shift）在 redundant 构造上全体 0 精度，说明该构造中颜色是唯一线索——这是设计如此，不是模型缺陷。
