# 06 训练前自检记录（合成数据，CPU）— 2026-11-17b 修订

**这不是实验结果。** 合成数据极小、1–2 个种子、CPU；仅用于检查管线和早期诊断。原始输出原样保存在 `results/early_diagnostics/README.md`。

## 已确认并修复的缺陷（对 HEAD 逐项核对）
| # | 缺口 | 状态 | 证据 |
|---|---|---|---|
| 1 | 可学习 1×1 读出未进 AdamW（影响 plain/cutout/multitask/fate_learned）。`a7c2297`、`39c0bcf` 上确实如此；`22b2907` 用“排除 stem/body”补过但无测试 | **已修复并加护栏** | `make_optimizer` 断言无遗漏/重复；`test_optimizer_covers_every_param_and_learned_readouts_update` 对每个臂验证读出权重一步后改变；变异检验：回退旧写法该测试失败 |
| 2 | `plain` 其实是 K 通道瓶颈，不是标准 GAP+分类 ResNet | **已拆分** | `gap_resnet` / `bottleneck`；`test_controls_are_distinct_and_capacity_matched`（bottleneck 与 fate_learned 参数量相等） |
| 3 | LCMV vs learned 初始化/标定不同 | **已对齐** | 两者共享标定层；`fate_learned` 以首批 LCMV 闭式解初始化；`test_learned_init_from_lcmv…` |
| 4 | 沉默损失可能作用在没有真实遮挡的样本 | **已修复** | 仅对部件可见的样本施加 |
| 5 | 空/小部件把签名 EMA 拖向 −外部均值 | **已修复** | 覆盖门控；`test_signature_ema_not_dragged_by_absent_parts`；诊断记录有效覆盖与约束残差 |
| 6 | 合成 train/test 用不同 seed，连调色板也不同；类别由缺哪个部件决定（擦除即改类） | **已修复** | 固定调色板；冗余构造（任一剩余部件可分类）；换色 `test_shift` 单列；旧构造保留作对照 |
| 7 | contour 替换 region 权重，并用粗网格 >0.5 二值化，小部件整体消失；“区域+轮廓”不共享区域基项 | **已修复** | 三臂共享区域基项（γ=0 退回区域项，有测试）；全分辨率边界带；消失率见 `results/small_part_vanish.md`（旧版直径小于约一个特征格的部件 100% 消失） |
| 8 | 评估只有 Top‑1 / 范数类指标，无成对干预、无同面积对照、阈值可能看到测试集 | **已重写** | 见 docs/02 修订、`evaluate.py`；`test_eval_does_not_use_test_masks_and_thresholds_are_val_defined` |
| 9 | 沉默率在“干净时不活跃”的样本上记 0 | **已修复**（`22b2907`） | 仅统计干净时活跃的样本 |

## 保留的早期负结果（不删、不当成结论）
- 沉默率 0.000（LCMV 臂）与干净精度 0.53 vs 0.74：产生于缺陷 1/6 存在之时，**不能归因于方法**，也不能据此宣称方法有效；已在固定设置下复核，见 `results/early_diagnostics/rerun.md`。
- 复核**不同时改动**分辨率、loading、pooling（固定为 64px、1e-2、amax），避免无法归因。

## 固定设置复核（`results/early_diagnostics/rerun.md`，1 种子，CPU，评估集 128）
- 早期“LCMV 臂沉默率 0、精度 0.53”**不是可归因于方法的现象**：在 `distinct/redundant` 构造下两个 FATE 臂都塌到 ≈0.258，并伴随 LCMV 诊断失效（rank 4/5、cond_G ~1e8–1e9、约束残差 ~1–2）；
  只把 `fate_learned` 的初始化改成 default，干净精度恢复到 0.750（单因素，1 种子）→ `fate_learned` 的塌缩来自病态 LCMV 初始化。
- `loading` 提到 1.0 不能修复秩亏。曾提出的“部件共现导致共线”被冻结骨干检查证伪（G0 全通过）；秩亏出现在训练的头两个 epoch，机制未知。
- 因此新增 G0/G1 门槛（docs/07 §D2），LCMV 求解对奇异 G 容错并计数（`refresh_failures`），不再崩溃。
- 对 FATE 的含义：**没有任何一次运行同时满足 LCMV 有效 + 优于对照**；没有证据支持 FATE，也没有证据能证伪 LCMV 假设（运行无效）。

## 已验证的数学事实
- 零陷约束对**估计签名**精确成立：对本部件增益 1、对其他签名泄漏 ≈ 1e-6；这**不**等于实际遮挡下的沉默或 ERF 无外泄。

## 尚无证据的东西
FATE 优于任何对照；LCMV 闭式读出优于学出的读出；ERF 约束有效。这些全部要等 docs/07 的首个 R18 机制检验。
