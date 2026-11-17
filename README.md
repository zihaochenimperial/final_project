# FATE-CNN

**FATE = Filter Annulment under Targeted Erasure**（目标擦除下的卷积核湮灭）。
备选释义：Filter-level Attribution via Targeted Erasure。

在 ResNet 的 layer3 上给每个部件分配一组读出通道，用闭式 LCMV 零陷约束读出
（对本部件单位增益，对其他部件签名与共模分量置零），再用反事实部件遮挡损失训练：
**遮挡部件 p ⇒ p 的卷积核沉默（annul），其余部件不变**。

| 名称含义 | 对应实现 |
|---|---|
| Filter Annulment | `fatecnn/lcmv.py`（零陷读出）、`losses.silence_loss` |
| Targeted Erasure | `data.erase`、`losses.stability_loss`、`losses.erf_leak_loss` |

## 结构
```
fatecnn/lcmv.py     LCMV 零陷读出 + 部件签名估计（轮廓内外特征差分）
fatecnn/losses.py   沉默 / 稳定 / ERF 泄漏损失
fatecnn/metrics.py  沉默率、他核稳定性、匹配通道抑制特异性
fatecnn/model.py    ResNet(stem..layer3) -> LCMV 读出 -> 可见性门控 -> 分类头
fatecnn/data.py     SyntheticParts 冒烟数据集；RIVAL10/CUB 沿用同一接口
configs/            实验配置
docs/PLAN.md        选题方案总览（2026-11-17 调研记录）
docs/01_objectives.md  目标
docs/02_protocol.md    协议（数据划分/遮挡/指标/基线/降档）
docs/03_settings.md    设置（模型/损失/训练超参）
docs/04_requirements.md 需求（数据/算力/HPC 清单/待办）
docs/06_pre_training_findings.md 训练前缺陷与自检
docs/07_acceptance.md 冻结的最小验收单（含首个 R18 机制检验通过/失败条件）
docs/05_unified_protocol.md 统一协议（仓库/本机/GPU：profile、路径、同步、manifest）
fatecnn/datasets.py CUB 加载（关键点→8 部件）+ 通用文件夹格式
fatecnn/occlusion.py 遮挡协议（灰/噪声/纹理/块）
fatecnn/evaluate.py  评估脚本
hpc/                 环境脚本与作业模板（调度器/队列需核实）
```

## 快速开始
```bash
pip install -e . pytest
pytest -q
python -m fatecnn.train --config configs/synthetic.yaml
```

## 状态（2026-11-17b）
- 训练前修复与回归测试已完成；验收单冻结于 `docs/07_acceptance.md`；早期负结果与固定设置复核在 `results/early_diagnostics/`。
- **没有任何真实数据训练结果**；合成自检不构成方法证据，且在 `distinct/redundant` 构造下 LCMV 训练中失效（docs/06、docs/07 §D2）。
- **阻塞**：RIVAL10 数据与转换脚本、CUB 数据（容器不能联网）、外部基线、HPC 调度器核实。正式训练与 HPC 提交未获授权。
