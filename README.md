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

## 状态
训练前的准备已完成（9 个测试通过；预检中 10 种方法/消融配置在合成数据上各跑通一步训练+评估）。
**尚未做**：真实数据（CUB/RIVAL10）下载与 RIVAL10 转换、外部基线（ICNN/ProtoPNet/PIP-Net/CompNets/MPM）、真实数据上的超参调试、
HPC 调度器核实（清单见 docs/04_requirements.md）。尚无任何真实训练结果；涨点预期均为推测。
