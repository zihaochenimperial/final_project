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
docs/PLAN.md        选题方案、实验与风险（来自 2026-11-17 的调研记录）
```

## 快速开始
```bash
pip install -e . pytest
pytest -q
python -m fatecnn.train --config configs/synthetic.yaml
```

## 状态
骨架与单元测试完成（零陷约束、签名估计、损失、训练冒烟）。**尚未实现**：RIVAL10 / CUB 数据加载、
基线（ICNN、ProtoPNet/PIP-Net、CompNets、MPM）、多填充遮挡评估、ResNet-50 实验。所有涨点预期均为推测，未经实验验证。
