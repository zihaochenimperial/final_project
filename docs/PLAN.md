# FATE-CNN 选题方案（2026-11-17）

## 结论
主选题：零陷约束的部件卷积核 + 反事实遮挡沉默训练（FATE-CNN）。主张帕累托占优：
干净精度不降超过 1 个点，同时提升部件遮挡精度、被遮挡卷积核沉默率、Part Specificity。
稳 CCF-B，必须加 CUB（第二主数据集）和遮挡基准才有冲 A 的资格。

## 三项原创
1. LCMV 零陷部件读出层：w_p = R⁻¹C(CᵀR⁻¹C)⁻¹e_p，C=[s_1..s_K,1]；签名由部件轮廓内外特征差分估计；可加 LEACE 式守护残差。
2. 反事实部件遮挡损失：L = L_cls + λ₁Σ‖a_p(x⊙¬M_p)‖₁ + λ₂Σ_{q≠p}‖a_q(x⊙¬M_p) − a_q(x)‖² + λ₃·ERF-leak。
3. 评估协议：沉默率、他核稳定性、匹配通道抑制特异性、CNN 版 PS、自然缺失 vs OOD 遮挡分离。

## 数学依据
- 命题1 零陷最优性：线性模型下该解为约束最小方差唯一解，对其他部件/共模的线性串扰恒为 0（`tests/test_fate.py::test_lcmv_null_constraint`）。
- 命题2 签名误差界：|w_pᵀs_q| ≤ ‖w_p‖‖δ_q‖，需对 R 做对角加载。预测"部件轮廓签名 > 通用边缘 ≈ 仅区域"，对应三臂对照。
- 命题3 感受野溢出：ERF 近似高斯、尺寸 O(√n)（Luo 等 2016）；非线性泄漏上界∝ERF 落在他部件掩码上的质量。
- 命题4 可见性门控组合：条件独立下在可见子集上求和 log-odds 为 Bayes 最优。
- 边界：命题1–2 只管线性泄漏；非线性部分靠沉默/ERF 损失与实验验证（Mahinpei 等 2021 指出去相关不足以消除泄漏）。

## 撞车与必比工作
最近邻：Metonymy（ViT，2026-05）、PF-CBM、OA-CBM、PartImageNet++/MPM、Towards Interpretable Face Recognition（FAD 损失）、
Interpretable CNN（ICNN）、CompositionalNets、PCMNet。本轮检索未发现 CNN 卷积核层面的"零陷 + 遮挡沉默"工作（推测，非穷尽）。

## 实验
- 数据：RIVAL10（11 个局部属性做部件，7 个整物体属性作对照）、CUB-200-2011（必选）、PartImageNet++/PASCAL-Part、OccludedPASCAL3D+。
- 遮挡：部件掩码（灰/噪声/纹理）、随机块 10/20/30%、自然遮挡。
- 基线：ResNet-18/50、SE/CBAM、ICNN、ProtoPNet/PIP-Net、CompNets、Concept Whitening、MPM、Metonymy 式两阶段上界。
- 控制：同调参预算、参数对齐、≥3 种子。
- 三臂对照：仅区域 / 区域+部件轮廓 / 区域+通用边缘（对应签名估计来源）。

## 风险与降档
1. "只是部件分割多任务" → 与 MPM/多任务头同监督消融。
2. RIVAL10 太小 → CUB/PIN++ 作主表。
3. 沉默来自填充捷径 → 多种填充交叉测试 + 自然遮挡协议。
降档：CUB 部件遮挡精度提升 <2 点或 PS 不优于 ICNN → 转 B 档"协议+方法"；沉默率也不显著 → 备选 CMF-Part（条件式匹配滤波）。

## 待向老师确认
1. "filter 提升"指学出的部件滤波器还是手工 Gabor/匹配滤波先验？
2. 部件是否允许强监督？
3. 是否同意 CUB 为第二主数据集？
4. 能否接受"精度持平、鲁棒/可解释提升"的帕累托叙事？
5. 能否用 DINO/SAM 3 伪掩码扩展到 PartImageNet++？

## 时间线（推测）
2026-12~2027-02 R18 原型 + RIVAL10；2027-01 可投 IJCAI；2027-03 ICCV；2027-05 补齐 CUB/PIN++ 后投 NeurIPS。
