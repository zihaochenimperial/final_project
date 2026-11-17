# 03 设置（Settings）

## 模型（2026-11-17b 修订：臂的定义以 docs/07 §B 为准）
- 骨干：torchvision ResNet-18（原型）/ ResNet-50（主表）；ImageNet 预训练；使用 stem..layer3。
- 读出：LCMV 1×1，K 个部件；协方差 EMA 动量 0.1；对角加载 1e-2·mean(diag R)；签名 EMA 动量 0.1。
- 读出计算固定 fp32（含 solve/inv），不进 AMP。
- 可见性门控阈值 0.05；分类头 Linear(K→类数)。

## 部件定义
- CUB K=8：beak{2}, head{5,6,7,10,11,15}, back{1}, belly{3}, breast{4}, wing{9,13}, tail{14}, legs{8,12}（括号为 CUB part id）；关键点处画半径 r=输入边长/10 的圆盘作部件掩码。
- RIVAL10 K=11：取 11 个局部属性（见 Moayeri 等 2022）；掩码来自官方 entity masks（转换脚本待数据到位后实现）。

## 损失
L = L_cls + λ₁·silence + λ₂·stability + λ₃·ERF-leak；默认 λ₁=0.1, λ₂=0.1, λ₃=0.1，前 30% epoch 线性预热；只对部件在图中可见的样本施加擦除损失；ERF 只在每 4 步启用一次（双反传昂贵）；每步随机抽一个部件 p 擦除。

## 训练
- 输入 224×224（CUB 先 resize 256 再 crop 224）；AdamW，lr 1e-3（头/读出）、1e-4（骨干）；cosine；weight decay 1e-4；
- batch 64；epoch：R18 30 / R50 40；种子 0,1,2。
- 增广：随机裁剪 + 水平翻转（翻转时对称部件掩码随图像同步翻转）；训练遮挡填充仅灰/噪声。

## 评估
遮挡面积 10/20/30%；填充 灰/噪声/纹理；每个部件单独遮挡；见 02_protocol.md。

## 工程
- 配置：`configs/*.yaml`；输出：`runs/<name>/<seed>/`（config、log、ckpt、metrics.json）。
- 随机性：固定种子；`torch.backends.cudnn.deterministic=True`（评估时）。
- 环境：Python ≥3.10，torch ≥2.2，torchvision，pyyaml，numpy，pillow。
- HPC：Imperial RCS；作业调度器与 GPU 队列名待核实（见 04_requirements.md）；模板 `hpc/job_train.pbs`。

## 修订 2026-11-17b
- 所有 K 通道臂共享同一读出标定层（每部件增益、偏置，初值偏置 −0.5）；读出核无偏置。`fate_learned` 以首个训练批的 LCMV 闭式解初始化。
- 签名 EMA 只在批内有效覆盖 ≥ `min_cov`（默认 1.0 个特征格质量）时更新，防止缺席部件把签名拖向 −外部均值；记录 `ema_cov`、`n_upd`。
- ERF 正则关闭（`lambda_erf: 0.0`）直到 stage‑1 判定；R50 配置保留但不使用。
- 优化器：骨干（stem/body/layer4）与其余参数两组，并断言无遗漏/重复。
