# 早期合成诊断（原样保留，均为 CPU、合成数据、1–2 种子；**不是实验结果**）

所有表都产生于下列已知缺陷存在时，仅作负面/诊断记录，不得用于对照结论：
- 表 A、B 产生于 `39c0bcf` 之前后：可学习 1×1 读出（plain/cutout/multitask/fate_learned）**未进优化器**；
- 合成 train/test 用不同 seed，连调色板也不同；类别由“缺哪个部件”决定，擦除一个存在的部件会使类别歧义；
- 沉默率在“干净时不活跃”的样本上记 0（定义缺陷，已于 22b2907 修）。

## 表 A（HEAD=a7c2297 时代，12 epoch，lr 1e-3，n=512，size 64，读出无标定）
method        clean occ(gray) occ(tex) sil(gray) sil(tex) stab
plain         0.736 0.738 0.738 0.501 0.320 0.456
cutout        0.736 0.738 0.738 0.301 0.293 0.570
multitask     0.736 0.663 0.724 0.545 0.378 0.602
fate_learned  0.506 0.509 0.584 0.620 0.636 0.494
fate_region   0.535 0.443 0.468 0.000 0.000 0.709
fate_contour  0.535 0.377 0.516 0.000 0.000 0.473

## 表 B（LCMV 读出诊断，同时间）
W 范数 1.2–1.7；签名范数 8–15；非对角泄漏 9.5e-7（零陷精确）；
部件证据 在/不在：p0 1.48/1.13，p1 1.47/1.63，p2 1.34/0.81，p3 2.13/1.40（几乎不可分）。

## 表 C（22b2907 之前的一版，lr 3e-4，20 epoch，λ=1:1，无预热）
seed0: plain .553 | multitask .736 | fate_learned .082 | fate_region .264
seed1: plain .736 | multitask .736 | fate_learned .271 | fate_region .244
（此表中 plain/multitask 仍受优化器缺陷影响。）

## 表 D（22b2907，λ=0.1+预热+仅可见样本，seed0）
fate_learned clean=0.506 sil_gray=0.133 | fate_region clean=0.264 sil_gray=0.267 sil_tex=0.047

保留原因：0 沉默率、0.53 vs 0.74 是早期诊断；在 docs/07 定义的固定设置下复核（见 results/early_diagnostics/rerun.md），不得先改分辨率/loading/pooling。
