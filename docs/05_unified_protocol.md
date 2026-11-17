# 05 统一协议：仓库 · 本机 · GPU(HPC)

**原则：一份代码、一份协议；三种环境的差异只来自 ① `FATE_*` 环境变量 ② `configs/profiles/<profile>.yaml`。**
结果可追溯到 git commit + config 哈希；数据不进 git；大文件不进 git。

## 1. 三个环境的角色
| 环境 | profile | 用途 | 能否产出论文数字 |
|---|---|---|---|
| 仓库（GitHub 分支） | — | 唯一真源：代码、配置、文档、`results/` 小结果 | — |
| 本机 / 云 CPU | `local` | 开发、单元测试、冒烟（1 epoch、小 batch、无 ERF、无预训练） | **否** |
| 本机/云 单 GPU | `gpu` | 原型与调参 | 仅当 manifest `git_dirty=false` 且为完整配置 |
| Imperial HPC GPU | `hpc` | 正式实验 | **是（唯一）** |

profile 自动判定：`$FATE_PROFILE` > 存在 `PBS_JOBID/SLURM_JOB_ID` → `hpc` > 有 CUDA → `gpu` > `local`。

## 2. 配置优先级
`configs/<实验>.yaml`  <  `configs/profiles/<profile>.yaml`  <  `--set key=value`。
`hpc` profile 只改 `workers`，**完整协议（epoch、λ、ERF）不被覆盖**，所以 `hpc` 上跑的就是 `configs/*.yaml` 本身。
配置里用 `${FATE_DATA}` 引用数据路径，不写机器相关的绝对路径。

## 3. 三个环境变量（路径约定）
| 变量 | 默认 | 含义 |
|---|---|---|
| `FATE_DATA` | `<repo>/data` | 数据根目录；三处**目录结构必须一致**，如 `$FATE_DATA/CUB_200_2011/...` |
| `FATE_RUNS` | `<repo>/runs` | 运行输出根目录（含 ckpt，不入 git） |
| `FATE_PROFILE` | 自动 | 强制 profile |

HPC 上 `FATE_DATA/FATE_RUNS` 指向 RDS/scratch（配额核实后定，`hpc/job_train.pbs` 里有 TODO）。

## 4. 同步协议（谁去哪）
| 内容 | 位置 | 同步方式 |
|---|---|---|
| 代码、配置、文档 | git（分支 `zihaochen_claude/festive-franklin-hk743v`） | `git push` / `git pull`（HPC 登录节点 pull） |
| 数据集 | `$FATE_DATA`，**不入 git** | 各机器各自下载；`python scripts/check_data.py` 校验布局 |
| 预训练权重 | torch hub 缓存 | 在 HPC **登录节点**预下载（计算节点可能无外网） |
| 运行产物 | `$FATE_RUNS/<exp>/<seed>/`：`ckpt.pt`、`config.yaml`、`manifest.json`、`metrics_*.json` | 不入 git |
| 可发表的小结果 | `results/<exp>/<seed>/`：manifest + config + metrics（无 ckpt） | `python scripts/collect.py [exp]` → `git add results` → 提交 |

## 5. 每次运行必写 `manifest.json`
包含：时间、主机、profile、`git_commit`、`git_dirty`、python/torch 版本、设备名、`config_sha`、调度器 job id。
**入表规则**：只收 `profile ∈ {gpu,hpc}` 且 `git_dirty=false` 的结果；同一张表的所有行 `git_commit` 应一致或在表注说明。

## 6. 标准工作流
```bash
# ① 本机/云：改代码 → 冒烟（profile 自动为 local）
pytest -q && python -m fatecnn.train --config configs/synthetic.yaml
git commit && git push

# ② HPC 登录节点：同步 + 一次性环境
git pull && bash hpc/setup_env.sh          # 仅首次
python scripts/check_data.py               # 数据布局一致性
SEED=0 qsub hpc/job_train.pbs              # 作业内 profile=hpc，训练+评估
# 3 个种子：SEED=0/1/2

# ③ 回收结果
python scripts/collect.py cub_r18 && git add results && git commit -m "results: cub_r18" && git push
```

## 7. 一致性与可复现约定
- 种子固定（`seed_all`）；CPU 与 GPU 数值**不保证逐位一致**，对等性标准：合成数据冒烟在三种 profile 上都能跑通且指标同量级；正式结论只依据 `hpc`。
- 读出层的协方差求解始终 fp32，不进 AMP。
- 评估协议（遮挡类型/面积/填充）写在代码里（`fatecnn/occlusion.py`），不随 profile 改变。
- 改动协议文件（`docs/02_protocol.md`、`occlusion.py`、`metrics.py`）须单独提交并在 commit message 标明，旧结果作废需重跑。

## 8. 未决项
- Imperial 调度器（PBS/Slurm）、队列、`gpu_type`、walltime、配额：待核实，只改 `hpc/` 与 `configs/profiles/hpc.yaml`，不动代码。
- `hpc` 上 git 访问方式（HTTPS token / SSH key）：待你配置。
