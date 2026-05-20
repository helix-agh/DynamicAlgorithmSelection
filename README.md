# DynamicAlgorithmSelection2

RL-based Dynamic Algorithm Selection (DAS) on the [BBOB benchmark](https://numbbo.github.io/coco/). A controller learns to switch between a portfolio of black-box optimizers at runtime — allocating function-evaluation budget to whichever optimizer is most promising at each checkpoint.

---

## Agents

Three agent families share the same BBOB problem set and evaluation protocol:

| Agent | Description | Key reference |
|---|---|---|
| **PPO** | Stable Baselines 3 PPO with VecNormalize; multi-dimensional; ELA-based observations | — |
| **RL-DAS** | Custom single-dimension PyTorch PPO; DE-only portfolio; population-state features with local sampling | Guo et al., 2024 |
| **Exp-DAS** | Custom PyTorch PPO with exponential checkpoint spacing; flexible portfolio | — |

### PPO
Uses `DASEnv` — a Gymnasium environment that wraps a warm-started portfolio of arbitrary optimizers. Observations are 22-dimensional ELA landscape features plus per-optimizer movement history. Trains across multiple dimensions simultaneously.

### RL-DAS
Faithful port of [Guo et al. 2024](https://doi.org/10.1145/3638529.3654223) with BBOB adaptations:
- Fixed DE portfolio: **NL_SHADE_RSP**, **MadDE**, **JDE21** (all share a single `Population` object as mutable warm-started state).
- 9-dimensional population-state features computed via local sampling (2 independent forward passes on a population deepcopy).
- Movement embedder networks compress D-dim displacement vectors to scalars; the backbone is dimension-specific (one model per `--dim`).
- Hand-rolled PPO training loop (no SB3 dependency for this agent).

### Exp-DAS
Evolution of the original DAS `policy-gradient` agent. Uses `DASEnv` (same as PPO) but replaces uniform checkpoint spacing with an **exponential schedule** controlled by the Checkpoint Division Base (`--cdb`):

- **`cdb = 1.0` (uniform):** every checkpoint covers the same number of function evaluations — consistent monitoring throughout the run.
- **`cdb > 1.0` (exponential):** early checkpoints are short (frequent switching during initial exploration) and later checkpoints are long (uninterrupted convergence during exploitation).

The agent uses separate actor and critic learning rates and a configurable number of PPO gradient epochs per update. Like PPO, it supports multiple dimensions simultaneously and an arbitrary optimizer portfolio.

---

## Installation

Requires Python 3.11. Dependency management via [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

---

## Quick start

`run_local.sh` runs a single agent with tiny settings (fast smoke test):

```bash
bash run_local.sh [seed] [agent] [portfolio...]

# Examples
bash run_local.sh 42 ppo          CPSO NM TDE
bash run_local.sh 42 ppo-cv       CPSO NM TDE
bash run_local.sh 42 rl-das               # DE portfolio fixed; no -p needed
bash run_local.sh 42 rl-das-cv
bash run_local.sh 42 exp-das      CPSO NM TDE
bash run_local.sh 42 exp-das-cv   CPSO NM TDE
bash run_local.sh 42 baselines    CPSO NM TDE
```

Run the full smoke-test suite (all agent types):

```bash
bash smoke_test.sh
# or selectively
bash smoke_test.sh rl-das rl-das-cv
```

---

## Training

```bash
python train.py {ppo,rl-das,exp-das} <name> [options]
```

### PPO

```bash
python train.py ppo MY_PPO \
    -p CPSO NM TDE \
    -d 2 5 10 \
    -E 20 \
    --fe-multiplier 10000 \
    --n-checkpoints 10 \
    --seed 42
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `-p / --portfolio` | `SPSO IPSO SPSOL` | Sub-optimizer names |
| `-d / --dims` | all | Problem dimensions |
| `-E / --n-epochs` | 20 | Passes over the training set |
| `--fe-multiplier` | 10 000 | Budget = multiplier × dimension |
| `--n-checkpoints` | 10 | Optimizer-selection steps per episode |
| `-x / --cdb` | 1.0 | Checkpoint division base (1 = uniform) |
| `-O / --reward-option` | 1 | Reward shaping (1–4) |
| `--wandb` | off | Log to Weights & Biases |

Outputs: `models/<name>.zip`, `models/<name>_vecnorm.pkl`

### RL-DAS

```bash
python train.py rl-das MY_RLDAS \
    --dim 10 \
    --n-epochs 20 \
    --fe-multiplier 10000 \
    --seed 42
```

The portfolio is fixed to `NL_SHADE_RSP MADDE JDE21` and `--n-individuals` defaults to 170 (matching the original paper). Use `--portfolio` to override.

Key options:

| Flag | Default | Description |
|---|---|---|
| `--dim` | 10 | Problem dimension (one model per dim) |
| `--n-epochs` | 20 | Training epochs |
| `--lr` | 1e-5 | Learning rate |
| `--k-epoch` | `0.3 × n_checkpoints` | PPO gradient steps per episode |
| `--device` | cpu | PyTorch device |

Outputs: `models/<name>_final.pt`, `models/<name>_epoch<N>.pt`, `models/<name>_train_log.jsonl`

### Exp-DAS

```bash
python train.py exp-das MY_EXPDAS \
    -p CPSO NM TDE \
    --dims 2 5 10 \
    -E 3 \
    --cdb 2.0 \
    --reward-option 1 \
    --seed 42
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--dims` | `2 5 10` | Problem dimensions |
| `--cdb` | 2.0 | Checkpoint Division Base (see below) |
| `-E / --n-epochs` | 3 | Passes over the training set |
| `--actor-lr` | 3e-5 | Actor learning rate |
| `--critic-lr` | 1e-5 | Critic learning rate |
| `--ppo-epochs` | 6 | PPO gradient epochs per update |
| `--buffer-capacity` | `16 × n_checkpoints` | PPO rollout buffer size in steps |
| `-O / --reward-option` | 1 | Reward shaping strategy (1–4, see below) |
| `--save-interval` | 500 | Save a checkpoint every N episodes |
| `--device` | cpu | PyTorch device |

Outputs: `models/<name>_best.pt`, `models/<name>_final.pt`, `models/<name>_ep<N>.pt`, `models/<name>_train_log.jsonl`

---

## Checkpoint Division Base (CDB)

The `--cdb` argument controls how the total FE budget is distributed across the `n_checkpoints` decision points in each episode.

With `cdb = 1.0` every checkpoint covers the same number of FEs (uniform). With `cdb > 1.0` checkpoint durations grow exponentially: the first checkpoints are short (fast switching during early exploration) and the last are long (uninterrupted convergence during exploitation).

```
cdb = 1.0  →  [───][───][───][───][───]   uniform
cdb = 2.0  →  [─][──][────][────────]    exponential
```

**When to use each value:**

| Value | Effect | Use case |
|---|---|---|
| `1.0` | Equal-length checkpoints | Consistent monitoring; PPO default |
| `2.0` | Moderate exponential growth | Exp-DAS default; balances exploration and exploitation |
| `> 2.0` | Aggressive early switching | Portfolios where early optimizer choice is decisive |

The `--cdb` flag is available for all three agents (`ppo`, `rl-das` ignores it, `exp-das`).

---

## Reward options

The `-O / --reward-option` flag selects the reward signal used at each checkpoint. All options measure improvement in the best objective value found so far and scale it by the initial value range.

| Option | Name | Description |
|---|---|---|
| `1` | Log-scaled improvement | `improvement` between consecutive checkpoints, clipped to `[0, 1]`, then `log(r + 1e-5)`. Smooths large variance. **Default.** |
| `2` | Linear clipped improvement | Same as option 1 but without the log transform: `clip(improvement, 0, 1)`. |
| `3` | Sparse total improvement | Returns `0` at every intermediate checkpoint; at the final checkpoint returns the log-scaled total improvement from episode start. Focuses the agent on end-of-run quality. |
| `4` | Binary threshold | Returns `1` if scaled improvement ≥ `1e-3`, else `0`. Simple binary feedback. |

---

## Cross-validation

```bash
python cv.py {ppo,rl-das,exp-das} <name> [options]
```

Two CV modes:

- **LOIO** (Leave-One-Instance-Out): hold out a subset of BBOB instances per fold.
- **LOPO** (Leave-One-Problem-Out): hold out a subset of BBOB functions per fold.

```bash
# PPO – 3-fold LOIO
python cv.py ppo MY_PPO_CV \
    -p CPSO NM TDE -d 5 10 \
    --cv-mode LOIO --n-folds 3 --n-epochs 10 --seed 42

# RL-DAS – 3-fold LOPO, dim 10 only
python cv.py rl-das MY_RLDAS_CV \
    --dim 10 --cv-mode LOPO --n-folds 3 --n-epochs 20 --seed 42

# Run only folds 0 and 2
python cv.py exp-das MY_EXPDAS_CV \
    -p CPSO NM TDE --dims 5 10 \
    --cv-mode LOIO --folds 0 2 --n-epochs 3 --seed 42
```

Outputs per fold: `results/<name>_cv_<fold_tag>.jsonl`
Aggregated: `results/<name>_cv_summary.jsonl`

---

## Baselines

```bash
python baselines.py <name> --agent <agent_type> [options]
```

Agent types:

| Type | Description |
|---|---|
| `random` | Uniform random selection at each checkpoint |
| `fixed:<name>` | Always pick one optimizer, e.g. `fixed:CPSO` |
| `single:<name>` | One optimizer runs the full budget (no checkpointing) |
| `all` | All of the above; derives oracle-best / oracle-worst |

```bash
python baselines.py MY_BASELINES --agent all \
    -p CPSO NM TDE -d 2 5 10 --seed 42
```

---

## Evaluation

Load a trained PPO model and evaluate it on the BBOB test set:

```bash
python evaluate.py MY_PPO \
    -p CPSO NM TDE -d 5 10 --seed 42
```

Add `--coco-observer` to write COCO-compatible data for `cocopp` post-processing.

---

## Problem set

The BBOB benchmark provides 24 functions × 15 instances × 6 dimensions = 2 160 problems per dimension.

**Dimensions:** `2, 3, 5, 10, 20, 40`

**Default train/test split** (`--mode easy`): trains on 14 structurally simpler functions and tests on the remaining 10 harder functions.

| Mode | Train | Test |
|---|---|---|
| `easy` | functions {4,6–14,18–20,22–24} | remaining 10 functions |
| `hard` | inverse of easy | — |
| `random` | 2/3 of all problems | 1/3 |

---

## Optimizer portfolio

Available sub-optimizers (pass names via `-p / --portfolio`):

| Family | Names |
|---|---|
| PSO | `SPSO`, `IPSO`, `SPSOL`, `CPSO` |
| DE | `NL_SHADE_RSP`, `MADDE`, `JDE21`, `TDE` |
| ES | `NM` (Nelder-Mead) |
| BO | `BO` |
| DS | `DS` (Direct Search) |

RL-DAS always uses the DE trio `NL_SHADE_RSP / MADDE / JDE21` — overridable with `--portfolio`.

---

## HPC / SLURM

Submit all agents for a given seed and portfolio:

```bash
bash runner.sh
```

Individual SLURM scripts:

| Script | Agent |
|---|---|
| `ppo_study.slurm` | PPO |
| `rl_das_study.slurm` | RL-DAS |
| `exp_das_study.slurm` | Exp-DAS |
| `baselines.slurm` | Baselines |

---

## Project structure

```
DynamicAlgorithmSelection2/
├── train.py              # Unified training entry point
├── cv.py                 # Cross-validation entry point
├── baselines.py          # Baseline agents
├── evaluate.py           # Model evaluation
├── run_local.sh          # Local smoke-test runner
├── smoke_test.sh         # Full smoke-test suite
├── runner.sh             # SLURM batch submission
│
├── agents/
│   ├── rl_das/           # RL-DAS (Guo et al. 2024 port)
│   │   ├── env.py        # RLDASEnv: Population-based Gymnasium env
│   │   ├── optimizers.py # NL_SHADE_RSP, JDE21, MadDE (BBOB-adapted)
│   │   ├── population.py # Shared mutable Population state (NLPSR)
│   │   ├── agent.py      # PPOAgent (actor-critic)
│   │   ├── network.py    # Movement embedder + backbone
│   │   └── trainer.py    # train() / evaluate() loops
│   └── exponential_das/  # Exp-DAS agent
│
├── das/
│   ├── env/
│   │   ├── das_env.py    # DASEnv: Gymnasium env for PPO / Exp-DAS
│   │   ├── bbob_splits.py# BBOB problem IDs, train/test/CV splits
│   │   ├── observation.py# ELA feature extraction (22-dim)
│   │   └── reward.py     # Reward shaping options
│   ├── optimizers/
│   │   ├── portfolio.py  # get_portfolio() factory
│   │   └── {PSO,DE,ES,BO,DS}/  # Sub-optimizer implementations
│   └── training/
│       ├── ppo.py        # run_ppo() / run_cv_ppo()
│       ├── rldas.py      # run_rl_das() / run_cv_rl_das()
│       ├── expdas.py     # run_exp_das() / run_cv_exp_das()
│       └── common.py     # Shared utilities (JSONL writer, etc.)
│
├── tests/                # pytest test suite
└── pyproject.toml
```

---

## References

- Guo, Y. et al. (2024). *Deep Reinforcement Learning for Dynamic Algorithm Selection: A Proof-of-Principle Study on Differential Evolution*. GECCO 2024. https://doi.org/10.1145/3638529.3654223
- Hansen, N. et al. (2021). *COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting*. Optimization Methods and Software.