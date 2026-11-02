# Forgetting to Improve

Code for the paper **Forgetting to Improve: Principled Data Removal in Active Learning**:

The uncertainty of a statistical model is most commonly factorized into an aleatoric and an epistemic part. This factorization changes how predictions are interpreted in downstream decision tasks. Importantly, except for the idealistic scenario with no model mismatch, the quantification is a characteristic of the model and not the data generating process. In this paper, we propose Forgetting to Improve a method that reduces this discrepancy by incorporating the task into the modeling framework. Our key insight is to acknowledge that in scenarios of model mismatch, data can have a detrimental effect on the modeling for a specific task. Based on this insight, we propose an influence function for Gaussian process models that allows for principled removal of detrimental data samples. We showcase the flexibility of this approach by demonstrating significant improvements across a range of tasks, including Bayesian optimization, model-based reinforcement learning, and transductive learning.


This repository contains:

| Package | What it does |
|---|---|
| [`forgetting_to_improve/forgetting_to_improve`](forgetting_to_improve/forgetting_to_improve) | F2I for GP regression on a target region (transductive benchmarks) and the restricted-curvature / gradient-dominance analysis |
| [`forgetting_to_improve/forgetful_bayesian_optimization`](forgetting_to_improve/forgetful_bayesian_optimization) | Forgetful BO: F2I applied to the GP surrogate in every BO iteration, with the target region around promising points |
| [`forgetting_to_improve/forgetful_rl`](forgetting_to_improve/forgetful_rl) | Forgetful RL: MBPO with a GP dynamics model whose real-data replay buffer is filtered with F2I |

## Installation

Tested with Python 3.12 on Linux.

```bash
git clone <repository-url> ForgettingToImprove
cd ForgettingToImprove
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All commands below are run from the repository root. Figures use LaTeX text
rendering when a `latex` binary is on the `PATH` and fall back to matplotlib's
mathtext otherwise. The RL experiments use `cuda:0` by default; add `device=cpu`
to run them on CPU.

## Reproducing the experiments

### 1. Twitter flash-crash comparison (GP vs. F2I)

```bash
python -m forgetting_to_improve.forgetting_to_improve.main \
    -c forgetting_to_improve/forgetting_to_improve/configs/comparison_twitter.yaml
```

Writes `forgetting_to_improve/forgetting_to_improve/figures/comparison_twitter_seed5.pdf`
(predictions) and `..._calibration.pdf`. Runs in a few seconds.

### 2. Transductive regression benchmarks (100 seeds)

Six benchmarks: Twitter flash crash, Yacht, Boston, CASP, and two synthetic
functions with symmetric / asymmetric lengthscale increase. Each compares a
homoscedastic GP (`none`), F2I (`sequential`, with the joint and the
aleatoric-only criterion), target-region sampling, relevance pursuit, a warped GP
and a heteroscedastic GP.

```bash
bash scripts/run_transductive.sh                # all six
bash scripts/run_transductive.sh twitter yacht  # a subset
```

Each benchmark writes one table (mean ± std over 100 seeds of RMSE, MAE, NLL,
calibration error, sharpness, runtime) to
`forgetting_to_improve/forgetting_to_improve/results/comparison_<benchmark>_method_noise_<timestamp>.txt`,
and the per-seed values to `..._per_seed.json` next to it.
All six take about 1.5 to 3 hours on a 22-core CPU.
These configs set `noise_init: data_variance`: the GP noise variance from which the
removal criterion starts is the variance of each seed's training targets (instead of
the kernel config's `white: noise_level`, the default for all other experiments).

Bar comparison of all methods (newest table of each benchmark, written to
`forgetting_to_improve/forgetting_to_improve/figures/benchmark_bar_comparison.pdf`):

```bash
python -m forgetting_to_improve.forgetting_to_improve.plot_transductive bar
```

**Noise ablation** (joint vs aleatoric-only removal criterion on Twitter and the
asymmetric synthetic function, Forgetting to Improve only, initial GP noise from
the kernel config; about 15 minutes):

```bash
bash scripts/run_transductive.sh noise_ablation
python -m forgetting_to_improve.forgetting_to_improve.plot_transductive noise_ablation
```

Writes `figures/noise_ablation_bars.pdf` (mean over seeds) and
`figures/noise_ablation_paired.pdf` (per-seed comparison with Wilcoxon tests).

The datasets are included in `forgetting_to_improve/forgetting_to_improve/helper/`:
Twitter flash crash (from the BoTorch tutorials), Boston housing, and Yacht
hydrodynamics and CASP (both from the UCI Machine Learning Repository).

### 3. Restricted curvature and gradient dominance

```bash
# lambda_min / lambda_max of the restricted Hessian, the empirical ratio rho_k and the
# gradient-dominance upper bound along the removal path (symmetric, asymmetric, twitter)
python -m forgetting_to_improve.forgetting_to_improve.run_curvature_single_grid

# the same curvature quantities for different Gamma noise-prior concentrations a
python -m forgetting_to_improve.forgetting_to_improve.run_curvature_ablations
```

Outputs: `figures/restricted_curvature_grid.pdf` and `figures/curvature_ablation_grid.pdf`
(in `forgetting_to_improve/forgetting_to_improve/`); about 5 and 10 minutes.

The closed-form gradients and Hessians used here are checked against autodiff /
finite differences in `tests/`:

```bash
python -m pytest tests
```

### 4. Forgetful Bayesian optimization

Eight BoTorch test functions (Ackley 2D/6D, augmented Rosenbrock, Branin,
Griewank, Hartmann 6D, Holder table, Levy), 10 seeds each, with UCB or LogEI and
with clean or corrupted observations (`*_noisy`: with probability 0.1 an
observation is replaced by -100). Methods: standard BO (`none`), forgetful BO
(`joint`), warped GP, relevance pursuit, trimmed MLL, Student-t GP and target-region
filtering (the last three only with UCB).

```bash
bash scripts/run_bo.sh            # all 32 configurations
bash scripts/run_bo.sh branin     # only configurations containing "branin"
```

The full suite is compute-heavy (about 50 hours when run sequentially, most of it relevance
pursuit); configurations are independent and can be run in parallel.

Results are written per group to
`forgetting_to_improve/forgetful_bayesian_optimization/results/{ucb,ucb_noisy,lei,lei_noisy}/`.
To plot simple regret, cumulative regret, best value and runtime of one group:

```bash
python -m forgetting_to_improve.forgetful_bayesian_optimization.helper.plot_saved_results \
    -f forgetting_to_improve/forgetful_bayesian_optimization/results/ucb/ \
    -o forgetting_to_improve/forgetful_bayesian_optimization/figures/ucb/
```

Illustration of forgetful BO on a 1D problem (GP before/after forgetting, target
region, next query):

```bash
python -m forgetting_to_improve.forgetful_bayesian_optimization.plot_bo_snapshots
```

### 5. Forgetful RL: cartpole balance

MBPO with a linear-kernel GP dynamics model on continuous cartpole, with
(`forgetful_rl=true`) and without (GP-MBPO) F2I filtering of the real-data
replay buffer, 5 seeds, 14k environment steps. The configuration is in
`forgetting_to_improve/forgetful_rl/config/` (`main.yaml`, `algorithm/mbpo.yaml`,
`dynamics_model/gp_model.yaml`, `overrides/mbpo_cartpole.yaml`).

```bash
bash scripts/run_rl_cartpole.sh            # 5 seeds x {Forgetful RL, GP-MBPO}
bash scripts/run_rl_cartpole.sh removal    # one run that stores the removed samples

python -m forgetting_to_improve.forgetful_rl.plotting.plot_cartpole_balance \
    --runs-dir exp/mbpo/cartpole_balance
python -m forgetting_to_improve.forgetful_rl.plotting.plot_removal_visualization \
    --runs-dir exp/mbpo/cartpole_balance_removal
```

Each run takes 15-25 minutes on a GPU. Runs are written to
`exp/mbpo/<experiment>/...`; each run folder contains its
Hydra config, all logged metrics (`results.jsonl`) and, for the removal run, the
kept/removed samples of every filtering step (`data_removal/`). Runs are also
logged to Weights & Biases in offline mode (`wandb.mode=online
wandb.entity=<entity>` to upload, `wandb.mode=disabled` to turn it off).

### Neural-network kernel variants

F2I also works with a neural-network kernel: the last-layer linearized Laplace
(Jacobian) kernel in
[`helper/nn_kernel`](forgetting_to_improve/forgetting_to_improve/helper/nn_kernel).
The corresponding configurations end in `_jacobian`:

```bash
bash scripts/run_transductive.sh nn        # statistical_*_jacobian.yaml
python -m forgetting_to_improve.forgetting_to_improve.main \
    -c forgetting_to_improve/forgetting_to_improve/configs/comparison_twitter_jacobian.yaml
bash scripts/run_bo.sh nn                   # *_jacobian(_noisy).yaml
```

## Repository structure

```
forgetting_to_improve/
├── forgetting_to_improve/              # F2I for GP regression
│   ├── main.py                         # entry point: python -m ... -c <config.yaml>
│   ├── evaluate.py                     # one experiment: fit, remove samples, evaluate on A
│   ├── optimize.py                     # sequential / batch sample removal
│   ├── uncertainty.py                  # epistemic + aleatoric removal criterion
│   ├── curvature_bounds.py             # restricted curvature, gradient-dominance bound
│   ├── models.py                       # baseline models (heteroscedastic GP, ...)
│   ├── objectives.py                   # synthetic objectives
│   ├── run_curvature_single_grid.py
│   ├── run_curvature_ablations.py
│   ├── plot_transductive.py            # bar comparison and noise-ablation figures
│   ├── configs/
│   └── helper/                         # data, kernels, metrics, plotting, nn_kernel/
├── forgetful_bayesian_optimization/
│   ├── main.py                         # entry point: python -m ... -c <config.yaml>
│   ├── opti_loop.py                    # BO loop for all methods
│   ├── filter_samples.py, target_region.py, baseline_models.py
│   ├── plot_bo_snapshots.py
│   ├── configs/
│   └── helper/                         # config parsing, result writing, plotting
└── forgetful_rl/
    ├── main.py                         # Hydra entry point
    ├── algorithm/mbpo.py               # MBPO training loop with forgetting
    ├── forgetting/                     # replay-buffer filtering and target region
    ├── model/                          # GP dynamics models
    ├── env/                            # continuous cartpole, reward/termination functions
    ├── helper/                         # SAC, logging, utilities
    ├── plotting/
    └── config/
scripts/                                # run_transductive.sh, run_bo.sh, run_rl_cartpole.sh
tests/                                  # derivative checks for the curvature analysis
```

## Citation

```bibtex
@inproceedings{
wendl2026forgetting,
title={Forgetting to Improve: Principled Data Removal in Active Learning},
author={Manuel Wendl, Erik Englesson, Andreas Krause, Carl Henrik Ek},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=wF1w6w0jmM}
}
```

## Acknowledgements

The RL code builds on [MBRL-Lib](https://github.com/facebookresearch/mbrl-lib)
(MIT license; files keep their original headers) and
[pytorch-soft-actor-critic](https://github.com/pranz24/pytorch-soft-actor-critic)
(MIT license, see `forgetting_to_improve/forgetful_rl/helper/pytorch_sac_pranz24/LICENSE`).
The DeepMind Control wrapper is [dmc2gym](https://github.com/denisyarats/dmc2gym)
(MIT license, see `forgetting_to_improve/forgetful_rl/helper/dmc2gym/LICENSE`).
