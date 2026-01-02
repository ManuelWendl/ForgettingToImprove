# Forgetting to Improve

A Python package implementing robust Gaussian Process regression with sample optimization and forgetful Bayesian optimization techniques.

## Overview

This package contains two main components:

1. **Forgetting to Improve** - Robust GP regression with selective sample removal
2. **Forgetful Bayesian Optimization** - Bayesian optimization with target region sampling

## Table of Contents

- [Installation](#installation)
- [Components](#components)
  - [Forgetting to Improve](#forgetting-to-improve-robust-regression)
  - [Forgetful Bayesian Optimization](#forgetful-bayesian-optimization)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Examples](#examples)
- [Requirements](#requirements)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch
- GPyTorch
- BoTorch

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd ForgettingToImproveBO

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Components

### Forgetting to Improve (Robust Regression)

A robust Gaussian Process regression framework that improves model performance by selectively removing training samples that negatively impact predictions on a target region.

#### Key Features

- **Sample Optimization**: Sequential and batch methods for removing harmful training samples
- **Uncertainty Decomposition**: Separate handling of epistemic and aleatoric uncertainty
- **Multiple Methods**:
  - `sequential`: Iteratively remove worst samples
  - `batch`: Remove multiple samples at once
  - `targetSampling`: Adaptive target region sampling
  - `relevance_pursuit`: BoTorch robust relevance pursuit
  - `warped_gp`: Warped GP with input transformation
  - `heteroscedastic_gp`: GP with heteroscedastic noise
  - `none`: Standard GP baseline

#### Noise Types for Influence Calculation in sequential or batch methods

- `joint`: Consider both epistemic and aleatoric uncertainty (default)
- `epistemic`: Focus on model uncertainty
- `aleatoric`: Focus on data/observation noise

#### Usage

```bash
python -m forgetting_to_improve.forgetting_to_improve.main \
  -c forgetting_to_improve/forgetting_to_improve/configs/single_experiment_symmetric.yaml
```

#### Configuration Example

```yaml
experiment:
  type: single  # or 'statistic' for multiple runs
  method: sequential
  objective: sin_symmetric_lengthscale_increase
  noise: epistemic
  A_limits: [-3, 3]  # Active learning region
  X_limits: [-10, 10]  # Full domain
  num_samples: 1000
  num_train_samples: 20
  num_A_samples: 50
  seed: 0
  fixed_kernel: true
  show_plots: true
  show_hessian: false

kernel_config:
  - type: 'rbf'
    length_scale: 1.0
```

#### Supported Objectives

- Synthetic functions: `botorch`, `sin_symmetric_lengthscale_increase`, `sin_asymmetric_lengthscale_increase`, `sin_periodic_lengthscale_increase`
- Real datasets: `boston`, `twitter`

---

### Forgetful Bayesian Optimization

A Bayesian optimization framework that incorporates sample forgetting and target region focusing for improved optimization performance.

#### Key Features

- **Target Region Sampling**: Focus optimization on promising regions
- **Sample Filtering**: Remove unhelpful samples during optimization
- **Multiple Algorithms**:
  - `none`: Standard Bayesian optimization
  - `joint`: Joint epistemic and aleatoric filtering with target regions
  - `epistemic`: Epistemic-only filtering with target regions
- **Acquisition Functions**: Support for various BoTorch acquisition functions (UCB, EI, PI, etc.)
- **Reproducible**: Deterministic sampling with seed control

#### Usage

```bash
python -m forgetting_to_improve.forgetfull_bayesian_optimization.main \
  -c forgetting_to_improve/forgetfull_bayesian_optimization/configs/config_example.yaml
```

#### Configuration Example

```yaml
bayes_opt:
  type: comparative
  objective: botorch_ackley_2d  # or other BoTorch test functions
  n_seeds: 20
  init_points: 10
  n_iter: 50
  acquisition: qUpperConfidenceBound
  acquisition_params:
    kappa: 2.576
  save_results: true
  plot_results: true
  results_path: botorch_ackley_2d_statistical.txt
  
  sample_optimization:
    algorithm: [none, joint]  # Compare multiple algorithms
    method: sequential
    min_samples: 5
    
  kernel:
    - type: matern
      nu: 2.5
      length_scale: 1.0
      
  alpha: 0.1  # GP noise parameter
```

#### Supported Test Functions

Any BoTorch test function can be used by prefixing with `botorch_`:
- `botorch_ackley_2d`
- `botorch_branin`
- `botorch_hartmann_6d`
- etc.

---

## Quick Start

### Example 1: Robust GP Regression

```python
from forgetting_to_improve.forgetting_to_improve.main import run_experiments_from_config

# Run single experiment
run_experiments_from_config(
    'forgetting_to_improve/forgetting_to_improve/configs/single_experiment_symmetric.yaml'
)
```

### Example 2: Bayesian Optimization

```python
from forgetting_to_improve.forgetfull_bayesian_optimization.main import (
    run_bayesian_optimization_experiment
)

# Run Bayesian optimization experiment
run_bayesian_optimization_experiment(
    'forgetting_to_improve/forgetfull_bayesian_optimization/configs/config_example.yaml'
)
```

### Example 3: Statistical Comparison

```bash
# Compare multiple methods across many seeds
python -m forgetting_to_improve.forgetting_to_improve.main \
  -c forgetting_to_improve/forgetting_to_improve/configs/statistical_comparison_sin_symmetric_lengthscale_increase.yaml
```

---

## Configuration

### Experiment Types

#### Forgetting to Improve
- `single`: Run a single experiment with given parameters
- `statistic`: Run statistical comparison across multiple seeds

#### Forgetful Bayesian Optimization
- `comparative`: Compare multiple algorithms on the same problem

### Output

Results are saved to:
- **Forgetting to Improve**: `forgetting_to_improve/forgetting_to_improve/results/`
- **Forgetful Bayesian Optimization**: `forgetting_to_improve/forgetfull_bayesian_optimization/results/`

Figures are saved to:
- **Forgetting to Improve**: `forgetting_to_improve/forgetting_to_improve/figures/`
- **Forgetful Bayesian Optimization**: `forgetting_to_improve/forgetfull_bayesian_optimization/figures/`
---

## Examples

### Running Experiments

```bash
# Robust GP with sequential optimization
python -m forgetting_to_improve.forgetting_to_improve.main \
  -c forgetting_to_improve/forgetting_to_improve/configs/single_experiment_symmetric.yaml

# Statistical comparison on Boston housing dataset
python -m forgetting_to_improve.forgetting_to_improve.main \
  -c forgetting_to_improve/forgetting_to_improve/configs/boston.yaml

# Forgetful Bayesian optimization on Ackley function
python -m forgetting_to_improve.forgetfull_bayesian_optimization.main \
  -c forgetting_to_improve/forgetfull_bayesian_optimization/configs/config_example.yaml
```

---

## Requirements

See `requirements.txt` for complete list.

---

## Project Structure

```
ForgettingToImproveBO/
├── forgetting_to_improve/
│   ├── forgetting_to_improve/          # Robust regression
│   │   ├── main.py
│   │   ├── evaluate.py
│   │   ├── optimize.py
│   │   ├── models.py
│   │   ├── objectives.py
│   │   ├── uncertainty.py
│   │   ├── configs/
│   │   └── helper/
│   └── forgetfull_bayesian_optimization/  # BO with forgetting
│       ├── main.py
│       ├── opti_loop.py
│       ├── target_region.py
│       ├── filter_samples.py
│       ├── configs/
│       └── helper/
├── results/
├── figures/
├── requirements.txt
└── README.md
```

---

## Key Algorithms

### Sample Influence Calculation

The package computes the influence of each training sample on prediction quality in a target region by analyzing:
- **Epistemic uncertainty**: How much each sample reduces model uncertainty
- **Aleatoric uncertainty**: How each sample affects prediction variance

Samples with negative total influence are candidates for removal.

### Target Region Optimization

For Bayesian optimization, the algorithm:
1. Identifies promising regions using upper confidence bounds
2. Filters training samples based on their influence
3. Optimizes acquisition function within target regions
4. Updates the GP model with new observations

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{tbd,
  title={TBD},
  author={[Author Names]},
  year={2026}
}
```
