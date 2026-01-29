# RA-UCB

Implementation of the **Resource Allocation UCB (RA-UCB)** algorithm for multi-armed bandits with censored feedback under Weibull-distributed delays.

## Overview

This repository contains the code for experiments on resource allocation bandits where:
- Each arm has a conversion probability `p` and a delay distribution (Weibull)
- The learner allocates a budget `B` across `K` arms at each round
- Feedback is censored: conversions are only observed if they occur before the allocated time

## Repository Structure

### Core Modules

| File | Description |
|------|-------------|
| `utils_weibull.py` | Core functions for RA-UCB with **continuous Weibull** distribution. Includes oracle optimization, g-function utilities, confidence bounds, and simulation functions. |
| `Utils_weibull_discrete.py` | Core functions for RA-UCB with **discrete Weibull** distribution. Adapted for integer-valued delay times with dynamic programming optimization. |

### Experiments

| Notebook | Description |
|----------|-------------|
| `main_exponential.ipynb` | Baseline experiments with exponential delay distribution (Weibull with k=1) |
| `main_expe_criteo.ipynb` | Experiments using Criteo advertising dataset with discrete Weibull model |
| `main_expe_ednet.ipynb` | Experiments using EdNet educational dataset with continuous Weibull model |

### Data

| File | Description |
|------|-------------|
| `criteo_exposures_before_click.csv.zip` | Criteo dataset: ad exposures and click delays |
| `pseudo_users_dataset.csv` | Processed user data for experiments |

## Key Functions

### Continuous Weibull (`utils_weibull.py`)
- `oracle_weibull_softmax()` - Solves the budget allocation optimization problem
- `g_weibull()` / `g_inv_weibull()` - Functions for parameter estimation
- `compute_confidence_bounds_weibull_vec()` - UCB confidence bounds
- `simulate_one_round_weibull()` - Simulation of one bandit round

### Discrete Weibull (`Utils_weibull_discrete.py`)
- `oracle_discrete_weibull()` - Budget allocation with optional DP solver
- `discrete_weibull_pmf()` / `discrete_weibull_cdf()` - Distribution functions
- `compute_confidence_bounds_discrete_weibull_vec()` - UCB confidence bounds
- `simulate_one_round_discrete_weibull()` - Simulation of one bandit round

## Dependencies

```
numpy
scipy
pandas
matplotlib
tqdm
```

## Usage

Run the sanity checks to verify the implementation:

```python
from utils_weibull import sanity_check
sanity_check(T=5000, K=3, B=40.0, seed=0)

from Utils_weibull_discrete import sanity_check_discrete
sanity_check_discrete(T=5000, K=3, B=40, seed=0)
```

Or explore the Jupyter notebooks for full experiments.
