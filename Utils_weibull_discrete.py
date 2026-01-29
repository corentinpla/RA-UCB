"""
RA-UCB Functions for Discrete Weibull Censored Feedback Experiments

This module contains all the core functions for the RA-UCB algorithm
with Discrete Weibull distributed censoring.

Discrete Weibull Distribution:
- PMF: P(X = k) = q^(k^β) - q^((k+1)^β) for k = 0, 1, 2, ...
- CDF: F(k) = P(X <= k) = 1 - q^((k+1)^β)
- Survival: S(k) = P(X > k) = q^((k+1)^β)

Parameterization:
- β (beta): shape parameter (known)
- q: scale parameter (unknown), where 0 < q < 1
- Alternatively: λ (lambda) where q = exp(-λ^β)

Relationship to continuous Weibull:
- Continuous: S(t) = exp(-(λt)^β)
- Discrete: S(k) = exp(-(λ(k+1))^β) = q^((k+1)^β)
"""

import numpy as np
import math
from typing import Callable, Optional

from scipy.optimize import fsolve, brentq, minimize, LinearConstraint, Bounds
from scipy.special import logsumexp


# =============================================================================
# Discrete Weibull Distribution Functions
# =============================================================================

def discrete_weibull_pmf(k: int, q: float, beta: float) -> float:
    """
    PMF of discrete Weibull distribution.
    P(X = k) = q^(k^β) - q^((k+1)^β)
    
    Parameters:
    - k: non-negative integer
    - q: scale parameter, 0 < q < 1
    - beta: shape parameter, β > 0
    """
    if k < 0:
        return 0.0
    return q ** (k ** beta) - q ** ((k + 1) ** beta)


def discrete_weibull_cdf(k: int, q: float, beta: float) -> float:
    """
    CDF of discrete Weibull distribution.
    F(k) = P(X <= k) = 1 - q^((k+1)^β)
    
    Parameters:
    - k: non-negative integer
    - q: scale parameter, 0 < q < 1
    - beta: shape parameter, β > 0
    """
    if k < 0:
        return 0.0
    return 1.0 - q ** ((k + 1) ** beta)


def discrete_weibull_survival(k: int, q: float, beta: float) -> float:
    """
    Survival function of discrete Weibull distribution.
    S(k) = P(X > k) = q^((k+1)^β)
    
    Parameters:
    - k: non-negative integer
    - q: scale parameter, 0 < q < 1
    - beta: shape parameter, β > 0
    """
    if k < 0:
        return 1.0
    return q ** ((k + 1) ** beta)


def discrete_weibull_mean(q: float, beta: float, max_k: int = 10000) -> float:
    """
    Compute the mean of discrete Weibull distribution.
    E[X] = Σ_{k=0}^{∞} S(k) = Σ_{k=0}^{∞} q^((k+1)^β)
    
    Parameters:
    - q: scale parameter, 0 < q < 1
    - beta: shape parameter, β > 0
    - max_k: maximum k for summation (truncation)
    """
    mean = 0.0
    for k in range(max_k):
        survival = q ** ((k + 1) ** beta)
        if survival < 1e-15:
            break
        mean += survival
    return mean


def discrete_weibull_sample(q: float, beta: float, rng=None) -> int:
    """
    Sample from discrete Weibull distribution using inverse CDF method.
    
    Parameters:
    - q: scale parameter, 0 < q < 1
    - beta: shape parameter, β > 0
    - rng: random number generator (numpy)
    
    Returns:
    - k: sampled value (non-negative integer)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    u = rng.random()
    # F(k) = 1 - q^((k+1)^β) = u
    # q^((k+1)^β) = 1 - u
    # (k+1)^β = log(1-u) / log(q)
    # k+1 = (log(1-u) / log(q))^(1/β)
    # k = (log(1-u) / log(q))^(1/β) - 1
    
    if u >= 1.0 - 1e-15:
        return 0
    
    log_q = np.log(q)
    if log_q >= 0:
        return 0
    
    k_continuous = (np.log(1 - u) / log_q) ** (1.0 / beta) - 1
    return max(0, int(np.floor(k_continuous)))


# =============================================================================
# Oracle and g-function utilities for Discrete Weibull
# =============================================================================

def oracle_discrete_weibull(p, q, q_prime, B, beta, n_restarts=5, seed=0, method="SLSQP", discrete=True):
    """
    Solve the budget allocation problem for discrete Weibull:
    max_{x>=0, sum x = B}  sum_i p_i * (1 - (q_prime_i/q_i) * q_i^((x_i+1)^β_i))
    
    Note: x_i are treated as continuous for optimization, then can be rounded.
    For discrete Weibull: P(T <= x) = 1 - q^((x+1)^β)
    
    Parameters:
    - p: success probabilities for each arm
    - q: scale parameters (0 < q < 1)
    - q_prime: scale parameters for UCB (usually same as q)
    - B: total budget (integer)
    - beta: shape parameters (known)
    - n_restarts: number of random restarts
    - seed: random seed
    - method: optimization method ("SLSQP", "trust-constr", "diff_evol", or "dp" for dynamic programming)
    - discrete: if True, use discrete optimization (DP or rounding), else continuous relaxation
    
    Returns:
    - x_opt: optimal allocation (integers if discrete=True)
    """
    p = np.asarray(p, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float).reshape(-1)
    q_prime = np.asarray(q_prime, dtype=float).reshape(-1)
    K = p.size
    assert q.shape == (K,)
    
    beta_arr = np.asarray(beta, dtype=float)
    if beta_arr.ndim == 0:
        beta_arr = np.full(K, float(beta_arr))
    else:
        beta_arr = beta_arr.reshape(-1)
        assert beta_arr.shape == (K,)
    
    rng = np.random.default_rng(seed)
    
    # Reward function for arm i with allocation x_i
    def reward_arm(i, x_i):
        """Expected reward for arm i with allocation x_i."""
        return p[i] * (1.0 - q_prime[i] ** ((x_i + 1) ** beta_arr[i]))
    
    # ===========================================
    # DISCRETE OPTIMIZATION: Dynamic Programming
    # ===========================================
    if method == "dp" or (discrete and B <= 200):
        # Use dynamic programming for exact discrete solution
        # dp[k][b] = max expected reward using arms 0..k-1 with budget b
        B_int = int(B)
        
        # Precompute rewards for each arm and each possible allocation
        rewards = np.zeros((K, B_int + 1))
        for i in range(K):
            for x in range(B_int + 1):
                rewards[i, x] = reward_arm(i, x)
        
        # DP with backtracking
        dp = np.zeros((K + 1, B_int + 1))
        choice = np.zeros((K + 1, B_int + 1), dtype=int)
        
        for i in range(1, K + 1):
            for b in range(B_int + 1):
                best_val = -np.inf
                best_x = 0
                for x in range(b + 1):
                    val = rewards[i - 1, x] + dp[i - 1, b - x]
                    if val > best_val:
                        best_val = val
                        best_x = x
                dp[i, b] = best_val
                choice[i, b] = best_x
        
        # Backtrack to find optimal allocation
        x_opt = np.zeros(K, dtype=int)
        remaining = B_int
        for i in range(K, 0, -1):
            x_opt[i - 1] = choice[i, remaining]
            remaining -= x_opt[i - 1]
        
        return x_opt.astype(float)
    
    # ===========================================
    # CONTINUOUS RELAXATION with optional rounding
    # ===========================================
    
    # Objective function (to minimize, so -f)
    def objective_neg(x):
        x = np.maximum(x, 0)
        # P(T <= x) = 1 - q^((x+1)^β)
        cdf_vals = 1.0 - q ** ((x + 1) ** beta_arr)
        # For UCB: use q_prime/q ratio
        val = np.sum(p * (1.0 - (q_prime / q) * (1.0 - cdf_vals)))
        # Simplify: val = sum p_i * (1 - q_prime_i^((x_i+1)^β_i) / q_i^((x_i+1)^β_i) * q_i^((x_i+1)^β_i))
        # val = sum p_i * (1 - q_prime_i^((x_i+1)^β_i))
        val = np.sum(p * (1.0 - q_prime ** ((x + 1) ** beta_arr)))
        return -val
    
    # Analytical gradient for SLSQP/trust-constr
    def gradient_neg(x):
        x = np.maximum(x, 1e-10)
        # d/dx [q^((x+1)^β)] = q^((x+1)^β) * ln(q) * β * (x+1)^(β-1)
        log_q_prime = np.log(np.maximum(q_prime, 1e-15))
        grad = -p * (q_prime ** ((x + 1) ** beta_arr)) * log_q_prime * beta_arr * ((x + 1) ** (beta_arr - 1))
        return grad
    
    if method == "diff_evol":
        from scipy.optimize import differential_evolution
        
        def objective_simplex(y):
            x = B * y / np.sum(y) if np.sum(y) > 0 else np.full(K, B/K)
            return objective_neg(x)
        
        bounds_de = [(1e-6, 1.0) for _ in range(K)]
        
        result = differential_evolution(
            objective_simplex, 
            bounds_de, 
            seed=seed,
            maxiter=1000,
            tol=1e-10,
            atol=1e-10,
            workers=1,
            updating='deferred',
            polish=True
        )
        
        x_opt = B * result.x / np.sum(result.x)
        return x_opt
    
    else:
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - B}
        bounds = [(0, B) for _ in range(K)]
        
        # Initialize with score proportional to expected reward
        score = np.maximum(1e-12, p * (-np.log(np.maximum(q, 1e-15))))
        x0_base = B * score / np.sum(score)
        
        best = None
        best_val = np.inf
        
        for r in range(n_restarts):
            if r == 0:
                x_init = x0_base.copy()
            else:
                noise = rng.uniform(-0.2, 0.2, K) * x0_base
                x_init = np.maximum(x0_base + noise, 0)
                x_init = B * x_init / np.sum(x_init)
            
            try:
                if method == "trust-constr":
                    linear_constraint = LinearConstraint(np.ones((1, K)), B, B)
                    bounds_tc = Bounds(np.zeros(K), np.full(K, B))
                    res = minimize(objective_neg, x_init, method='trust-constr',
                                   jac=gradient_neg, constraints=linear_constraint,
                                   bounds=bounds_tc, options={'maxiter': 500})
                else:  # SLSQP by default
                    res = minimize(objective_neg, x_init, method='SLSQP',
                                   jac=gradient_neg, bounds=bounds, 
                                   constraints=constraints,
                                   options={'maxiter': 500, 'ftol': 1e-10})
                
                if res.fun < best_val:
                    best_val = res.fun
                    best = res
            except Exception as e:
                continue
        
        if best is None:
            return x0_base
        
        x_opt = np.maximum(best.x, 0)
        x_opt = B * x_opt / np.sum(x_opt)
        return x_opt


def g_discrete_weibull(x: float, beta: float, q: float) -> float:
    """
    The g-function for discrete Weibull.
    
    For truncated expectation E[T | T <= x]:
    g(x, β, q) = E[T | T <= x] / x
    
    For discrete Weibull:
    E[T | T <= x] = Σ_{k=0}^{floor(x)} k * P(T=k) / P(T <= x)
    
    Parameters:
    - x: budget allocation (can be continuous, we use floor)
    - beta: shape parameter
    - q: scale parameter (0 < q < 1)
    
    Returns:
    - g value: E[T | T <= x] / x
    """
    if x <= 0:
        return 0.0
    if q <= 0 or q >= 1:
        raise ValueError("q must be in (0, 1)")
    if beta <= 0:
        raise ValueError("beta must be > 0")
    
    x_int = int(np.floor(x))
    
    # Compute E[T | T <= x]
    cdf_x = discrete_weibull_cdf(x_int, q, beta)
    if cdf_x < 1e-15:
        return 0.0
    
    # E[T | T <= x] = Σ_{k=0}^{x} k * P(T=k) / P(T <= x)
    numerator = 0.0
    for k in range(x_int + 1):
        pmf_k = discrete_weibull_pmf(k, q, beta)
        numerator += k * pmf_k
    
    conditional_mean = numerator / cdf_x
    
    return conditional_mean / x if x > 0 else 0.0


def g_inv_discrete_weibull(
    y: float,
    beta: float,
    q_min: float = 1e-6,
    q_max: float = 1.0 - 1e-6,
    x: float = 10.0,
    *,
    tol: float = 1e-10
) -> float:
    """
    Numerical inverse of g_discrete_weibull to find q given y = g(x, β, q).
    
    Given observed y = E[T | T <= x] / x, find q.
    
    Parameters:
    - y: observed g value (E[T|T<=x] / x)
    - beta: shape parameter (known)
    - q_min: minimum q to search
    - q_max: maximum q to search
    - x: budget allocation used
    - tol: tolerance for root finding
    
    Returns:
    - q: estimated scale parameter
    """
    if q_min <= 0 or q_max >= 1 or q_min >= q_max:
        raise ValueError("Require 0 < q_min < q_max < 1")
    if beta <= 0:
        raise ValueError("beta must be > 0")
    
    def f(q):
        return g_discrete_weibull(x, beta, q) - y
    
    try:
        fmin, fmax = f(q_min), f(q_max)
        
        if fmin == 0.0:
            return q_min
        if fmax == 0.0:
            return q_max
        if fmin * fmax > 0:
            # No bracketing, return closest bound
            if abs(fmin) < abs(fmax):
                return q_min
            else:
                return q_max
        
        return brentq(f, q_min, q_max, xtol=tol, rtol=tol, maxiter=500)
    except Exception:
        return 0.5  # Default fallback


# =============================================================================
# Confidence bounds and allocation selection
# =============================================================================

def compute_confidence_bounds_discrete_weibull_vec(
    t, i, p_hat, q_hat, n, x_est, B, D, K, beta_vec,
    eps_n=1e-2, eps=1e-12
):
    """
    Discrete Weibull version of compute_confidence_bounds.
    For discrete Weibull with survival S(k) = q^((k+1)^β)
    
    Parameters:
    - t: current time step
    - i: arm being selected
    - p_hat: estimated p values
    - q_hat: estimated q values (scale parameter)
    - n: counts array
    - x_est: estimated allocations
    - B: total budget
    - D: confidence parameter
    - K: number of arms
    - beta_vec: shape parameters (known)
    """
    p_bar = np.zeros(K)
    q_bar = np.zeros(K)
    q_bar_prime = np.zeros(K)

    for j in range(K):
        beta_j = float(beta_vec[j])
        
        # Confidence radius for q (in log scale for stability)
        c_q = 0.1 * np.sqrt(3 * np.log(t + 1) / (2 * (n[t, j] + eps_n)))
        
        # Confidence radius for p
        x_j = max(1, x_est[t, j])
        cdf_j = 1.0 - q_hat[t, j] ** ((x_j + 1) ** beta_j)
        c_p = 0.01 * (1 + p_hat[t, j]) / (cdf_j * D + eps_n) * np.sqrt(3 * np.log(t + 1) / (2 * (n[t, j] + eps_n)))
        
        if j == i:
            # Optimistic for the selected arm i (lower q = faster events)
            q_bar[j] = max(eps, min(1 - eps, q_hat[t, j] - c_q))
            q_bar_prime[j] = max(eps, min(1 - eps, q_hat[t, j] - c_q))
            p_bar[j] = max(0.001, min(1, p_hat[t, j] + c_p))
        else:
            # Pessimistic for other arms (higher q = slower events)
            q_bar[j] = max(eps, min(1 - eps, q_hat[t, j] + c_q))
            q_bar_prime[j] = max(eps, min(1 - eps, q_hat[t, j] + c_q))
            p_bar[j] = max(0.001, min(1, p_hat[t, j] - c_p))
    
    return p_bar, q_bar, q_bar_prime


def select_allocation_discrete(q_bar, q_bar_prime, p_bar, B, beta, n_restarts=5, method="SLSQP"):
    """Select allocation using the oracle solver for discrete Weibull."""
    return oracle_discrete_weibull(p_bar, q_bar, q_bar_prime, B, beta, n_restarts=n_restarts, seed=None, method=method)


# =============================================================================
# Simulation and update functions
# =============================================================================

def update_estimates_discrete_weibull(
    t, i, beta_i, phi, psi, n, sum_obs, q_hat, p_hat, x_est, B
):
    """
    Discrete Weibull version of update_estimates.
    
    Parameters:
    - t: time step
    - i: arm index
    - beta_i: shape parameter for arm i (known)
    - phi: indicator array (1 if event observed, 0 otherwise)
    - psi: observed time array (time of event or censoring time)
    - n: count of events
    - sum_obs: sum of observed times (for mean estimation)
    - q_hat: estimated q values
    - p_hat: estimated p values
    - x_est: allocation estimates
    - B: total budget
    """
    phi_val = phi[t + 1, i]
    psi_val = psi[t + 1, i]
    beta_i = float(beta_i)
    
    # Step 1: Update n and sum of observed times
    if phi_val == 1:
        n[t + 1, i] = n[t, i] + 1
        sum_obs[t + 1, i] = (sum_obs[t, i] * n[t, i] + psi_val) / n[t + 1, i]
    else:
        n[t + 1, i] = n[t, i]
        sum_obs[t + 1, i] = sum_obs[t, i]
    
    mean_obs = sum_obs[t + 1, i]
    
    # Step 2: Estimate q using method of moments or MLE approximation
    # For discrete Weibull, mean ≈ Σ q^((k+1)^β) for k=0,1,2,...
    # Approximate: mean ≈ q^(1) + q^(2^β) + ... 
    # Simple estimator: use the g-function inverse
    if mean_obs > 0 and n[t + 1, i] > 0:
        x_mean = np.mean(x_est[:t + 2, i][x_est[:t + 2, i] > 0]) if np.any(x_est[:t + 2, i] > 0) else B
        g_val = min(0.9, mean_obs / max(1, x_mean))
        q_hat[t + 1, i] = g_inv_discrete_weibull(g_val, beta_i, x=max(1, x_mean))
    else:
        q_hat[t + 1, i] = 0.5  # Default
    
    # Step 3: Estimate p
    x_i = max(1, x_est[t + 1, i]) if t + 1 < len(x_est) else B / 10
    cdf_vals = np.array([1.0 - q_hat[t + 1, i] ** ((max(1, x_est[s, i]) + 1) ** beta_i) 
                         for s in range(t + 2)])
    denom = np.sum(cdf_vals) + 1e-6
    p_hat[t + 1, i] = np.sum(phi[:t + 2, i]) / denom


def compute_regret(t1, reward, opt_reward, regret, cum_regret, f, f_star):
    """Compute regret at time t1."""
    reward[t1 + 1] = np.sum(f[t1 + 1])
    opt_reward[t1 + 1] = np.sum(f_star[t1 + 1])
    regret[t1 + 1] = opt_reward[t1 + 1] - reward[t1 + 1]
    cum_regret[t1 + 1] = np.sum(regret[:t1 + 2])


def simulate_one_round_discrete_weibull(p_true, q_true, beta_true, x_alloc, rng):
    """
    Simulate one round with discrete Weibull censoring.
    
    Model:
      Y ~ Bern(p_true)
      T ~ DiscreteWeibull(q, β)
      event if Y=1 and x >= T
      z = min(T, x)
      delta = 1{event}
    
    Parameters:
    - p_true: success probabilities
    - q_true: scale parameters (0 < q < 1)
    - beta_true: shape parameters
    - x_alloc: budget allocations (integers)
    - rng: random number generator
    
    Returns:
    - delta: indicator of event (1 if observed, 0 if censored)
    - z: observed time (event time if delta=1, allocation if delta=0)
    """
    K = len(p_true)
    delta = np.zeros(K, dtype=float)
    z = np.zeros(K, dtype=float)

    for i in range(K):
        y = rng.random() < p_true[i]
        T = discrete_weibull_sample(q_true[i], beta_true[i], rng)
        x = int(x_alloc[i])
        
        if y and (x >= T):
            delta[i] = 1.0
            z[i] = T
        else:
            delta[i] = 0.0
            z[i] = x  # right censoring
    
    return delta, z


# =============================================================================
# Sanity check and testing
# =============================================================================

def sanity_check_discrete(
    T=10000,
    K=3,
    B=40,
    seed=0,
    arm_to_plot=0,
    D=1.0,
    alloc_mode="equal",
):
    """
    Sanity check for the Discrete Weibull RA-UCB algorithm.
    Tests the estimator convergence with fixed allocations.
    """
    import matplotlib.pyplot as plt
    
    rng = np.random.default_rng(seed)

    # True parameters
    p_true = np.array([0.5, 0.3, 0.7][:K], dtype=float)
    q_true = np.array([0.7, 0.5, 0.8][:K], dtype=float)  # scale parameter (0 < q < 1)
    beta_true = np.array([1.5, 2.0, 1.2][:K], dtype=float)

    # Allocations (fixed, for testing)
    if alloc_mode == "equal":
        x_fixed = np.full(K, B // K, dtype=int)
    elif alloc_mode == "full_arm0":
        x_fixed = np.zeros(K, dtype=int)
        x_fixed[0] = B
    else:
        raise ValueError("alloc_mode must be 'equal' or 'full_arm0'")

    # Arrays (time index: 0..T)
    phi = np.zeros((T + 1, K))
    psi = np.zeros((T + 1, K))
    x_est = np.zeros((T + 1, K))

    n = np.zeros((T + 1, K))
    sum_obs = np.zeros((T + 1, K))
    q_hat = np.zeros((T + 1, K))
    p_hat = np.zeros((T + 1, K))

    # Safe init
    q_hat[0, :] = 0.5
    p_hat[0, :] = 0.5

    # Logs for the arm to plot
    q_est_hist = np.zeros(T + 1)
    p_est_hist = np.zeros(T + 1)
    
    q_est_hist[0] = q_hat[0, arm_to_plot]
    p_est_hist[0] = p_hat[0, arm_to_plot]

    # Main loop
    for t in range(0, T):
        x_est[t + 1, :] = x_fixed

        delta, z = simulate_one_round_discrete_weibull(p_true, q_true, beta_true, x_fixed, rng)
        phi[t + 1, :] = delta
        psi[t + 1, :] = z

        for i in range(K):
            update_estimates_discrete_weibull(
                t=t,
                i=i,
                beta_i=beta_true[i],
                phi=phi,
                psi=psi,
                n=n,
                sum_obs=sum_obs,
                q_hat=q_hat,
                p_hat=p_hat,
                x_est=x_est,
                B=B
            )

        q_est_hist[t + 1] = q_hat[t + 1, arm_to_plot]
        p_est_hist[t + 1] = p_hat[t + 1, arm_to_plot]

    # DEBUG
    print(f"\n=== DEBUG: Arm {arm_to_plot} ===")
    print(f"True parameters: p={p_true[arm_to_plot]:.3f}, q={q_true[arm_to_plot]:.3f}, beta={beta_true[arm_to_plot]:.3f}")
    print(f"Final estimates: p_hat={p_hat[T, arm_to_plot]:.3f}, q_hat={q_hat[T, arm_to_plot]:.3f}")
    print(f"n[T, arm] = {n[T, arm_to_plot]} (number of successes)")
    print(f"Ratio n/T = {n[T, arm_to_plot]/T:.2%}")
    
    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    tt = np.arange(T + 1)

    # n evolution
    axes[0].plot(tt, n[:T+1, arm_to_plot], label=f'n[t, arm {arm_to_plot}]')
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('n (number of successes)')
    axes[0].set_title(f'Evolution of n for arm {arm_to_plot}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # q estimate
    axes[1].plot(tt, q_est_hist, label=r"$\hat{q}$")
    axes[1].axhline(q_true[arm_to_plot], linestyle="--", color='r', label=r"$q^\star$")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("q")
    axes[1].set_title(f"Arm {arm_to_plot}: q estimate (alloc={alloc_mode})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # p estimate
    axes[2].plot(tt, p_est_hist, label=r"$\hat{p}$")
    axes[2].axhline(p_true[arm_to_plot], linestyle="--", color='r', label=r"$p^\star$")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("p")
    axes[2].set_title(f"Arm {arm_to_plot}: p estimate (alloc={alloc_mode})")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        'p_true': p_true,
        'q_true': q_true,
        'beta_true': beta_true,
        'p_hat_final': p_hat[T],
        'q_hat_final': q_hat[T],
        'n_final': n[T]
    }


if __name__ == "__main__":
    sanity_check_discrete(
        T=5000,
        K=3,
        B=40,
        seed=0,
        arm_to_plot=0,
        D=1.0,
        alloc_mode="equal",
    )
