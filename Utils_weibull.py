"""
RA-UCB Functions for Weibull Censored Feedback Experiments

This module contains all the core functions for the RA-UCB algorithm
with Weibull distributed censoring.
"""

import numpy as np
import math
from typing import Callable, Optional

from scipy.optimize import fsolve, brentq, minimize, LinearConstraint, Bounds
from scipy.special import logsumexp, gamma as Gamma, gammainc
from scipy.stats import weibull_min


# =============================================================================
# Oracle and g-function utilities
# =============================================================================

def oracle_weibull_softmax(p, lambda_, lambda_prime, B, k, n_restarts=5, seed=0, method="trust-constr"):
    """
    Solve: max_{x>=0, sum x = B}  sum_i p_i * (1 - (lambda_prime/lambda) * exp(-(lambda_i x_i)^k))

    Available methods:
    - "diff_evol": Differential Evolution (GLOBAL optimizer, recommended for k>1)
    - "SLSQP": Sequential Least Squares Programming
    - "trust-constr": Trust-region constrained
    - "softmax": Legacy method based on L-BFGS-B with a softmax reparameterization
    """
    p = np.asarray(p, dtype=float).reshape(-1)
    lam = np.asarray(lambda_, dtype=float).reshape(-1)
    lam_prime = np.asarray(lambda_prime, dtype=float).reshape(-1)
    K = p.size
    assert lam.shape == (K,)


    k_arr = np.asarray(k, dtype=float)
    if k_arr.ndim == 0:
        k_arr = np.full(K, float(k_arr))
    else:
        k_arr = k_arr.reshape(-1)
        assert k_arr.shape == (K,)

    rng = np.random.default_rng(seed)
    

    def objective_neg(x):
        x = np.maximum(x, 1e-10)  # avoid log(0) in gradient
        val = np.sum(p * (1.0 - (lam_prime/lam) * np.exp(-(lam * x) ** k_arr)))
        return -val
    
    # Analytical gradient for SLSQP/trust-constr
    def gradient_neg(x):
        x = np.maximum(x, 1e-10)
        exp_term = np.exp(-(lam * x) ** k_arr)
        grad = -p * (lam_prime/lam) * k_arr * (lam ** k_arr) * (x ** (k_arr - 1)) * exp_term
        return grad
    
    if method == "softmax":

        def z_to_x(z):
            z = np.asarray(z, float)
            z = z - logsumexp(z)
            w = np.exp(z)
            return B * w

        def objective_neg_softmax(z):
            x = z_to_x(z)
            val = np.sum(p * (1.0 - (lam_prime/lam) * np.exp(-(lam * x) ** k_arr)))
            return -val

        score = np.maximum(1e-12, p * (lam ** k_arr))
        z0 = np.log(score) - np.mean(np.log(score))
        
        best = None
        for r in range(n_restarts):
            if r == 0:
                z_init = z0
            else:
                z_init = z0 + 0.5 * rng.standard_normal(K)

            res = minimize(objective_neg_softmax, z_init, method="L-BFGS-B")
            if best is None or res.fun < best.fun:
                best = res

        return z_to_x(best.x)
    
    elif method == "diff_evol":

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
        
        score = np.maximum(1e-12, p * (lam ** k_arr))
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
                else: 
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


def g_weibull(x: float, k: float) -> float:
    """
    For X ~ Weibull(k, lambda) and x = lambda * B,
    we have mu/B = g_k(x) with

      g_k(x) = (1/x) * gamma_lower(1+1/k, x^k) / (1 - e^{-x^k})

    where gamma_lower(s,z) = ∫_0^z t^{s-1} e^{-t} dt (lower incomplete gamma).
    In scipy:
      gammainc(s,z) = gamma_lower(s,z) / Gamma(s) (regularized version)
    so gamma_lower(s,z) = gammainc(s,z) * Gamma(s).
    """
    if x <= 0:
        raise ValueError("x must be > 0")
    if k <= 0:
        raise ValueError("k must be > 0")

    s = 1.0 + 1.0 / k
    z = x ** k

    denom = 1.0 - math.exp(-z)
    if denom <= 0:
        raise ValueError("Denominator numerically zero; x too small.")

    gamma_lower = gammainc(s, z) * Gamma(s)
    return (gamma_lower / denom) / x


def g_inv_weibull(
    y: float,
    k: float,
    xmin: float = 1e-2,
    xmax: float = 1e12,
    *,
    tol: float = 1e-12
) -> float:
    """
    Numerical inverse of g_weibull(., k) on [xmin, xmax] via brentq.
    g_weibull is decreasing in x (x>0), so the inversion is well-posed.
    """
    if xmin <= 0 or xmax <= xmin:
        raise ValueError("Require 0 < xmin < xmax")
    if k <= 0:
        raise ValueError("k must be > 0")

    f = lambda x: g_weibull(x, k) - y
    fmin, fmax = f(xmin), f(xmax)

    if fmin == 0.0:
        return xmin
    if fmax == 0.0:
        return xmax
    if fmin * fmax > 0:
        raise ValueError(
            f"y={y} is not bracketed by g_k on [{xmin},{xmax}]. "
            f"g_k(xmin)={g_weibull(xmin,k)}, g_k(xmax)={g_weibull(xmax,k)}"
        )

    return brentq(f, xmin, xmax, xtol=tol, rtol=tol, maxiter=500)


# =============================================================================
# Confidence bounds and allocation selection
# =============================================================================

def compute_confidence_bounds_weibull_vec(
    t, i, p_hat, lambda_hat, n, x_est, B, D, K, k_vec,
    eps_n=1e-2, eps=1e-12
):
    """
    Weibull version of compute_confidence_bounds (inspired by Giovanni's exponential version).
    For Weibull with survival S(t) = exp(-(lambda*t)^k)
    """
    p_bar = np.zeros(K)
    lambda_bar = np.zeros(K)
    lambda_bar_prime = np.zeros(K)

    for j in range(K):
        k_j = float(k_vec[j])
        
        # Confidence radius for lambda
        c_lambda = 1 / (B * D) * np.sqrt(3 * np.log(t + 1) / (2 * (n[t, j] + eps_n)))
        
        # Confidence radius for p
        c_p = 0.01 * (1 + p_hat[t, j]) / ((1 - np.exp(-(lambda_hat[t, j] * x_est[t, j]) ** k_j)) * D + eps_n) * np.sqrt(3 * np.log(t + 1) / (2 * (n[t, j] + eps_n)))
        
        if j == i:
            # Optimistic for the selected arm i
            lambda_bar[j] = max(1 / B, lambda_hat[t, j] + c_lambda)
            lambda_bar_prime[j] = max(1 / B, lambda_hat[t, j] + c_lambda)
            p_bar[j] = max(0.001, min(1, p_hat[t, j] + c_p))
        else:
            # Pessimistic for other arms
            lambda_bar[j] = max(1 / B, lambda_hat[t, j] - c_lambda)
            lambda_bar_prime[j] = max(1 / B, lambda_hat[t, j] - c_lambda)
            p_bar[j] = max(0.001, min(1, p_hat[t, j] - c_p))
    
    return p_bar, lambda_bar, lambda_bar_prime


def select_allocation(lambda_bar, lambda_bar_prime, p_bar, B, k, n_restarts=5, method="trust-constr"):
    """Select allocation using the oracle solver."""
    return oracle_weibull_softmax(p_bar, lambda_bar, lambda_bar_prime, B, k, n_restarts=n_restarts, seed=None, method=method)


# =============================================================================
# Simulation and update functions
# =============================================================================



def update_estimates_weibull(
    t, i, k_i, phi, psi, n, Szk, theta_hat, lambda_hat, p_hat, x_est, B
):
    """
    Weibull version of update_estimates (following Giovanni's pattern exactly).
    
    Giovanni's exponential version:
    1. if phi_val == 1: update n and mu_hat (empirical mean)
    2. lambda_hat = (1/B) * g_inv(min(0.5, mu_hat/B))
    3. p_hat = (sum delta) / sum(1 - exp(-lambda * x))
    
    This Weibull version:
    1. if phi_val == 1: update n and mu_hat (empirical mean of observed times)
    2. lambda_hat = (1/B) * g_inv_weibull(min(0.5, mu_hat/B), k_i)
    3. p_hat = (sum delta) / sum(1 - exp(-(lambda * x)^k))
    
    NOTE: B is now passed as parameter to avoid using global B!
    """
    phi_val = phi[t + 1, i]
    psi_val = psi[t + 1, i]
    k_i = float(k_i)
    
    # Step 1: Update n and mu_hat (same as Giovanni)
    if phi_val == 1:
        n[t + 1, i] = n[t, i] + 1
        Szk[t + 1, i] = (Szk[t, i] * n[t, i] + psi_val) / n[t + 1, i]
    else:
        n[t + 1, i] = n[t, i]
        Szk[t + 1, i] = Szk[t, i]
    
    mu_hat = Szk[t + 1, i]
    
    # Step 2: Estimate lambda using g_inv_weibull
    if mu_hat == 0:
        lambda_hat[t + 1, i] = 0
    else:
        lambda_hat[t + 1, i] = (1 / B) * g_inv_weibull(min(0.5, mu_hat / B), k_i)
    
    # Step 3: Estimate p
    denom = np.sum(1 - np.exp(-(lambda_hat[t + 1, i] * x_est[:t + 2, i]) ** k_i)) + 1e-6
    p_hat[t + 1, i] = np.sum(phi[:t + 2, i]) / denom


def compute_regret(t1, reward, opt_reward, regret, cum_regret, f, f_star):
    """Compute regret at time t1."""
    reward[t1 + 1] = np.sum(f[t1 + 1])
    opt_reward[t1 + 1] = np.sum(f_star[t1 + 1])
    regret[t1 + 1] = opt_reward[t1 + 1] - reward[t1 + 1]
    cum_regret[t1 + 1] = np.sum(regret[:t1 + 2])


def simulate_one_round_weibull(p_true, lam_true, k_true, x_alloc, rng):
    """
    Model:
      Y ~ Bern(p_true)
      T ~ Weibull(k, scale = 1/lam_true)  <=> survival exp(-(lam*t)^k)
      event if Y=1 and x >= T
      z = min(T, x)
      delta = 1{event}
    """
    K = len(p_true)
    delta = np.zeros(K, dtype=float)
    z = np.zeros(K, dtype=float)

    for i in range(K):
        y = rng.random() < p_true[i]
        T = (1.0 / lam_true[i]) * rng.weibull(k_true[i])
        x = x_alloc[i]
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

def sanity_check(
    T=10000,
    K=3,
    B=40.0,
    seed=0,
    arm_to_plot=0,
    D=0.01,
    alloc_mode="equal",
):
    """
    Sanity check for the Weibull RA-UCB algorithm.
    Tests the estimator convergence with fixed allocations.
    """
    import matplotlib.pyplot as plt
    
    rng = np.random.default_rng(seed)

    # true parameters
    p_true = np.array([0.9, 0.20, 0.50][:K], dtype=float)
    lam_true = np.array([0.3, 0.04, 0.12][:K], dtype=float)
    k_true = np.array([1.6, 2.2, 1.2][:K], dtype=float)

    # allocations (fixed, for testing)
    if alloc_mode == "equal":
        x_fixed = np.full(K, B / K, dtype=float)
    elif alloc_mode == "full_arm0":
        x_fixed = np.zeros(K, dtype=float)
        x_fixed[0] = B
    else:
        raise ValueError("alloc_mode must be 'equal' or 'full_arm0'")

    # arrays (time index: 0..T)
    phi = np.zeros((T + 1, K))
    psi = np.zeros((T + 1, K))
    x_est = np.zeros((T + 1, K))

    n = np.zeros((T + 1, K))
    Szk = np.zeros((T + 1, K))
    theta_hat = np.zeros((T + 1, K))
    lambda_hat = np.zeros((T + 1, K))
    p_hat = np.zeros((T + 1, K))

    # safe init (avoid lambda=0)
    lambda_hat[0, :] = 1.0 / B
    theta_hat[0, :] = (lambda_hat[0, :] ** k_true)
    p_hat[0, :] = 0.5

    # logs for the arm to plot
    pbar_hist = np.zeros(T + 1)
    plow_hist = np.zeros(T + 1)
    phigh_hist = np.zeros(T + 1)

    lam_est_hist = np.zeros(T + 1)
    lam_low_hist = np.zeros(T + 1)
    lam_high_hist = np.zeros(T + 1)

    # at time 0
    p_bar, lam_bar, lam_bar_p = compute_confidence_bounds_weibull_vec(
        t=0, i=arm_to_plot, p_hat=p_hat, lambda_hat=lambda_hat, n=n, x_est=x_est,
        B=B, D=D, K=K, k_vec=k_true
    )
    pbar_hist[0] = p_hat[0, arm_to_plot]
    plow_hist[0] = p_bar[arm_to_plot]
    phigh_hist[0] = p_bar[arm_to_plot]

    lam_est_hist[0] = lambda_hat[0, arm_to_plot]
    lam_low_hist[0] = lam_bar[arm_to_plot]
    lam_high_hist[0] = lam_bar_p[arm_to_plot]

    # main loop
    for t in range(0, T):
        x_est[t + 1, :] = x_fixed

        delta, z = simulate_one_round_weibull(p_true, lam_true, k_true, x_fixed, rng)
        phi[t + 1, :] = delta
        psi[t + 1, :] = z

        for i in range(K):
            update_estimates_weibull(
                t=t,
                i=i,
                k_i=k_true[i],
                phi=phi,
                psi=psi,
                n=n,
                Szk=Szk,
                theta_hat=theta_hat,
                lambda_hat=lambda_hat,
                p_hat=p_hat,
                x_est=x_est,
                B=B
            )

        p_bar, lam_bar, lam_bar_p = compute_confidence_bounds_weibull_vec(
            t=t + 1, i=arm_to_plot, p_hat=p_hat, lambda_hat=lambda_hat, n=n, x_est=x_est,
            B=B, D=D, K=K, k_vec=k_true
        )

        pbar_hist[t + 1] = p_hat[t + 1, arm_to_plot]
        plow_hist[t + 1] = p_bar[arm_to_plot]
        phigh_hist[t + 1] = p_bar[arm_to_plot]

        lam_est_hist[t + 1] = lambda_hat[t + 1, arm_to_plot]
        lam_low_hist[t + 1] = lam_bar[arm_to_plot]
        lam_high_hist[t + 1] = lam_bar_p[arm_to_plot]

    # DEBUG
    print(f"\n=== DEBUG: Arm {arm_to_plot} ===")
    print(f"n[T, arm] = {n[T, arm_to_plot]} (number of successes)")
    print(f"T = {T}")
    print(f"Ratio n/T = {n[T, arm_to_plot]/T:.2%}")
    
    for t_check in [100, 1000, 5000, T]:
        if t_check <= T:
            n_val = n[t_check, arm_to_plot]
            c_lam = 1 / (B * D) * np.sqrt(3 * np.log(t_check + 1) / (2 * (n_val + 1e-2)))
            print(f"  t={t_check}: n={n_val:.0f}, c_lambda={c_lam:.6f}, width=2*c_lambda={2*c_lam:.6f}")
    
    # Plots
    plt.figure()
    plt.plot(np.arange(T+1), n[:T+1, arm_to_plot], label=f'n[t, arm {arm_to_plot}]')
    plt.xlabel('t')
    plt.ylabel('n (number of successes)')
    plt.title(f'Evolution of n for arm {arm_to_plot}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    tt = np.arange(T + 1)

    # lambda
    plt.figure()
    plt.plot(tt, lam_est_hist, label=r"$\hat{\lambda}$")
    plt.plot(tt, lam_low_hist, label=r"$\bar{\lambda}$")
    plt.plot(tt, lam_high_hist, label=r"$\bar{\lambda}'$")
    plt.axhline(lam_true[arm_to_plot], linestyle="--", label=r"$\lambda^\star$")
    plt.title(f"Arm {arm_to_plot}: lambda estimate & bounds (alloc={alloc_mode})")
    plt.xlabel("t")
    plt.ylabel("lambda")
    plt.legend()
    plt.tight_layout()

    # p
    plt.figure()
    plt.plot(tt, pbar_hist, label=r"$\hat{p}$")
    plt.plot(tt, plow_hist, label=r"$\bar{p}$ (boosted)")
    plt.axhline(p_true[arm_to_plot], linestyle="--", label=r"$p^\star$")
    plt.title(f"Arm {arm_to_plot}: p estimate & boosted p (alloc={alloc_mode})")
    plt.xlabel("t")
    plt.ylabel("p")
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    sanity_check(
        T=5000,
        K=3,
        B=40.0,
        seed=0,
        arm_to_plot=1,
        D=1.0,
        alloc_mode="equal",
    )
