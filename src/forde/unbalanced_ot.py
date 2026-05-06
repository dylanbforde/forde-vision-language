"""Dense fixed-support unbalanced OT reference routines.

This module is intentionally small and theorem-oriented. It is not a production
kernel path; it provides a dense reference for the KL-penalized fixed-support
R-step tail used in the innovation-track proofs and validators.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def lambda_from_tau(tau: float, epsilon: float) -> float:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    return float(tau) / float(epsilon)


def alpha_from_tau(tau: float, epsilon: float) -> float:
    if tau < 0:
        raise ValueError("tau must be non-negative.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    return float(tau) / (float(tau) + float(epsilon))


def alpha_from_lambda(lam: float) -> float:
    if lam < 0:
        raise ValueError("lambda must be non-negative.")
    return float(lam) / (1.0 + float(lam))


def alphas_from_taus(tau_q: float, tau_k: float, epsilon: float):
    return alpha_from_tau(tau_q, epsilon), alpha_from_tau(tau_k, epsilon)


def score_from_qk(Q, K, epsilon: float):
    return (Q @ K.T) / (jnp.sqrt(Q.shape[-1]) * epsilon)


def make_uniform_log_masses(q_mask, k_mask):
    q_mass = jnp.asarray(q_mask, dtype=jnp.float32)
    k_mass = jnp.asarray(k_mask, dtype=jnp.float32)
    q_mass = q_mass / jnp.sum(q_mass)
    k_mass = k_mass / jnp.sum(k_mass)
    log_a = jnp.where(q_mask, jnp.log(q_mass), 0.0)
    log_b = jnp.where(k_mask, jnp.log(k_mass), 0.0)
    return log_a, log_b


def make_unit_log_masses(q_mask, k_mask):
    log_a = jnp.where(q_mask, 0.0, 0.0)
    log_b = jnp.where(k_mask, 0.0, 0.0)
    return log_a, log_b


def _masked_logsumexp(logits, mask, axis):
    return jax.nn.logsumexp(jnp.where(mask, logits, -jnp.inf), axis=axis)


def next_u_from_score(score, v, support, log_a, alpha_q, q_mask):
    lse = _masked_logsumexp(score + v[None, :], support, axis=-1)
    u = alpha_q * (log_a - lse)
    return jnp.where(q_mask, u, 0.0)


def next_v_from_score(score, u, support, log_b, alpha_k, k_mask):
    lse = _masked_logsumexp(score + u[:, None], support, axis=0)
    v = alpha_k * (log_b - lse)
    return jnp.where(k_mask, v, 0.0)


def compute_uv_unbalanced_dense_from_score(
    score,
    support,
    log_a,
    log_b,
    q_mask,
    k_mask,
    *,
    n_iters: int,
    alpha_q: float,
    alpha_k: float,
):
    u = jnp.zeros_like(log_a)
    v = jnp.zeros_like(log_b)
    for _ in range(int(n_iters)):
        u = next_u_from_score(score, v, support, log_a, alpha_q, q_mask)
        v = next_v_from_score(score, u, support, log_b, alpha_k, k_mask)
    return u, v


def fixed_tail_history_unbalanced_dense_from_score(
    score,
    support,
    log_a,
    log_b,
    q_mask,
    k_mask,
    *,
    n_iters: int,
    refine_steps: int,
    alpha_q: float,
    alpha_k: float,
):
    u0, v0 = compute_uv_unbalanced_dense_from_score(
        score,
        support,
        log_a,
        log_b,
        q_mask,
        k_mask,
        n_iters=n_iters,
        alpha_q=alpha_q,
        alpha_k=alpha_k,
    )
    u0 = jax.lax.stop_gradient(u0)
    v0 = jax.lax.stop_gradient(v0)
    u_hist = [u0]
    v_hist = [v0]
    u_curr, v_curr = u0, v0
    for _ in range(int(refine_steps)):
        u_curr = next_u_from_score(score, v_curr, support, log_a, alpha_q, q_mask)
        v_curr = next_v_from_score(score, u_curr, support, log_b, alpha_k, k_mask)
        u_hist.append(u_curr)
        v_hist.append(v_curr)
    return u_hist, v_hist


def plan_from_score_and_duals(score, u, v, support):
    return jnp.where(support, jnp.exp(score + u[:, None] + v[None, :]), 0.0)


def surrogate_output_unbalanced_dense_from_score(
    score,
    V,
    support,
    log_a,
    log_b,
    q_mask,
    k_mask,
    *,
    n_iters: int,
    refine_steps: int,
    alpha_q: float,
    alpha_k: float,
):
    u_hist, v_hist = fixed_tail_history_unbalanced_dense_from_score(
        score,
        support,
        log_a,
        log_b,
        q_mask,
        k_mask,
        n_iters=n_iters,
        refine_steps=refine_steps,
        alpha_q=alpha_q,
        alpha_k=alpha_k,
    )
    p_final = plan_from_score_and_duals(score, u_hist[-1], v_hist[-1], support)
    out = p_final @ V
    return jnp.where(q_mask[:, None], out, 0.0)


def _row_normalized(plan):
    row_sum = jnp.sum(plan, axis=-1)
    safe = jnp.where(row_sum > 0, row_sum, 1.0)
    return plan / safe[:, None], row_sum


def _col_normalized(plan):
    col_sum = jnp.sum(plan, axis=0)
    safe = jnp.where(col_sum > 0, col_sum, 1.0)
    return plan / safe[None, :], col_sum


def fixed_r_tail_unbalanced_grads_dense_from_score(
    score,
    V,
    dO,
    support,
    log_a,
    log_b,
    q_mask,
    k_mask,
    *,
    n_iters: int,
    refine_steps: int,
    alpha_q: float,
    alpha_k: float,
):
    u_hist, v_hist = fixed_tail_history_unbalanced_dense_from_score(
        score,
        support,
        log_a,
        log_b,
        q_mask,
        k_mask,
        n_iters=n_iters,
        refine_steps=refine_steps,
        alpha_q=alpha_q,
        alpha_k=alpha_k,
    )

    p_final = plan_from_score_and_duals(score, u_hist[-1], v_hist[-1], support)
    Z = dO @ V.T
    g_u = jnp.sum(p_final * Z, axis=-1)
    g_v = jnp.sum(p_final * Z, axis=0)
    dV = p_final.T @ dO

    zero_u = jnp.zeros_like(g_u)
    zero_v = jnp.zeros_like(g_v)
    du_hist = [zero_u] * (refine_steps + 1)
    dv_hist = [zero_v] * (refine_steps + 1)
    dv_curr = g_v
    for t in range(refine_steps, 0, -1):
        dv_hist[t] = dv_curr
        p_same = plan_from_score_and_duals(score, u_hist[t], v_hist[t], support)
        c_same_norm, _ = _col_normalized(p_same)
        direct_u = g_u if t == refine_steps else zero_u
        du_curr = direct_u - alpha_k * (c_same_norm @ dv_curr)
        du_hist[t] = du_curr

        p_prev = plan_from_score_and_duals(score, u_hist[t], v_hist[t - 1], support)
        r_prev_norm, _ = _row_normalized(p_prev)
        dv_curr = -alpha_q * (r_prev_norm.T @ du_curr)

    dS = p_final * Z
    for t in range(refine_steps, 0, -1):
        p_same = plan_from_score_and_duals(score, u_hist[t], v_hist[t], support)
        _, c_same = _col_normalized(p_same)
        safe_c = jnp.where(c_same > 0, c_same, 1.0)
        dS = dS - alpha_k * p_same * (dv_hist[t] / safe_c)[None, :]

        p_prev = plan_from_score_and_duals(score, u_hist[t], v_hist[t - 1], support)
        _, r_prev = _row_normalized(p_prev)
        safe_r = jnp.where(r_prev > 0, r_prev, 1.0)
        dS = dS - alpha_q * p_prev * (du_hist[t] / safe_r)[:, None]

    dS = jnp.where(support, dS, 0.0)
    return {
        "dS": dS,
        "dV": dV,
        "u_hist": u_hist,
        "v_hist": v_hist,
    }
