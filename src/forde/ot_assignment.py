"""Unbalanced-OT expert role assignment for the FORDE slow loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

try:
    from src.forde.unbalanced_ot import (
        alpha_from_tau,
        fixed_tail_history_unbalanced_dense_from_score,
        plan_from_score_and_duals,
    )
except ModuleNotFoundError:
    from unbalanced_ot import (
        alpha_from_tau,
        fixed_tail_history_unbalanced_dense_from_score,
        plan_from_score_and_duals,
    )


@dataclass(frozen=True)
class OTAssignmentConfig:
    """Configuration for expert-to-role assignment via dense unbalanced OT."""

    num_roles: int = 3
    use_dustbin: bool = True
    epsilon: float = 1.0
    tau_q: float = 1.0
    tau_k: float = 1.0
    n_iters: int = 30
    refine_steps: int = 2
    role_priors: tuple[float, ...] = (0.35, 0.45, 0.15, 0.05)
    temperature: float = 1.0
    router_adjustment_scale: float = 0.1


@dataclass(frozen=True)
class OTAssignmentResult:
    """Result of assigning experts to functional roles."""

    transport: jnp.ndarray
    role_probs: jnp.ndarray
    role_ids: jnp.ndarray
    dustbin_fraction: jnp.ndarray
    dustbin_mass: jnp.ndarray
    cost_matrix: jnp.ndarray
    diagnostics: dict[str, jnp.ndarray]


def ot_config_from_model_config(config: Any) -> OTAssignmentConfig:
    """Build an OT assignment config from an LLMConfig-like object."""
    return OTAssignmentConfig(
        num_roles=int(getattr(config, "ot_num_roles", 3)),
        use_dustbin=bool(getattr(config, "ot_use_dustbin", True)),
        epsilon=float(getattr(config, "ot_epsilon", 1.0)),
        tau_q=float(getattr(config, "ot_tau_q", 1.0)),
        tau_k=float(getattr(config, "ot_tau_k", 1.0)),
        n_iters=int(getattr(config, "ot_n_iters", 30)),
        refine_steps=int(getattr(config, "ot_refine_steps", 2)),
        role_priors=tuple(getattr(config, "ot_role_priors", (0.35, 0.45, 0.15, 0.05))),
        temperature=float(getattr(config, "ot_temperature", 1.0)),
        router_adjustment_scale=float(
            getattr(config, "ot_router_adjustment_scale", 0.1)
        ),
    )


def default_role_prototypes(dtype=jnp.float32) -> jnp.ndarray:
    """Role prototypes over usage, usage variance, and selection confidence."""
    return jnp.asarray(
        [
            [0.75, 0.25, 0.50],  # generalist
            [0.50, 0.75, 0.85],  # specialist
            [0.10, 0.25, 0.10],  # under-used
        ],
        dtype=dtype,
    )


def normalize_expert_features(expert_features: jnp.ndarray) -> jnp.ndarray:
    """Column-wise min-max normalize expert features with degenerate columns zeroed."""
    features = jnp.nan_to_num(jnp.asarray(expert_features, dtype=jnp.float32))
    mins = jnp.min(features, axis=0, keepdims=True)
    maxs = jnp.max(features, axis=0, keepdims=True)
    span = maxs - mins
    normalized = (features - mins) / jnp.where(span > 0, span, 1.0)
    return jnp.where(span > 0, normalized, 0.0)


def role_priors_for_config(config: OTAssignmentConfig) -> jnp.ndarray:
    """Return normalized role priors, including dustbin when enabled."""
    if config.num_roles != 3:
        raise ValueError("OT expert assignment currently supports exactly 3 roles.")

    priors = jnp.asarray(config.role_priors, dtype=jnp.float32)
    expected = config.num_roles + int(config.use_dustbin)
    if config.use_dustbin and priors.shape[0] == config.num_roles:
        priors = jnp.concatenate([priors, jnp.asarray([0.05], dtype=priors.dtype)])
    elif priors.shape[0] < expected:
        raise ValueError(
            f"role_priors must have at least {expected} entries, got {priors.shape[0]}."
        )
    priors = priors[:expected]
    priors = jnp.maximum(priors, 1e-8)
    return priors / jnp.sum(priors)


def build_role_cost_matrix(
    expert_features: jnp.ndarray, config: OTAssignmentConfig
) -> jnp.ndarray:
    """Compute squared-distance costs from experts to functional role prototypes."""
    normalized = normalize_expert_features(expert_features)
    prototypes = default_role_prototypes(dtype=normalized.dtype)
    costs = jnp.sum((normalized[:, None, :] - prototypes[None, :, :]) ** 2, axis=-1)

    if config.use_dustbin:
        dustbin_cost = jnp.zeros((normalized.shape[0], 1), dtype=normalized.dtype)
        costs = jnp.concatenate([costs, dustbin_cost], axis=-1)

    return costs


def assign_expert_roles_ot(
    expert_features: jnp.ndarray,
    expert_usage: jnp.ndarray,
    config: OTAssignmentConfig | None = None,
) -> OTAssignmentResult:
    """Assign experts to generalist, specialist, and under-used roles via OT."""
    if config is None:
        config = OTAssignmentConfig()

    features = jnp.asarray(expert_features, dtype=jnp.float32)
    usage = jnp.asarray(expert_usage, dtype=jnp.float32)
    if features.ndim != 2:
        raise ValueError(f"expert_features must be rank-2, got {features.shape}.")
    if features.shape[0] != usage.shape[0]:
        raise ValueError(
            "expert_features and expert_usage must agree on number of experts."
        )

    cost_matrix = build_role_cost_matrix(features, config)
    temperature = max(float(config.temperature), 1e-6)
    score = -cost_matrix / temperature

    num_experts, num_targets = score.shape
    support = jnp.ones((num_experts, num_targets), dtype=bool)
    q_mask = jnp.ones((num_experts,), dtype=bool)
    k_mask = jnp.ones((num_targets,), dtype=bool)

    source_mass = jnp.ones((num_experts,), dtype=jnp.float32) / num_experts
    target_mass = role_priors_for_config(config)
    log_a = jnp.log(source_mass)
    log_b = jnp.log(target_mass)

    alpha_q = alpha_from_tau(config.tau_q, config.epsilon)
    alpha_k = alpha_from_tau(config.tau_k, config.epsilon)
    u_hist, v_hist = fixed_tail_history_unbalanced_dense_from_score(
        score,
        support,
        log_a,
        log_b,
        q_mask,
        k_mask,
        n_iters=config.n_iters,
        refine_steps=config.refine_steps,
        alpha_q=alpha_q,
        alpha_k=alpha_k,
    )
    transport = plan_from_score_and_duals(score, u_hist[-1], v_hist[-1], support)

    row_mass = jnp.sum(transport, axis=-1, keepdims=True)
    row_probs_full = transport / jnp.where(row_mass > 0, row_mass, 1.0)
    role_probs_raw = row_probs_full[:, : config.num_roles]
    role_prob_mass = jnp.sum(role_probs_raw, axis=-1, keepdims=True)
    role_probs = role_probs_raw / jnp.where(role_prob_mass > 0, role_prob_mass, 1.0)
    role_ids = jnp.argmax(role_probs, axis=-1).astype(jnp.int32)

    if config.use_dustbin:
        dustbin_fraction = row_probs_full[:, -1]
    else:
        dustbin_fraction = jnp.zeros((num_experts,), dtype=transport.dtype)

    total_mass = jnp.maximum(jnp.sum(transport), 1e-8)
    transport_probs = transport / total_mass
    role_mass_full = jnp.sum(transport, axis=0) / total_mass
    entropy = -jnp.sum(
        jnp.where(transport_probs > 0, transport_probs * jnp.log(transport_probs), 0.0)
    )
    entropy_norm = entropy / jnp.log(jnp.asarray(transport.size, dtype=transport.dtype))
    uniform_usage = 1.0 / num_experts
    load_imbalance = jnp.mean(jnp.abs(usage - uniform_usage))

    diagnostics = {
        "role_masses": role_mass_full[: config.num_roles],
        "transport_entropy": entropy_norm,
        "load_imbalance": load_imbalance,
        "dustbin_mass": role_mass_full[-1] if config.use_dustbin else jnp.array(0.0),
        "mean_dustbin_fraction": jnp.mean(dustbin_fraction),
        "mean_role_confidence": jnp.mean(jnp.max(role_probs, axis=-1)),
    }

    return OTAssignmentResult(
        transport=transport,
        role_probs=role_probs,
        role_ids=role_ids,
        dustbin_fraction=dustbin_fraction,
        dustbin_mass=diagnostics["dustbin_mass"],
        cost_matrix=cost_matrix,
        diagnostics=diagnostics,
    )
