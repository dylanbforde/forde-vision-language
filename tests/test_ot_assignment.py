import pytest
import jax.numpy as jnp

from src.forde.ot_assignment import (
    OTAssignmentConfig,
    assign_expert_roles_ot,
    normalize_expert_features,
)


def test_ot_assignment_shapes_and_diagnostics_with_dustbin():
    features = jnp.array(
        [
            [0.75, 0.05, 0.45],
            [0.50, 0.30, 0.90],
            [0.10, 0.02, 0.10],
            [0.30, 0.10, 0.20],
        ]
    )
    usage = jnp.array([0.45, 0.30, 0.15, 0.10])
    config = OTAssignmentConfig(n_iters=5, refine_steps=1)

    result = assign_expert_roles_ot(features, usage, config)

    assert result.transport.shape == (4, 4)
    assert result.role_probs.shape == (4, 3)
    assert result.role_ids.shape == (4,)
    assert result.dustbin_fraction.shape == (4,)
    assert jnp.allclose(result.role_probs.sum(axis=-1), 1.0, atol=1e-5)
    assert jnp.isfinite(result.transport).all()
    assert 0.0 <= float(result.dustbin_mass) <= 1.0
    assert result.diagnostics["role_masses"].shape == (3,)


def test_ot_assignment_can_disable_dustbin():
    features = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    usage = jnp.array([0.5, 0.5])
    config = OTAssignmentConfig(use_dustbin=False, n_iters=3, refine_steps=1)

    result = assign_expert_roles_ot(features, usage, config)

    assert result.transport.shape == (2, 3)
    assert jnp.all(result.dustbin_fraction == 0.0)
    assert result.dustbin_mass == 0.0


def test_normalize_expert_features_handles_degenerate_columns():
    features = jnp.array([[1.0, 2.0, 5.0], [3.0, 2.0, 5.0]])
    normalized = normalize_expert_features(features)

    assert jnp.allclose(normalized[:, 0], jnp.array([0.0, 1.0]))
    assert jnp.all(normalized[:, 1:] == 0.0)


def test_ot_assignment_rejects_unsupported_role_count():
    features = jnp.ones((2, 3))
    usage = jnp.array([0.5, 0.5])
    config = OTAssignmentConfig(num_roles=4)

    with pytest.raises(ValueError):
        assign_expert_roles_ot(features, usage, config)

