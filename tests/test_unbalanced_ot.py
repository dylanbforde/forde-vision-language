import pytest
import jax
import jax.numpy as jnp

from src.forde import unbalanced_ot


def test_alpha_lambda_conversions_validate_inputs():
    assert unbalanced_ot.lambda_from_tau(2.0, 0.5) == pytest.approx(4.0)
    assert unbalanced_ot.alpha_from_tau(1.0, 1.0) == pytest.approx(0.5)
    assert unbalanced_ot.alpha_from_lambda(3.0) == pytest.approx(0.75)

    with pytest.raises(ValueError):
        unbalanced_ot.alpha_from_tau(-1.0, 1.0)
    with pytest.raises(ValueError):
        unbalanced_ot.lambda_from_tau(1.0, 0.0)


def test_unbalanced_fixed_tail_gradients_match_autodiff():
    key = jax.random.PRNGKey(0)
    score_key, value_key, cotangent_key = jax.random.split(key, 3)
    score = jax.random.normal(score_key, (5, 4)) * 0.1
    values = jax.random.normal(value_key, (4, 3))
    cotangent = jax.random.normal(cotangent_key, (5, 3))
    support = jnp.ones((5, 4), dtype=bool)
    q_mask = jnp.ones((5,), dtype=bool)
    k_mask = jnp.ones((4,), dtype=bool)
    log_a, log_b = unbalanced_ot.make_uniform_log_masses(q_mask, k_mask)
    kwargs = {"n_iters": 3, "refine_steps": 2, "alpha_q": 0.6, "alpha_k": 0.7}

    reference = unbalanced_ot.fixed_r_tail_unbalanced_grads_dense_from_score(
        score, values, cotangent, support, log_a, log_b, q_mask, k_mask, **kwargs
    )

    def loss_fn(score_arg, values_arg):
        output = unbalanced_ot.surrogate_output_unbalanced_dense_from_score(
            score_arg,
            values_arg,
            support,
            log_a,
            log_b,
            q_mask,
            k_mask,
            **kwargs,
        )
        return jnp.sum(output * cotangent)

    grad_score, grad_values = jax.grad(loss_fn, argnums=(0, 1))(score, values)

    assert jnp.allclose(grad_score, reference["dS"], atol=1e-6)
    assert jnp.allclose(grad_values, reference["dV"], atol=1e-6)


def test_unbalanced_plan_respects_masked_support():
    score = jnp.array([[1.0, 0.0], [0.5, 0.25]])
    support = jnp.array([[True, False], [True, True]])
    q_mask = jnp.array([True, True])
    k_mask = jnp.array([True, True])
    log_a, log_b = unbalanced_ot.make_uniform_log_masses(q_mask, k_mask)

    u, v = unbalanced_ot.compute_uv_unbalanced_dense_from_score(
        score,
        support,
        log_a,
        log_b,
        q_mask,
        k_mask,
        n_iters=4,
        alpha_q=0.5,
        alpha_k=0.5,
    )
    plan = unbalanced_ot.plan_from_score_and_duals(score, u, v, support)

    assert plan.shape == score.shape
    assert plan[0, 1] == 0.0
    assert jnp.isfinite(plan).all()

