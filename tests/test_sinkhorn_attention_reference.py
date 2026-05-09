import jax
import jax.numpy as jnp

from src.forde.experimental import sinkhorn_attention


def test_sinkhorn_attention_exact_and_tail_forward_match_on_tiny_case():
    key = jax.random.PRNGKey(0)
    q_key, k_key, v_key = jax.random.split(key, 3)
    queries = jax.random.normal(q_key, (1, 1, 128, 8))
    keys = jax.random.normal(k_key, (1, 1, 128, 8))
    values = jax.random.normal(v_key, (1, 1, 128, 8))

    exact = sinkhorn_attention.sinkhorn_attention(
        queries,
        keys,
        values,
        n_iters=2,
        refine_steps=1,
        gradient_mode="exact_autodiff",
    )
    tail = sinkhorn_attention.sinkhorn_attention(
        queries,
        keys,
        values,
        n_iters=2,
        refine_steps=1,
        gradient_mode="tail_refinement",
    )

    assert exact.shape == queries.shape
    assert tail.shape == queries.shape
    assert jnp.isfinite(exact).all()
    assert jnp.allclose(exact, tail, atol=1e-5)
