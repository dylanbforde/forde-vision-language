import jax
import jax.numpy as jnp

from src.forde.sensing import calculate_neuron_stats, hoyer_sparsity


def _hoyer_sparsity_reference(x):
    n = x.shape[-1]
    l1_norm = jnp.sum(jnp.abs(x), axis=-1)
    l2_norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=-1))
    safe_l2_norm = jnp.where(l2_norm == 0, 1.0, l2_norm)
    denominator = jnp.where(n == 1, 1.0, jnp.sqrt(n) - 1)
    sparsity = (jnp.sqrt(n) - (l1_norm / safe_l2_norm)) / denominator
    sparsity = jnp.where(l2_norm == 0, 0.0, sparsity)
    return jnp.where(n == 1, 0.0, sparsity)


def _calculate_neuron_stats_reference(activations, gradients):
    num_features = activations.shape[-1]
    activations = activations.reshape(-1, num_features).astype(jnp.float32)
    gradients = gradients.reshape(-1, num_features).astype(jnp.float32)

    act_gini = jax.vmap(_hoyer_sparsity_reference)(activations.T)
    act_gdp = jnp.mean(jnp.abs(activations), axis=0)
    act_variance = jnp.var(activations, axis=0)
    grad_gini = jax.vmap(_hoyer_sparsity_reference)(gradients.T)
    grad_gdp = jnp.mean(jnp.abs(gradients), axis=0)

    return jnp.stack(
        [grad_gini, grad_gdp, act_gini, act_gdp, act_variance],
        axis=-1,
    )


def test_hoyer_sparsity_default_axis_matches_reference():
    key = jax.random.PRNGKey(0)
    values = jax.random.normal(key, (8, 16))

    assert jnp.allclose(hoyer_sparsity(values), _hoyer_sparsity_reference(values))


def test_hoyer_sparsity_axis_zero_matches_transposed_reference():
    key = jax.random.PRNGKey(1)
    values = jax.random.normal(key, (8, 16))

    expected = _hoyer_sparsity_reference(values.T)
    actual = hoyer_sparsity(values, axis=0)

    assert jnp.allclose(actual, expected)


def test_calculate_neuron_stats_matches_vmap_reference():
    key = jax.random.PRNGKey(2)
    activations = jax.random.normal(key, (4, 6, 12))
    gradients = jax.random.normal(jax.random.fold_in(key, 1), (4, 6, 12))

    expected = _calculate_neuron_stats_reference(activations, gradients)
    actual = calculate_neuron_stats(activations, gradients)

    assert actual.shape == (12, 5)
    assert jnp.allclose(actual, expected, atol=1e-5)
