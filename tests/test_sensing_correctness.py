
import jax
import jax.numpy as jnp
import pytest
from src.forde.sensing import calculate_neuron_stats, hoyer_sparsity

def hoyer_sparsity_orig(x):
    """Original implementation of Hoyer's sparsity (always along last axis)"""
    n = x.shape[-1]
    l1_norm = jnp.sum(jnp.abs(x), axis=-1)
    l2_norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=-1))

    # Avoid division by zero if l2_norm is zero
    safe_l2_norm = jnp.where(l2_norm == 0, 1.0, l2_norm)

    # Avoid division by zero if n is 1
    denominator = jnp.where(n == 1, 1.0, jnp.sqrt(n) - 1)

    sparsity = (jnp.sqrt(n) - (l1_norm / safe_l2_norm)) / denominator

    # If l2_norm is 0, the vector is all zeros. Sparsity is undefined/0.
    sparsity = jnp.where(l2_norm == 0, 0.0, sparsity)

    return jnp.where(n == 1, 0.0, sparsity)

def calculate_neuron_stats_orig(activations, gradients):
    """Original implementation of neuron stats calculation"""
    num_features = activations.shape[-1]
    activations = activations.reshape(-1, num_features)
    gradients = gradients.reshape(-1, num_features)

    activations = activations.astype(jnp.float32)
    gradients = gradients.astype(jnp.float32)

    # Original logic: vmap over transposed matrices
    act_gini = jax.vmap(hoyer_sparsity_orig)(activations.T)
    act_gdp = jnp.mean(jnp.abs(activations), axis=0)
    act_variance = jnp.var(activations, axis=0)

    grad_gini = jax.vmap(hoyer_sparsity_orig)(gradients.T)
    grad_gdp = jnp.mean(jnp.abs(gradients), axis=0)

    neuron_stats = jnp.stack([grad_gini, grad_gdp, act_gini, act_gdp, act_variance], axis=-1)
    return neuron_stats

def test_hoyer_sparsity_compatibility():
    """Verify hoyer_sparsity behavior on default axis -1 matches original"""
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (10, 100))

    orig = hoyer_sparsity_orig(x)
    new_default = hoyer_sparsity(x) # Should default to axis=-1
    new_explicit = hoyer_sparsity(x, axis=-1)

    assert jnp.allclose(orig, new_default, atol=1e-6)
    assert jnp.allclose(orig, new_explicit, atol=1e-6)

def test_hoyer_sparsity_axis():
    """Verify hoyer_sparsity works correctly on other axes"""
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (10, 50))

    # Sparsity along axis 0
    # Manually transpose and compute on last axis to verify
    x_T = x.T
    expected_axis0 = hoyer_sparsity_orig(x_T)
    actual_axis0 = hoyer_sparsity(x, axis=0)

    assert jnp.allclose(expected_axis0, actual_axis0, atol=1e-6)

def test_calculate_neuron_stats_correctness():
    """Verify optimized calculate_neuron_stats matches original logic"""
    key = jax.random.PRNGKey(123)
    batch_size = 16
    seq_len = 64
    features = 256

    activations = jax.random.normal(key, (batch_size, seq_len, features))
    gradients = jax.random.normal(key, (batch_size, seq_len, features))

    # Run original logic
    expected_stats = calculate_neuron_stats_orig(activations, gradients)

    # Run optimized logic
    actual_stats = calculate_neuron_stats(activations, gradients)

    # Check shape
    assert actual_stats.shape == (features, 5)

    # Check values
    max_diff = jnp.max(jnp.abs(expected_stats - actual_stats))
    print(f"Max difference: {max_diff}")

    assert jnp.allclose(expected_stats, actual_stats, atol=1e-5)

if __name__ == "__main__":
    test_hoyer_sparsity_compatibility()
    test_hoyer_sparsity_axis()
    test_calculate_neuron_stats_correctness()
    print("All correctness tests passed!")
