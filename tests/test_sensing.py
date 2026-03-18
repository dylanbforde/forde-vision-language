import jax
import jax.numpy as jnp
from src.forde.sensing import hoyer_sparsity, calculate_neuron_stats

def test_hoyer_sparsity_axis():
    # Test that hoyer_sparsity works with a specified axis
    x = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0]])

    # Axis=-1 (default)
    res_default = hoyer_sparsity(x)
    assert res_default.shape == (4,)

    # Axis=0
    res_axis0 = hoyer_sparsity(x, axis=0)
    assert res_axis0.shape == (2,)

def test_calculate_neuron_stats():
    # Test that calculate_neuron_stats produces expected output shapes
    batch_size = 8
    seq_len = 16
    features = 128

    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)

    activations = jax.random.normal(k1, (batch_size, seq_len, features))
    gradients = jax.random.normal(k2, (batch_size, seq_len, features))

    stats = calculate_neuron_stats(activations, gradients)

    assert stats.shape == (features, 5)

    # Test with 2D input (after reshaping)
    activations_2d = jax.random.normal(k1, (batch_size * seq_len, features))
    gradients_2d = jax.random.normal(k2, (batch_size * seq_len, features))

    stats_2d = calculate_neuron_stats(activations_2d, gradients_2d)

    assert stats_2d.shape == (features, 5)

if __name__ == "__main__":
    test_hoyer_sparsity_axis()
    test_calculate_neuron_stats()
    print("All tests passed.")
