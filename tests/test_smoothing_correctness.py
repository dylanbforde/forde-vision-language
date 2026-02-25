
import jax
import jax.numpy as jnp
import pytest
from src.forde.smoothing import smooth_assignments, smooth_assignments_3d

def test_smooth_assignments_basic():
    H, W = 16, 16
    num_clusters = 3
    kernel_size = 3

    key = jax.random.PRNGKey(0)
    assignments = jax.random.randint(key, (H, W), 0, num_clusters)

    smoothed = smooth_assignments(assignments, kernel_size, num_clusters)

    assert smoothed.shape == assignments.shape
    assert smoothed.dtype == assignments.dtype
    assert jnp.all(smoothed >= 0)
    assert jnp.all(smoothed < num_clusters)

def test_smooth_assignments_3d_basic():
    D, H, W = 4, 16, 16
    num_clusters = 3
    kernel_size = 3

    key = jax.random.PRNGKey(0)
    assignments = jax.random.randint(key, (D, H, W), 0, num_clusters)

    smoothed = smooth_assignments_3d(assignments, kernel_size, num_clusters)

    assert smoothed.shape == assignments.shape
    assert smoothed.dtype == assignments.dtype
    assert jnp.all(smoothed >= 0)
    assert jnp.all(smoothed < num_clusters)

def test_smooth_assignments_3d_padding():
    # Test case where input size < kernel size
    D, H, W = 1, 2, 2
    num_clusters = 2
    kernel_size = 3 # Larger than input

    key = jax.random.PRNGKey(0)
    assignments = jax.random.randint(key, (D, H, W), 0, num_clusters)

    # This should not crash due to padding
    smoothed = smooth_assignments_3d(assignments, kernel_size, num_clusters)

    assert smoothed.shape == assignments.shape
    assert smoothed.dtype == assignments.dtype
