import jax
import jax.numpy as jnp
import pytest
from src.forde.smoothing import smooth_assignments, smooth_assignments_3d

def test_smooth_assignments():
    # Grid sizes have to be relatively larger to be meaningful, but
    # let's test a simple grid without any padding needed just for correctness
    grid = jnp.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 2, 2],
        [2, 2, 2, 2]
    ])
    smoothed = smooth_assignments(grid, kernel_size=3, num_clusters=3)
    assert smoothed.shape == grid.shape

def test_smooth_assignments_3d():
    # Grid sizes needing padding (dim < 3)
    grid = jnp.array([
        [[0, 0, 1, 1], [0, 0, 1, 1]],
        [[2, 2, 2, 2], [2, 2, 2, 2]]
    ])
    smoothed = smooth_assignments_3d(grid, kernel_size=3, num_clusters=3)
    assert smoothed.shape == grid.shape

    # Larger grid sizes
    grid = jnp.zeros((8, 8, 8), dtype=jnp.int32)
    smoothed = smooth_assignments_3d(grid, kernel_size=3, num_clusters=3)
    assert smoothed.shape == grid.shape
