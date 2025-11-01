"""
Implements the "Smoothing" stage of the FORDE model.

This stage takes the raw, 1D neuron cluster assignments and smooths them over a 2D grid
to encourage the formation of spatially contiguous functional areas. This is a key
step in creating the desired "brain map" structure.
"""

import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d

def assignments_to_grid(assignments: jnp.ndarray, grid_size: tuple[int, int]) -> jnp.ndarray:
    """
    Reshapes the 1D neuron assignments into a 2D grid.

    Args:
        assignments: A 1D array of integer cluster assignments for each neuron.
        grid_size: A tuple (height, width) specifying the desired 2D grid dimensions.

    Returns:
        A 2D grid of neuron assignments.
    """
    return assignments.reshape(grid_size)

def smooth_assignments(
    assignment_grid: jnp.ndarray,
    kernel_size: int = 3,
    num_clusters: int = 3
) -> jnp.ndarray:
    """
    Smooths the 2D grid of neuron assignments using a 2D convolution.

    This function applies a separate convolution for each cluster, effectively performing a
    "mode filter" or "majority vote" within a local neighborhood. It helps to eliminate
    isolated "salt-and-pepper" noise in the assignments and encourages the formation of
    larger, contiguous regions of neurons with the same assignment.

    Args:
        assignment_grid: A 2D array of integer cluster assignments.
        kernel_size: The size of the square convolutional kernel (e.g., 3 for a 3x3 neighborhood).
        num_clusters: The total number of possible cluster assignments.

    Returns:
        A smoothed 2D grid of neuron assignments.
    """
    kernel = jnp.ones((kernel_size, kernel_size)) / (kernel_size**2)
    
    # Create a one-hot encoded representation of the assignment grid
    one_hot_grid = jax.nn.one_hot(assignment_grid, num_clusters)
    
    # Apply convolution to each cluster's one-hot map
    # The result is a grid where each cell contains the density of each cluster in its neighborhood
    smoothed_one_hot = jnp.stack(
        [convolve2d(one_hot_grid[:, :, i], kernel, mode='same') for i in range(num_clusters)],
        axis=-1
    )
    
    # Find the cluster with the highest density in each neighborhood
    smoothed_assignments = jnp.argmax(smoothed_one_hot, axis=-1)
    
    return smoothed_assignments
