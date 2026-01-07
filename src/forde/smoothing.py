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
    # Calculate padding needed to ensure grid dimensions are > kernel_size
    original_h, original_w, _ = one_hot_grid.shape
    
    pad_h = max(0, kernel_size + 1 - original_h)
    pad_w = max(0, kernel_size + 1 - original_w)

    # Apply padding to one_hot_grid
    # Pad symmetrically
    padding_config = (
        (pad_h // 2, pad_h - pad_h // 2),
        (pad_w // 2, pad_w - pad_w // 2),
        (0, 0) # No padding for num_clusters dimension
    )
    padded_one_hot_grid = jnp.pad(one_hot_grid, padding_config, 'constant')

    # Convolve on padded grid
    smoothed_padded_one_hot_grid = jnp.stack(
        [convolve2d(padded_one_hot_grid[:, :, i], kernel, mode='same') for i in range(num_clusters)],
        axis=-1
    )

    # Unpad the result to original one_hot_grid size
    unpadded_smoothed_one_hot_grid = smoothed_padded_one_hot_grid[
        padding_config[0][0] : padding_config[0][0] + original_h,
        padding_config[1][0] : padding_config[1][0] + original_w,
        :
    ]
    smoothed_one_hot = unpadded_smoothed_one_hot_grid
    
    # Find the cluster with the highest density in each neighborhood
    smoothed_assignments = jnp.argmax(smoothed_one_hot, axis=-1)
    
    return smoothed_assignments


def smooth_assignments_3d(
    assignment_grid: jnp.ndarray,
    kernel_size: int = 3,
    num_clusters: int = 3
) -> jnp.ndarray:
    """
    Smooths a 3D grid of assignments using 3D convolution.
    
    Useful for MoE architectures where we have (experts, neuron_grid_h, neuron_grid_w).
    Encourages consistency across experts and within expert neuron grids.
    
    Args:
        assignment_grid: 3D array of integer cluster assignments (D, H, W).
        kernel_size: Size of the cubic kernel (e.g., 3 for 3x3x3).
        num_clusters: Total number of clusters.
        
    Returns:
        Smoothed 3D grid of assignments.
    """
    from jax.scipy.signal import convolve
    
    # Create normalized 3D kernel
    kernel = jnp.ones((kernel_size, kernel_size, kernel_size)) / (kernel_size**3)
    
    # One-hot encode: (D, H, W, num_clusters)
    one_hot_grid = jax.nn.one_hot(assignment_grid, num_clusters)
    
    # Pad to handle boundaries
    d, h, w, _ = one_hot_grid.shape
    pad = kernel_size // 2
    
    # Apply 3D convolution per cluster channel
    smoothed_channels = []
    for c in range(num_clusters):
        # Extract channel c: (D, H, W)
        channel = one_hot_grid[..., c]
        
        # JAX convolve requires one input to be smaller than the other in EVERY dimension
        # If grid is small (e.g. 1x2x4) and kernel is 3x3x3, this fails.
        # We must pad the channel to be at least kernel size in all dims.
        
        pad_d = max(0, kernel_size - channel.shape[0])
        pad_h = max(0, kernel_size - channel.shape[1])
        pad_w = max(0, kernel_size - channel.shape[2])
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            # Pad symmetrically where possible
            padding = (
                (pad_d // 2, pad_d - pad_d // 2),
                (pad_h // 2, pad_h - pad_h // 2),
                (pad_w // 2, pad_w - pad_w // 2)
            )
            padded_channel = jnp.pad(channel, padding, 'edge') # Use edge padding to extend values
        else:
            padded_channel = channel
            padding = ((0,0), (0,0), (0,0))
            
        # Convolve with mode='same'
        smoothed = convolve(padded_channel, kernel, mode='same')
        
        # If we padded, we need to crop back to original size
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            start_d = padding[0][0]
            start_h = padding[1][0]
            start_w = padding[2][0]
            smoothed = smoothed[
                start_d : start_d + d,
                start_h : start_h + h,
                start_w : start_w + w
            ]
            
        smoothed_channels.append(smoothed)
        
    # Stack back: (D, H, W, num_clusters)
    smoothed_one_hot = jnp.stack(smoothed_channels, axis=-1)
    
    # Argmax to get smoothed assignments
    smoothed_assignments = jnp.argmax(smoothed_one_hot, axis=-1)
    
    return smoothed_assignments
