import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d

def smooth_assignments(
    assignment_grid: jnp.ndarray, kernel_size: int = 3, num_clusters: int = 3
) -> jnp.ndarray:
    kernel = jnp.ones((kernel_size, kernel_size)) / (kernel_size**2)

    # Create a one-hot encoded representation of the assignment grid
    one_hot_grid = jax.nn.one_hot(assignment_grid, num_clusters)

    original_h, original_w, _ = one_hot_grid.shape

    pad_h = max(0, kernel_size + 1 - original_h)
    pad_w = max(0, kernel_size + 1 - original_w)

    padding_config = (
        (pad_h // 2, pad_h - pad_h // 2),
        (pad_w // 2, pad_w - pad_w // 2),
        (0, 0),  # No padding for num_clusters dimension
    )
    padded_one_hot_grid = jnp.pad(one_hot_grid, padding_config, "constant")

    # Vectorize convolution across clusters using vmap
    channels = jnp.transpose(padded_one_hot_grid, (2, 0, 1))

    def conv_fn(x):
        return convolve2d(x, kernel, mode="same")

    smoothed_channels = jax.vmap(conv_fn)(channels)
    smoothed_padded_one_hot_grid = jnp.transpose(smoothed_channels, (1, 2, 0))

    # Unpad the result to original one_hot_grid size
    unpadded_smoothed_one_hot_grid = smoothed_padded_one_hot_grid[
        padding_config[0][0] : padding_config[0][0] + original_h,
        padding_config[1][0] : padding_config[1][0] + original_w,
        :,
    ]
    smoothed_one_hot = unpadded_smoothed_one_hot_grid

    # Find the cluster with the highest density in each neighborhood
    smoothed_assignments = jnp.argmax(smoothed_one_hot, axis=-1)

    return smoothed_assignments
