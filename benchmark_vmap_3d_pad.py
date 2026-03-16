import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve
import time

def smooth_loop_3d(one_hot_grid, kernel, num_clusters):
    d, h, w, _ = one_hot_grid.shape
    kernel_size = kernel.shape[0]
    # Apply 3D convolution per cluster channel
    smoothed_channels = []
    for c in range(num_clusters):
        channel = one_hot_grid[..., c]
        pad_d = max(0, kernel_size - channel.shape[0])
        pad_h = max(0, kernel_size - channel.shape[1])
        pad_w = max(0, kernel_size - channel.shape[2])

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            padding = (
                (pad_d // 2, pad_d - pad_d // 2),
                (pad_h // 2, pad_h - pad_h // 2),
                (pad_w // 2, pad_w - pad_w // 2),
            )
            padded_channel = jnp.pad(channel, padding, "edge")
        else:
            padded_channel = channel
            padding = ((0, 0), (0, 0), (0, 0))

        smoothed = convolve(padded_channel, kernel, mode="same")

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            start_d = padding[0][0]
            start_h = padding[1][0]
            start_w = padding[2][0]
            smoothed = smoothed[
                start_d : start_d + d, start_h : start_h + h, start_w : start_w + w
            ]

        smoothed_channels.append(smoothed)
    return jnp.stack(smoothed_channels, axis=-1)

def _conv3d(channel, kernel):
    return convolve(channel, kernel, mode="same")

def smooth_vmap_3d(one_hot_grid, kernel, num_clusters):
    d, h, w, _ = one_hot_grid.shape
    kernel_size = kernel.shape[0]

    pad_d = max(0, kernel_size - d)
    pad_h = max(0, kernel_size - h)
    pad_w = max(0, kernel_size - w)

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        padding = (
            (pad_d // 2, pad_d - pad_d // 2),
            (pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2),
            (0, 0)
        )
        padded_one_hot_grid = jnp.pad(one_hot_grid, padding, "edge")
    else:
        padded_one_hot_grid = one_hot_grid
        padding = ((0, 0), (0, 0), (0, 0), (0, 0))

    out = jax.vmap(_conv3d, in_axes=(3, None))(padded_one_hot_grid, kernel)
    smoothed_one_hot = jnp.moveaxis(out, 0, -1)

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        start_d = padding[0][0]
        start_h = padding[1][0]
        start_w = padding[2][0]
        smoothed_one_hot = smoothed_one_hot[
            start_d : start_d + d, start_h : start_h + h, start_w : start_w + w
        ]

    return smoothed_one_hot

# test correctness
kernel = jnp.ones((3, 3, 3)) / 27.0
grid = jax.random.normal(jax.random.PRNGKey(0), (1, 2, 4, 16))

res_loop = smooth_loop_3d(grid, kernel, 16)
res_vmap = smooth_vmap_3d(grid, kernel, 16)
print("Correctness 3D:", jnp.allclose(res_loop, res_vmap))

# test speed/compile time
start = time.time()
jax.jit(smooth_loop_3d, static_argnums=(2,))(grid, kernel, 16).block_until_ready()
print("Loop compile + run 3D:", time.time() - start)

start = time.time()
jax.jit(smooth_vmap_3d, static_argnums=(2,))(grid, kernel, 16).block_until_ready()
print("Vmap compile + run 3D:", time.time() - start)

# check performance on larger inputs
grid_large = jax.random.normal(jax.random.PRNGKey(1), (32, 32, 32, 64))

start = time.time()
res1 = jax.jit(smooth_loop_3d, static_argnums=(2,))(grid_large, kernel, 64).block_until_ready()
print("Loop run large 3D:", time.time() - start)

start = time.time()
res2 = jax.jit(smooth_vmap_3d, static_argnums=(2,))(grid_large, kernel, 64).block_until_ready()
print("Vmap run large 3D:", time.time() - start)
