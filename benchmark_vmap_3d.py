import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve
import time

def smooth_loop_3d(one_hot_grid, kernel, num_clusters):
    # Apply 3D convolution per cluster channel
    smoothed_channels = []
    for c in range(num_clusters):
        channel = one_hot_grid[..., c]
        smoothed = convolve(channel, kernel, mode="same")
        smoothed_channels.append(smoothed)
    return jnp.stack(smoothed_channels, axis=-1)

def _conv3d(channel, kernel):
    return convolve(channel, kernel, mode="same")

def smooth_vmap_3d(one_hot_grid, kernel, num_clusters):
    out = jax.vmap(_conv3d, in_axes=(3, None))(one_hot_grid, kernel)
    return jnp.moveaxis(out, 0, -1)

# test correctness
kernel = jnp.ones((3, 3, 3)) / 27.0
grid = jax.random.normal(jax.random.PRNGKey(0), (8, 8, 8, 16))

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
