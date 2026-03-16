import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d
import time

def smooth_loop(padded_one_hot_grid, kernel, num_clusters):
    return jnp.stack(
        [
            convolve2d(padded_one_hot_grid[:, :, i], kernel, mode="same")
            for i in range(num_clusters)
        ],
        axis=-1,
    )

def _conv(grid, kernel):
    return convolve2d(grid, kernel, mode="same")

def smooth_vmap(padded_one_hot_grid, kernel, num_clusters):
    # in_axes=(2, None) maps over the last dimension of padded_one_hot_grid
    # out is (num_clusters, H, W)
    out = jax.vmap(_conv, in_axes=(2, None))(padded_one_hot_grid, kernel)
    return jnp.moveaxis(out, 0, -1)

# test correctness
kernel = jnp.ones((3, 3)) / 9.0
grid = jax.random.normal(jax.random.PRNGKey(0), (64, 64, 16))

res_loop = smooth_loop(grid, kernel, 16)
res_vmap = smooth_vmap(grid, kernel, 16)
print("Correctness:", jnp.allclose(res_loop, res_vmap))

# test speed/compile time
start = time.time()
jax.jit(smooth_loop, static_argnums=(2,))(grid, kernel, 16).block_until_ready()
print("Loop compile + run:", time.time() - start)

start = time.time()
jax.jit(smooth_vmap, static_argnums=(2,))(grid, kernel, 16).block_until_ready()
print("Vmap compile + run:", time.time() - start)

# check performance on larger inputs
grid_large = jax.random.normal(jax.random.PRNGKey(1), (256, 256, 64))

start = time.time()
res1 = jax.jit(smooth_loop, static_argnums=(2,))(grid_large, kernel, 64).block_until_ready()
print("Loop run large:", time.time() - start)

start = time.time()
res2 = jax.jit(smooth_vmap, static_argnums=(2,))(grid_large, kernel, 64).block_until_ready()
print("Vmap run large:", time.time() - start)
