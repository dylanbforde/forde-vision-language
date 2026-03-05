import jax
import jax.numpy as jnp
import time
import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

def hoyer_sparsity_old(x):
    n = x.shape[-1]
    l1_norm = jnp.sum(jnp.abs(x), axis=-1)
    l2_norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=-1))
    safe_l2_norm = jnp.where(l2_norm == 0, 1.0, l2_norm)
    denominator = jnp.where(n == 1, 1.0, jnp.sqrt(n) - 1)
    sparsity = (jnp.sqrt(n) - (l1_norm / safe_l2_norm)) / denominator
    sparsity = jnp.where(l2_norm == 0, 0.0, sparsity)
    return jnp.where(n == 1, 0.0, sparsity)

def calc_old(activations):
    return jax.vmap(hoyer_sparsity_old)(activations.T)

def hoyer_sparsity_new(x, axis=-1):
    n = x.shape[axis]
    l1_norm = jnp.sum(jnp.abs(x), axis=axis)
    l2_norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=axis))
    safe_l2_norm = jnp.where(l2_norm == 0, 1.0, l2_norm)
    denominator = jnp.where(n == 1, 1.0, jnp.sqrt(n) - 1)
    sparsity = (jnp.sqrt(n) - (l1_norm / safe_l2_norm)) / denominator
    sparsity = jnp.where(l2_norm == 0, 0.0, sparsity)
    return jnp.where(n == 1, 0.0, sparsity)

def calc_new(activations):
    return hoyer_sparsity_new(activations, axis=0)

calc_old_jit = jax.jit(calc_old)
calc_new_jit = jax.jit(calc_new)

key = jax.random.PRNGKey(42)
# Simulating a smaller batch
data = jax.random.normal(key, (1024 * 16, 768))

calc_old_jit(data[:10, :10]).block_until_ready()
calc_new_jit(data[:10, :10]).block_until_ready()

print("Benchmarking old...")
start = time.time()
for _ in range(10):
    out_old = calc_old_jit(data).block_until_ready()
end = time.time()
old_time = end - start
print(f"Old time: {old_time:.4f}s")

print("Benchmarking new...")
start = time.time()
for _ in range(10):
    out_new = calc_new_jit(data).block_until_ready()
end = time.time()
new_time = end - start
print(f"New time: {new_time:.4f}s")

print(f"Speedup: {old_time/new_time:.2f}x")
print(f"Results match: {jnp.allclose(out_old, out_new)}")
