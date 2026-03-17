import jax
import jax.numpy as jnp
import time
import os

# Prevent XLA preallocation to get accurate timing
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

# Original implementation
def hoyer_sparsity_orig(x):
    n = x.shape[-1]
    l1_norm = jnp.sum(jnp.abs(x), axis=-1)
    l2_norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=-1))
    safe_l2_norm = jnp.where(l2_norm == 0, 1.0, l2_norm)
    denominator = jnp.where(n == 1, 1.0, jnp.sqrt(n) - 1)
    sparsity = (jnp.sqrt(n) - (l1_norm / safe_l2_norm)) / denominator
    sparsity = jnp.where(l2_norm == 0, 0.0, sparsity)
    return jnp.where(n == 1, 0.0, sparsity)

@jax.jit
def calc_stats_orig(activations):
    return jax.vmap(hoyer_sparsity_orig)(activations.T)

# Optimized implementation
def hoyer_sparsity_opt(x, axis=-1):
    n = x.shape[axis]
    l1_norm = jnp.sum(jnp.abs(x), axis=axis)
    l2_norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=axis))
    safe_l2_norm = jnp.where(l2_norm == 0, 1.0, l2_norm)
    denominator = jnp.where(n == 1, 1.0, jnp.sqrt(n) - 1)
    sparsity = (jnp.sqrt(n) - (l1_norm / safe_l2_norm)) / denominator
    sparsity = jnp.where(l2_norm == 0, 0.0, sparsity)
    return jnp.where(n == 1, 0.0, sparsity)

@jax.jit
def calc_stats_opt(activations):
    return hoyer_sparsity_opt(activations, axis=0)

# Generate dummy data
# Use large arrays to show memory/time impact
key = jax.random.PRNGKey(0)
# (batch * seq_len, features)
# e.g. batch=32, seq_len=2048 -> 65536
dummy_activations = jax.random.normal(key, (65536, 1024))

print("Warming up original...")
_ = calc_stats_orig(dummy_activations)
print("Warming up optimized...")
_ = calc_stats_opt(dummy_activations)

print("\nBenchmarking Original (vmap + T)...")
start = time.time()
for _ in range(100):
    res_orig = calc_stats_orig(dummy_activations).block_until_ready()
orig_time = time.time() - start
print(f"Original Time: {orig_time:.4f}s")

print("\nBenchmarking Optimized (axis=0)...")
start = time.time()
for _ in range(100):
    res_opt = calc_stats_opt(dummy_activations).block_until_ready()
opt_time = time.time() - start
print(f"Optimized Time: {opt_time:.4f}s")

print(f"\nSpeedup: {orig_time / opt_time:.2f}x")
print("Outputs match:", jnp.allclose(res_orig, res_opt, atol=1e-5))
