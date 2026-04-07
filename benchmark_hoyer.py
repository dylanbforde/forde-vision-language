import jax
import jax.numpy as jnp
import time

def hoyer_sparsity_old(x):
    n = x.shape[-1]
    l1_norm = jnp.sum(jnp.abs(x), axis=-1)
    l2_norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=-1))

    safe_l2_norm = jnp.where(l2_norm == 0, 1.0, l2_norm)
    denominator = jnp.where(n == 1, 1.0, jnp.sqrt(n) - 1)

    sparsity = (jnp.sqrt(n) - (l1_norm / safe_l2_norm)) / denominator
    sparsity = jnp.where(l2_norm == 0, 0.0, sparsity)
    return jnp.where(n == 1, 0.0, sparsity)

def hoyer_sparsity_new(x, axis=-1):
    n = x.shape[axis]
    l1_norm = jnp.sum(jnp.abs(x), axis=axis)
    l2_norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=axis))

    safe_l2_norm = jnp.where(l2_norm == 0, 1.0, l2_norm)
    denominator = jnp.where(n == 1, 1.0, jnp.sqrt(n) - 1)

    sparsity = (jnp.sqrt(n) - (l1_norm / safe_l2_norm)) / denominator
    sparsity = jnp.where(l2_norm == 0, 0.0, sparsity)
    return jnp.where(n == 1, 0.0, sparsity)

@jax.jit
def old_way(activations):
    return jax.vmap(hoyer_sparsity_old)(activations.T)

@jax.jit
def new_way(activations):
    return hoyer_sparsity_new(activations, axis=0)

# Simulate a large batch and sequence length
# Batch size: 32, Seq Len: 1024, Features: 768
activations = jax.random.normal(jax.random.PRNGKey(0), (32 * 1024, 768))

# Warmup
old_way(activations).block_until_ready()
new_way(activations).block_until_ready()

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

# Benchmark
start = time.time()
for _ in range(100):
    res1 = old_way(activations)
res1.block_until_ready()
print(f"Old way time: {time.time() - start:.4f}s")

start = time.time()
for _ in range(100):
    res2 = new_way(activations)
res2.block_until_ready()
print(f"New way time: {time.time() - start:.4f}s")

# Check correctness
assert jnp.allclose(res1, res2, atol=1e-5), "Results don't match!"
print("Results match.")
