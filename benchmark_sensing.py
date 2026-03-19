import jax
import jax.numpy as jnp
import time
import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

# Original
def hoyer_sparsity_orig(x):
    n = x.shape[-1]
    l1_norm = jnp.sum(jnp.abs(x), axis=-1)
    l2_norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=-1))
    safe_l2_norm = jnp.where(l2_norm == 0, 1.0, l2_norm)
    denominator = jnp.where(n == 1, 1.0, jnp.sqrt(n) - 1)
    sparsity = (jnp.sqrt(n) - (l1_norm / safe_l2_norm)) / denominator
    sparsity = jnp.where(l2_norm == 0, 0.0, sparsity)
    return jnp.where(n == 1, 0.0, sparsity)

def calculate_neuron_stats_orig(activations, gradients):
    num_features = activations.shape[-1]
    activations = activations.reshape(-1, num_features)
    gradients = gradients.reshape(-1, num_features)
    activations = activations.astype(jnp.float32)
    gradients = gradients.astype(jnp.float32)
    act_gini = jax.vmap(hoyer_sparsity_orig)(activations.T)
    act_gdp = jnp.mean(jnp.abs(activations), axis=0)
    act_variance = jnp.var(activations, axis=0)
    grad_gini = jax.vmap(hoyer_sparsity_orig)(gradients.T)
    grad_gdp = jnp.mean(jnp.abs(gradients), axis=0)
    neuron_stats = jnp.stack([grad_gini, grad_gdp, act_gini, act_gdp, act_variance], axis=-1)
    return neuron_stats

# New
def hoyer_sparsity_new(x, axis=-1):
    n = x.shape[axis]
    l1_norm = jnp.sum(jnp.abs(x), axis=axis)
    l2_norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=axis))
    safe_l2_norm = jnp.where(l2_norm == 0, 1.0, l2_norm)
    denominator = jnp.where(n == 1, 1.0, jnp.sqrt(n) - 1)
    sparsity = (jnp.sqrt(n) - (l1_norm / safe_l2_norm)) / denominator
    sparsity = jnp.where(l2_norm == 0, 0.0, sparsity)
    return jnp.where(n == 1, 0.0, sparsity)

def calculate_neuron_stats_new(activations, gradients):
    num_features = activations.shape[-1]
    activations = activations.reshape(-1, num_features)
    gradients = gradients.reshape(-1, num_features)
    activations = activations.astype(jnp.float32)
    gradients = gradients.astype(jnp.float32)
    act_gini = hoyer_sparsity_new(activations, axis=0)
    act_gdp = jnp.mean(jnp.abs(activations), axis=0)
    act_variance = jnp.var(activations, axis=0)
    grad_gini = hoyer_sparsity_new(gradients, axis=0)
    grad_gdp = jnp.mean(jnp.abs(gradients), axis=0)
    neuron_stats = jnp.stack([grad_gini, grad_gdp, act_gini, act_gdp, act_variance], axis=-1)
    return neuron_stats

# Benchmark
key = jax.random.PRNGKey(0)
# Use a relatively large batch to see the vmap + transpose overhead
batch_size, seq_len, features = 1024, 64, 512
activations = jax.random.normal(key, (batch_size, seq_len, features))
gradients = jax.random.normal(key, (batch_size, seq_len, features))

# JIT compile both
orig_jit = jax.jit(calculate_neuron_stats_orig)
new_jit = jax.jit(calculate_neuron_stats_new)

print("Warming up original...")
res_orig = orig_jit(activations, gradients).block_until_ready()
print("Warming up new...")
res_new = new_jit(activations, gradients).block_until_ready()

import numpy as np
np.testing.assert_allclose(res_orig, res_new, rtol=1e-5, atol=1e-5)
print("Results match!")

print("Benchmarking original...")
t0 = time.time()
for _ in range(10):
    res = orig_jit(activations, gradients).block_until_ready()
t1 = time.time()
orig_time = (t1 - t0) / 10
print(f"Original: {orig_time:.5f} s")

print("Benchmarking new...")
t0 = time.time()
for _ in range(10):
    res = new_jit(activations, gradients).block_until_ready()
t1 = time.time()
new_time = (t1 - t0) / 10
print(f"New: {new_time:.5f} s")

speedup = orig_time / new_time
print(f"Speedup: {speedup:.2f}x")
