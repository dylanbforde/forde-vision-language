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

def calculate_neuron_stats_old(activations, gradients):
    num_features = activations.shape[-1]
    activations = activations.reshape(-1, num_features).astype(jnp.float32)
    gradients = gradients.reshape(-1, num_features).astype(jnp.float32)

    act_gini = jax.vmap(hoyer_sparsity_old)(activations.T)
    act_gdp = jnp.mean(jnp.abs(activations), axis=0)
    act_variance = jnp.var(activations, axis=0)

    grad_gini = jax.vmap(hoyer_sparsity_old)(gradients.T)
    grad_gdp = jnp.mean(jnp.abs(gradients), axis=0)

    return jnp.stack([grad_gini, grad_gdp, act_gini, act_gdp, act_variance], axis=-1)

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
    activations = activations.reshape(-1, num_features).astype(jnp.float32)
    gradients = gradients.reshape(-1, num_features).astype(jnp.float32)

    act_gini = hoyer_sparsity_new(activations, axis=0)
    act_gdp = jnp.mean(jnp.abs(activations), axis=0)
    act_variance = jnp.var(activations, axis=0)

    grad_gini = hoyer_sparsity_new(gradients, axis=0)
    grad_gdp = jnp.mean(jnp.abs(gradients), axis=0)

    return jnp.stack([grad_gini, grad_gdp, act_gini, act_gdp, act_variance], axis=-1)

# Compile functions
jit_old = jax.jit(calculate_neuron_stats_old)
jit_new = jax.jit(calculate_neuron_stats_new)

key = jax.random.PRNGKey(0)
# Smaller batch/seq length so it won't time out, but we still measure speedup
acts = jax.random.normal(key, (8, 512, 256))
grads = jax.random.normal(key, (8, 512, 256))

# Warmup
res_old = jit_old(acts, grads)
res_new = jit_new(acts, grads)
res_old.block_until_ready()
res_new.block_until_ready()

print("Differences:", jnp.max(jnp.abs(res_old - res_new)))

start = time.perf_counter()
for _ in range(100):
    res_old = jit_old(acts, grads)
    res_old.block_until_ready()
time_old = time.perf_counter() - start

start = time.perf_counter()
for _ in range(100):
    res_new = jit_new(acts, grads)
    res_new.block_until_ready()
time_new = time.perf_counter() - start

print(f"Old time: {time_old:.5f}s")
print(f"New time: {time_new:.5f}s")
