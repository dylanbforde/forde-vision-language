import jax
import jax.numpy as jnp
import time
from src.forde.sensing import hoyer_sparsity

# Original vmap implementation
def original_calculate_neuron_stats(activations, gradients):
    num_features = activations.shape[-1]
    activations = activations.reshape(-1, num_features).astype(jnp.float32)
    gradients = gradients.reshape(-1, num_features).astype(jnp.float32)

    act_gini = jax.vmap(hoyer_sparsity)(activations.T)
    act_gdp = jnp.mean(jnp.abs(activations), axis=0)
    act_variance = jnp.var(activations, axis=0)

    grad_gini = jax.vmap(hoyer_sparsity)(gradients.T)
    grad_gdp = jnp.mean(jnp.abs(gradients), axis=0)

    return jnp.stack([grad_gini, grad_gdp, act_gini, act_gdp, act_variance], axis=-1)

# New optimized version
def optimized_hoyer_sparsity(x, axis=-1):
    n = x.shape[axis]
    l1_norm = jnp.sum(jnp.abs(x), axis=axis)
    l2_norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=axis))
    safe_l2_norm = jnp.where(l2_norm == 0, 1.0, l2_norm)
    denominator = jnp.where(n == 1, 1.0, jnp.sqrt(n) - 1)
    sparsity = (jnp.sqrt(n) - (l1_norm / safe_l2_norm)) / denominator
    sparsity = jnp.where(l2_norm == 0, 0.0, sparsity)
    return jnp.where(n == 1, 0.0, sparsity)

def optimized_calculate_neuron_stats(activations, gradients):
    num_features = activations.shape[-1]
    activations = activations.reshape(-1, num_features).astype(jnp.float32)
    gradients = gradients.reshape(-1, num_features).astype(jnp.float32)

    act_gini = optimized_hoyer_sparsity(activations, axis=0)
    act_gdp = jnp.mean(jnp.abs(activations), axis=0)
    act_variance = jnp.var(activations, axis=0)

    grad_gini = optimized_hoyer_sparsity(gradients, axis=0)
    grad_gdp = jnp.mean(jnp.abs(gradients), axis=0)

    return jnp.stack([grad_gini, grad_gdp, act_gini, act_gdp, act_variance], axis=-1)

jitted_original = jax.jit(original_calculate_neuron_stats)
jitted_optimized = jax.jit(optimized_calculate_neuron_stats)

batch_size = 64
seq_len = 2048
features = 1024

key = jax.random.PRNGKey(0)
key1, key2 = jax.random.split(key)
activations = jax.random.normal(key1, (batch_size, seq_len, features))
gradients = jax.random.normal(key2, (batch_size, seq_len, features))

# Warmup
o1 = jitted_original(activations, gradients)
o2 = jitted_optimized(activations, gradients)
jax.block_until_ready(o1)
jax.block_until_ready(o2)

# Check correctness
assert jnp.allclose(o1, o2, atol=1e-5), "Results do not match!"

# Benchmark original
start = time.time()
for _ in range(10):
    res = jitted_original(activations, gradients)
    jax.block_until_ready(res)
end = time.time()
orig_time = (end - start) / 10

# Benchmark optimized
start = time.time()
for _ in range(10):
    res = jitted_optimized(activations, gradients)
    jax.block_until_ready(res)
end = time.time()
opt_time = (end - start) / 10

print(f"Original time: {orig_time:.5f} s")
print(f"Optimized time: {opt_time:.5f} s")
print(f"Speedup: {orig_time / opt_time:.2f}x")
