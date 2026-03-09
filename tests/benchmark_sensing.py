import time
import jax
import jax.numpy as jnp
from src.forde.sensing import hoyer_sparsity

def calculate_neuron_stats_old(activations, gradients):
    num_features = activations.shape[-1]
    activations = activations.reshape(-1, num_features).astype(jnp.float32)
    gradients = gradients.reshape(-1, num_features).astype(jnp.float32)

    act_gini = jax.vmap(hoyer_sparsity)(activations.T)
    act_gdp = jnp.mean(jnp.abs(activations), axis=0)
    act_variance = jnp.var(activations, axis=0)

    grad_gini = jax.vmap(hoyer_sparsity)(gradients.T)
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

old_jitted = jax.jit(calculate_neuron_stats_old)
new_jitted = jax.jit(calculate_neuron_stats_new)

batch_size = 4096
seq_len = 1
features = 4096
activations = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, features))
gradients = jax.random.normal(jax.random.PRNGKey(1), (batch_size, seq_len, features))

print("Compiling old...")
_ = old_jitted(activations, gradients).block_until_ready()
print("Compiling new...")
_ = new_jitted(activations, gradients).block_until_ready()

print("Benchmarking old...")
start = time.time()
for _ in range(100):
    _ = old_jitted(activations, gradients).block_until_ready()
end = time.time()
old_time = end - start
print(f"Old time: {old_time:.4f}s")

print("Benchmarking new...")
start = time.time()
for _ in range(100):
    _ = new_jitted(activations, gradients).block_until_ready()
end = time.time()
new_time = end - start
print(f"New time: {new_time:.4f}s")

print(f"Speedup: {old_time / new_time:.2f}x")
