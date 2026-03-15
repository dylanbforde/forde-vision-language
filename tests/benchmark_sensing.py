import jax
import jax.numpy as jnp
import time

def hoyer_sparsity_vmap(x):
    n = x.shape[-1]
    l1_norm = jnp.sum(jnp.abs(x), axis=-1)
    l2_norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=-1))

    safe_l2_norm = jnp.where(l2_norm == 0, 1.0, l2_norm)
    denominator = jnp.where(n == 1, 1.0, jnp.sqrt(n) - 1)

    sparsity = (jnp.sqrt(n) - (l1_norm / safe_l2_norm)) / denominator
    sparsity = jnp.where(l2_norm == 0, 0.0, sparsity)
    return jnp.where(n == 1, 0.0, sparsity)

def calculate_neuron_stats_vmap(activations, gradients):
    num_features = activations.shape[-1]
    activations = activations.reshape(-1, num_features).astype(jnp.float32)
    gradients = gradients.reshape(-1, num_features).astype(jnp.float32)

    act_gini = jax.vmap(hoyer_sparsity_vmap)(activations.T)
    act_gdp = jnp.mean(jnp.abs(activations), axis=0)
    act_variance = jnp.var(activations, axis=0)

    grad_gini = jax.vmap(hoyer_sparsity_vmap)(gradients.T)
    grad_gdp = jnp.mean(jnp.abs(gradients), axis=0)

    return jnp.stack([grad_gini, grad_gdp, act_gini, act_gdp, act_variance], axis=-1)


def hoyer_sparsity_axis(x, axis=-1):
    n = x.shape[axis]
    l1_norm = jnp.sum(jnp.abs(x), axis=axis)
    l2_norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=axis))

    safe_l2_norm = jnp.where(l2_norm == 0, 1.0, l2_norm)
    denominator = jnp.where(n == 1, 1.0, jnp.sqrt(n) - 1)

    sparsity = (jnp.sqrt(n) - (l1_norm / safe_l2_norm)) / denominator
    sparsity = jnp.where(l2_norm == 0, 0.0, sparsity)
    return jnp.where(n == 1, 0.0, sparsity)

def calculate_neuron_stats_axis(activations, gradients):
    num_features = activations.shape[-1]
    activations = activations.reshape(-1, num_features).astype(jnp.float32)
    gradients = gradients.reshape(-1, num_features).astype(jnp.float32)

    act_gini = hoyer_sparsity_axis(activations, axis=0)
    act_gdp = jnp.mean(jnp.abs(activations), axis=0)
    act_variance = jnp.var(activations, axis=0)

    grad_gini = hoyer_sparsity_axis(gradients, axis=0)
    grad_gdp = jnp.mean(jnp.abs(gradients), axis=0)

    return jnp.stack([grad_gini, grad_gdp, act_gini, act_gdp, act_variance], axis=-1)

# Compile first
activations = jax.random.normal(jax.random.PRNGKey(0), (128, 1024))
gradients = jax.random.normal(jax.random.PRNGKey(1), (128, 1024))

print("Compiling vmap...")
start = time.time()
jax.block_until_ready(calculate_neuron_stats_vmap(activations, gradients))
print(f"vmap compile: {time.time() - start:.4f}s")

print("Compiling axis...")
start = time.time()
jax.block_until_ready(calculate_neuron_stats_axis(activations, gradients))
print(f"axis compile: {time.time() - start:.4f}s")

# Benchmark large scale
activations = jax.random.normal(jax.random.PRNGKey(0), (8192, 4096))
gradients = jax.random.normal(jax.random.PRNGKey(1), (8192, 4096))

print("Benchmarking vmap...")
start = time.time()
for _ in range(10):
    res_vmap = jax.block_until_ready(calculate_neuron_stats_vmap(activations, gradients))
vmap_time = (time.time() - start) / 10
print(f"vmap time: {vmap_time:.4f}s")

print("Benchmarking axis...")
start = time.time()
for _ in range(10):
    res_axis = jax.block_until_ready(calculate_neuron_stats_axis(activations, gradients))
axis_time = (time.time() - start) / 10
print(f"axis time: {axis_time:.4f}s")

# Verification
diff = jnp.abs(res_vmap - res_axis).max()
print(f"Max difference: {diff}")
