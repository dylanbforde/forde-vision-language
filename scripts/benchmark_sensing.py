
import jax
import jax.numpy as jnp
import time

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

    # Note: vmap over transpose implies computing sparsity per neuron across the batch
    act_gini = jax.vmap(hoyer_sparsity_orig)(activations.T)
    act_gdp = jnp.mean(jnp.abs(activations), axis=0)
    act_variance = jnp.var(activations, axis=0)

    grad_gini = jax.vmap(hoyer_sparsity_orig)(gradients.T)
    grad_gdp = jnp.mean(jnp.abs(gradients), axis=0)

    neuron_stats = jnp.stack([grad_gini, grad_gdp, act_gini, act_gdp, act_variance], axis=-1)
    return neuron_stats

# Optimized version: Avoids vmap(transpose) and explicit reshaping where possible
def calculate_neuron_stats_opt(activations, gradients):
    # Inputs: (batch, seq, features) or (batch, features)
    # We want stats per feature, aggregating over batch (and seq)

    # Flatten batch dims
    activations = activations.reshape(-1, activations.shape[-1])
    gradients = gradients.reshape(-1, gradients.shape[-1])

    activations = activations.astype(jnp.float32)
    gradients = gradients.astype(jnp.float32)

    # 1. Hoyer sparsity: Instead of vmap(transpose), compute directly along axis=0
    # Formula: (sqrt(N) - L1/L2) / (sqrt(N) - 1)
    # where N is batch_size (number of samples per feature)

    def hoyer_per_feature(x):
        n = x.shape[0] # Batch dimension
        l1 = jnp.sum(jnp.abs(x), axis=0)
        l2 = jnp.sqrt(jnp.sum(jnp.square(x), axis=0))

        safe_l2 = jnp.where(l2 == 0, 1.0, l2)
        denom = jnp.sqrt(n) - 1
        denom = jnp.where(denom == 0, 1.0, denom)

        sparsity = (jnp.sqrt(n) - (l1 / safe_l2)) / denom
        sparsity = jnp.where(l2 == 0, 0.0, sparsity)
        return sparsity

    act_gini = hoyer_per_feature(activations)
    act_gdp = jnp.mean(jnp.abs(activations), axis=0)
    act_variance = jnp.var(activations, axis=0)

    grad_gini = hoyer_per_feature(gradients)
    grad_gdp = jnp.mean(jnp.abs(gradients), axis=0)

    neuron_stats = jnp.stack([grad_gini, grad_gdp, act_gini, act_gdp, act_variance], axis=-1)
    return neuron_stats

def benchmark_sensing():
    print("Benchmarking Sensing: Vmap vs Vectorized...")
    key = jax.random.PRNGKey(42)

    # Simulate realistic sizes
    # Batch=32, Seq=1024, Features=4096 (e.g., hidden dim)
    batch_size = 32
    seq_len = 128
    features = 4096

    activations = jax.random.normal(key, (batch_size, seq_len, features))
    gradients = jax.random.normal(key, (batch_size, seq_len, features))

    # JIT compile
    orig_jit = jax.jit(calculate_neuron_stats_orig)
    opt_jit = jax.jit(calculate_neuron_stats_opt)

    print("Verifying correctness...")
    res_orig = orig_jit(activations, gradients)
    res_opt = opt_jit(activations, gradients)

    max_diff = jnp.max(jnp.abs(res_orig - res_opt))
    print(f"Max difference: {max_diff:.6f}")
    assert max_diff < 1e-4, "Implementations do not match!"

    # Benchmark
    steps = 50

    # Warmup
    _ = orig_jit(activations, gradients).block_until_ready()
    _ = opt_jit(activations, gradients).block_until_ready()

    start = time.time()
    for _ in range(steps):
        _ = orig_jit(activations, gradients).block_until_ready()
    end = time.time()
    orig_time = (end - start) / steps

    start = time.time()
    for _ in range(steps):
        _ = opt_jit(activations, gradients).block_until_ready()
    end = time.time()
    opt_time = (end - start) / steps

    print(f"Original (vmap+transpose): {orig_time*1000:.4f} ms")
    print(f"Optimized (vectorized): {opt_time*1000:.4f} ms")
    print(f"Speedup: {orig_time/opt_time:.2f}x")

if __name__ == "__main__":
    benchmark_sensing()
