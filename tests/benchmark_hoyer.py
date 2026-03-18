import jax
import jax.numpy as jnp
import time
import os

# Prevent pre-allocation for accurate benchmarking
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

def hoyer_sparsity_vmap(x):
    n = x.shape[-1]
    l1_norm = jnp.sum(jnp.abs(x), axis=-1)
    l2_norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=-1))
    safe_l2_norm = jnp.where(l2_norm == 0, 1.0, l2_norm)
    denominator = jnp.where(n == 1, 1.0, jnp.sqrt(n) - 1)
    sparsity = (jnp.sqrt(n) - (l1_norm / safe_l2_norm)) / denominator
    sparsity = jnp.where(l2_norm == 0, 0.0, sparsity)
    return jnp.where(n == 1, 0.0, sparsity)

def hoyer_sparsity_axis(x, axis=-1):
    n = x.shape[axis]
    l1_norm = jnp.sum(jnp.abs(x), axis=axis)
    l2_norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=axis))
    safe_l2_norm = jnp.where(l2_norm == 0, 1.0, l2_norm)
    denominator = jnp.where(n == 1, 1.0, jnp.sqrt(n) - 1)
    sparsity = (jnp.sqrt(n) - (l1_norm / safe_l2_norm)) / denominator
    sparsity = jnp.where(l2_norm == 0, 0.0, sparsity)
    return jnp.where(n == 1, 0.0, sparsity)

@jax.jit
def vmap_method(activations):
    return jax.vmap(hoyer_sparsity_vmap)(activations.T)

@jax.jit
def axis_method(activations):
    return hoyer_sparsity_axis(activations, axis=0)

def main():
    print("Benchmarking Hoyer Sparsity: vmap vs explicit axis")
    key = jax.random.PRNGKey(42)
    # Use a large batch size and feature dimension
    activations = jax.random.normal(key, (100000, 1024))

    # Warmup
    vmap_method(activations).block_until_ready()
    axis_method(activations).block_until_ready()

    # Benchmark vmap
    start = time.time()
    for _ in range(100):
        res1 = vmap_method(activations).block_until_ready()
    vmap_time = time.time() - start

    # Benchmark axis
    start = time.time()
    for _ in range(100):
        res2 = axis_method(activations).block_until_ready()
    axis_time = time.time() - start

    print(f"vmap + transpose time: {vmap_time:.4f}s")
    print(f"explicit axis time:    {axis_time:.4f}s")
    print(f"Speedup:               {vmap_time / axis_time:.2f}x")

    # Verify correctness
    assert jnp.allclose(res1, res2, atol=1e-5), "Results do not match!"

if __name__ == "__main__":
    main()
