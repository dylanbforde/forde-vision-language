import time
import jax
import jax.numpy as jnp
from src.forde.sensing import hoyer_sparsity

def hoyer_sparsity_new(x, axis=-1):
    n = x.shape[axis]
    l1_norm = jnp.sum(jnp.abs(x), axis=axis)
    l2_norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=axis))

    # Avoid division by zero if l2_norm is zero
    safe_l2_norm = jnp.where(l2_norm == 0, 1.0, l2_norm)

    # Avoid division by zero if n is 1
    denominator = jnp.where(n == 1, 1.0, jnp.sqrt(n) - 1)

    sparsity = (jnp.sqrt(n) - (l1_norm / safe_l2_norm)) / denominator

    # If l2_norm is 0, the vector is all zeros. Sparsity is undefined/0.
    sparsity = jnp.where(l2_norm == 0, 0.0, sparsity)

    return jnp.where(n == 1, 0.0, sparsity)

def test_perf():
    key = jax.random.PRNGKey(0)
    activations = jax.random.normal(key, (8192, 4096))

    @jax.jit
    def old_way(x):
        return jax.vmap(hoyer_sparsity)(x.T)

    @jax.jit
    def new_way(x):
        return hoyer_sparsity_new(x, axis=0)

    # Warmup
    o1 = old_way(activations).block_until_ready()
    o2 = new_way(activations).block_until_ready()

    assert jnp.allclose(o1, o2)

    start = time.perf_counter()
    for _ in range(100):
        old_way(activations).block_until_ready()
    old_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(100):
        new_way(activations).block_until_ready()
    new_time = time.perf_counter() - start

    print(f"Old time: {old_time:.4f}s")
    print(f"New time: {new_time:.4f}s")
    print(f"Speedup: {old_time/new_time:.2f}x")

if __name__ == '__main__':
    test_perf()
