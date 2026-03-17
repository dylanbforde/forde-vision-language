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
def test_orig(activations):
    return jax.vmap(hoyer_sparsity_orig)(activations.T)

@jax.jit
def test_new(activations):
    return hoyer_sparsity_axis(activations, axis=0)

activations = jax.random.normal(jax.random.PRNGKey(0), (8192, 4096))

# Warmup
test_orig(activations).block_until_ready()
test_new(activations).block_until_ready()

start = time.time()
for _ in range(100):
    test_orig(activations).block_until_ready()
orig_time = time.time() - start

start = time.time()
for _ in range(100):
    test_new(activations).block_until_ready()
new_time = time.time() - start

print(f"Original (vmap + T): {orig_time:.4f}s")
print(f"New (explicit axis): {new_time:.4f}s")

# Ensure results match
diff = jnp.max(jnp.abs(test_orig(activations) - test_new(activations)))
print(f"Max difference: {diff}")
