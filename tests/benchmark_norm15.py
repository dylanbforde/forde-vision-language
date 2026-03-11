import jax
import jax.numpy as jnp
import optax
import time

def generate_large_pytree(depth=5, width=6):
    if depth == 0:
        return jax.random.normal(jax.random.PRNGKey(0), (10, 10))
    return {f"key_{i}": generate_large_pytree(depth - 1, width) for i in range(width)}

def bench():
    print("Generating tree...")
    tree = generate_large_pytree(depth=5, width=5)
    leaves = len(jax.tree.leaves(tree))
    print(f"Number of leaves: {leaves}")

    # Pre-compile
    print("Compiling manual_norm...")
    start = time.time()
    @jax.jit
    def manual_norm(grads):
        return jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree.leaves(grads)))
    manual_norm(tree).block_until_ready()
    print(f"Compile time manual_norm: {time.time() - start:.2f}s")

    print("Compiling optax.global_norm...")
    start = time.time()
    @jax.jit
    def optax_global_norm(grads):
        return optax.global_norm(grads)
    optax_global_norm(tree).block_until_ready()
    print(f"Compile time optax.global_norm: {time.time() - start:.2f}s")

    print("Compiling global_norm_fast...")
    start = time.time()
    @jax.jit
    def global_norm_fast(grads):
        # We can avoid the large graph tree reduce issues and manual unroll issues
        # by flattening to a 1D array using jax.flatten_util.ravel_pytree
        flat_grads, _ = jax.flatten_util.ravel_pytree(grads)
        return jnp.sqrt(jnp.sum(jnp.square(flat_grads)))
    global_norm_fast(tree).block_until_ready()
    print(f"Compile time global_norm_fast: {time.time() - start:.2f}s")

    print("Benchmarking...")
    start = time.time()
    for _ in range(10):
        manual_norm(tree).block_until_ready()
    end = time.time()
    print(f"Manual norm execution: {(end - start) * 100} ms/op")

    start = time.time()
    for _ in range(10):
        optax_global_norm(tree).block_until_ready()
    end = time.time()
    print(f"optax.global_norm execution: {(end - start) * 100} ms/op")

    start = time.time()
    for _ in range(10):
        global_norm_fast(tree).block_until_ready()
    end = time.time()
    print(f"global_norm_fast execution: {(end - start) * 100} ms/op")

bench()
