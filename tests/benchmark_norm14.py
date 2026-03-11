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
    tree = generate_large_pytree(depth=4, width=6)
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

    print("Compiling fast_global_norm...")
    start = time.time()
    @jax.jit
    def fast_global_norm(grads):
        flat_grads, _ = jax.tree_util.tree_flatten(grads)
        # Vectorize sum of squares
        sq_norms = [jnp.sum(jnp.square(x)) for x in flat_grads]
        # Use jnp.sum to avoid building massive unrolled expression trees
        return jnp.sqrt(jnp.sum(jnp.array(sq_norms)))
    fast_global_norm(tree).block_until_ready()
    print(f"Compile time fast_global_norm: {time.time() - start:.2f}s")

    print("Compiling optax_tree_l2_norm...")
    start = time.time()
    @jax.jit
    def optax_tree_l2_norm(grads):
        return optax.tree_utils.tree_l2_norm(grads)
    optax_tree_l2_norm(tree).block_until_ready()
    print(f"Compile time optax_tree_l2_norm: {time.time() - start:.2f}s")

    print("Benchmarking...")
    start = time.time()
    for _ in range(10):
        manual_norm(tree).block_until_ready()
    end = time.time()
    print(f"Manual norm execution: {(end - start) * 100} ms/op")

    start = time.time()
    for _ in range(10):
        fast_global_norm(tree).block_until_ready()
    end = time.time()
    print(f"Fast global norm execution: {(end - start) * 100} ms/op")

    start = time.time()
    for _ in range(10):
        optax_tree_l2_norm(tree).block_until_ready()
    end = time.time()
    print(f"optax_tree_l2_norm execution: {(end - start) * 100} ms/op")

bench()
