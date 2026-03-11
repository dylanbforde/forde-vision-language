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

    print("Compiling tree_map_norm...")
    start = time.time()
    @jax.jit
    def tree_map_norm(grads):
        sq_norms = jax.tree.map(lambda x: jnp.sum(jnp.square(x)), grads)
        # sum over all sq_norms
        flat_sq, _ = jax.tree_util.tree_flatten(sq_norms)
        return jnp.sqrt(jnp.sum(jnp.array(flat_sq)))
    tree_map_norm(tree).block_until_ready()
    print(f"Compile time tree_map_norm: {time.time() - start:.2f}s")

    print("Benchmarking...")
    start = time.time()
    for _ in range(10):
        manual_norm(tree).block_until_ready()
    end = time.time()
    print(f"Manual norm execution: {(end - start) * 100} ms/op")

    start = time.time()
    for _ in range(10):
        tree_map_norm(tree).block_until_ready()
    end = time.time()
    print(f"Tree map norm execution: {(end - start) * 100} ms/op")

bench()
