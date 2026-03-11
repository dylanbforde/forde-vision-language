import jax
import jax.numpy as jnp
import optax
import time
from jax.flatten_util import ravel_pytree

def generate_large_pytree(depth=5, width=6):
    if depth == 0:
        return jax.random.normal(jax.random.PRNGKey(0), (10, 10))
    return {f"key_{i}": generate_large_pytree(depth - 1, width) for i in range(width)}

def bench():
    print("Generating tree...")
    tree = generate_large_pytree(depth=4, width=6)
    leaves = len(jax.tree.leaves(tree))
    print(f"Number of leaves: {leaves}")

    print("Compiling manual_norm...")
    start = time.time()
    @jax.jit
    def manual_norm(grads):
        return jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree.leaves(grads)))
    manual_norm(tree).block_until_ready()
    print(f"Compile time manual_norm: {time.time() - start:.2f}s")

    print("Compiling optax_global_norm...")
    start = time.time()
    @jax.jit
    def optax_global_norm(grads):
        return optax.global_norm(grads)
    optax_global_norm(tree).block_until_ready()
    print(f"Compile time optax_global_norm: {time.time() - start:.2f}s")

    print("Compiling tree_map_norm...")
    start = time.time()
    @jax.jit
    def tree_map_norm(grads):
        # Flatten before squaring! This allows a single compilation node for stack and sum
        leaves, _ = jax.tree_util.tree_flatten(grads)
        # We can't stack them unless they are the same shape, which they are in this dummy tree,
        # but in real life they aren't. We must square then sum, then stack
        sq_norms = [jnp.sum(jnp.square(x)) for x in leaves]
        # jnp.sum on list of scalars creates a huge graph, so array then sum!
        return jnp.sqrt(jnp.sum(jnp.array(sq_norms)))
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
        optax_global_norm(tree).block_until_ready()
    end = time.time()
    print(f"Optax global norm execution: {(end - start) * 100} ms/op")

    start = time.time()
    for _ in range(10):
        tree_map_norm(tree).block_until_ready()
    end = time.time()
    print(f"Tree map norm execution: {(end - start) * 100} ms/op")

bench()
