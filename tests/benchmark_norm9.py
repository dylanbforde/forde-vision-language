import jax
import jax.numpy as jnp
import optax
import time

def tree_l2_norm(tree):
    """Compute the L2 norm of a PyTree of arrays."""
    # We use tree_leaves to extract arrays, square them, flatten them,
    # and sum them. This is faster than optax.global_norm which doesn't
    # use tree_flatten+flatten correctly, but is faster than python sum
    leaves, _ = jax.tree_util.tree_flatten(tree)
    # Using tree_map on the flattened list + sum
    return jnp.sqrt(sum(jnp.sum(x**2) for x in leaves))

def opt_tree_l2_norm(tree):
    # Flattening using jax.flatten_util.ravel_pytree
    flat, _ = jax.flatten_util.ravel_pytree(tree)
    return jnp.sqrt(jnp.sum(jnp.square(flat)))


import jax.flatten_util

def bench():
    print("Generating tree...")
    tree = {f"key_{i}": jax.random.normal(jax.random.PRNGKey(i), (100, 100)) for i in range(100)}
    for _ in range(4):
        tree = {f"k_{i}": tree for i in range(4)}

    leaves = len(jax.tree.leaves(tree))
    print(f"Number of leaves: {leaves}")

    print("Compiling manual_norm...")
    start = time.time()
    @jax.jit
    def manual_norm(grads):
        return jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree.leaves(grads)))
    manual_norm(tree).block_until_ready()
    print(f"Compile time manual_norm: {time.time() - start:.2f}s")

    print("Compiling opt_norm...")
    start = time.time()
    @jax.jit
    def opt_norm(grads):
        flat, _ = jax.flatten_util.ravel_pytree(grads)
        return jnp.sqrt(jnp.sum(jnp.square(flat)))
    opt_norm(tree).block_until_ready()
    print(f"Compile time opt_norm: {time.time() - start:.2f}s")

bench()
