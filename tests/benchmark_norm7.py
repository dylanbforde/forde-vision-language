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

    print("Compiling ravel_norm...")
    start = time.time()
    @jax.jit
    def ravel_norm(grads):
        flat, _ = ravel_pytree(grads)
        return jnp.sqrt(jnp.sum(jnp.square(flat)))
    ravel_norm(tree).block_until_ready()
    print(f"Compile time ravel_norm: {time.time() - start:.2f}s")

    print("Benchmarking...")
    start = time.time()
    for _ in range(10):
        manual_norm(tree).block_until_ready()
    end = time.time()
    print(f"Manual norm execution: {(end - start) * 100} ms/op")

    start = time.time()
    for _ in range(10):
        ravel_norm(tree).block_until_ready()
    end = time.time()
    print(f"Ravel norm execution: {(end - start) * 100} ms/op")

bench()
