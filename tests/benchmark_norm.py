import jax
import jax.numpy as jnp
import optax
import time

def generate_large_pytree(depth=5, width=10):
    if depth == 0:
        return jax.random.normal(jax.random.PRNGKey(0), (100, 100))
    return {f"key_{i}": generate_large_pytree(depth - 1, width) for i in range(width)}

def bench():
    print("Generating tree...")
    tree = generate_large_pytree(depth=4, width=5)

    # Pre-compile
    print("Pre-compiling...")
    @jax.jit
    def manual_norm(grads):
        return jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree.leaves(grads)))

    @jax.jit
    def global_norm(grads):
        return optax.global_norm(grads)

    manual_norm(tree).block_until_ready()
    global_norm(tree).block_until_ready()

    print("Benchmarking...")
    start = time.time()
    for _ in range(10):
        manual_norm(tree).block_until_ready()
    end = time.time()
    print(f"Manual norm: {(end - start) * 100} ms/op")

    start = time.time()
    for _ in range(10):
        global_norm(tree).block_until_ready()
    end = time.time()
    print(f"Global norm: {(end - start) * 100} ms/op")

bench()
