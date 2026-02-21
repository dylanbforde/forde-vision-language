
import jax
import jax.numpy as jnp
import time
import functools

def verify_top_k_optimization():
    print("Verifying Top-k Optimization...")

    # Setup
    batch_size = 32
    seq_len = 2048
    k = 64

    key = jax.random.PRNGKey(0)
    scores = jax.random.normal(key, (batch_size, seq_len))

    # 1. argsort method (original)
    @jax.jit
    def method_argsort(scores):
        return jnp.argsort(scores, axis=-1)[:, -k:]

    # 2. top_k method (optimized)
    @jax.jit
    def method_top_k(scores):
        return jax.lax.top_k(scores, k)[1]

    # Warmup
    print("Warming up...")
    res_argsort = method_argsort(scores).block_until_ready()
    res_top_k = method_top_k(scores).block_until_ready()

    # Correctness Check
    print("Checking correctness...")
    # Since top_k and argsort might return different indices for duplicate values (rare for floats),
    # and top_k returns descending while argsort returns ascending,
    # we verify that the SET of selected indices is the same.

    # Sort indices per row to compare sets
    sorted_argsort = jnp.sort(res_argsort, axis=-1)
    sorted_top_k = jnp.sort(res_top_k, axis=-1)

    # Check if they are equal
    match = jnp.array_equal(sorted_argsort, sorted_top_k)
    print(f"Indices match (ignoring order): {match}")

    if not match:
        print("Mismatch found!")
        print("Argsort indices (sorted):", sorted_argsort[0, :10])
        print("Top-k indices (sorted):  ", sorted_top_k[0, :10])
        return

    # Benchmark
    print("\nBenchmarking...")

    # Measure argsort
    start = time.time()
    for _ in range(100):
        _ = method_argsort(scores).block_until_ready()
    end = time.time()
    argsort_time = (end - start) / 100
    print(f"argsort time: {argsort_time*1000:.4f} ms")

    # Measure top_k
    start = time.time()
    for _ in range(100):
        _ = method_top_k(scores).block_until_ready()
    end = time.time()
    top_k_time = (end - start) / 100
    print(f"top_k time:   {top_k_time*1000:.4f} ms")

    speedup = argsort_time / top_k_time
    print(f"Speedup: {speedup:.2f}x")

    assert speedup > 1.0, "Optimization failed to improve performance!"
    print("\nTop-k optimization verification passed!")

if __name__ == "__main__":
    verify_top_k_optimization()
