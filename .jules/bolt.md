## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2025-04-19 - JAX Vectorization over Loops in Smoothing
**Learning:** Using explicit Python `for` loops with `jnp.stack` to apply operations (like convolutions) across independent channels causes JAX to unroll the loop, resulting in massive XLA compile-time bloat (e.g., 1.5s vs 0.1s for a simple 3D smoothing function).
**Action:** Always prefer `jax.vmap` combined with axis transpositions or parameter adjustments to process independent channels simultaneously. Additionally, hoist common operations like `jnp.pad` outside the mapped function to act on the entire N-dimensional array at once.
