## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-24 - JAX Compilation XLA Graph Bloat Optimization
**Learning:** In JAX, replacing explicit Python loops with list comprehensions and `jnp.stack` for channel-wise operations (like `convolve`) creates immense XLA graph bloat, drastically inflating JIT compilation times (e.g. from 0.24s up to 1.5s for 2D, and 0.5s up to 2.1s for 3D). Using `jax.vmap` along the channel dimension, combined with hoisting padding operations out of loops to process an entire grid at once, resolves the bloat and provides ~4x-6x compilation speedups.
**Action:** Always prefer `jax.vmap` and explicitly avoid unrolled Python loops when applying the same JAX operation across independent dimensions (like image/tensor channels) to ensure efficient compilation and avoid XLA memory overhead.
