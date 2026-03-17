## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2025-03-17 - JAX Reduction Vectorization Optimization
**Learning:** For reduction operations on large batch tensors in JAX (like Hoyer sparsity calculation), explicitly vectorizing operations along a specific axis (e.g., `axis=0`) avoids the massive pre-allocation memory overhead of `jax.vmap` combined with `transpose`. This yields significant speedups (e.g., ~4.5x faster on 8192x4096 arrays) and prevents OOMs on large arrays.
**Action:** When implementing or optimizing custom reduction functions across a specific dimension, always prefer adding an `axis` parameter and explicitly summing/reducing over it rather than relying on `jax.vmap` and transposing the array, especially for large multidimensional tensors.
