## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - Hoyer Sparsity Optimization
**Learning:** For reduction operations on large batch tensors in JAX (like Hoyer sparsity calculation), explicit vectorization along a specific axis (e.g., `axis=0`) is much faster and more memory-efficient than using `jax.vmap` combined with `transpose`. The `vmap` + `transpose` approach causes massive pre-allocation memory overhead leading to OOMs on large arrays.
**Action:** Prefer explicit axis parameters for tensor reductions instead of relying on `jax.vmap` with transpositions when calculating statistics over large batches.
