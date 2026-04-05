## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-24 - JAX Reductions Vectorization
**Learning:** For reduction operations on large batch tensors in JAX (like Hoyer sparsity calculation), prefer explicitly vectorizing operations along a specific axis (e.g., `axis=0`) over `jax.vmap` combined with `transpose`. The `vmap` + `transpose` approach causes massive pre-allocation memory overhead leading to OOMs on large arrays, whereas explicit vectorization avoids this and yields ~2x to 4x speedups.
**Action:** When performing reductions (sum, mean, sparsity), always update the function to accept an `axis` argument rather than relying on `vmap` + `.T`, which is unsafe for higher dimensional tensors and inefficient.
