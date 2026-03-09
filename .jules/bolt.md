## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2025-02-20 - Vectorizing Norm Computations
**Learning:** In JAX, calculating norms over large multi-dimensional matrices by transposing and applying `jax.vmap` causes compilation overhead and sub-optimal memory access patterns. Direct vectorization using an explicit `axis` parameter yields significant speedups (~4x faster during JIT-compiled inference for Hoyer's Sparsity).
**Action:** Always prefer explicit axis arguments in reduction operations (like `jnp.sum`, `jnp.mean`, etc.) over `vmap` combined with matrix transposes for operations applied along a specific dimension.
