## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2025-03-03 - Vectorization over Vmap
**Learning:** For reduction operations like Hoyer sparsity calculation on large multi-dimensional arrays (e.g., `(batch, seq, features)`), explicitly vectorizing operations along a specific axis (`axis=0`) is significantly faster than using `jax.vmap` paired with array transpositions. The vmap compilation overhead and memory layout preservation result in a ~3.8x speedup locally.
**Action:** Always prefer native JAX parameter axes for reductions over relying on `jax.vmap` wrapping 1D implementations.
