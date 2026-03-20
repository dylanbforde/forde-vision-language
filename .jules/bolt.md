## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - Hoyer Sparsity Axis Vectorization Optimization
**Learning:** Using `jax.vmap` combined with `.T` (transpose) for reductions (like Hoyer Sparsity) over large batch arrays introduces significant execution overhead and pre-allocation memory bloat. Explicitly rewriting the reduction operation to support an `axis` argument (e.g., `axis=0`) avoids the `vmap` + `transpose` machinery, leading to considerable speedups (up to ~4.3x for certain shapes, consistently 2x faster for large arrays).
**Action:** When performing reduction operations along specific dimensions of multi-dimensional arrays, prefer explicit parameterization of the reduction axis over `jax.vmap` of transposed arrays to minimize memory and time overhead.
