## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - Hoyer Sparsity Vmap Overhead
**Learning:** Using `jax.vmap` with transpose for reduction operations on large batch tensors (like Hoyer sparsity calculation) causes significant pre-allocation memory overhead and execution time bloat in XLA. Explicitly vectorizing the operation by adding an `axis` parameter avoids this overhead and provides substantial speedups (~6x faster for large arrays).
**Action:** Always prefer explicit `axis` parameters for reductions over `jax.vmap(fn)(x.T)` patterns when processing large batched feature arrays in JAX.
