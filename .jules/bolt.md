## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - Hoyer Sparsity Vmap Overhead
**Learning:** Using `jax.vmap` combined with transpose (`.T`) for calculating metrics like Hoyer sparsity across the batch dimension of large tensors (e.g. `(batch_size, features)`) introduces massive pre-allocation memory overhead and severe compilation bloat (~15x slower to compile/execute on large arrays).
**Action:** Always favor explicit vectorization via an `axis` parameter inside reduction operations (e.g., `jnp.sum(x, axis=0)`) instead of implicitly wrapping a 1D function with `jax.vmap` when dealing with batch-wise statistics, particularly for sensing and statistics logging in large JAX loops.
