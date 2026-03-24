## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2025-02-21 - Vectorization Optimization
**Learning:** For reduction operations on large batch tensors (like Hoyer sparsity), explicitly vectorizing by adding an `axis` parameter is significantly faster than using `jax.vmap` combined with `.T` (transpose). `jax.vmap` combined with transpose creates massive XLA pre-allocation overhead and memory bloat, resulting in over ~10-15x slower execution in micro-benchmarks.
**Action:** Prefer explicit array operations along specified axes over `jax.vmap` for simple statistical reductions on large arrays.
