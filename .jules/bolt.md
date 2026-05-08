## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-18 - [Optimization Pattern] JAX bincount vs boolean masking
**Learning:** In JAX, computing histogram-like reductions (e.g. counting top-1 expert occurrences) using boolean masking and sum (like `jnp.where(mask, values, 0.0).sum(...)`) allocates massive full-size intermediate tensors and is very slow.
**Action:** Replace `mask` and `.sum(...)` with `jnp.argmax(...)` combined with `jnp.bincount(..., weights=...)` for massive performance improvements (~8x speedup) and lower memory overhead. Ensure you use `.reshape(-1)` to flatten inputs as `bincount` only accepts 1D arrays.
