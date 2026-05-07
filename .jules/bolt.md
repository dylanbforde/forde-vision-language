## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.
## 2024-05-18 - [Optimization Pattern: Bincount vs Where-Sum]
**Learning:** In JAX, calculating aggregated metrics for selected indices using `jnp.where(mask, values, 0.0).sum(...)` requires allocating large full-size tensors for the boolean mask and intermediate zeroed values, leading to memory bloat and slower execution, especially for large sequences and expert counts.
**Action:** Replace `where`/`sum` patterns and direct `mask.sum()` calculations with `jnp.bincount(indices, weights=values)` and `jnp.bincount(indices)`. This flattens operations and avoids large intermediate memory allocations, resulting in significantly faster runtime.
