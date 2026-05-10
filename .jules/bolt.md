## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-24 - JAX Metric Aggregation Optimization
**Learning:** In JAX, calculating aggregated metrics for selected indices (such as top-1 router confidence sums) using boolean masking combined with `jnp.where` and `.sum(...)` causes excessive pre-allocation of full-sized intermediate tensors. Using `jnp.argmax` and `jnp.bincount` avoids this bloat and provides a ~3-4x performance speedup. Furthermore, `bincount` explicitly enforces selection of only one class in cases of identical probability ties, preventing subtle double-counting bugs.
**Action:** When calculating statistics over subsets of batch items (like in MoE router statistics or attention mapping), prefer `jnp.bincount(indices, weights=values)` over `jnp.where(mask, values, 0.0).sum(...)` to optimize speed and stability. Ensure `indices` and `weights` are appropriately flattened before using `bincount`.
