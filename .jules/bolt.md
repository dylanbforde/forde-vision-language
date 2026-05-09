## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-25 - Efficient Expert Stat Aggregation
**Learning:** In JAX, calculating aggregated metrics for selected indices (like top-1 expert statistics) using `mask = probs == probs.max(...)` followed by `jnp.where(mask, values, 0.0).sum(...)` allocates large full-size intermediate tensors and scales poorly ($O(B \times L \times E)$). By replacing this pattern with `jnp.argmax(probs)` combined with `jnp.bincount(indices, weights=max_probs)`, you avoid massive boolean mask allocations, enforce strict tie-breaking (picking the first occurrence instead of erroneously double-counting), and achieve significant speedups (~2x-8x in isolated component benchmarks).
**Action:** Always replace multi-dimensional boolean masks and `.sum(...)` reductions for top-k/top-1 aggregation with `argmax` and `bincount` on flattened arrays.
