## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-06-11 - MoE Router Stat Calculation
**Learning:** When calculating top-1 expert routing statistics (like confidence sum or top-1 count), using boolean masks (`probs == probs.max(...)`) followed by `jnp.where` causes two major issues: 1) it allocates massive boolean masks, and 2) it double-counts statistics if there are ties (duplicate max probabilities).
**Action:** Replace this pattern with `jnp.argmax(...)` combined with `jnp.bincount(...)`. `jnp.argmax` naturally breaks ties by picking the first occurrence, and `jnp.bincount` avoids the large boolean mask allocations, resulting in an ~8x performance speedup. Note that `bincount` requires flattening multi-dimensional indices and weights using `.reshape(-1)` first.
