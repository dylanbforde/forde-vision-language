## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.
## 2024-05-23 - Top-1 MoE Routing Statistics Aggregation
**Learning:** Using `mask = probs == probs.max(...)` followed by `jnp.where` for routing statistics will erroneously double-count confidence sums and usage counts if there are exact ties in routing probabilities. In JAX, using `jnp.argmax` combined with `jnp.bincount` not only enforces strict top-1 selection (resolving ties by index), but also provides an ~8x performance speedup by avoiding large intermediate tensor allocations `(batch, seq, num_experts)`.
**Action:** Always prefer `jnp.bincount` and `jnp.argmax` over boolean mask summation for categorical counting operations in JAX.
