## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-06-01 - Expert Execution Loop Optimization
**Learning:** In JAX/Flax Mixture of Experts (`MoELayer`), eagerly computing and stacking all expert outputs into a massive intermediate tensor using `jnp.stack([expert(x) for expert in experts])` causes excessive memory allocation and slower execution times (e.g., 23.2s -> 19.5s when optimized).
**Action:** Iterate over experts and accumulate the results (e.g., using `jnp.where` with routing weights) to drastically improve execution times and reduce memory overhead.
