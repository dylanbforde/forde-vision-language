## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - MoE Routing Optimization (Memory)
**Learning:** In JAX/Flax Mixture of Experts (`MoELayer`), avoid eagerly computing and stacking all expert outputs into a massive intermediate tensor using `jnp.stack`. This causes severe compilation bloat and silent OOMs on large XLA compilations when scaling the number of experts.
**Action:** Iterate over experts individually, calculate their outputs, and accumulate the results using boolean masking (e.g., `jnp.where`) on the gating weights. This avoids massive tensor allocations while remaining XLA-compatible.
