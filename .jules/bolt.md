## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - MoE Gather Optimization
**Learning:** In JAX/Flax Mixture of Experts (`MoELayer`), avoid eagerly computing and stacking all expert outputs into a massive intermediate tensor using `jnp.stack` prior to gathering. This causes severe XLA compilation bloat and silent OOMs on large compilations. Instead, iterating over experts to compute their output and conditionally accumulating it using a mask drastically reduces peak memory and speeds up compilation times significantly, with identical mathematical results.
**Action:** When implementing custom scatter/gather operations across multiple independent modules in JAX, prefer loop-based accumulation with masking over creating an explicitly stacked dense intermediate tensor.
