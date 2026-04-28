## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-24 - Avoid eager jnp.stack in MoE for memory efficiency
**Learning:** In JAX/Flax Mixture of Experts (`MoELayer`), eagerly computing and stacking all expert outputs into a massive intermediate tensor using `jnp.stack` causes massive intermediate tensor memory allocations leading to silent OOMs on large XLA compilations.
**Action:** Instead, iterate over experts individually, calculate their outputs lazily, and accumulate the results using boolean masking (`jnp.where` on top-k assignments). This drastically reduces memory allocations and OOM crashes during XLA compilation while being equivalent mathematically.
