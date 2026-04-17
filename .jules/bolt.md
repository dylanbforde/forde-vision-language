## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2025-02-28 - MoE Routing XLA Memory Optimization
**Learning:** In JAX/Flax MoE implementations, eagerly computing and stacking all expert outputs into a massive intermediate tensor using `jnp.stack` causes massive memory allocation during compilation and very slow execution (O(N^2) scaling with batch and sequence length due to the gather operations). Instead, iterating over experts and accumulating outputs conditionally using `jnp.where` on a boolean mask prevents this pre-allocation and is massively faster (~18x speedup observed in benchmarks).
**Action:** Always avoid `jnp.stack` across all expert outputs in XLA-compiled routing layers; prefer iterative evaluation and boolean accumulation (`jnp.where`) for scalable MoE routing.
