## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-24 - JAX MoE Eager Stacking OOM
**Learning:** In JAX/Flax Mixture of Experts (`MoELayer`), eagerly computing and stacking all expert outputs into a massive intermediate tensor using `jnp.stack([expert(x) ...])` causes severe XLA compilation bloat and silent Out-Of-Memory errors on large batch/sequence sizes. The tensor allocation grows at $O(\text{num\_experts} \times \text{batch} \times \text{seq} \times d\_model)$, which is highly inefficient for routing layers.
**Action:** Always replace eager expert output stacking with iterative accumulation loops using boolean masking (e.g., `jnp.where(top_k_indices == i, top_k_probs, 0)`). This prevents large tensor allocations, keeps the memory footprint at $O(\text{batch} \times \text{seq} \times d\_model)$, and dramatically speeds up both XLA compilation time (~5.8x faster) and runtime execution (~1.6x faster).
