## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-19 - [Avoid jnp.stack in MoELayer]
**Learning:** In JAX/Flax Mixture of Experts (`MoELayer`), eagerly computing and stacking all expert outputs into a massive intermediate tensor (`jnp.stack([expert(x) for expert in experts])`) causes significant memory bloat and XLA compilation overhead, particularly as the number of experts and sequence length scale up.
**Action:** Instead of stacking, iterate over experts, compute their outputs independently, and conditionally accumulate the weighted results (e.g., using `jnp.sum(expert_mask * top_k_probs, axis=-1, keepdims=True)`). This prevents large intermediate tensor allocations and improves memory efficiency and JIT compilation times without affecting correctness.
