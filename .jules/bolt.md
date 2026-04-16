## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-18 - JAX MoE Eager Stacking OOM
**Learning:** In JAX/Flax MoE implementations, eagerly computing all expert outputs and stacking them with `jnp.stack([expert(x) for expert in experts])` creates a massive `(num_experts, batch_size, seq_len, d_model)` tensor. This consistently causes silent OOMs during XLA compilation for large networks.
**Action:** Always iterate over experts sequentially, calculating their output individually, and accumulating the weighted results into a running sum tensor using boolean masking (`jnp.where(weights > 0, ...)`). This keeps peak memory constrained to `(batch_size, seq_len, d_model)`.
