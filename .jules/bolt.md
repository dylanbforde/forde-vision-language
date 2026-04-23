## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-24 - MoE OOM with jnp.stack
**Learning:** In JAX/Flax Mixture of Experts, using `jnp.stack` to aggregate outputs from all experts into a single massive tensor (e.g. shape `(num_experts, batch, seq, d_model)`) before gathering selected expert outputs causes huge memory overhead during XLA compilation and execution, frequently leading to silent OOMs on large models.
**Action:** Always prefer iterating over individual experts and accumulating their contributions using boolean masking (`jnp.where`) into a single output buffer. This approach prevents eager large intermediate tensor allocations and improves memory efficiency during XLA compilation.
