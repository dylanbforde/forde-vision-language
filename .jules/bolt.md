## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-18 - [Optimization] JAX MoE Routing: Avoid Eagerly Stacking Expert Outputs
**Learning:** In JAX/Flax Mixture of Experts (`MoELayer`), eagerly computing and stacking all expert outputs into a single massive tensor (`jnp.stack([expert(x) for expert in experts])`) followed by gather operations creates massive intermediate allocations (`(num_experts, batch, seq, d_model)`). This causes significant memory bloat, silent OOMs, and heavily inflates XLA compilation time.
**Action:** Always process individual experts dynamically by iterating over them and accumulating the results directly (`expert_out * mask`), which prevents massive intermediate graph nodes and results in faster compilation and reduced peak memory overhead without sacrificing runtime execution speed.
