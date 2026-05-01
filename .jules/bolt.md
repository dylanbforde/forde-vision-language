## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - MoE Expert Output Masking Optimization
**Learning:** Eagerly evaluating all experts and stacking their outputs into a single massive tensor `(num_experts, batch, seq, d_model)` using `jnp.stack` before routing causes silent OOM errors and significant XLA compilation bloat in large models. Iterating over experts and accumulating their outputs via boolean masking (`jnp.where`) drastically reduces memory allocations and compile times without slowing down runtime.
**Action:** Always prefer conditional accumulation or masking over eager materialization of all possible routing paths in JAX MoE architectures.
