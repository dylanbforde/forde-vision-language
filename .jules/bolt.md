## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2026-03-25 - MoE Massive Tensor Stacking Bottleneck
**Learning:** In JAX/Flax MoE implementations, evaluating all experts via a list comprehension and eagerly stacking them `jnp.stack([expert(x) for expert in experts], axis=0)` forces XLA to compile an enormous graph and pre-allocate massive memory (e.g. `num_experts * batch * seq * d_model`). Iterating over experts and accumulating outputs instead significantly reduces JIT compile time and execution time, as well as preventing OOMs for a high number of experts.
**Action:** Avoid eager computation and `jnp.stack` across experts in MoE implementations. Prefer accumulating the outputs sequentially via loop and conditional masks (`jnp.where`).
