## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-24 - MoE Stack Compilation Bloat
**Learning:** In JAX/Flax Mixture of Experts (`MoELayer`), avoiding eager computation and stacking of all expert outputs via `jnp.stack([expert(x) for expert in experts])` is highly critical. The explicit stacking combined with dynamic indexing causes severe XLA compilation bloat and frequent OOMs. Iterating over experts and accumulating using `jnp.where` masking resolves the OOMs and roughly halves the compile time on realistic parameter scales.
**Action:** Always avoid massive intermediate `jnp.stack` allocations for conditional routing in JAX models.
