## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-04 - Optimize JAX MoE Expert Output Computation
**Learning:** In JAX/Flax Mixture of Experts (`MoELayer`), eagerly computing and stacking all expert outputs into a massive intermediate tensor using `jnp.stack([expert(x) for expert in experts])` causes excessive memory allocations during XLA compilation and execution, leading to extremely slow compile times (~2.2s vs ~0.27s) and slow execution times due to memory bloat.
**Action:** Instead, iterate over experts individually, calculate their outputs, and accumulate the results into a zero tensor using boolean masking (`jnp.where` on top-k indices). This drastically reduces XLA HLO bloat, lowering both memory allocations and execution time (~60x faster execution in microbenchmarks).
