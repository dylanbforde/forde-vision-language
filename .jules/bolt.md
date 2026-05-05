## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - MoE Expert Output Loop Optimization
**Learning:** Using `jnp.stack` to eagerly compute all expert outputs in JAX Mixture of Experts causes silent OOMs and massive compile times due to large intermediate tensor allocation during XLA compilation.
**Action:** Instead, iterate over experts individually, compute their outputs, and accumulate the results using boolean masking (`jnp.where`). This avoids large intermediate tensor allocations, radically lowering compilation time and memory footprint.
