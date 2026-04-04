## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-03-01 - Avoid `jnp.stack` for MoE Expert Execution
**Learning:** In JAX/Flax Mixture of Experts (`MoELayer`), eagerly computing and stacking all expert outputs into a massive intermediate tensor using `jnp.stack` causes silent OOMs on large compilations. The XLA compiler struggles to optimize away the unused branches when they are bundled into a large multi-dimensional stacked array, allocating massive amounts of memory during compilation and execution.
**Action:** Iterate over experts and accumulate the results (e.g., using `jnp.where` or element-wise multiplication with routing weights). This allows XLA to handle eliminating unused branches appropriately, drastically improving compilation times and eliminating OOMs for heavy MoE layers.
