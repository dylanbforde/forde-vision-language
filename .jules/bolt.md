## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-18 - [JAX MoE Compilation Memory Bloat]
**Learning:** In JAX/Flax Mixture of Experts (`MoELayer`), eagerly computing and stacking all expert outputs into a massive intermediate tensor using `jnp.stack` causes silent OOMs on large XLA compilations. Advanced indexing over this massive tensor further exacerbates the issue.
**Action:** Always iterate over experts individually, calculate their outputs only when processing them, and accumulate the results into a single output tensor using boolean masking (e.g., `jnp.where`). This drastically reduces memory allocations during compilation and lowers total execution time.
