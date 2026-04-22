## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - MoE JAX Compilation Memory Bottleneck
**Learning:** In JAX/Flax MoE implementations, eagerly computing and stacking all expert outputs into a single intermediate tensor (`jnp.stack`) for subsequent routing causes huge memory bloat during XLA compilation and execution, leading to silent OOMs. Iterating over experts individually and accumulating results using boolean masking (`jnp.where`) dramatically reduces memory pressure and overhead, maintaining a stable XLA graph.
**Action:** When working with independent processing paths (like MoE experts) in JAX, accumulate outputs incrementally via boolean masking rather than eagerly stacking them into large intermediate tensors.
