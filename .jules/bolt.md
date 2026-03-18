## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2025-03-03 - Avoid vmap+transpose for reductions on large batch tensors
**Learning:** For reduction operations on large batch tensors (e.g. `hoyer_sparsity`), `jax.vmap` combined with `.T` can cause massive pre-allocation memory overhead leading to OOMs and significant speed penalties. In `calculate_neuron_stats`, the `vmap` + `transpose` method caused a 4.6x speed penalty compared to directly supporting explicit `axis` parameters.
**Action:** Expose an `axis` argument on mathematical functions intended to be run across dimensions instead of relying on `jax.vmap` combined with transpose operations, especially when operating on very large arrays.
