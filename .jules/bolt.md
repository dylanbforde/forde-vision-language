## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2025-02-14 - Calculate Neuron Stats Optimization
**Learning:** For JAX array reductions (like calculating Hoyer sparsity across a large dimension), explicitly parameterizing the reduction `axis` on primitives like `jnp.sum` performs significantly faster (~3.4x) compared to a memory-expensive transpose operation paired with `jax.vmap()`.
**Action:** Avoid `jax.vmap()` in conjunction with `.T` (transpose) on large matrices. Rewrite functions to natively support an `axis` parameter instead.
