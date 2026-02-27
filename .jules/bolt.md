## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - Sensing Optimization
**Learning:** For reduction operations like Hoyer's sparsity calculation on large batch tensors, vectorizing the operation explicitly over an axis (e.g. `axis=0`) is much faster (~3.7x) than using `jax.vmap` with transposed inputs, due to avoiding the internal reshaping/vmap overhead.
**Action:** Default to vectorized axis operations for reductions rather than `jax.vmap` over matrix transposes in JAX.
