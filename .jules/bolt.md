## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-28 - Vectorized Reductions over Vmap
**Learning:** For reduction operations on large batch tensors in JAX (like Hoyer sparsity calculation), explicitly vectorizing the operation along a specific axis (e.g., `axis=0`) is much faster than using `jax.vmap` combined with `.T` (transpose). This avoids memory layout contiguity breakage and `vmap` overhead, yielding up to a ~5x execution speedup on large tensors.
**Action:** Always prefer explicit axis arguments for reduction operations instead of wrapping them in `jax.vmap` with transpositions when calculating statistics across specific dimensions.
