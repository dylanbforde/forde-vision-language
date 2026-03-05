## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2025-02-27 - Vectorizing Hoyer Sparsity
**Learning:** In JAX, using `jax.vmap` combined with `.T` (transposing) for row-wise or column-wise operations on matrices can introduce significant overhead compared to explicitly passing an `axis` parameter to native JAX numpy operations. For `hoyer_sparsity`, allowing `axis=0` instead of transposing and vmapping yielded a ~1.8x speedup.
**Action:** Prefer explicit `axis` parameters in reduction operations over `jax.vmap` with transposes when possible to minimize overhead.
