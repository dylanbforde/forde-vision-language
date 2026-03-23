## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - Hoyer Sparsity Vmap Overhead
**Learning:** For reduction operations on large batch tensors in JAX (like Hoyer sparsity calculation), using `jax.vmap` combined with `transpose` (`jax.vmap(hoyer_sparsity)(x.T)`) causes massive pre-allocation memory overhead and slower compilation. Explicitly vectorizing the operation by passing an `axis` parameter (`axis=0`) avoids this and yields significant speedups (~40% reduction in execution time in benchmarks) and cleaner code.
**Action:** Always prefer native JAX vectorization via `axis` arguments for reduction functions over wrapping them in `jax.vmap` and transposing inputs, especially when operating over batch dimensions of large tensors.
