## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.
## 2024-05-23 - Sensing Optimization
**Learning:** For reduction operations on large batch tensors in JAX (like Hoyer sparsity calculation), explicitly vectorizing operations along a specific axis (e.g., `axis=0`) avoids the massive pre-allocation memory overhead of using `jax.vmap` combined with `transpose`, leading to significant speedups (e.g., ~4.6x faster on large matrices) and preventing OOMs.
**Action:** Always prefer explicitly parameterizing the `axis` parameter in reduction operations and avoiding `jax.vmap` with transposes when aggregating along batch dimensions.
