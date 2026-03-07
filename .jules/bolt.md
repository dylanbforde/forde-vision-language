## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - Hoyer Sparsity Vmap Overhead
**Learning:** In JAX, using `jax.vmap` combined with transpose operations for reducing across batch dimensions introduces massive compilation and execution overhead compared to explicit vectorization via an `axis` parameter.
**Action:** Always prefer updating reduction operations (like sum, mean, norm) to accept an `axis` parameter and applying them natively to N-dimensional tensors rather than relying on `vmap` to vectorize 1D operations.
