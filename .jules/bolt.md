## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-15 - Vectorizing Hoyer Sparsity
**Learning:** In JAX, using `jax.vmap` combined with transpose (`activations.T`) to apply a function across the batch dimension for each feature can cause massive pre-allocation memory overhead and is significantly slower than explicit vectorization using the `axis` parameter.
**Action:** When calculating statistics across batch dimensions (like Hoyer sparsity), prefer modifying the function to accept an `axis` parameter and compute reductions directly along that axis, avoiding `vmap` and transpose overhead. Benchmarks showed a ~6.5x speedup for this specific case.
