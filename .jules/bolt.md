## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - Hoyer Sparsity Axis Vectorization Optimization
**Learning:** In JAX, for reduction operations on large batch tensors (like Hoyer sparsity calculation), explicitly vectorizing operations along a specific axis (e.g., `axis=0`) is much faster than using `jax.vmap` combined with `transpose`. The explicitly vectorized approach preserves memory layout and avoids vmap overhead, yielding a significant speedup (~5x in `calculate_neuron_stats` local tests).
**Action:** When performing reductions across a specific dimension, prefer rewriting the function to accept an `axis` argument over using `jax.vmap(func)(tensor.T)`.
