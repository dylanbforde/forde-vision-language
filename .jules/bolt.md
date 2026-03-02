## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-24 - JAX Vectorization Optimization
**Learning:** For reduction operations on large batch tensors in JAX (like Hoyer sparsity calculation), explicitly vectorizing operations along a specific axis (e.g., `axis=0`) is significantly faster than using `jax.vmap` combined with `transpose`. The vectorized approach preserves memory layout and avoids vmap overhead, yielding a ~5x speedup in `calculate_neuron_stats` for large batches.
**Action:** Prefer explicit axis parameters over `jax.vmap` + `transpose` for simple reduction operations on contiguous dimensions.
