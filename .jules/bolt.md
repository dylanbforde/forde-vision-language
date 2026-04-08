## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - Hoyer Sparsity JAX Vectorization
**Learning:** For reduction operations on large batch tensors in JAX (like Hoyer sparsity calculation), explicitly vectorizing operations along a specific axis (e.g., `axis=0`) over `jax.vmap` combined with `transpose` reduces pre-allocation memory overhead that can lead to OOMs on large arrays, and yields ~3x speedups.
**Action:** When calculating statistics across batch dimensions for each feature/neuron, prefer passing an explicit `axis` argument to reductions instead of transposing and using `jax.vmap`. Avoid committing transient benchmark scripts.
