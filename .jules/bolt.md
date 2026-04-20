## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - Hoyer Sparsity Vmap Transpose Memory Overhead
**Learning:** For reduction operations on large batch tensors in JAX (like Hoyer sparsity calculation), using `jax.vmap` combined with `transpose` causes massive pre-allocation memory overhead leading to OOMs on large arrays. Explicitly vectorizing operations along a specific axis (e.g., `axis=0`) avoids this and yields ~3x to 5x speedups.
**Action:** When performing reductions like sum or mean across specific dimensions in custom JAX functions, always implement an `axis` argument instead of relying on `vmap` + `transpose` to handle multi-dimensional data.
