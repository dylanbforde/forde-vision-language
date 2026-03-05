## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2026-03-05 - Explicit Vectorization over vmap+transpose
**Learning:** For reduction operations on large batch tensors in JAX (like Hoyer sparsity calculation), explicitly vectorizing operations along a specific axis (e.g., `axis=0`) over `jax.vmap` combined with `transpose` yields significant speedups (observed ~3x). The vectorized approach preserves memory layout and avoids vmap overhead.
**Action:** When performing reductions across specific dimensions of large tensors, prefer implementing an `axis` parameter in the reduction function over mapping the function across transposed tensors using `jax.vmap`.
