## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - Hoyer Sparsity Optimization
**Learning:** For reduction operations on large batch tensors in JAX (like Hoyer sparsity calculation), using `jax.vmap` combined with `transpose` causes massive pre-allocation memory overhead and slow compilation times in XLA.
**Action:** Prefer explicitly vectorizing operations along a specific axis (e.g., `axis=0`) over `vmap` + `transpose` for large batch/sequence data to drastically speed up compilation (~6x speedup) and execution (~5x speedup) while avoiding OOMs.
