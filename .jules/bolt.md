## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.
## 2026-03-19 - Hoyer Sparsity Optimization
**Learning:** For reduction operations on large batch tensors (like Hoyer sparsity calculation), replacing `jax.vmap` combined with `transpose` with explicit vectorization along an `axis` prevents massive pre-allocation memory overhead leading to OOMs and yields significant speedups (~4.7x speedup in benchmarks).
**Action:** Always prefer explicit vectorization using the `axis` parameter over `jax.vmap` combined with `transpose` for reduction operations.
