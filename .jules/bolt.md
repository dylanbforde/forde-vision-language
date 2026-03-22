## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.
## 2024-05-24 - VMAP + Transpose Overhead in JAX
**Learning:** For reduction operations on large batch tensors in JAX (like Hoyer sparsity calculation), using `jax.vmap` combined with `transpose` causes massive pre-allocation memory overhead and is significantly slower than explicitly vectorizing the operation along a specific axis (e.g., `axis=0`). In benchmarks, `vmap` + `transpose` took ~33s while `axis=0` vectorization took ~8s (a ~4x speedup).
**Action:** Prefer explicit vectorization with axis arguments over `jax.vmap` + `transpose` for large tensor reductions.
