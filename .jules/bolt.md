## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-04-03 - [Vectorize instead of vmap for reductions on large tensors]
**Learning:** For reduction operations on large batch tensors in JAX (like Hoyer sparsity calculation), using `jax.vmap` combined with `transpose` causes massive pre-allocation memory overhead and drastically slower XLA compilation/execution times compared to explicitly vectorizing the operation along a specific axis (e.g., `axis=0`).
**Action:** When calculating row/column-wise metrics on large arrays (like activations or gradients), update functions to accept an `axis` argument and explicitly vectorize the inner operations (like `jnp.sum(..., axis=axis)`) rather than relying on `jax.vmap(...)`.
