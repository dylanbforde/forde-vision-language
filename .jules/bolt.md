## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - Reduction Operation Optimization
**Learning:** For reduction operations on large batch tensors in JAX (like calculating Hoyer sparsity across a batch of activations), explicitly vectorizing operations along a specific axis (e.g., `axis=0`) preserves memory layout and avoids the compilation and runtime overhead of `jax.vmap` combined with `transpose`. This yields significant speedups.
**Action:** When performing reduction or statistical operations across a specific dimension of a multi-dimensional array, prefer adding an `axis` argument to the underlying function and passing it down to `jnp.sum`, `jnp.mean`, etc., instead of using `jax.vmap`.
