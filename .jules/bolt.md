## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - Hoyer Sparsity Vmap Transpose Optimization
**Learning:** In JAX, combining `vmap` with `transpose` to compute statistics over batch dimensions for large feature arrays causes severe performance overhead and memory pressure, especially on large inputs (e.g., 4x slower and potential OOMs).
**Action:** Replace `vmap(func)(array.T)` with explicit vectorization within the function by adding an `axis` argument and computing directly (e.g., `axis=0`).
