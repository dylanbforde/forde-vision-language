## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - Memory Pre-allocation Overhead with jax.vmap
**Learning:** Using `jax.vmap` combined with a `transpose` operation for reductions on large batch tensors (e.g., computing statistics across the batch dimension) can cause massive memory pre-allocation overhead in XLA, sometimes leading to Out-Of-Memory (OOM) errors and dramatically slower execution.
**Action:** When performing reductions along a specific axis on large tensors, explicitly pass the `axis` parameter down to the underlying operations instead of using `jax.vmap` and transposing the array. This avoids the memory bloat and is significantly faster (~3-4x speedup observed).
