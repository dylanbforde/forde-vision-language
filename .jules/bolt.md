## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-24 - Hoyer Sparsity Memory Bottleneck
**Learning:** For reduction operations on large batch tensors in JAX, using `jax.vmap` combined with transpose (e.g., `jax.vmap(func)(x.T)`) can cause massive pre-allocation memory overhead, leading to OOMs on large arrays.
**Action:** Always prefer explicitly vectorizing operations along a specific axis (e.g., using an `axis` parameter) over `vmap` + `transpose` to avoid intermediate representations and achieve significant speedups (e.g. 5x speedup and no OOM in `calculate_neuron_stats`).
