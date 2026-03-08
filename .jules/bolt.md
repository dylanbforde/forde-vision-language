## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - Vectorized Reduction Ops
**Learning:** For reduction operations on large batch tensors in JAX (like Hoyer sparsity calculation), `jax.vmap` combined with transpose operations incurs significant overhead. Explicitly vectorizing the operation along a specific axis (e.g., `axis=0`) preserves memory layout and avoids vmap overhead, yielding significant speedups (observed ~3.6x in `calculate_neuron_stats` block).
**Action:** When applying operations across batch/sequence dimensions, prefer explicit axis arguments in JAX/numpy functions over `jax.vmap` with transpositions.
