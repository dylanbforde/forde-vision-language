## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - explicit vectorization over vmap+transpose
**Learning:** For reduction operations on large batch tensors in JAX (like Hoyer sparsity calculation), `jax.vmap(fn)(x.T)` creates massive pre-allocation memory overhead and causes silent OOMs on large arrays. Explicitly vectorizing the operations along a specific axis (e.g., `axis=0`) avoids this completely and provides ~4x speedups.
**Action:** When performing reductions across a specific dimension, prefer explicit vectorization by threading an `axis` argument through the mathematical operations rather than relying on `vmap` + matrix transpositions.
