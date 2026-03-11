## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2025-03-11 - Hoyer Sparsity Explicit Vectorization
**Learning:** Using `jax.vmap` combined with `transpose` for reduction operations along large dimensions (like batch size) causes significant JIT compilation overhead and large memory pre-allocations in JAX. Explicit vectorization by adding an `axis` parameter to the reduction function provides a ~4x speedup for calculating `hoyer_sparsity`.
**Action:** When performing reductions along batch/sequence dimensions on large tensors in JAX, prefer explicitly vectorizing the function logic via an `axis` parameter instead of using `jax.vmap` with `transpose`.
