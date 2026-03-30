## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-24 - Vmap Transpose Memory Overhead
**Learning:** Using `jax.vmap` combined with matrix transpositions to apply reductions along a specific axis causes severe pre-allocation memory overhead and slows down XLA execution significantly. For example, using explicit `axis=0` in `hoyer_sparsity` is ~2.7x faster than transposing and running `jax.vmap` over the arrays for calculating neuron statistics. Additionally, using `jnp.linalg.norm(x, axis=axis)` is faster and cleaner than manual computation with squares and sums.
**Action:** Always prefer explicit axis reductions with vectorized JAX functions over `jax.vmap` with transpositions to prevent XLA pre-allocation bloat, especially on large matrices.
