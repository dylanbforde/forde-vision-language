## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2025-02-20 - JAX vmap Transpose Overhead
**Learning:** Using `jax.vmap` combined with matrix transposition (`x.T`) for reduction operations (like calculating Hoyer sparsity across a batch) creates massive pre-allocation memory overhead and trace bloat in JAX/XLA.
**Action:** Always prefer explicit parameterization of the reduction axis (e.g., passing `axis=0` to the reduction function) over transposing and vmapping. It yields substantial speedups (~3.5x in our benchmarks) and drastically reduces OOM risk on large batch arrays.
