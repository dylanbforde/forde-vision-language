## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-24 - JAX vmap + transpose overhead
**Learning:** Using `jax.vmap` combined with matrix transposition (`.T`) for reduction operations along an axis on large batched tensors (like Hoyer sparsity calculation) causes massive pre-allocation memory overhead and slowdowns. Benchmarking showed `vmap + transpose` took ~26.8s, while explicitly vectorizing by adding an `axis` parameter (e.g., `axis=0`) took ~6.6s on a 8192x2048 array (a ~4x speedup).
**Action:** Always prefer explicit parameterization of axes (like `axis=0`) for mathematical reduction functions in JAX rather than relying on `vmap` with transpositions, especially for large arrays, to avoid OOMs and improve speed.
