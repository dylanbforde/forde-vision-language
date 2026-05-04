## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - Hoyer Sparsity Vmap Transpose Optimization
**Learning:** Using `jax.vmap` combined with a matrix transpose (e.g. `jax.vmap(hoyer_sparsity)(activations.T)`) for reductions across a specific dimension creates massive memory overhead and slows down execution.
**Action:** Always modify reduction functions to take an explicit `axis` parameter and perform operations natively, avoiding `vmap` and transpose views when reducing along axes in JAX arrays.
