## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.
## 2024-05-24 - Vmap Hoyer Sparsity Performance Memory overhead
**Learning:** Using `jax.vmap(hoyer_sparsity)(activations.T)` to calculate sparsity per neuron creates massive memory allocations and slows down compilation and execution time by roughly 4x on large arrays (256x1024x256). Explicitly calculating metrics using the `axis=0` argument is significantly faster and uses less memory.
**Action:** Use `axis` parameter inside operations like Hoyer's Sparsity to avoid vmapping over large input dimensions when reduction is straightforward.
