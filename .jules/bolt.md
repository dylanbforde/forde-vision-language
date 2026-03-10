## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-24 - JAX Vectorization Optimization (avoiding vmap)
**Learning:** Using `jax.vmap(func)(x.T)` to apply a reduction function per-neuron across a large batch dimension incurs massive pre-allocation memory overhead and slower execution compared to adding an `axis` argument to the function and calling `func(x, axis=0)`. For operations like Hoyer's sparsity, explicit vectorization with `axis=0` prevented OOM errors at scale (1024x512x2048) and yielded a ~1.72x speedup on a single batch (16x512x1024).
**Action:** When calculating statistics over large feature dimensions (e.g. neuron stats), prioritize modifying the base mathematical functions to support an explicit `axis` argument over using `jax.vmap` combined with transpose operations.
