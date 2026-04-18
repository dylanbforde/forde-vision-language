## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - Avoid JAX JIT Stack OOM in MoE routing
**Learning:** In JAX/Flax Mixture of Experts, eagerly computing and stacking all expert outputs into a single intermediate tensor (`jnp.stack([expert(x) for expert in experts])`) can cause massive memory spikes and silent OOMs during XLA compilation and execution, especially at large sequence lengths.
**Action:** Replace `jnp.stack` and advanced indexing with an explicit loop over experts. Compute each expert's output lazily within the loop and accumulate results using boolean masking (`jnp.where(mask, probs, 0.0)`). This dramatically reduces intermediate allocations and speeds up execution (observed ~5x speedup).
