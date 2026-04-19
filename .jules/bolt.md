## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - XLA JIT and MoE OOMs
**Learning:** In JAX/Flax Mixture of Experts (`MoELayer`), eagerly computing and stacking all expert outputs into a massive intermediate tensor using `jnp.stack([expert(x) for expert in experts], axis=0)` followed by advanced indexing causes silent OOMs and massive compile-time bloat on large XLA compilations. This requires allocating an array of size `(num_experts, batch, seq, d_model)`.
**Action:** Always iterate over experts individually, calculate their outputs, and accumulate the results using boolean masking (e.g., `jnp.where`). This drastically reduces memory allocations during compilation, making the model scalable to larger dimensions.
