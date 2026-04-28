## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.
## 2024-05-14 - [MoE Memory OOM via jnp.stack]
**Learning:** In JAX/Flax Mixture of Experts (`MoELayer`), eagerly computing and stacking all expert outputs into a massive intermediate tensor using `jnp.stack` (e.g., shape `(num_experts, batch, seq, d_model)`) causes huge memory spikes and silent XLA compilation OOMs.
**Action:** Instead, iterate over experts individually, evaluate them, mask via `jnp.where` on top-k routing, and accumulate the results continuously to avoid allocating the massive intermediate stack.
