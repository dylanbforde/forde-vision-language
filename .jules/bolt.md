## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-24 - JAX MoE Computation Optimization
**Learning:** In JAX/Flax Mixture of Experts (`MoELayer`), eagerly computing and stacking all expert outputs into a massive intermediate tensor using `jnp.stack` causes silent memory issues and massive XLA compilation bloat. My benchmark showed a compilation+execution time decrease from ~16.7s to ~14.0s (and prevents crashing out on larger shapes during compilation) by replacing the stack with a loop that iterators over experts, accumulates results sequentially, and avoids allocating space for all experts simultaneously.
**Action:** Always avoid `jnp.stack` for dynamically computing all expert outputs in MoE routing layers. Instead, use a loop that accumulates outputs using masked weights directly into the target output tensor.
