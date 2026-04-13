## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.
## 2024-05-23 - MoE Gather Optimization
**Learning:** In JAX/Flax Mixture of Experts (`MoELayer`), eagerly computing and stacking all expert outputs into a massive intermediate tensor using `jnp.stack` causes silent memory bloat and massive XLA compilation slowdowns.
**Action:** Always iterate over experts and accumulate the results via boolean masking (e.g., element-wise multiplication with routing weights) instead of computing a `(num_experts, batch, seq, d_model)` tensor. This drastically improves XLA compilation times (~1.7x faster) and avoids peak memory OOMs.
