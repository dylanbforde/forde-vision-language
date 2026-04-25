## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.
## 2024-05-23 - MoE Routing OOM Optimization
**Learning:** Eagerly computing and stacking expert outputs in JAX MoE layers (`jnp.stack`) creates massive intermediate tensors `(num_experts, batch, seq, d_model)` that lead to OOM errors during large XLA compilations. Iterating over experts and accumulating using `jnp.where` completely bypasses this issue.
**Action:** Always prefer iterative accumulation via `jnp.where` masking over explicit `jnp.stack` and advanced indexing for mixture-of-experts or similar gating structures to reduce XLA compilation peak memory.
