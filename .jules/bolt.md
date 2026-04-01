## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - MoE Expert Aggregation Optimization
**Learning:** Eagerly computing and stacking all expert outputs into a single massive intermediate tensor (`jnp.stack`) for advanced indexing causes extremely slow XLA compilations and high memory overhead, creating a hidden performance bottleneck. Iterating over experts and accumulating their outputs conditionally using `jnp.where` provides a drastic speedup (~40x in some benchmark configurations) and reduces memory pressure by evaluating paths sequentially in XLA.
**Action:** When gathering from a Mixture of Experts layer, always iterate over the experts and conditionally accumulate outputs (e.g., via `jnp.where` or multiplication masks) rather than stacking all expert outputs together upfront. Avoid large `jnp.stack` calls on parallel branches in JAX when not necessary.
