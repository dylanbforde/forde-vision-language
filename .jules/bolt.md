## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-24 - Efficient MoE Output Gathering
**Learning:** In JAX/Flax Mixture of Experts, avoiding `jnp.stack` across all expert outputs followed by advanced indexing significantly reduces memory overhead and speeds up execution. Using boolean masks (`jnp.where`) in a loop over experts avoids large intermediate tensor allocations and is ~35% faster.
**Action:** When gathering outputs from multiple conditional branches/experts in JAX, prefer iterating over the branches and accumulating with boolean masks instead of stacking all possible outputs and indexing into them.
