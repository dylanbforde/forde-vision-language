## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.
## 2024-05-23 - JAX MoE Intermediate Compilation Bloat
**Learning:** Eagerly stacking all expert outputs into a massive intermediate tensor using `jnp.stack([expert(x) for expert in experts])` causes extremely slow JIT compilation times and large memory footprint during XLA lowering. Iterating over experts and accumulating their outputs via boolean masks (`jnp.where`) dramatically reduces this overhead and executes slightly faster.
**Action:** Avoid large intermediate pre-allocations in MoE routing functions; prefer loop-based mathematical accumulation with masks.
