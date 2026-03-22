## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - Vectorizing Channel Operations
**Learning:** In JAX, using a Python loop to apply independent operations (like convolutions) across channels and combining them with `jnp.stack` is highly inefficient. Replacing this pattern with `jax.vmap` over the channel dimension (axis=-1) yields a ~2x speedup in execution time and halves the JIT compilation time, as it prevents XLA from unrolling the loop into a massive compute graph.
**Action:** Always prefer `jax.vmap` over Python loops with `jnp.stack` when applying identical operations independently across channels or other dimensions in JAX.
