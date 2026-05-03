## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - JAX vmap vs List Comprehensions for Convolutions
**Learning:** Using Python list comprehensions and `jnp.stack` to apply operations (like 2D or 3D convolutions) independently across multiple channels causes massive XLA loop unrolling bloat. This can cause JIT compilation to fail entirely with an `algebraic_simplifier` loop error on many channels or drastically increase compilation time. Using `jax.vmap` is structurally cleaner for XLA, leading to much faster JIT compilation (~20-80x faster for many channels) and avoiding circular simplification limits.
**Action:** Always prefer `jax.vmap` with appropriate `in_axes` and `out_axes` instead of manual Python iterations and stacking for independent channel-wise operations in JAX.
