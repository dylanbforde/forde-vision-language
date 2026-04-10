## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-24 - Vmap Independent Channel Operations
**Learning:** Replacing explicit Python loops + `jnp.stack` with `jax.vmap` for independent operations (like convolutions) across channels prevents loop unrolling in JAX and dramatically reduces JIT compilation time (~3x speedup observed).
**Action:** Prefer `jax.vmap` combined with axis transpositions when performing identical operations on multiple independent channels to keep XLA graphs compact.
