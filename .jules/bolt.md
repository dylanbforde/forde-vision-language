## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-24 - JAX Convolution Vectorization
**Learning:** In JAX, explicitly vectorizing independent channel convolutions via `jax.vmap` combined with axis transpositions (e.g., `jnp.moveaxis`) avoids loop unrolling, keeps XLA graphs compact, and dramatically reduces JIT compilation time (~4x to 5x speedups observed) compared to explicit Python loops accumulating with `jnp.stack`.
**Action:** Prefer `jax.vmap` for applying independent operations across channels rather than Python loops and `jnp.stack`.
