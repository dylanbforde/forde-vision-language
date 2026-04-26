## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-04-26 - [JAX Compiler Bloat from List Comprehensions]
**Learning:** Using Python list comprehensions and `jnp.stack` to apply independent operations (like convolutions) across channels in JAX causes XLA to unroll the loop, resulting in massive JIT compilation times and potential circular simplification loops (compilation bloat).
**Action:** Always prefer `jax.vmap` (e.g., `in_axes=(-1, None), out_axes=-1`) over explicit Python list comprehensions for applying operations across independent dimensions. To do this effectively, hoist any non-uniform operations (like `jnp.pad`) outside the mapped function so the entire N-dimensional array can be processed uniformly.
