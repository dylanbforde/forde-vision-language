## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - JAX vmap over Explicit Loops for Large Channel Counts in Convolutions
**Learning:** Using explicit Python loops and `jnp.stack` to apply operations like convolutions across many channels (e.g., smoothing assignments across hundreds of experts/clusters) causes massive XLA loop unrolling bloat. This can lead to compilation failures with "algebraic simplifier loop" errors and excessively long JIT times.
**Action:** Always prefer `jax.vmap` (e.g., `jax.vmap(conv_single, in_axes=(-1, None), out_axes=-1)`) over list comprehensions combined with `jnp.stack` when applying independent operations across a large, dynamic axis to keep the compiled HLO graph compact and compilation fast.
