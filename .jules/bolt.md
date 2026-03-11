## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-24 - Gradient Norm Calculation Compile Bloat
**Learning:** When calculating the gradient norm for PyTrees with thousands of leaves (like in deep LLMs), both manual Python generator loops `sum(jnp.sum(x**2) for x in jax.tree.leaves(grads))` and standard APIs like `optax.global_norm` suffer from massive compile-time bloat (e.g. taking >170s to compile vs 5s) because they unroll operations for every single leaf into the XLA HLO graph. This also causes slightly slower execution times. Using `jax.flatten_util.ravel_pytree` to flatten all leaves into a single contiguous 1D array before computing the norm dramatically reduces compile time (~30x faster) and improves execution time (~2x faster).
**Action:** Replace `sum(jnp.sum(...)` loops and `optax.global_norm` with `ravel_pytree`-based L2 norm calculations when monitoring or clipping gradient norms of large models.
