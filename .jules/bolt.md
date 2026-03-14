## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2025-02-27 - Gradient Norm JIT Compilation Optimization
**Learning:** Using Python's `sum()` with generator expressions over `jax.tree.leaves()` (e.g., for gradient norm computation) is a major JAX performance anti-pattern. It forces XLA to unroll thousands of scalar additions into the HLO graph during JIT compilation, causing massive compile-time bloat. `optax.global_norm` suffers from a similar issue due to its use of `tree_reduce`.
**Action:** Replace Python tree-level reductions with array-level reductions by stacking or arraying the leaves before reducing: `jnp.sqrt(jnp.sum(jnp.array(jax.tree.leaves(jax.tree.map(lambda x: jnp.sum(jnp.square(x)), grads)))))`. This drastically reduces XLA graph size and JIT compilation time.
