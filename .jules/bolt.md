## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2025-03-06 - Optax global_norm vs Manual Loop Compilation Overhead
**Learning:** Using a manual python loop with `jax.tree.leaves` for calculating gradient norms (e.g. `sum(jnp.sum(x**2) for x in jax.tree.leaves(grads))`) causes JAX/XLA to trace and unroll every single leaf into the HLO graph. This severely bloats compile time, especially for models with many parameter leaves like Transformers with MoE. `optax.global_norm` provides a native implementation that is far more compilation-friendly.
**Action:** Always prefer `optax.global_norm(grads)` over manual tree leaf reductions when calculating global norms to minimize JIT compilation time.
