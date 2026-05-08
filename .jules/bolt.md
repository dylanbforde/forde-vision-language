## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-25 - Top-1 Routing Statistics Pattern
**Learning:** In JAX, computing Top-1 selection confidence and counts by using a boolean mask (`probs == probs.max(...)`) followed by `jnp.where(...).sum()` creates massive intermediate allocations and is highly inefficient. It also causes incorrect double-counting if there are exactly tied max values.
**Action:** Replace this pattern with `jnp.argmax` combined with `jnp.max` and `jnp.bincount(indices.reshape(-1), weights.reshape(-1))`. This avoids $O(B \times T \times E)$ allocations, resolves ties deterministically, and provides a ~3x performance boost on large sequence/expert dimensions. Ensure inputs to `bincount` are flattened to 1D.
