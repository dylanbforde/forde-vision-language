## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-24 - Avoiding memory bloat in top-k selection
**Learning:** In JAX/Flax, when calculating top-1 expert statistics (like total confidence sum and selection count), using a boolean mask like `mask = probs == probs.max(...)` followed by `jnp.where(mask, probs, 0.0).sum(...)` erroneously double-counts if there are duplicate max values (ties). Furthermore, it is slow and allocates large full-sized tensors.
**Action:** Replace multi-dimensional boolean masks and sums with `jnp.argmax` and `jnp.bincount` (e.g., `jnp.bincount(indices, weights=values)`). This enforces strict top-1 selection (picks the first occurrence of a tie), completely avoids large boolean mask allocations, and yields ~3x-8x performance speedups on TPU/GPU backends. Note that indices and weights must be flattened using `.reshape(-1)` for `jnp.bincount`.
