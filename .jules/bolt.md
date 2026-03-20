## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2025-03-20 - [Performance] Use tree_map_with_path over flatten_dict
**Learning:** When updating specific nested parameters in a Flax/JAX `FrozenDict`, using `flax.traverse_util.flatten_dict` followed by a loop and `unflatten_dict` incurs significant overhead due to intermediate dictionary allocations.
**Action:** Always prefer `jax.tree_util.tree_map_with_path` for targeted parameter updates based on path keys. It traverses the PyTree in-place without flattening, yielding a ~2x speedup and preserving the `FrozenDict` structure natively.
