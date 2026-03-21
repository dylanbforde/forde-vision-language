## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - MoE Slow Loop PyTree Traversal Optimization
**Learning:** In JAX, manually unfreezing and flattening nested parameter dictionaries (like `FrozenDict`) via `flax.traverse_util.flatten_dict` during the slow loop introduces unnecessary intermediate allocations and latency when applying targeted updates.
**Action:** Replace `flatten_dict`/`unflatten_dict` cycles with `jax.tree_util.tree_map_with_path`. This directly applies conditional modifications without explicitly unfolding the entire parameter tree, resulting in faster and cleaner updates while natively handling `FrozenDict` types. When matching paths in the mapping function, use `hasattr(p, 'key')` to safely extract `DictKey` types.
