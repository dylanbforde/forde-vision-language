## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-24 - Parameter Tree Traversal Optimization
**Learning:** For large parameter trees (like LLMs), `flax.traverse_util.flatten_dict` combined with loop-based updates and `unflatten_dict` is significantly slower (~1.8x) than `jax.tree_util.tree_map_with_path`. The overhead comes from reconstructing the nested dictionary structure.
**Action:** Use `tree_map_with_path` for selective parameter updates instead of flattening/unflattening. Ensure test fixtures use correct key names (e.g., `layer_0` vs `layers_0`) when mocking structure-dependent logic, as string prefix checks can fail silently.
