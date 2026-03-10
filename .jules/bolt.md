## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-25 - Tree Map with Path Optimization
**Learning:** PyTree traversals and updates using `flax.traverse_util.flatten_dict` cycles have significant overhead, creating many intermediate structures. Replacing these cycles with `jax.tree.map_with_path` gives ~30% faster parameter updates for MoE slow loop steps when dynamically updating biases deep in the network.
**Action:** When updating nested model parameter dicts conditionally on path names in JAX, prefer `jax.tree.map_with_path` over flattening/unflattening PyTree dicts. Convert `path.key` from `DictKey` to strings to perform substring checks.