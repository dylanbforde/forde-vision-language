## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.
## 2024-06-13 - [Tree Map Optimization]
**Learning:** In JAX, using `flax.traverse_util.flatten_dict` combined with `unfreeze` to update specific parameters deep within a PyTree is computationally slow due to the intermediate dictionary allocations.
**Action:** Use `jax.tree_util.tree_map_with_path` instead for ~2x faster traversals, and safely extract the string from path objects by checking `hasattr(p, 'key')` before accessing `p.key`. Use a stateful callable class instead of a nested function with `nonlocal` state counters to avoid scoping errors.
