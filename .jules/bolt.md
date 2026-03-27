## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - MoE Expert Output Accumulation Optimization
**Learning:** Eagerly evaluating and stacking expert outputs in a list comprehension using `jnp.stack` creates massive intermediate memory overhead in JAX (e.g. allocating `num_experts * batch * seq * d_model` tensors). XLA compiler often fails or stalls when these buffers balloon up in size.
**Action:** Always prefer iterative accumulation (looping over experts and computing their outputs element-wise using `jnp.where` or direct multiplication with combined routing weights) in JAX MoE layers. This drastically reduces XLA compilation time and prevents memory overflow.
