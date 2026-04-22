## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-25 - MoE Expert Stack OOM Optimization
**Learning:** Eagerly computing and stacking expert outputs in JAX/Flax MoE layers (`jnp.stack([expert(x) for expert in experts])`) scales poorly with the number of experts, creating massive intermediate tensors of shape `(num_experts, batch_size, seq_len, d_model)` that lead to silent OOMs and massive XLA compilation times.
**Action:** Always accumulate MoE expert outputs iteratively using boolean masking (e.g., `jnp.where`) and an output accumulator (`jnp.zeros_like(x)`). This shifts peak memory complexity from $O(E \cdot B \cdot S \cdot D)$ down to $O(B \cdot S \cdot D)$ and drastically reduces JIT compilation times (e.g., from ~3.7s to ~3.6s and allows much larger models to compile at all).
