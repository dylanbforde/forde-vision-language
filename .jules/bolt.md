## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - JAX MoE Eager Compilation OOM
**Learning:** In JAX/Flax MoE implementations, evaluating and stacking all expert outputs using `jnp.stack([expert(x) for expert in experts], axis=0)` forces XLA to eagerly compute and maintain a massive tensor `(num_experts, batch_size, seq_len, d_model)` during compilation. This can cause silent OOM errors and significant overhead, even if the result is only queried dynamically (e.g. via `top_k_indices`).
**Action:** When implementing independent MoE layer execution, always accumulate weighted outputs iteratively using `jnp.where` instead of stacking and gathering them. Avoid building large `(num_experts, ...)` intermediate arrays anywhere in the forward pass unless executing sparse operations like Scatter/Gather on pre-allocated buffers.
