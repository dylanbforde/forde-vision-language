## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.
## 2024-05-24 - JAX MoE Output Accumulation Optimization
**Learning:** In JAX/Flax MoE implementations, eagerly computing and stacking all expert outputs into a massive intermediate tensor using `jnp.stack` prior to using gather operations causes massive VRAM bloat and can lead to silent Out-of-Memory (OOM) errors during large XLA compilations.
**Action:** Avoid `jnp.stack` for un-fused expert evaluations. Instead, loop over individual experts sequentially, evaluate their outputs, apply masking (`jnp.where`) based on top-$k$ assignments, and accumulate into a pre-allocated output tensor in-place. This significantly reduces memory allocations during tracing and execution.
