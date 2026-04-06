## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.
## 2025-02-18 - Avoid JAX stack loops on Expert outputs
**Learning:** Eagerly stacking a sequence of dynamically executed PyTree evaluations (like `jnp.stack([expert(x) for expert in experts])`) in JAX/Flax can lead to massive unroll blobs causing silent OOM and huge compilation times. Iterating and accumulating `jnp.where` elements avoids creating this intermediate tensor and vastly reduces HLO build time.
**Action:** When implementing mixtures or multi-model executions in JAX, accumulate using element-wise weights rather than building large multi-expert intermediate arrays and gathering them.
