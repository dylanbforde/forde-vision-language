## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - Avoiding jnp.stack in MoE Routing
**Learning:** Using `jnp.stack` to eagerly compute and stack all expert outputs into a single intermediate tensor `(num_experts, batch, seq, d_model)` inside MoE routing functions causes severe memory spikes and silent Out-Of-Memory (OOM) errors during large XLA compilations.
**Action:** Always iterate over experts individually, calculate their outputs, and dynamically accumulate the results using boolean masking (e.g., `jnp.where(is_selected, probs, 0.0)`) into the final output tensor. This prevents the compiler from allocating massive intermediate tensors while retaining mathematically equivalent logic.
