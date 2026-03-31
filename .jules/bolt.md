## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-03-31 - MoE Layer jnp.stack Intermediate Memory OOM
**Learning:** When computing Mixture of Experts outputs in JAX/Flax, using `jnp.stack([expert(x) for expert in experts])` forces XLA to pre-allocate a massive `(num_experts, batch_size, seq_len, d_model)` tensor. For large configurations (e.g., 32 experts, 1024 sequence length), this will silently kill the process during compilation due to OOM.
**Action:** Always iterate over experts using a standard Python `for` loop and accumulate the results element-wise (using `jnp.where` and probabilities). This allows XLA to optimize away the massive intermediate tensor while yielding the same results.
