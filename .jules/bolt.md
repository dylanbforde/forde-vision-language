## 2025-02-12 - JAX MoE Optimization: top_k vs argsort
**Learning:** `jax.lax.top_k` is ~10x faster than `jnp.argsort` for selecting top-k experts, even with small K=2 and N=16. `jnp.bincount` is ~8x faster than `jax.nn.one_hot` followed by `sum` for load balancing histograms.
**Action:** Always prefer `jax.lax.top_k` for top-k selection and `jnp.bincount` for histograms/counts in JAX unless specialized constraints exist.
