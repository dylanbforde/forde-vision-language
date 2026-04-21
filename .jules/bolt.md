## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - Vectorizing N-D Smoothing Convolutions
**Learning:** In JAX, explicitly iterating over channels (e.g., `num_clusters`) using Python list comprehensions and `jnp.stack` for independent channel operations like `convolve` or `convolve2d` causes severe loop unrolling. This bloats the XLA HLO graph and significantly increases JIT compilation times (e.g., from ~0.08s to ~0.23s).
**Action:** When applying identical operations across independent channels, hoist any `jnp.pad` operations out of the loop to pad the entire array at once, and replace the list comprehension + `jnp.stack` with `jax.vmap` applied along the channel axis. This keeps the XLA graph compact and compiles much faster.
