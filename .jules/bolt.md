## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-24 - Vectorized Sparsity and MoE Output Allocation
**Learning:** Explicit axis-based reduction (e.g. `axis=0`) avoids the large overhead of `jax.vmap` combined with `.T` transposition for calculating statistical functions like `hoyer_sparsity`. This can lead to >4x execution speedups and lowered compile times. Furthermore, replacing eager `jnp.stack` allocations of expert outputs with a loop applying `jnp.where` on scaled routing weights dramatically decreases XLA compile time and execution memory.
**Action:** Favor `jnp.where` inside standard loops to conditionally add results for dynamic execution routes (like MoE), and always explicitly vectorize computations via `axis` arguments instead of using `jax.vmap` when memory and speed optimizations are critical.
