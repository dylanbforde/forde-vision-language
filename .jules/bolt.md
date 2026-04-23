## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-04-23 - Optimize MoE Expert Routing Accumulation to Prevent XLA OOMs and Slowdowns
**Learning:** In JAX/Flax MoE routing, aggressively computing and gathering all expert outputs via `jnp.stack` forces XLA to realize massive intermediate tensors during execution, leading to significant memory allocations and sometimes OOMs.
**Action:** Instead of eagerly computing and stacking all experts (`jnp.stack([expert(x) for expert in experts])`), iterate over experts individually, calculate their respective routing weights via boolean masking (e.g., `jnp.where`), and accumulate their outputs directly. This enables XLA to fuse the element-wise operations and drastically reduces peak VRAM and execution times (e.g. from 171s to 135s in benchmarks, with 1.5x faster compile times).
