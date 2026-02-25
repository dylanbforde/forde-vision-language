## 2024-05-23 - MoE Routing Optimization
**Learning:** Significant speedups can be achieved in MoE routing by replacing `argsort` with `jax.lax.top_k` (~18x) and `one_hot().sum()` with `jnp.bincount` (~15x). However, these gains may be masked by inefficient expert execution loops in naive implementations.
**Action:** Always verify micro-benchmarks for component-level optimizations when end-to-end impact is limited by other bottlenecks. Ensure `uv.lock` is not accidentally modified during dependency resolution.

## 2024-05-23 - Native Sparse Attention Optimization
**Learning:** Conditional execution of heavy layers (like global attention) based on sequence length can yield significant speedups (~35%) for short sequences. However, this requires careful initialization with `max_seq_len` to ensure all parameters (even those in skipped branches) are created. Flax's lazy parameter initialization makes this tricky.
**Action:** When implementing conditional logic in Flax modules, always initialize the model with inputs that trigger ALL branches, or use `setup()` to define layers unconditionally. Update `train.py` to use `max_seq_len` for initialization.
