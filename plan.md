1. **Optimize `smooth_assignments` and `smooth_assignments_3d` in `src/forde/smoothing.py` by replacing explicit Python loops + `jnp.stack` with `jax.vmap` over the cluster channel.**
   - In `smooth_assignments`, transpose the one-hot padded grid to put the cluster channel first (from `H, W, C` to `C, H, W`), apply `jax.vmap(convolve2d)`, and transpose back. This replaces the slow list comprehension and `jnp.stack`.
   - In `smooth_assignments_3d`, similarly transpose to `C, D, H, W`, apply `jax.vmap(convolve)`, and transpose back. This replaces the explicit Python for-loop and `jnp.stack`.
   - This prevents loop unrolling in JAX compilation, significantly reducing compilation time and improving execution speed.
2. **Add a performance learning journal entry to `.jules/bolt.md`.**
   - Document that replacing Python loops + `jnp.stack` with `jax.vmap` for independent convolutions over channels in JAX significantly reduces compilation time (approx ~3x) and slightly improves execution speed.
3. **Verify the optimization.**
   - Run the linting and formatting on the modified files.
   - Run the full test suite (`PYTHONPATH=. uv run pytest tests/`).
4. **Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.**
5. **Submit the PR with performance metrics.**
   - Title: "⚡ Bolt: [performance improvement] Vectorize convolutions in smoothing.py using jax.vmap"
   - Include What, Why, Impact, and Measurement in the PR description.
