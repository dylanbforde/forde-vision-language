
import jax
import jax.numpy as jnp

def test_top_k_equivalence():
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (32, 100))
    k = 10

    # Original: argsort ascending, take last k
    indices_argsort = jnp.argsort(x, axis=-1)[..., -k:]
    # This gives smallest to largest among top k

    # New: top_k descending
    values_topk, indices_topk = jax.lax.top_k(x, k)
    # This gives largest to smallest

    # To compare, we must sort both sets of indices
    indices_argsort_sorted = jnp.sort(indices_argsort, axis=-1)
    indices_topk_sorted = jnp.sort(indices_topk, axis=-1)

    assert jnp.array_equal(indices_argsort_sorted, indices_topk_sorted)
    print("Indices match after sorting!")

if __name__ == "__main__":
    test_top_k_equivalence()
