import jax
import jax.numpy as jnp
import time

def test_sum_vs_bincount_slow_loop(batch_size, seq_len, num_experts):
    key = jax.random.PRNGKey(42)
    router_probs = jax.random.uniform(key, (batch_size, seq_len, num_experts))
    eps = 1e-8

    # Method 1: where + sum
    @jax.jit
    def f_conf_sum(probs):
        max_probs_mask = probs == probs.max(axis=-1, keepdims=True)
        conf = jnp.where(max_probs_mask, probs, 0.0).sum(axis=(0, 1)) / (max_probs_mask.sum(axis=(0, 1)) + eps)
        return conf

    # Method 2: max + bincount
    @jax.jit
    def f_conf_bincount(probs):
        max_probs = jnp.max(probs, axis=-1).reshape(-1)
        max_indices = jnp.argmax(probs, axis=-1).reshape(-1)
        conf_sum = jnp.bincount(max_indices, weights=max_probs, length=num_experts)
        count_sum = jnp.bincount(max_indices, length=num_experts)
        return conf_sum / (count_sum + eps)


    f_conf_sum(router_probs).block_until_ready()
    f_conf_bincount(router_probs).block_until_ready()

    n_runs = 1000

    start = time.time()
    for _ in range(n_runs):
        f_conf_sum(router_probs).block_until_ready()
    t_conf_sum = time.time() - start

    start = time.time()
    for _ in range(n_runs):
        f_conf_bincount(router_probs).block_until_ready()
    t_conf_bincount = time.time() - start

    print(f"Batch={batch_size}, Seq={seq_len}, Experts={num_experts}")
    print(f"Conf Sum time: {t_conf_sum:.4f}s")
    print(f"Conf Bincount time: {t_conf_bincount:.4f}s")

if __name__ == "__main__":
    test_sum_vs_bincount_slow_loop(8, 1024, 8)
    test_sum_vs_bincount_slow_loop(16, 2048, 16)
    test_sum_vs_bincount_slow_loop(32, 4096, 32)
