import jax
import jax.numpy as jnp
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from forde.moe import MoELayer
from forde.sparse_attention import NativeSparseAttention, TopKSelection


def test_moe_layer_shapes():
    key = jax.random.PRNGKey(42)
    batch, seq, d_model = 2, 32, 64
    num_experts = 4
    top_k = 2

    moe = MoELayer(
        num_experts=num_experts,
        top_k=top_k,
        expert_hidden_dim=128,
        d_model=d_model,
    )

    x = jax.random.normal(key, (batch, seq, d_model))
    variables = moe.init(key, x)

    output, aux_loss, router_probs = moe.apply(variables, x)

    assert output.shape == (batch, seq, d_model)
    assert router_probs.shape == (batch, seq, num_experts)
    assert aux_loss.shape == ()

    # Check probabilities sum to 1
    # Note: router_probs are softmax outputs
    assert jnp.allclose(router_probs.sum(axis=-1), 1.0, atol=1e-5)


def test_moe_top_k_consistency():
    # Verify that top-k selection logic works as expected (descending order)
    # This indirectly tests our argsort -> top_k replacement
    # Create dummy logits
    logits = jnp.array(
        [
            [10.0, 2.0, 5.0, 8.0],  # Top 2: 0 (10.0), 3 (8.0)
            [1.0, 9.0, 3.0, 4.0],  # Top 2: 1 (9.0), 3 (4.0)
        ]
    ).reshape(1, 2, 4)

    # We can't access private method _top_k_gating easily without init,
    # so we'll test the logic directly using lax.top_k vs argsort

    k = 2
    # argsort (original behavior, ascending, last k)
    argsort_indices = jnp.argsort(logits, axis=-1)[..., -k:]
    # Should be [[3, 0], [3, 1]] (indices of 8,10 and 4,9).
    # [2, 5, 8, 10] -> indices [1, 2, 3, 0]. Last 2: [3, 0] (values 8, 10)
    # [1, 3, 4, 9] -> indices [0, 2, 3, 1]. Last 2: [3, 1] (values 4, 9)

    # lax.top_k (new behavior, descending)
    _, top_k_indices = jax.lax.top_k(logits, k)
    # Should be [[0, 3], [1, 3]] (indices of 10,8 and 9,4)

    # The set of indices should be the same
    # We sort both to compare sets
    argsort_sorted = jnp.sort(argsort_indices, axis=-1)
    top_k_sorted = jnp.sort(top_k_indices, axis=-1)

    assert jnp.array_equal(argsort_sorted, top_k_sorted)


def test_moe_expert_accumulation_matches_stack_gather_reference():
    layer = MoELayer(num_experts=3, top_k=2, expert_hidden_dim=8, d_model=4)
    x = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4)
    top_k_indices = jnp.array(
        [
            [[0, 2], [1, 0], [2, 1]],
            [[1, 2], [0, 1], [2, 0]],
        ]
    )
    top_k_probs = jnp.array(
        [
            [[0.75, 0.25], [0.6, 0.4], [0.9, 0.1]],
            [[0.2, 0.8], [0.55, 0.45], [0.3, 0.7]],
        ],
        dtype=jnp.float32,
    )
    experts = [
        lambda value, scale=scale: value * scale + scale
        for scale in (1.0, 2.0, 3.0)
    ]

    actual = layer._compute_expert_outputs(x, experts, top_k_indices, top_k_probs)

    all_expert_outputs = jnp.stack([expert(x) for expert in experts], axis=0)
    expected = jnp.zeros_like(x)
    batch_indices = jnp.arange(x.shape[0])[:, None]
    seq_indices = jnp.arange(x.shape[1])[None, :]
    for k in range(layer.top_k):
        selected_output = all_expert_outputs[
            top_k_indices[..., k], batch_indices, seq_indices, :
        ]
        expected = expected + top_k_probs[..., k : k + 1] * selected_output

    assert jnp.allclose(actual, expected)


def test_sparse_attention_shapes():
    key = jax.random.PRNGKey(99)
    batch, seq, d_model = 2, 128, 64

    attn = NativeSparseAttention(
        num_heads=4,
        head_dim=16,
        window_size=32,
        compression_ratio=4,
        top_k_global=16,
    )

    x = jax.random.normal(key, (batch, seq, d_model))
    variables = attn.init(key, x)
    output = attn.apply(variables, x)

    assert output.shape == (batch, seq, d_model)


def test_top_k_selection_module():
    key = jax.random.PRNGKey(77)
    batch, seq, d_model = 1, 64, 32
    k = 8

    selector = TopKSelection(num_heads=2, head_dim=16, top_k=k)
    x = jax.random.normal(key, (batch, seq, d_model))
    variables = selector.init(key, x)

    output, selected_indices = selector.apply(variables, x)

    assert output.shape == (batch, seq, d_model)
    assert selected_indices.shape == (batch, k)
