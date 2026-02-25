
import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np

# Try importing from src
try:
    from src.forde.sparse_attention import TopKSelection, NativeSparseAttention
    from src.forde.moe import MoELayer
except ImportError:
    import sys
    sys.path.append(".")
    from src.forde.sparse_attention import TopKSelection, NativeSparseAttention
    from src.forde.moe import MoELayer

def test_topk_usage():
    key = jax.random.PRNGKey(0)
    batch_size = 2
    seq_len = 128
    d_model = 64
    k = 4

    # 1. Test TopKSelection
    print("Testing TopKSelection...")
    layer = TopKSelection(num_heads=2, head_dim=32, top_k=k)
    x = jax.random.normal(key, (batch_size, seq_len, d_model))
    variables = layer.init(key, x)
    output, indices = layer.apply(variables, x)

    assert output.shape == (batch_size, seq_len, d_model)
    assert indices.shape == (batch_size, k)
    print("TopKSelection passed.")

    # 2. Test NativeSparseAttention
    print("Testing NativeSparseAttention...")
    layer = NativeSparseAttention(
        num_heads=2, head_dim=32, d_model=d_model,
        window_size=32, compression_ratio=4, top_k_global=k,
        use_compressed=False, use_top_k=True
    )
    variables = layer.init(key, x)
    output = layer.apply(variables, x)

    assert output.shape == (batch_size, seq_len, d_model)
    print("NativeSparseAttention passed.")

    # 3. Test MoELayer
    print("Testing MoELayer...")
    num_experts = 8
    top_k_experts = 2
    layer = MoELayer(
        num_experts=num_experts, top_k=top_k_experts,
        expert_hidden_dim=128, d_model=d_model
    )
    variables = layer.init(key, x)
    output, aux_loss, router_probs = layer.apply(variables, x)

    assert output.shape == (batch_size, seq_len, d_model)
    # Check router_probs implies top_k selection happened
    # Verify aux_loss is scalar
    assert aux_loss.shape == ()
    print("MoELayer passed.")

if __name__ == "__main__":
    test_topk_usage()
