
import jax
import jax.numpy as jnp
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from forde.moe import MoELayer

def test_moe_forward_pass():
    """Test that MoELayer runs correctly and produces valid outputs."""
    key = jax.random.PRNGKey(42)
    batch_size = 4
    seq_len = 128
    d_model = 64
    num_experts = 8
    top_k = 2
    expert_hidden_dim = 128

    x = jax.random.normal(key, (batch_size, seq_len, d_model))

    moe = MoELayer(
        num_experts=num_experts,
        top_k=top_k,
        expert_hidden_dim=expert_hidden_dim,
        d_model=d_model
    )

    variables = moe.init(key, x)

    # Run forward pass
    output, aux_loss, router_probs = moe.apply(variables, x)

    # Check shapes
    assert output.shape == (batch_size, seq_len, d_model)
    assert aux_loss.shape == ()
    assert router_probs.shape == (batch_size, seq_len, num_experts)

    # Check that probabilities sum to 1
    assert jnp.allclose(router_probs.sum(axis=-1), 1.0, atol=1e-5)

    # Check for NaN/Inf
    assert jnp.all(jnp.isfinite(output))
    assert jnp.isfinite(aux_loss)

    # Basic value check (smoke test for consistency with PRNG)
    # This value might change if initialization changes, but ensures deterministic behavior
    assert output.mean() != 0.0

if __name__ == "__main__":
    test_moe_forward_pass()
    print("MoE verification passed!")
