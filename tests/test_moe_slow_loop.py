import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import FrozenDict
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from forde.moe_slow_loop import moe_slow_loop_step
from forde.model import LLMConfig

def test_moe_slow_loop_router_update():
    # Setup dummy params
    num_experts = 4
    d_model = 16
    config = LLMConfig(
        num_layers=2,
        num_experts=num_experts,
        d_model=d_model,
        expert_hidden_dim=32
    )

    # Create dummy params structure
        # Typically: params -> layer_0 -> moe -> router_linear -> bias
    params = {
        'params': {
                'layer_0': {
                'moe': {
                    'router_linear': {
                        'kernel': jnp.zeros((d_model, num_experts)),
                        'bias': jnp.zeros((num_experts,)),
                    }
                }
            },
                'layer_1': {
                'moe': {
                    'router_linear': {
                        'kernel': jnp.zeros((d_model, num_experts)),
                        'bias': jnp.zeros((num_experts,)),
                    }
                }
            }
        }
    }
    model_params = FrozenDict(params)

    # Create dummy mutable stats
    # expert_usage: (num_experts,)
    # Need to simulate accumulation over steps
    # Usage: [10, 50, 10, 30] -> Sum=100. Freq: [0.1, 0.5, 0.1, 0.3]
    expert_usage = jnp.array([10.0, 50.0, 10.0, 30.0])
    step_count = jnp.array(100)

    mutable_variables = FrozenDict({
        'stats_buffer': {
                'layer_0': {
                'moe': {
                    'expert_usage': expert_usage,
                    'step_count': step_count
                }
            },
                'layer_1': {
                'moe': {
                    'expert_usage': expert_usage,
                    'step_count': step_count
                }
            }
        }
    })

    key = jax.random.PRNGKey(42)

    # Run slow loop step
    # This currently uses flatten_dict
    updated_params, updated_vars, diagnostics = moe_slow_loop_step(
        model_params, mutable_variables, config, key, epoch=0, step=100
    )

    # Check that router biases were updated
    # Layers 0 and 1 should have updated biases

    bias_0 = updated_params['params']['layer_0']['moe']['router_linear']['bias']
    bias_1 = updated_params['params']['layer_1']['moe']['router_linear']['bias']

    # Check if updated (not all zeros)
    assert not jnp.allclose(bias_0, 0.0), "Bias 0 should be updated"
    assert not jnp.allclose(bias_1, 0.0), "Bias 1 should be updated"

    # Check logic: expert 1 (index 1) has high usage (0.5 > 0.25 uniform), should have negative bias adjustment
    # expert 0 has low usage (0.1 < 0.25), should have positive adjustment
    assert bias_0[1] < 0, f"Expert 1 bias should be negative, got {bias_0[1]}"
    assert bias_0[0] > 0, f"Expert 0 bias should be positive, got {bias_0[0]}"

    # Check stats buffer reset
    assert jnp.allclose(updated_vars['stats_buffer']['layer_0']['moe']['expert_usage'], 0.0)

if __name__ == "__main__":
    test_moe_slow_loop_router_update()
    print("Test passed!")
