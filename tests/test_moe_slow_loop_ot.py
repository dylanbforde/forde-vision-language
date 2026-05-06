import jax
import jax.numpy as jnp

from src.forde.model import create_default_config
from src.forde.moe_slow_loop import (
    collect_moe_stats_from_variables,
    moe_slow_loop_step,
)


def _stats_buffer(num_experts):
    return {
        "layer_0": {
            "moe": {
                "expert_usage": jnp.array([0.8, 0.6, 0.4, 0.2], dtype=jnp.float32),
                "expert_usage_sq": jnp.array(
                    [0.34, 0.20, 0.10, 0.04], dtype=jnp.float32
                ),
                "expert_top1_confidence_sum": jnp.array(
                    [1.6, 1.1, 0.5, 0.2], dtype=jnp.float32
                ),
                "expert_top1_count": jnp.array([2.0, 2.0, 1.0, 1.0]),
                "router_entropy": jnp.array(1.2, dtype=jnp.float32),
                "token_count": jnp.array(16, dtype=jnp.int32),
                "step_count": jnp.array(2, dtype=jnp.int32),
            }
        },
        "last_assignments": jnp.zeros(num_experts, dtype=jnp.int32),
    }


def test_collect_moe_stats_extracts_extended_routing_stats():
    stats, step_count = collect_moe_stats_from_variables(
        {"stats_buffer": _stats_buffer(4)}, num_layers=1, num_experts=4
    )

    assert step_count == 2
    assert stats["usage_mean"].shape == (1, 4)
    assert stats["usage_var"].shape == (1, 4)
    assert stats["selection_confidence"].shape == (1, 4)
    assert jnp.all(stats["usage_var"] >= 0.0)
    assert stats["token_count"][0] == 16


def test_moe_slow_loop_uses_ot_and_preserves_last_assignments():
    config = create_default_config()
    config.num_layers = 1
    config.num_experts = 4
    config.slow_loop_assignment_method = "ot"
    config.slow_loop_smoothing = False
    config.ot_n_iters = 5
    config.ot_refine_steps = 1

    model_params = {
        "layer_0": {
            "moe": {"router_linear": {"bias": jnp.zeros(config.num_experts)}}
        }
    }
    mutable_variables = {"stats_buffer": _stats_buffer(config.num_experts)}

    updated_params, updated_mutable, diagnostics = moe_slow_loop_step(
        model_params=model_params,
        mutable_variables=mutable_variables,
        config=config,
        key=jax.random.PRNGKey(0),
        epoch=0,
        step=10,
    )

    bias = updated_params["layer_0"]["moe"]["router_linear"]["bias"]
    assert diagnostics["assignment_method"] == "ot"
    assert diagnostics["role_probs"].shape == (config.num_experts, 3)
    assert "dustbin_mass" in diagnostics
    assert jnp.linalg.norm(bias) > 0.0
    assert updated_mutable["stats_buffer"]["last_assignments"].shape == (
        config.num_experts,
    )
    assert updated_mutable["stats_buffer"]["layer_0"]["moe"]["step_count"] == 0

