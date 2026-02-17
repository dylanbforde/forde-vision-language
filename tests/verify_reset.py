import jax
import jax.numpy as jnp
from src.forde.moe_slow_loop import moe_slow_loop_step
from src.forde.model import create_default_config


def main():
    print("Testing MoE Slow Loop Stats Buffer Reset...")

    # 1. Setup Mock Config
    config = create_default_config()
    config.num_layers = 1
    config.num_experts = 4
    config.d_model = 32

    # 2. Setup Mock Model Params
    model_params = {
        "params": {
            "layer_0": {
                "moe": {"router_linear": {"bias": jnp.zeros(config.num_experts)}}
            }
        }
    }

    # 3. Setup Mock Mutable Variables (Stats Buffer)
    stats_buffer = {
        "layer_0": {
            "moe": {
                "expert_usage": jnp.ones(config.num_experts, dtype=jnp.float32),
                "step_count": jnp.array(10, dtype=jnp.int32),
            }
        }
    }

    mutable_variables = {"stats_buffer": stats_buffer}

    print(f"Initial stats buffer: {mutable_variables}")

    # 4. Run Slow Loop
    key = jax.random.PRNGKey(42)
    epoch = 0
    step = 100

    updated_params, updated_mutable, diagnostics = moe_slow_loop_step(
        model_params=model_params,
        mutable_variables=mutable_variables,
        config=config,
        key=key,
        epoch=epoch,
        step=step,
    )

    # 5. Verify Reset
    print("\nVerifying reset...")

    new_stats_buffer = updated_mutable["stats_buffer"]

    # Just check that it's all zeros now
    # The structure should be preserved but values reset

    def check_zeros(x):
        return jnp.all(x == 0)

    # Fix: use jax.tree.map instead of deprecated jax.tree_map
    all_zeros = jax.tree_util.tree_reduce(
        lambda x, y: x and y, jax.tree.map(check_zeros, new_stats_buffer)
    )

    if all_zeros:
        print("SUCCESS: Stats buffer reset correctly.")
    else:
        print("FAILURE: Stats buffer NOT reset.")
        print(f"New buffer: {new_stats_buffer}")


if __name__ == "__main__":
    main()
