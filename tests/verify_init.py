import jax
import jax.numpy as jnp
from src.forde.model import FORDEDecoderLM, create_default_config


def main():
    print("Testing FORDEDecoderLM Initialization...")

    key = jax.random.PRNGKey(0)
    config = create_default_config()

    # Override config for smaller test model
    config.d_model = 32
    config.num_layers = 1
    config.num_heads = 2
    config.head_dim = 16
    config.max_seq_len = 16
    config.vocab_size = 100
    config.num_experts = 4
    config.top_k_experts = 2
    config.expert_hidden_dim = 64

    print(f"Config: {config}")

    model = FORDEDecoderLM(config=config)

    batch_size = 2
    seq_len = 16
    input_ids = jax.random.randint(key, (batch_size, seq_len), 0, config.vocab_size)

    print("\nInitializing model...")
    variables = model.init(key, input_ids)

    print("\nVariable collections found:")
    for collection_name in variables.keys():
        print(f"- {collection_name}")

    # Check for expected collections
    if "params" in variables:
        print("SUCCESS: 'params' collection found.")
    else:
        print("FAILURE: 'params' collection NOT found.")

    if "stats_buffer" in variables:
        print("SUCCESS: 'stats_buffer' collection found (for MoE stats).")
        # Check if 'expert_usage' is being tracked implicitly or structurally
        print(f"Stats buffer keys: {variables['stats_buffer'].keys()}")
    else:
        print(
            "WARNING: 'stats_buffer' collection NOT found (expected for MoEStatefulLayer)."
        )

    print("\nRunning forward pass...")
    # We must mark 'stats_buffer' as mutable since MoEStatefulLayer updates it
    (logits, aux_loss), new_vars = model.apply(
        variables, input_ids, mutable=["stats_buffer"]
    )
    print(f"Logits shape: {logits.shape}")
    print(f"Aux loss: {aux_loss}")

    if "stats_buffer" in new_vars:
        print("SUCCESS: 'stats_buffer' updated during forward pass.")
    else:
        print("WARNING: 'stats_buffer' not in output variables.")

    if logits.shape == (batch_size, seq_len, config.vocab_size):
        print("SUCCESS: Forward pass output shape matches.")
    else:
        print(
            f"FAILURE: Output shape mismatch. Expected {(batch_size, seq_len, config.vocab_size)}"
        )


if __name__ == "__main__":
    main()
