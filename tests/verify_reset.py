
import jax
import jax.numpy as jnp
from src.training.train import slow_loop_step
from src.forde.model import VisionConfig, TextConfig

def main():
    print("Testing Stats Buffer Reset...")
    
    # Mock mutable variables
    # Structure: stats_buffer -> data -> neuron_stats -> neuron_0 -> [stats]
    #            stats_buffer -> data -> step_count
    
    features = 10
    D = 5
    
    neuron_stats = {f'neuron_{i}': jnp.ones((D,), dtype=jnp.float32) for i in range(features)}
    stats_buffer_data = {
        'neuron_stats': neuron_stats,
        'step_count': jnp.array(10, dtype=jnp.int32)
    }
    
    mutable_variables = {
        'stats_buffer': {'data': stats_buffer_data},
        'state': {'assignments': jnp.zeros((features,), dtype=jnp.int32)}
    }
    
    # Mock configs
    vision_config = VisionConfig(16, 1, features, 1)
    text_config = TextConfig(100, 1, features, 1, 16)
    projection_dim = 2
    key = jax.random.PRNGKey(0)
    
    # Run slow loop
    updated_vars, _ = slow_loop_step(mutable_variables, vision_config, text_config, projection_dim, key, 0, 0)
    
    # Check if stats_buffer is reset
    new_stats_buffer = updated_vars['stats_buffer']
    new_step_count = new_stats_buffer['data']['step_count']
    new_neuron_0_stats = new_stats_buffer['data']['neuron_stats']['neuron_0']
    
    print(f"New step count: {new_step_count}")
    print(f"New neuron 0 stats: {new_neuron_0_stats}")
    
    if new_step_count == 0 and jnp.all(new_neuron_0_stats == 0):
        print("SUCCESS: Stats buffer reset correctly.")
    else:
        print("FAILURE: Stats buffer NOT reset.")

if __name__ == "__main__":
    main()
