
import jax
import jax.numpy as jnp
from src.forde.model import FORDEModel, VisionConfig, TextConfig

def main():
    key = jax.random.PRNGKey(0)
    batch_size = 2
    
    vision_config = VisionConfig(patch_size=16, num_layers=1, features=32, num_heads=2)
    text_config = TextConfig(vocab_size=100, num_layers=1, features=32, num_heads=2, max_len=16)
    
    model = FORDEModel(vision_config=vision_config, text_config=text_config, projection_dim=16)
    
    dummy_image = jnp.ones((batch_size, 224, 224, 3))
    dummy_text = jnp.ones((batch_size, 16), dtype=jnp.int32)
    
    print("Initializing model...")
    variables = model.init(key, dummy_image, dummy_text)
    
    print("\nVariable collections:")
    for collection_name in variables.keys():
        print(f"- {collection_name}")
        
    if 'grad_sinks' in variables:
        print("\nSUCCESS: 'grad_sinks' collection found.")
        # Check if 'grad_buffer' is also there
        if 'grad_buffer' in variables:
             print("SUCCESS: 'grad_buffer' collection found.")
        else:
             print("FAILURE: 'grad_buffer' collection NOT found.")
    else:
        print("\nFAILURE: 'grad_sinks' collection NOT found.")

if __name__ == "__main__":
    main()
