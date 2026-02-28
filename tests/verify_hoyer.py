
import jax.numpy as jnp
from src.forde.sensing import hoyer_sparsity

def main():
    print("Testing Hoyer Sparsity...")
    
    # Test case 1: Zero vector
    zero_vec = jnp.zeros((10,))
    sparsity_zero = hoyer_sparsity(zero_vec)
    print(f"Sparsity of zero vector: {sparsity_zero}")
    
    # Test case 2: Uniform vector (should be 0 sparsity)
    uniform_vec = jnp.ones((10,))
    sparsity_uniform = hoyer_sparsity(uniform_vec)
    print(f"Sparsity of uniform vector: {sparsity_uniform}")
    
    # Test case 3: One-hot vector (should be 1 sparsity)
    one_hot = jnp.zeros((10,)).at[0].set(1.0)
    sparsity_one_hot = hoyer_sparsity(one_hot)
    print(f"Sparsity of one-hot vector: {sparsity_one_hot}")
    
    # Analysis
    # Formula: (sqrt(n) - L1/L2) / (sqrt(n) - 1)
    # For zero vector: L1=0, L2=0.
    # Implementation uses safe_l2 = 1.0 if l2=0.
    # So term = 0/1 = 0.
    # Result = sqrt(n) / (sqrt(n)-1).
    # This is > 1. This is technically incorrect/undefined.
    # Ideally it should probably be 0 (undefined sparsity usually treated as 0 or NaN).
    
    expected_zero_result = 0.0
    print(f"Expected result for zero vector with fixed logic: {expected_zero_result}")
    
    if jnp.allclose(sparsity_zero, expected_zero_result):
        print("CONFIRMED: Zero vector yields 0.0 sparsity.")
    else:
        print("Unexpected behavior.")

if __name__ == "__main__":
    main()
