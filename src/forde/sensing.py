
import jax
import jax.numpy as jnp

def hoyer_sparsity(x):
    """Calculates Hoyer's sparsity for a given vector or matrix.
    
    Hoyer's sparsity is defined as (sqrt(N) - (sum(|x_i|) / sqrt(sum(x_i^2)))) / (sqrt(N) - 1)
    where N is the number of elements.
    
    Args:
        x: A JAX array (vector or matrix). Sparsity is calculated along the last dimension.
    
    Returns:
        A scalar or vector representing the Hoyer's sparsity.
    """
    n = x.shape[-1]
    l1_norm = jnp.sum(jnp.abs(x), axis=-1)
    l2_norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=-1))
    
    # Avoid division by zero if l2_norm is zero
    safe_l2_norm = jnp.where(l2_norm == 0, 1.0, l2_norm)
    
    # Avoid division by zero if n is 1
    denominator = jnp.where(n == 1, 1.0, jnp.sqrt(n) - 1)
        
    sparsity = (jnp.sqrt(n) - (l1_norm / safe_l2_norm)) / denominator
    
    # If l2_norm is 0, the vector is all zeros. Sparsity is undefined/0.
    sparsity = jnp.where(l2_norm == 0, 0.0, sparsity)
    
    return jnp.where(n == 1, 0.0, sparsity)

def calculate_neuron_stats(activations, gradients):
    """
    Calculates a feature vector of statistics for each neuron.

    Args:
        activations: JAX array of shape (batch_size, seq_len, features) representing 
                     the output of the dense layer before activation.
        gradients: JAX array of shape (batch_size, seq_len, features) representing 
                   gradients w.r.t. the activations.

    Returns:
        JAX array of shape (features, D) where D is the number of statistics (5).
    """
    # Reshape to flatten batch and sequence dimensions
    num_features = activations.shape[-1]
    activations = activations.reshape(-1, num_features)
    gradients = gradients.reshape(-1, num_features)

    # Ensure inputs are float32 for consistency
    activations = activations.astype(jnp.float32)
    gradients = gradients.astype(jnp.float32)

    # Activation Statistics (across the batch dimension for each neuron)
    act_gini = jax.vmap(hoyer_sparsity)(activations.T) # Transpose to apply Hoyer's per neuron
    act_gdp = jnp.mean(jnp.abs(activations), axis=0) # Mean L1 norm across batch
    act_variance = jnp.var(activations, axis=0)

    # Gradient Statistics (across the batch dimension for each neuron)
    grad_gini = jax.vmap(hoyer_sparsity)(gradients.T)
    grad_gdp = jnp.mean(jnp.abs(gradients), axis=0) # Mean L1 norm across batch

    # Stack all statistics into a (features, D) array
    # D = 5: [grad_gini, grad_gdp, act_gini, act_gdp, act_variance]
    neuron_stats = jnp.stack([grad_gini, grad_gdp, act_gini, act_gdp, act_variance], axis=-1)

    return neuron_stats

if __name__ == '__main__':
    # Example usage
    batch_size = 8
    features = 128
    
    dummy_activations = jax.random.normal(jax.random.PRNGKey(0), (batch_size, features))
    dummy_gradients = jax.random.normal(jax.random.PRNGKey(1), (batch_size, features))

    stats = calculate_neuron_stats(dummy_activations, dummy_gradients)
    
    print(f"Dummy Activations shape: {dummy_activations.shape}")
    print(f"Dummy Gradients shape: {dummy_gradients.shape}")
    print(f"Calculated Neuron Stats shape: {stats.shape}")
    print("Neuron stats calculated successfully.")

    # Test with a sparse activation pattern
    sparse_activations = jnp.zeros((batch_size, features))
    sparse_activations = sparse_activations.at[0, 0].set(10.0) # One active neuron
    sparse_stats = calculate_neuron_stats(sparse_activations, dummy_gradients)
    print(f"\nSparse Activations Gini (first neuron): {sparse_stats[0, 2]}") # act_gini is 3rd stat
    # Expected: high sparsity for a single active neuron

    # Test with a dense activation pattern
    dense_activations = jnp.ones((batch_size, features))
    dense_stats = calculate_neuron_stats(dense_activations, dummy_gradients)
    print(f"Dense Activations Gini (first neuron): {dense_stats[0, 2]}")
    # Expected: low sparsity for all active neurons
