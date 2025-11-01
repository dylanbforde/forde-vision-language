
import jax
import jax.numpy as jnp
from sklearn.mixture import GaussianMixture

def cluster_neurons_gmm(aggregated_stats, num_clusters, random_key):
    """
    Clusters neurons based on their aggregated statistics using Gaussian Mixture Model.

    Args:
        aggregated_stats: JAX array of shape (num_neurons, D) containing the aggregated statistics.
        num_clusters: The number of clusters (neuron types) to find.
        random_key: JAX PRNGKey for reproducibility.

    Returns:
        A tuple: (assignments, gmm_params)
        assignments: JAX array of shape (num_neurons,) with integer cluster assignments.
        gmm_params: A dictionary containing the GMM parameters (weights, means, covariances).
    """
    # Convert JAX array to NumPy for scikit-learn
    numpy_stats = jnp.asarray(aggregated_stats)

    # Use a fixed seed for scikit-learn GMM for reproducibility
    # JAX random_key is not directly compatible, so we convert it to an int.
    seed = int(jax.random.randint(random_key, (), 0, 2**31 - 1))

    gmm = GaussianMixture(n_components=num_clusters, random_state=seed)
    gmm.fit(numpy_stats)

    assignments = gmm.predict(numpy_stats)
    
    # Extract GMM parameters
    gmm_params = {
        'weights': jnp.asarray(gmm.weights_),
        'means': jnp.asarray(gmm.means_),
        'covariances': jnp.asarray(gmm.covariances_)
    }

    return jnp.asarray(assignments, dtype=jnp.int32), gmm_params

if __name__ == '__main__':
    # Example usage
    key = jax.random.PRNGKey(0)
    num_neurons = 100
    D = 5 # Number of statistics per neuron
    num_clusters = 3

    # Create some dummy aggregated stats with clear clusters
    stats_key, subkey = jax.random.split(key)
    dummy_stats = jax.random.normal(stats_key, (num_neurons, D)) * 0.5
    # Introduce 3 distinct clusters
    dummy_stats = dummy_stats.at[0:30, :].add(2.0) # Cluster 0
    dummy_stats = dummy_stats.at[30:70, :].add(-2.0) # Cluster 1
    # Remaining 30 neurons are Cluster 2 (around 0.0)

    assignments, gmm_params = cluster_neurons_gmm(dummy_stats, num_clusters, subkey)

    print(f"Aggregated Stats shape: {dummy_stats.shape}")
    print(f"Assignments shape: {assignments.shape}")
    print(f"Assignments (first 10): {assignments[:10]}")
    print(f"GMM Means shape: {gmm_params['means'].shape}")
    print("Neuron clustering successful.")

    # Verify assignments for dummy clusters
    print(f"\nAssignments for first 30 neurons (expected mostly one cluster): {assignments[0:30]}")
    print(f"Assignments for next 40 neurons (expected mostly another cluster): {assignments[30:70]}")
    print(f"Assignments for last 30 neurons (expected mostly a third cluster): {assignments[70:100]}")
