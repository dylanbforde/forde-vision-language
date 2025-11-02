import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
import numpy as np # For converting JAX arrays to NumPy for plotting

def plot_brain_scan(smoothed_assignments_grid, step, epoch):
    """
    Generates and saves a heatmap of the smoothed neuron assignments (Brain Scan).
    """
    plt.figure(figsize=(8, 8))
    sns.heatmap(np.asarray(smoothed_assignments_grid), cmap='viridis', cbar=True, square=True)
    plt.title(f"Brain Scan - Smoothed Assignments (Epoch {epoch}, Step {step})")
    plt.xlabel("Grid Width")
    plt.ylabel("Grid Height")
    plt.savefig(f"brain_scan_epoch{epoch}_step{step}.png")
    plt.close()

def plot_feature_space(flattened_stats, assignments, step, epoch):
    """
    Generates and saves a scatter plot of neuron features, colored by cluster assignment.
    Assumes flattened_stats is (num_neurons, num_features).
    For visualization, we'll use the first two features if available, or PCA if more.
    For simplicity, let's assume we can plot the first two features directly.
    """
    if flattened_stats.shape[1] < 2:
        print(f"Warning: Cannot plot feature space for {flattened_stats.shape[1]} features. Need at least 2.")
        return

    plt.figure(figsize=(10, 8))
    # Convert JAX arrays to NumPy for plotting
    numpy_flattened_stats = np.asarray(flattened_stats)
    numpy_assignments = np.asarray(assignments)

    sns.scatterplot(
        x=numpy_flattened_stats[:, 0],
        y=numpy_flattened_stats[:, 1],
        hue=numpy_assignments,
        palette='tab10', # A good palette for categorical data
        legend='full',
        s=50 # Marker size
    )
    plt.title(f"Feature Space - Neuron Stats (Epoch {epoch}, Step {step})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig(f"feature_space_epoch{epoch}_step{step}.png")
    plt.close()
