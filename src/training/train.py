import jax
import jax.numpy as jnp
import optax
import torch
from torch.utils.data import DataLoader
from flax.training import train_state
from transformers import AutoTokenizer
from tqdm.auto import tqdm

# Assuming the script is run from the project root
from src.data.dataset import create_dataset, MAX_TEXT_LENGTH
from src.forde.model import FORDEModel, VisionConfig, TextConfig
from src.forde.sensing import calculate_neuron_stats
from src.forde.clustering import cluster_neurons_gmm as cluster_neurons
from src.forde.smoothing import assignments_to_grid, smooth_assignments
from src.forde.actuation import update_neuron_assignments

class TrainState(train_state.TrainState):
    # We can add more things to the state here later, like metrics
    pass

def create_train_state(model_cls, key, learning_rate, dummy_image, dummy_text, vision_config, text_config, projection_dim):
    """Creates initial TrainState and mutable variables."""
    model = model_cls(
        vision_config=vision_config,
        text_config=text_config,
        projection_dim=projection_dim
    )
    
    # Initialize the model parameters and all mutable variables (state, stats_buffer)
    variables = model.init(key, dummy_image, dummy_text)
    params = variables['params']
    mutable_variables = {k: v for k, v in variables.items() if k != 'params'}

    # Create an optimizer
    tx = optax.adam(learning_rate)

    # Create and return the training state and initial mutable variables
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx), mutable_variables

@jax.jit
def train_step(state, mutable_variables, batch):
    """Performs a single training step with contrastive loss and updates mutable variables."""
    def loss_fn(params, mutable_variables):
        # Apply the model to get image and text embeddings, logit_scale, and updated mutable variables
        (image_embed, text_embed, logit_scale), updated_mutable_variables = state.apply_fn(
            {'params': params, **mutable_variables},
            batch['image'],
            batch['input_ids'],
            mutable=['state', 'stats_buffer'] # Specify which collections are mutable
        )

        # Normalize embeddings to unit length
        image_embed = image_embed / jnp.linalg.norm(image_embed, axis=-1, keepdims=True)
        text_embed = text_embed / jnp.linalg.norm(text_embed, axis=-1, keepdims=True)

        # Compute similarity matrix (batch_size, batch_size)
        logits = jnp.matmul(image_embed, text_embed.T) * jnp.exp(logit_scale)

        # Create labels for cross-entropy loss (identity matrix for matching pairs)
        labels = jnp.arange(batch['image'].shape[0])

        # Calculate contrastive loss
        image_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        text_loss = optax.softmax_cross_entropy_with_integer_labels(logits.T, labels).mean()

        total_loss = (image_loss + text_loss) / 2
        return total_loss, updated_mutable_variables

    # Use has_aux=True to get updated_mutable_variables from loss_fn
    # argnums=0 means only params are differentiated
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True, argnums=0)
    (loss, updated_mutable_variables), grads = grad_fn(state.params, mutable_variables)
    state = state.apply_gradients(grads=grads)
    return state, mutable_variables, loss

def slow_loop_step(mutable_variables, vision_config, text_config, projection_dim, key):
    """Performs the FORDE slow loop: sense, cluster, smooth, actuate."""
    print("--- Running Slow Loop ---")
    
    # 1. Sense: Aggregate statistics from the buffer
    # The stats_buffer is a list of pytrees, where each leaf is a list of stats arrays
    # We need to aggregate them into a single pytree of aggregated stats.
    all_leaves = jax.tree.leaves(mutable_variables['stats_buffer'])
    if all(l.size == 0 for l in all_leaves):
        print("Stats buffer is empty, skipping slow loop.")
        return jnp.array([]) # Return an empty array for new_assignments

    def aggregate_and_resize(l):
        # Ensure it's at least 1D
        if l.ndim == 0:
            aggregated_val = jnp.atleast_1d(l)
        else:
            aggregated_val = jnp.atleast_1d(jnp.mean(l, axis=0))

        # Pad or truncate to projection_dim
        current_dim = aggregated_val.shape[0]
        if current_dim < projection_dim:
            padding_needed = projection_dim - current_dim
            return jnp.pad(aggregated_val, (0, padding_needed), 'constant')
        elif current_dim > projection_dim:
            return aggregated_val[:projection_dim]
        else:
            return aggregated_val

    aggregated_stats = jax.tree.map(aggregate_and_resize, mutable_variables['stats_buffer'])

    # Now, we need to flatten the aggregated stats into a (num_neurons, num_features) array
    # This depends on the structure of the model. For now, we assume a simple structure
    # where the stats are stored in a way that can be easily concatenated.
    # This is a placeholder and will need to be updated based on the final model structure.
    # For now, we will assume the stats are in a dictionary and we can concatenate them.
    flattened_stats = jnp.stack(list(jax.tree_util.tree_flatten(aggregated_stats)[0]), axis=0)

    # 2. Cluster: Run GMM on aggregated stats
    assignments, gmm = cluster_neurons(flattened_stats, num_clusters=3, random_key=key)
    print(f"Clustering complete. Neuron assignments shape: {assignments.shape}")

    # 3. Smooth: Apply convolutional smoothing
    num_neurons = flattened_stats.shape[0]
    grid_size = (int(jnp.sqrt(num_neurons)), -1)
    assignment_grid = assignments_to_grid(assignments, grid_size)
    
    smoothed_assignments_grid = smooth_assignments(assignment_grid, kernel_size=3, num_clusters=3)
    
    # Reshape back to 1D
    smoothed_assignments = smoothed_assignments_grid.flatten()
    print("Smoothing complete.")

    # 4. Actuate: Update neuron assignments in the model state
    print("Actuation complete.")
    return smoothed_assignments

def collate_fn(examples):
    """Custom collate function to handle dictionary-based batches."""
    batch = {
        'image': jnp.stack([ex['image'] for ex in examples]),
        'input_ids': jnp.stack([ex['input_ids'] for ex in examples]),
    }
    return batch

def main():
    # --- Configuration ---
    learning_rate = 1e-4
    num_epochs = 10 # Number of epochs instead of steps
    slow_loop_freq = 10 # Run slow loop every 10 steps
    batch_size = 4 
    features = 128 # Embedding dimension for transformers
    projection_dim = 64 # Dimension of the shared embedding space

    # Model configurations
    vision_config = VisionConfig(
        patch_size=16,
        num_layers=2,
        features=features,
        num_heads=4
    )
    text_config = TextConfig(
        vocab_size=30522, # BERT vocab size
        num_layers=2,
        features=features,
        num_heads=4,
        max_len=MAX_TEXT_LENGTH
    )

    # --- Initialization ---
    key = jax.random.PRNGKey(0)
    
    # 1. Initialize Model and Train State
    dummy_image = jnp.ones((batch_size, 224, 224, 3))
    dummy_text = jnp.ones((batch_size, MAX_TEXT_LENGTH), dtype=jnp.int32)
    
    state, mutable_variables = create_train_state(
        FORDEModel, key, learning_rate, dummy_image, dummy_text,
        vision_config, text_config, projection_dim
    )
    print(f"Train state and mutable variables created. Mutable variable keys: {mutable_variables.keys()}")

    # 2. Initialize Dataset and DataLoader
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = create_dataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)
    print(f"DataLoader num_workers: {dataloader.num_workers}")
    print("DataLoader created.")

    # --- Training Loop ---
    print("Starting training loop...")
    step = 0
    for epoch in range(num_epochs):
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            state, mutable_variables, loss = train_step(state, mutable_variables, batch)

            if step % 5 == 0: # Increased frequency for loss printing
                print(f"Step {step}, Epoch {epoch}, Loss: {loss}")

            # --- Slow Loop ---
            if (step + 1) % 10 == 0: # Increased frequency for slow loop
                key, slow_loop_key = jax.random.split(key)
                new_assignments = slow_loop_step(mutable_variables, vision_config, text_config, projection_dim, slow_loop_key)
                
                # The actuation step needs to be performed on the actual state.
                state = update_neuron_assignments(state, new_assignments)
                print("Model state updated with new assignments.")
            
            step += 1

if __name__ == "__main__":
    main()
