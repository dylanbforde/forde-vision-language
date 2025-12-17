import jax
import jax.numpy as jnp
import optax
import torch
from torch.utils.data import DataLoader
from flax.training import train_state
from flax.core import unfreeze
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

    # Ensure parameters are float32 for gradient computation
    params = jax.tree.map(lambda x: x.astype(jnp.float32) if jnp.issubdtype(x.dtype, jnp.integer) else x, params)
    mutable_variables = {k: v for k, v in variables.items() if k != 'params'}

    # Ensure mutable_variables are also float32 for gradient computation compatibility
    mutable_variables = jax.tree.map(lambda x: x.astype(jnp.float32) if jnp.issubdtype(x.dtype, jnp.integer) else x, mutable_variables)

    # Create an optimizer
    tx = optax.adam(learning_rate)

    # Create and return the training state and initial mutable variables
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx), mutable_variables

from flax.core import unfreeze

def train_step(state, mutable_variables, batch):
    """
    Performs a single training step, including gradient capture and cycling.
    This function is not JIT-compiled itself, but orchestrates JIT-compiled sub-functions.
    """
    
    # Define the loss function for both parameter and intermediate gradient calculation.
    def loss_fn(params, mutable_vars, batch):
        # Collections to be made mutable during the apply call.
        # We include 'grad_sinks' here because we might want to update them (though they are just sinks)
        # but crucially, we need to pass them in.
        mutable_collections = ['state', 'stats_buffer', 'grad_buffer', 'grad_sinks']
        
        (image_embed, text_embed, logit_scale), updated_vars = state.apply_fn(
            {'params': params, **mutable_vars},
            batch['image'],
            batch['input_ids'],
            mutable=mutable_collections
        )

        # Contrastive loss calculation
        image_embed = image_embed / jnp.linalg.norm(image_embed, axis=-1, keepdims=True)
        text_embed = text_embed / jnp.linalg.norm(text_embed, axis=-1, keepdims=True)
        logits = jnp.matmul(image_embed, text_embed.T) * jnp.exp(logit_scale)
        labels = jnp.arange(batch['image'].shape[0])
        image_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        text_loss = optax.softmax_cross_entropy_with_integer_labels(logits.T, labels).mean()
        total_loss = (image_loss + text_loss) / 2
        
        # Return loss and the sown activations as auxiliary data for grad calculation
        return total_loss, (updated_vars, updated_vars.get('activations_to_grad'))

    # JIT-compile the function that calculates gradients.
    # argnums=0: gradients w.r.t. params
    # argnums=1: gradients w.r.t. mutable_variables (specifically 'grad_sinks' inside it)
    # We need to restructure how we pass arguments to differentiate w.r.t. a subset of mutable_variables.
    # To make it cleaner, we can split mutable_variables.
    
    # Helper wrapper to handle the splitting of mutable variables
    def loss_wrapper(params, grad_sinks, other_mutable_vars, batch):
        # Recombine variables
        all_mutable_vars = {**other_mutable_vars, 'grad_sinks': grad_sinks}
        return loss_fn(params, all_mutable_vars, batch)

    grad_fn = jax.value_and_grad(loss_wrapper, argnums=(0, 1), has_aux=True)

    # Separate 'grad_sinks' from other mutable variables
    grad_sinks = mutable_variables.pop('grad_sinks')
    other_mutable_vars = mutable_variables

    # --- Execute the training step ---
    
    # 1. Calculate loss, parameter gradients, and sink gradients
    (loss, (updated_mutable_vars, _)), (param_grads, sink_grads) = grad_fn(state.params, grad_sinks, other_mutable_vars, batch)
    
    # The 'sink_grads' are the gradients of the loss w.r.t. the 'grad_sinks' variables.
    # Since 'grad_sinks' were added to activations, these are exactly dL/d(activation).
    intermediate_grads = sink_grads 

    # 2. Apply parameter gradients to the optimizer state
    state = state.apply_gradients(grads=param_grads)

    # 3. Cycle the intermediate gradients for the next step.
    # We take the intermediate gradients we just calculated and put them
    # into the 'grad_buffer' for the next iteration to use.
    # Note: updated_mutable_vars contains the updated state/stats_buffer/grad_buffer/grad_sinks
    mutable_variables = unfreeze(updated_mutable_vars)
    
    # We need to map the structure of intermediate_grads (which is {'sink': ...}) 
    # to the structure of grad_buffer (which is {'pre_activation_grad': ...})
    # The structure of the pytree should be identical down to the leaf names.
    
    def map_sink_to_buffer(sink_grad_leaf):
        return sink_grad_leaf

    # We assume the tree structure of 'grad_sinks' matches 'grad_buffer' 
    # except for the leaf name ('sink' vs 'pre_activation_grad').
    # However, JAX tree_map works on structure.
    # Let's manually traverse or assume the model structure ensures alignment.
    # A safer way is to traverse the 'grad_buffer' in mutable_variables and fill it.
    
    # Actually, the simplest way is to realize that 'intermediate_grads' is a dictionary 
    # mirroring the model structure. We can just iterate and assign.
    # But since we are using Flax variables, we can try to just swap the collection.
    # But the collection names are different in the variable definition ('grad_sinks' vs 'grad_buffer').
    # And the variable names are different ('sink' vs 'pre_activation_grad').
    
    # Let's use a recursive update that ignores the specific leaf key name
    def update_grad_buffer(buffer_tree, sink_tree):
        if isinstance(buffer_tree, dict) and isinstance(sink_tree, dict):
            # If we are at the leaf container level
            if 'pre_activation_grad' in buffer_tree and 'sink' in sink_tree:
                buffer_tree['pre_activation_grad'] = sink_tree['sink']
            else:
                for k in buffer_tree.keys():
                    if k in sink_tree:
                        update_grad_buffer(buffer_tree[k], sink_tree[k])
        return buffer_tree

    if intermediate_grads:
        mutable_variables['grad_buffer'] = update_grad_buffer(mutable_variables['grad_buffer'], intermediate_grads)
    
    return state, mutable_variables, loss

def slow_loop_step(mutable_variables, vision_config, text_config, projection_dim, key, epoch, step):
    """Performs the FORDE slow loop: sense, cluster, smooth, actuate."""
    print("--- Running Slow Loop ---")

    # Helper to recursively find all 'data' dictionaries for stats
    def get_all_stats_data(pytree):
        all_data = []
        if isinstance(pytree, dict):
            # Using 'neuron_stats' as a unique marker for the dictionary we want
            if 'neuron_stats' in pytree and 'step_count' in pytree:
                all_data.append(pytree)
            else:
                for k in sorted(pytree.keys()): # Sort keys for deterministic order
                    all_data.extend(get_all_stats_data(pytree[k]))
        return all_data

    # 1. Sense: Aggregate statistics from all StatefulLayers
    all_stats_data_list = get_all_stats_data(mutable_variables['stats_buffer'])

    if not all_stats_data_list:
        print("Could not find any stats data, skipping slow loop.")
        return mutable_variables, jnp.array([])

    # All step_counts should be the same, take the first one.
    step_count = all_stats_data_list[0]['step_count']
    if step_count == 0:
        print("Stats buffer is empty (step_count is 0), skipping slow loop.")
        return mutable_variables, jnp.array([])

    # Define the aggregation and resizing function for each neuron's accumulated stats
    def aggregate_and_resize_per_neuron(neuron_accumulated_stats):
        mean_stats = neuron_accumulated_stats / step_count
        current_dim = mean_stats.shape[0]
        if current_dim < projection_dim:
            padding_needed = projection_dim - current_dim
            return jnp.pad(mean_stats, (0, padding_needed), 'constant')
        elif current_dim > projection_dim:
            return mean_stats[:projection_dim]
        else:
            return mean_stats

    # Apply aggregation to all collected neuron stats
    all_aggregated_stats = []
    for stats_data in all_stats_data_list:
        aggregated_dict = jax.tree.map(aggregate_and_resize_per_neuron, stats_data['neuron_stats'])
        # .values() are the arrays for each neuron, sort keys to be safe
        all_aggregated_stats.extend([aggregated_dict[k] for k in sorted(aggregated_dict.keys())])

    flattened_stats = jnp.stack(all_aggregated_stats, axis=0)

    # 2. Cluster: Run GMM on aggregated stats
    assignments, gmm = cluster_neurons(flattened_stats, num_clusters=3, random_key=key)
    print(f"Clustering complete. Neuron assignments shape: {assignments.shape}")

    # 3. Smooth: Apply convolutional smoothing
    num_neurons = flattened_stats.shape[0]
    
    # Find factors for a rectangular grid that is as square as possible
    r = int(jnp.sqrt(num_neurons))
    while num_neurons % r != 0:
        r -= 1
    c = num_neurons // r
    grid_size = (r, c)

    assignment_grid = assignments_to_grid(assignments, grid_size)

    kernel_size = 3
    num_clusters = 3
    
    smoothed_assignments_grid = smooth_assignments(assignment_grid, kernel_size=kernel_size, num_clusters=num_clusters)
    
    # --- Diagnostics ---
    from src.utils.logging import plot_brain_scan, plot_feature_space
    plot_brain_scan(smoothed_assignments_grid, step, epoch)
    plot_feature_space(flattened_stats, assignments, step, epoch)
    # --- End Diagnostics ---

    # Reshape back to 1D
    smoothed_assignments = smoothed_assignments_grid.flatten()
    print("Smoothing complete.")

    # 4. Actuate: Update neuron assignments in the model state
    print("Actuation complete.")
    
    # Call the new update_neuron_assignments function
    mutable_variables = update_neuron_assignments(mutable_variables, smoothed_assignments)
    
    # 5. Reset Stats Buffer
    # We need to reset the stats buffer to zeros for the next cycle.
    # We can use the same structure as the input but filled with zeros.
    
    def reset_leaf(x):
        return jnp.zeros_like(x)
        
    reset_stats_buffer = jax.tree.map(reset_leaf, mutable_variables['stats_buffer'])
    mutable_variables['stats_buffer'] = reset_stats_buffer
    print("Stats buffer reset.")

    return mutable_variables, smoothed_assignments

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
    slow_loop_freq = 150 # Run slow loop every 150 steps
    batch_size = 32 
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
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4)
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
            if (step + 1) % slow_loop_freq == 0: # Run slow loop every slow_loop_freq steps
                key, slow_loop_key = jax.random.split(key)
                updated_mutable_variables, new_assignments = slow_loop_step(mutable_variables, vision_config, text_config, projection_dim, slow_loop_key, epoch, step)
                
                mutable_variables = updated_mutable_variables # Update mutable_variables in main scope

                # The actuation step is now handled within slow_loop_step
                # The line below is no longer needed if update_neuron_assignments is called inside slow_loop_step
                # state = update_neuron_assignments(state, new_assignments)
                print("Model state updated with new assignments.")
            
            step += 1

if __name__ == "__main__":
    main()
