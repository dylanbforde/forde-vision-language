"""
Implements the "Actuation" stage of the FORDE model.
"""

import jax.numpy as jnp
from flax.core import unfreeze

def update_neuron_assignments(mutable_variables: dict, new_assignments: jnp.ndarray) -> dict:
    """
    Updates the neuron assignments in the model's state by traversing the state pytree.
    """
    
    # Flax returns FrozenDicts, we need to unfreeze to modify.
    mutable_vars_dict = unfreeze(mutable_variables)
    
    assignment_offset = 0

    # This helper function will recursively search for the 'assignments' leaf
    # in the state pytree and update it with a slice from the new_assignments.
    def find_and_update(pytree_node):
        nonlocal assignment_offset
        if isinstance(pytree_node, dict):
            if 'assignments' in pytree_node and isinstance(pytree_node['assignments'], jnp.ndarray):
                num_layer_neurons = pytree_node['assignments'].shape[0]
                
                # Take the appropriate slice from the global new_assignments array
                assignment_chunk = new_assignments[assignment_offset : assignment_offset + num_layer_neurons]
                
                # Update the assignments for this layer
                pytree_node['assignments'] = assignment_chunk
                
                # Move the offset for the next layer
                assignment_offset += num_layer_neurons
            else:
                # Recursively traverse the dictionary
                for key in sorted(pytree_node.keys()): # Sort keys for deterministic order
                    find_and_update(pytree_node[key])

    # Start the traversal from the 'state' collection
    find_and_update(mutable_vars_dict['state'])
    
    return mutable_vars_dict