"""
Implements the "Actuation" stage of the FORDE model.

This is the final step of the slow loop, where the newly computed and smoothed
neuron assignments are written back into the model's state. This makes the
updated functional map available to the fast loop for the next N training steps.
"""

import jax.numpy as jnp
from flax.core import frozen_dict

def update_neuron_assignments(state: frozen_dict.FrozenDict, new_assignments: jnp.ndarray) -> frozen_dict.FrozenDict:
    """
    Updates the neuron assignments in the model's state.

    This function takes the current model state and a new set of assignments,
    unfreezes the state, updates the relevant part of the state tree, and then
    returns the new, frozen state.

    Args:
        state: The current Flax model state (a FrozenDict).
        new_assignments: A 1D array of the new integer assignments for each neuron.

    Returns:
        An updated Flax model state with the new neuron assignments.
    """
    # Unfreeze the state to allow modification
    state_dict = state.unfreeze()

    # NOTE: The exact path to the assignments will depend on the final model structure.
    # We assume the assignments are stored in a way that can be accessed and updated.
    # This is a placeholder for the actual path.
    # For example, it could be: state_dict['params']['StatefulLayer_0']['neuron_assignments']
    # We will need to adjust this based on the final implementation in train.py

    # For now, let's assume a simple structure where assignments are directly in the state
    # This will likely need to be updated to traverse the nested state dict.
    # A more robust solution will be implemented in train.py where the state structure is known.
    if 'neuron_assignments' in state_dict.get('params', {}):
        state_dict['params']['neuron_assignments'] = new_assignments
    else:
        # This is a fallback, the actual implementation will be more specific
        # and we will need to update this file once we have the full model structure
        # For now, we will create it if it does not exist
        if 'params' not in state_dict:
            state_dict['params'] = {}
        state_dict['params']['neuron_assignments'] = new_assignments

    # Re-freeze the state
    return frozen_dict.freeze(state_dict)
