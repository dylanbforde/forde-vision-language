"""
Implements the "Actuation" stage of the FORDE model.

This is the final step of the slow loop, where the newly computed and smoothed
neuron assignments are written back into the model's state. This makes the
updated functional map available to the fast loop for the next N training steps.
"""

import jax.numpy as jnp
from flax.core import frozen_dict
from flax.training import train_state

def update_neuron_assignments(state: train_state.TrainState, new_assignments: jnp.ndarray) -> train_state.TrainState:
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
    # Unfreeze the parameters to allow modification
    # state.params is a FrozenDict, so we unfreeze it.
    mutable_params = state.params.unfreeze()

    # Update the relevant part of the parameters
    # Assuming 'neuron_assignments' is directly under 'params'
    # This part needs to be consistent with how 'neuron_assignments' is stored in the model.
    if 'neuron_assignments' in mutable_params:
        mutable_params['neuron_assignments'] = new_assignments
    else:
        # If 'neuron_assignments' is not directly in params, it might be nested.
        # For now, let's assume it's directly in params.
        # If it's not there, we add it.
        mutable_params['neuron_assignments'] = new_assignments

    # Re-freeze the parameters
    new_params = frozen_dict.freeze(mutable_params)

    # Create a new TrainState with the updated parameters
    return state.replace(params=new_params)
