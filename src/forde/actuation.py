"""
Implements the "Actuation" stage of the FORDE model.

This is the final step of the slow loop, where the newly computed and smoothed
neuron assignments are written back into the model's state. This makes the
updated functional map available to the fast loop for the next N training steps.
"""

import jax.numpy as jnp
from flax.core import frozen_dict

def update_neuron_assignments(mutable_variables: dict, new_assignments: jnp.ndarray) -> dict:
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
    # Get a mutable copy of mutable_variables
    # Assuming mutable_variables is a regular Python dictionary.
    mutable_vars_dict = mutable_variables.copy()

    # Update the relevant part of the mutable variables
    # Assuming 'neuron_assignments' is directly in mutable_variables
    mutable_vars_dict['neuron_assignments'] = new_assignments

    # Return the updated mutable variables (as a dict)
    return mutable_vars_dict
