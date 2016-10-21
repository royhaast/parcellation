"""Functions for surface-based parcellation using freesurfer files."""
import numpy as np


def generate_weights(n_wcombs, parameters):
    """Generate combinations of weights.

    Every combination will be equal to 1.
    """
    weightings = np.zeros((n_wcombs, len(parameters)))
    for i in range(n_wcombs):
        weightings[i] = np.random.dirichlet(np.ones(len(parameters)), size=1)
    return weightings
