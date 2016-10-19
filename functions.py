"""Functions for surface-based parcellation using freesurfer files.
Based on the sci-kit learn package.
"""
import numpy as np

def generate_weights(n_wcombs, parameters):
    weightings = np.zeros((n_wcombs, len(parameters)))
    for i in range(n_wcombs):
        weightings[i] = np.random.dirichlet(np.ones(len(parameters)),size=1)
    return weightings
    
###############################################################################
# Compute Ward distance

def ward_dist(moments_1, moments_2, coord_row, coord_col, inertia): 
    size_max = coord_row.shape[0]
    n_features = moments_2.shape[1]
    
    for i in range(size_max):
        row = coord_row[i]
        col = coord_col[i]
        n = (moments_1[row] * moments_1[col]) / (moments_1[row] + moments_1[col])
        pa = 0.
        for j in range(n_features):
            pa += (moments_2[row, j] / moments_1[row] - moments_2[col, j] / moments_1[col]) ** 2
        inertia[i] = pa * n
    return inertia
