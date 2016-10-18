"""Script for surface-based parcellation using freesurfer files.

Based on the Frontiers paper and codes by Thririon, B. (2014).
"""

import nibabel
import numpy as np
import time as time
from scipy.sparse import coo_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer

from functions import (
    unique_rows, generate_weights, agglomerativeclustering)

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

# =============================================================================
# Loading data & perform data preprocessing for each parameter (e.g. R1, T2*...)
# =============================================================================
parameters = ['R1', 'T2', 'CBF', 'thickness']  # which parameters to include
n_wcombs = 100
weightings = unique_rows(generate_weights(n_wcombs, parameters))

# Generate an array of different weightings for each parameter that sum up to 1
n_wcombs = 100
weightings = np.zeros((n_wcombs, len(parameters)))
for i in range(n_wcombs):
    weightings[i] = np.random.dirichlet(np.ones(len(parameters)), size=1)
weightings = unique_rows(weightings)

print 'loading surface and input data for parcellation...'
coords, faces = nibabel.freesurfer.io.read_geometry(
    'sample_data/rh.inflated')
nverts = len(coords)
nfaces = len(faces)

data = np.zeros((nverts, len(parameters)))
for j in range(0, len(parameters)):
    data_tmp = nibabel.freesurfer.io.read_morph_data(
        'sample_data/01/rh.%s' % parameters[j])
    if parameters[j] != 'thickness':
        data_tmp[data_tmp <= 0] = np.nan
        data_tmp[data_tmp < np.percentile(
            data_tmp, 3)] = np.percentile(data_tmp, 3)
        data_tmp[data_tmp > np.percentile(
            data_tmp, 97)] = np.percentile(data_tmp, 97)
    for i in range(0, nverts):
        data[i, j] = data_tmp[i]

# Perform whitening (decorrelation) of the data (after imputation to
# replace NaNs by mean of each feature)
    data = PCA(whiten=True).fit_transform(imp.fit_transform(data))

print '1. finding number of nearest neighbors...'
st = time.time()
num_nbrs = [0] * nverts

for i in range(0, nfaces):
    num_nbrs[faces[i, 0]] = num_nbrs[faces[i, 0]] + 1
    num_nbrs[faces[i, 1]] = num_nbrs[faces[i, 1]] + 1
    num_nbrs[faces[i, 2]] = num_nbrs[faces[i, 2]] + 1

max_num_nbrs = max(num_nbrs)
print '     Done. Elapsed time (sec): ', time.time() - st

print '2. finding nearest neighbors...'
st = time.time()
nbrs = np.zeros((nverts, max_num_nbrs))

for i in range(0, nfaces):
    for j in range(0, 3):
        vcur = faces[i, j]
        for k in range(0, 3):
            if j != k:
                vnbr = faces[i, k]
                if vnbr in nbrs[vcur, :]:
                    pass
                else:
                    n_nbr = min(np.where(nbrs[vcur, :] == 0))
                    n_nbr = min(n_nbr)
                    nbrs[vcur, n_nbr] = vnbr

print '     Done. Elapsed time (sec): ', time.time() - st

print '3. computing connectivity matrix...'
st = time.time()

num_non_zero_entries = np.sum(num_nbrs)
num_non_zero_entries = np.int64(num_non_zero_entries)
IndX = np.zeros(num_non_zero_entries, dtype=np.int64)
IndY = np.zeros(num_non_zero_entries, dtype=np.int64)

count = np.int(0)
for i in range(0, len(num_nbrs)):
    ThisSize = num_nbrs[i]
    IndX[range(0 + count, ThisSize + count)] = i
    IndY[range(0 + count, ThisSize + count)] = nbrs[i, range(0, ThisSize)]
    count = count + ThisSize

connectivity = coo_matrix(
    ([1] * num_non_zero_entries, (IndX, IndY)), shape=(nverts, nverts))

print '     Done. Elapsed time (sec): ', time.time() - st

# =============================================================================
# Generate vector of cluster numbers
# =============================================================================
# min_n_cluster=2
# max_n_cluster=50
# max_n_cluster2=100
# max_n_cluster3=1050
# range_n_clusters_tmp = np.arange(min_n_cluster,max_n_cluster,1).tolist()
# range_n_clusters_tmp2 = np.arange(max_n_cluster,max_n_cluster2,5).tolist()
# range_n_clusters_tmp3 = np.arange(max_n_cluster2,max_n_cluster3,50).tolist()
# range_n_clusters = range_n_clusters_tmp+range_n_clusters_tmp2+range_n_clusters_tmp3
# range_n_clusters.reverse()
#
# del min_n_cluster, max_n_cluster, max_n_cluster2, max_n_cluster3, range_n_clusters_tmp, range_n_clusters_tmp2, range_n_clusters_tmp3

range_n_clusters = [160]

print '4. compute structural hierarchical (Ward) clustering and silhouette scores...'

X = np.zeros((nverts, 3))
X = data

for c in range_n_clusters:
    print '   - %d clusters' % c
    st = time.time()

    # We should come up with a method to test different weightings for the different
    # features, probably within 'euclidean_distances' function
    #
    # e.g. something like:
    # distance(vertex A, connected vertex B) = sqrt(sum_across_features(beta*(feature 1 vertex A - feature 1 vertex B)^2))
    # beta are the weightings for each features
    #
    # We should rewrite the AgglomerativeClustering to accept a list of different weightings
    #
    ward = agglomerativeclustering(
        linkage='ward', n_clusters=c, connectivity=connectivity).fit(X)

#==============================================================================
# Get euclidean distance for each pair of vertices
#==============================================================================

    euclidean_distances = np.zeros((nverts, max_num_nbrs))
    k = 0
    for i in range(0, nverts):
        for j in range(0, num_nbrs[i]):
            euclidean_distances[IndX[k], j] = connectivity.data[k]
            k = k + 1
    exec('distances_%d_clusters' % c + " = euclidean_distances")

    exec('labels_%d_clusters' % c + " = ward.labels_")
    labels = ward.labels_

    ward_parcel_txt_string = 'results/rh_ward_%d_clusters_noCBF.txt' % c
    np.savetxt(ward_parcel_txt_string, labels % c, fmt='%1.1i')

    # =========================================================================
    # Generate borders/contours using parcellation
    # =========================================================================
    borders = np.zeros((nverts, 1))
    c_nbrs = np.zeros((nverts, 6))
    for v in range(0, nverts):
        for v_nbr in range(0, num_nbrs[v]):
            c_nbrs[v, v_nbr] = labels[np.int(nbrs[v, v_nbr])]
        if len(np.unique(c_nbrs[v])) > 1:
            borders[v] = labels[v]

    borders_txt_string = 'results/rh_ward_%d_clusters_borders_noCBF.txt' % c
    np.savetxt(borders_txt_string, borders, fmt='%1.1i')

    # =========================================================================
    # Find neighbouring clusters for each cluster
    # =========================================================================
    nbrs_clusters = np.zeros((nverts, max_num_nbrs))
    for i in range(0, nverts):
        for j in range(0, 6):
            vcur = np.int(nbrs[i, j])
            nbrs_clusters[i, j] = labels[vcur]

    unique_nbrs_clusters = np.empty((c, c))
    unique_nbrs_clusters[:] = np.NAN
    for i in range(0, c):
        unique_nbrs_clusters[0:len(np.unique(nbrs_clusters[np.where(
            labels == i)])), i] = np.unique(nbrs_clusters[np.where(labels == i)])
    mask = np.all(np.isnan(unique_nbrs_clusters), axis=1)
    unique_nbrs_clusters = unique_nbrs_clusters[~mask]

    # =========================================================================
    # Concatenate data+labels connected clusters
    # =========================================================================
    X_labeled = np.empty((nverts, 4))
    X_labeled[:, 0:3] = X[:, 0:3]
    X_labeled[:, 3] = labels
