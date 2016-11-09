""" Script for surface-based parcellation using freesurfer files.
Based on the Frontiers paper and codes by Thririon, B. (2014).
"""

import numpy as np
from time import time
from nibabel import freesurfer as fs
import nibabel.gifti.giftiio as gio
from nibabel import save
from scipy.sparse import coo_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from utility_functions import (
    generate_weights, reproducibility_rating, parameter_map)
from sklearn.cluster import AgglomerativeClustering

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

# =============================================================================
# Load .mgh and .shape.gii file to use its headers.
# =============================================================================
mgh = fs.mghformat.load('sample_data/01/rh.rs-fMRI.mgh')
gifti = gio.read('sample_data/rh.mask.shape.gii')

# =============================================================================
# Load data & perform data preprocessing for each parameter (e.g. R1, T2*...)
# =============================================================================

parameters = ['R1', 'T2', 'CBF', 'thickness']  # which parameters to include
n_components = len(parameters)
n_wcombs = 100
weightings = generate_weights(n_wcombs, parameters)

print 'loading surface and input data for parcellation...'
coords, faces = fs.io.read_geometry('sample_data/rh.inflated')
nverts, nfaces = coords.shape[0], faces.shape[0]

# Truncation
data = np.zeros((nverts, len(parameters)))
for i, string in enumerate(parameters):
    data_tmp = fs.io.read_morph_data('sample_data/01/rh.%s' % string)
    if string != 'thickness':
        data_tmp[data_tmp <= 0] = np.nan
        perc_min = np.nanpercentile(data_tmp, 3)
        perc_max = np.nanpercentile(data_tmp, 97)
        data_tmp_finite = data_tmp[np.isfinite(data_tmp)]
        data_tmp_finite[data_tmp_finite < perc_min] = perc_min
        data_tmp_finite[data_tmp_finite > perc_max] = perc_max
        data_tmp[np.isfinite(data_tmp)] = data_tmp_finite
        data[:, i] = data_tmp

    # Perform whitening (decorrelation) of the data (after imputation to
    # replace NaNs by mean of each feature)
    data = PCA(whiten=True).fit_transform(imp.fit_transform(data))

print '1. finding number of nearest neighbors...'
st = time()
num_nbrs = [0] * nverts

for i in range(nfaces):
    num_nbrs[faces[i, 0]] += 1
    num_nbrs[faces[i, 1]] += 1
    num_nbrs[faces[i, 2]] += 1

max_num_nbrs = max(num_nbrs)
print '     Done. Elapsed time (sec): ', time() - st

print '2. finding nearest neighbors...'
st = time()
nbrs = np.zeros((nverts, max_num_nbrs))

for i in range(nfaces):
    for j in range(3):
        vcur = faces[i, j]
        for k in range(3):
            if j != k:
                vnbr = faces[i, k]
                if vnbr in nbrs[vcur, :]:
                    pass
                else:
                    n_nbr = min(np.where(nbrs[vcur, :] == 0))
                    n_nbr = min(n_nbr)
                    nbrs[vcur, n_nbr] = vnbr

print '     Done. Elapsed time (sec): ', time() - st

print '3. computing connectivity matrix...'
st = time()

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

print '     Done. Elapsed time (sec): ', time() - st

k_range = [10, 50, 100, 200, 500]
range_weightings = [0, 19, 39, 59, 79, 99]
ars_file_string = 'results/rh_ward_ars_score.txt'
ars_txt = open(ars_file_string, 'w')
ami_file_string = 'results/rh_ward_ami_score.txt'
ami_txt = open(ami_file_string, 'w')
vm_file_string = 'results/rh_ward_vm_score.txt'
vm_txt = open(vm_file_string, 'w')

print '4. compute structural hierarchical (Ward) clustering...'

X = np.copy(data)
niter = 2
maps = []
for i in range(niter):
    bootstrap = (np.random.rand(X.shape[1]) * X.shape[1]).astype(int)
    X_ = X[:, bootstrap]
    maps.append(PCA(n_components=n_components).fit_transform(X_))

all_bic = np.zeros((len(range_weightings), len(k_range)))
all_crit = np.zeros((len(range_weightings), len(k_range)))

for w in range_weightings:
    ars_score = {}
    ami_score = {}
    vm_score = {}

    for (ik, k_) in enumerate(k_range):
        label_ = []
        print '   - %d clusters, weighting %d' % (k_, w)
        st = time()

        ward = AgglomerativeClustering(linkage='ward', n_clusters=k_,
                                       weights=weightings[w],
                                       connectivity=connectivity).fit(X)

        exec('labels_%d_clusters' % k_ + " = ward.labels_")
        labels = ward.labels_

        # =====================================================================
        # Compute reproducibility and BIC critera
        # =====================================================================
        for i in range(niter):
            ward = AgglomerativeClustering(linkage='ward', n_clusters=k_,
                                           weights=weightings[w],
                                           connectivity=connectivity).fit(maps[i])
            labels = ward.labels_
            label_.append(labels)

        ars_score[k_] = reproducibility_rating(label_, 'ars')
        ars_txt.write("Weighting %s, %s clusters: %f\n" %
                      (w, k_, ars_score[k_]))

        ami_score[k_] = reproducibility_rating(label_, 'ami')
        ami_txt.write("Weighting %s, %s clusters: %f\n" %
                      (w, k_, ami_score[k_]))

        vm_score[k_] = reproducibility_rating(label_, 'vm')
        vm_txt.write("Weighting %s, %s clusters: %f\n" % (w, k_, vm_score[k_]))

        ll, bic = 0, 0
        for component in range(n_components):
            ll1, mu_, sigma1_, sigma2_, bic_ = parameter_map(
                X[:, component], labels, null=False)
            bic += bic_.sum()
            ll += np.sum(ll1)
        all_crit[w, ik] = ll
        all_bic[w, ik] = bic

        # =====================================================================
        # Generate borders/contours using parcellation
        # =====================================================================
        borders = np.zeros((nverts, 1))
        c_nbrs = np.zeros((nverts, 6))
        for v in range(0, nverts):
            for v_nbr in range(0, num_nbrs[v]):
                c_nbrs[v, v_nbr] = labels[np.int(nbrs[v, v_nbr])]
            if len(np.unique(c_nbrs[v])) > 1:
                borders[v] = labels[v]

        # =====================================================================
        # Find neighbouring clusters for each cluster
        # =====================================================================
        nbrs_clusters = np.zeros((nverts, max_num_nbrs))
        for i in range(0, nverts):
            for j in range(0, 6):
                vcur = np.int(nbrs[i, j])
                nbrs_clusters[i, j] = labels[vcur]

        unique_nbrs_clusters = np.empty((k_, k_))
        unique_nbrs_clusters[:] = np.NAN
        for i in range(0, k_):
            temp = nbrs_clusters[labels == i]
            unique_nbrs_clusters[0:len(np.unique(temp)), i] = np.unique(temp)
        mask = np.all(np.isnan(unique_nbrs_clusters), axis=1)
        unique_nbrs_clusters = unique_nbrs_clusters[~mask]

        print 'Done. Saving...'

        ward_parcel_file_string = 'results/rh_ward_%d_clusters_weighting_%d' % (
            k_, w)
        borders_file_string = 'results/rh_ward_%d_clusters_weighting_%d_borders' % (
            k_, w)

        # save as mgh
        temp = mgh.get_data()
        temp[:, 0, 0] = np.squeeze(labels)
        out = fs.MGHImage(temp, mgh.affine, mgh.header)
        save(out, ward_parcel_file_string + '.mgh')

        temp[:, 0, 0] = np.squeeze(borders)
        out = fs.MGHImage(temp, mgh.affine, mgh.header)
        save(out, borders_file_string + '.mgh')

        # save as .shape.gii
        gifti.darrays[0].data = labels.astype('<f4')
        gio.write(gifti, ward_parcel_file_string + '.shape.gii')

        gifti.darrays[0].data = borders.astype('<f4')
        gio.write(gifti, borders_file_string + '.shape.gii')

        print 'All outputs saved.'
ars_txt.close()
ami_txt.close()
vm_txt.close()
