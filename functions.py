"""Functions for surface-based parcellation using freesurfer files.
Based on the sci-kit learn package.
"""
import numpy as np
from sklearn.decomposition import PCA

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

###############################################################################
# Compute scores to evaluate performance (e.g. ARI, see Thirion [2014])

def reproducibility_selection(
    X, grp_mask, niter=2, method='AgglomerativeClustering', k_range=KRANGE, write_dir='/tmp',
    verbose=True):
    """ Returns a reproducibility metric on bootstraped models
    
    Parameters
    ----------
    X: array of shape (n_voxels, n_contrasts, n_subjects)
       the input data
    grp_mask: array of shape (image_shape),
              the non-zeros elements yield the spatial model
    niter: int, number of bootstrap samples estimated
    method: string, one of 'ward', 'kmeans', 'spectral'
    k_range: list of ints, 
             the possible number of parcels to be tested
    """
    n_voxels, n_contrasts, n_subjects = X.shape
    n_components = 4

    # Define a spatial model
    connectivity
        
    # concatenate the data spatially
    Xv = np.reshape(X, (n_voxels, n_contrasts * n_subjects))
    # pre-computed stuff
    ic, jc = connectivity.nonzero()
    sigma = np.sum((Xv[ic] - Xv[jc]) ** 2, 1).mean()
    
    maps = []
    for i in range(niter):
        bootstrap = (np.random.rand(Xv.shape[1]) * Xv.shape[1]).astype(int)
        X_ = Xv[:, bootstrap]
        if method == 'spectral':
            connectivity.data = np.exp( 
                - np.sum((X_[ic] - X_[jc]) ** 2, 1) / (2 * sigma))
            maps.append(spectral_embedding(connectivity,
                                           n_components=n_components,
                                           eigen_solver='arpack',
                                           random_state=None,
                                           eigen_tol=0.0, drop_first=False))
        else:
            maps.append(PCA(n_components=n_components).fit_transform(X_))
            
    ars_score = {}
    ami_score = {}
    vm_score = {}
    for (ik, k_) in enumerate(k_range):
        label_ = []
        for i in range(niter):
            bootstrap = (np.random.rand(Xv.shape[1]) * Xv.shape[1]).astype(int)
            if method == 'spectral':
                if k_ <= n_components:
                    for _ in range(10):
                        labels = discretize(maps[i][:, :k_])
                        if len(np.unique(labels)) == k_:
                            break
                else:
                    _, labels, _ = k_means(
                        maps[i], n_clusters=k_, n_init=1,
                        precompute_distances=False, max_iter=10)
            elif method == 'AgglomerativeClustering':
                    ward = AgglomerativeClustering(linkage='ward', n_clusters=k_, 
                                connectivity=connectivity).fit(maps[i])
                    labels = ward.labels_
            elif method in ['k-means', 'kmeans']:
                _, labels, _ = k_means(maps[i], n_clusters=k_, n_init=1,
                                       precompute_distances=False, max_iter=10)
            elif method == 'geometric':
                xyz = np.array(np.where(grp_mask)).T
                _, labels, _ = k_means(xyz, n_clusters=k_, n_init=1,
                                       precompute_distances=False, max_iter=10)
            label_.append(labels)
        ars_score[k_] = reproducibility_rating(label_, 'ars')
        ami_score[k_] = reproducibility_rating(label_, 'ami')
        vm_score[k_] = reproducibility_rating(label_, 'vm')
        if verbose:
            print 'k: ', k_, '  ari: ', ars_score[k_], 'ami: ',ami_score[k_],\
                ' vm: ', vm_score[k_]
    file = open(path.join(write_dir, 'ari_score_%s.pck' % method), 'w')
    pickle.dump(ars_score, file)
    file = open(path.join(write_dir, 'ami_score_%s.pck' % method), 'w')
    pickle.dump(ami_score, file)
    file = open(path.join(write_dir, 'vm_score_%s.pck' % method), 'w')
    pickle.dump(vm_score, file)
    return ars_score, ami_score, vm_score    


