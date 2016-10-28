"""Functions for surface-based parcellation using freesurfer files."""
from os import path
import numpy as np
import pickle
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

def generate_weights(n_wcombs, parameters):
    """Generate combinations of weights.

    Every combination will be equal to 1.
    """
    weightings = np.zeros((n_wcombs, len(parameters)))
    for i in range(n_wcombs):
        weightings[i] = np.random.dirichlet(np.ones(len(parameters)), size=1)
    return weightings
    
def supervised_rating(true_label, pred_label, score='ars', verbose=False):
    """Provide assessment of a certain labeling given a ground truth

    Parameters
    ----------
    true_label: array of shape (n_samples), the true labels
    pred_label: array of shape (n_samples), the predicted labels
    score: string, on of 'ars', 'ami', 'vm'
    """
    ars = metrics.adjusted_rand_score(true_label, pred_label)
    ami = metrics.adjusted_mutual_info_score(true_label, pred_label)
    vm = metrics.v_measure_score(true_label, pred_label)
    if verbose:
        print 'Adjusted rand score:', ars
        print 'Adjusted MI', ami
        print 'Homogeneity', metrics.homogeneity_score(true_label, pred_label)
        print 'Completeness', metrics.completeness_score(true_label, pred_label)
        print 'V-measure', vm
    if score == 'ars':
        return ars
    elif score == 'vm':
        return vm
    else:
        return ami    
    
def reproducibility_selection(
    X, niter=2, method='AgglomerativeClustering', k_range=range_n_clusters, write_dir='/tmp',
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
    maps = []
    for i in range(niter):
        bootstrap = (np.random.rand(X.shape[1]) * X.shape[1]).astype(int)
        X_ = X[:, bootstrap]
        maps.append(PCA(n_components=n_components).fit_transform(X_))
            
    ars_score = {}
    ami_score = {}
    vm_score = {}
    for (ik, k_) in enumerate(k_range):
        label_ = []
        for i in range(niter):
            bootstrap = (np.random.rand(X.shape[1]) * X.shape[1]).astype(int)
            ward = AgglomerativeClustering(linkage='ward', n_clusters=k_, connectivity=connectivity).fit(maps[i])
            labels = ward.labels_
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

def reproducibility_rating(labels, score='ars', verbose=False):
    """ Run mutliple pairwise supervised ratings to obtain an average
    rating

    Parameters
    ----------
    labels: list of label vectors to be compared
    score: string, on of 'ars', 'ami', 'vm'
    verbose: bool, verbosity

    Returns
    -------
    av_score:  float, a scalar summarizing the reproducibility of pairs of maps
    """
    av_score = 0
    niter = len(labels) 
    for i in range(1, niter):
        for j in range(i):
            av_score += supervised_rating(labels[j], labels[i], score=score,
                                       verbose=verbose)
            av_score += supervised_rating(labels[i], labels[j], score=score,
                                       verbose=verbose)
    av_score /= (niter * (niter - 1))
    return av_score

