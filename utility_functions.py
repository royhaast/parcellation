"""Functions for surface-based parcellation using freesurfer files."""
import numpy as np
from sklearn import metrics

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

