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
    """Provide assessment of a certain labeling given a ground truth.
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


def log_likelihood_fast(s0, s1, s2, mu, sigma1, sigma2):
    """Return the data loglikelihood (fast method specific to the block model).
    Parameters
    ==========
    s0, s1, s2: moments of order 0, 1 and 2 of the data
    mu: float, mean parameter
    sigma1: float, first level variance
    sigma2: float, seond level variance
    u: array of shape(n_samples), the associated index
    Returns
    =======
    ll: float, the log-likelihood of the data under the proposed model
    """
    log_det, quad = 0, 0
    for (s0_, s1_, s2_) in zip(s0, s1, s2):
        if s0_ > 0:
            log_det += np.log(s0_ * sigma2 ** 2 + sigma1 ** 2) +\
                (s0_ - 1) * np.log(sigma1 ** 2)
            prec = - 1. / (s0_ * sigma1 ** 2 + sigma1 ** 4 / sigma2 ** 2)
            quad += prec * (s0_ ** 2) * (s1_ / s0_ - mu) ** 2
            quad += (s2_ + mu * (s0_ * mu - 2 * s1_)) / sigma1 ** 2
    return - 0.5 * (log_det + quad + s0.sum() * np.log(2 * np.pi))


def em_inference_fast(y, u, mu=None, sigma1=None, sigma2=None, niter=30,
                      eps=1.e-3, mins=1.e-6, verbose=False):
    """Use an EM algorithm to compute sigma1, sigma2, mu -- fast version.
    Parameters
    ==========
    y: array of shape(n_samples), the input data
    u: array of shape(n_samples), the associated index
    sigma1: float, optional, initial value for sigma1
    sigma2: float, optional, initial value for sigma2
    niter: int, optional, max number of EM iterations
    eps: float, optional, convergence criteria on log-likelihood
    mins: float, optional, lower bound on variance values (numerical issues)
    verbose: bool, optional, verbosity mode
    Returns
    =======
    mu: float, estimatred mean
    sigma1: float, estimated first-level variance
    sigma2: float, second-level variance
    ll: float, the log-likelihood of the data
    Note
    ====
    use summary statistics for speed-up
    """
    if mu is None:
        mu = y.mean()
        learn_mu = True
    else:
        learn_mu = False
    s0 = np.array([np.sum(u == k) for k in np.unique(u)])
    s1 = np.array([np.sum(y[u == k]) for k in np.unique(u)])
    s2 = np.array([np.sum(y[u == k] ** 2) for k in np.unique(u)])

    # initialization
    if sigma1 is None:
        sigma1 = np.sqrt((s2 - (s1 ** 2) / s0).sum() / y.size)
    if sigma2 is None:
        sigma2 = np.std(s1 / s0)

    # EM iterations
    ll_old = - np.infty
    for j in range(niter):
        sigma1, sigma2 = np.maximum(sigma1, mins), np.maximum(sigma2, mins)
        ll_ = log_likelihood_fast(s0, s1, s2, mu, sigma1, sigma2)
        # ll_ = log_likelihood_(y, mu, sigma1, sigma2, u)
        # assert(np.abs(ll - ll_) < 1.e-8)
        if verbose:
            print j, ll_
        if ll_ < ll_old - eps and len(np.unique(u)) < u.size:
            raise ValueError('LL should not decrease during EM')
        if ll_ - ll_old < eps:
            break
        else:
            ll_old = ll_
        var = 1. / (s0 * 1. / sigma1 ** 2 + 1. / sigma2 ** 2)
        cbeta = var * (s1 / sigma1 ** 2 + mu / sigma2 ** 2)
        if learn_mu:
            mu = cbeta.mean()
        sigma2 = np.sqrt(((cbeta - mu) ** 2 + var).mean())
        sigma1 = np.sqrt(
            (s0 * var + s2 + cbeta * (s0 * cbeta - 2 * s1)).sum() / y.size)
    return mu, sigma1, sigma2, ll_


def reproducibility_rating(labels, score='ars', verbose=False):
    """Run multiple pairwise supervised ratings to obtain an average rating.
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


def parameter_map(X, labels, two_level=True, null=False, sigma1=None,
                  sigma2=None, niter=30, eps=1.e-3, verbose=False):
    """Return the likelihood of the model per label.
    Parameters
    ==========
    X: array of shape(n_voxels, n_subjects) the data to be parcelled
    label: array of shape (n_voxels) an index array describing the parcellation
    two_level: bool, optional
               Whether the model contains two levels or variance or not.
    null: bool, optional,
          whether parameters should be estimated under the null (mu=0) or not
    Returns
    =======
    ll: array_of shape(len(np.unique(label)))
    mu: array_of shape(len(np.unique(label)))
    sigma1: array_of shape(len(np.unique(label)))
    sigma2: array_of shape(len(np.unique(label)))
    bic: array_of shape(len(np.unique(label)))
    """
    n_labels = len(np.unique(labels))
    ll, bic = np.zeros(n_labels), np.zeros(n_labels)
    mu = np.zeros(n_labels)
    sigma1 = np.zeros(n_labels)
    sigma2 = np.zeros(n_labels)
    for k in np.unique(labels):
        y = X[labels == k].T.ravel()
        u = np.repeat(0, X[labels == k].shape[0])
        mu[k], sigma1[k], sigma2[k], ll[k] = em_inference_fast(y, u, mu[k],
                                                               sigma1[k],
                                                               sigma2[k],
                                                               niter, eps,
                                                               verbose=verbose)
        bic[k] = -2 * ll[k] + 3 * np.log(X[labels == k].size)
    return ll, mu, sigma1, sigma2, bic