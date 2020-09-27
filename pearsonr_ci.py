import numpy as np
from scipy import stats, special


def pearsonr_ci(x=None, y=None, alpha=0.05, n_test=None, r=None):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''
    if not n_test:
        r, p = stats.pearsonr(x, y)
    else:
        r = r
        ab = n_test/2 - 1
        p = 2*special.btdtr(ab, ab, 0.5*(1 - abs(np.float64(r))))
    r_z = np.arctanh(r)
    if not n_test:
        se = 1/np.sqrt(x.size-3)
    else:
        se = 1/np.sqrt(n_test-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi