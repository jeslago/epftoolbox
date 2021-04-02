'''
Functions to compute and plot the univariate and multivariate versions of the Giacomini-White (GW) test for Conditional Predictive Ability
'''

import numpy as np
import scipy
from scipy import stats

def gwtest(loss1, loss2, tau=1, conditional=1):
    d = loss1 - loss2
    TT = np.max(d.shape)

    if conditional:
        instruments = np.stack([np.ones_like(d[:-tau]), d[:-tau]])
        d = d[tau:]
        T = TT - tau
    else:
        instruments = np.ones_like(d)
        T = TT
    
    instruments = np.array(instruments, ndmin=2)

    reg = np.ones_like(instruments) * -999
    for jj in range(instruments.shape[0]):
        reg[jj, :] = instruments[jj, :] * d
    
    if tau == 1:
        # print(reg.shape, T)
        # print(reg.T)
        betas = np.linalg.lstsq(reg.T, np.ones(T), rcond=None)[0]
        # print(np.dot(reg.T, betas).shape)
        err = np.ones((T, 1)) - np.dot(reg.T, betas)
        r2 = 1 - np.mean(err**2)
        GWstat = T * r2
    else:
        raise NotImplementedError
        zbar = np.mean(reg, -1)
        nlags = tau - 1
        # ...
    
    GWstat *= np.sign(np.mean(d))
    # pval = 1 - scipy.stats.norm.cdf(GWstat)
    # if np.isnan(pval) or pval > .1:
    #     pval = .1
    # return pval
    
    q = reg.shape[0]
    pval = 1 - scipy.stats.chi2.cdf(GWstat, q)
    # if np.isnan(pval) or pval > .1:
    #     pval = .1
    return pval
