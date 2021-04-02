'''
Functions to compute and plot the univariate and multivariate versions of the
Giacomini-White (GW) test for Conditional Predictive Ability
'''

import numpy as np
import scipy
from scipy import stats
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

def GW(p_real, p_pred_1, p_pred_2, norm=1, version='univariate'):
    """Perform the one-sided GW test
    
    The test compares the Conditional Predictive Accuracy of two forecasts
    ``p_pred_1`` and ``p_pred_2``. The null H0 is that the CPA of errors ``p_pred_1``
    is higher (better) or equal to the errors of ``p_pred_2`` vs. the alternative H1
    that the CPA of ``p_pred_2`` is higher. Rejecting H0 means that the forecasts
    ``p_pred_2`` are significantly more accurate than forecasts ``p_pred_1``.
    (Note that this is an informal definition. For a formal one we refer 
    `here <https://epftoolbox.readthedocs.io/en/latest/modules/cite.html>`_)


    Parameters
    ----------
    p_real : numpy.ndarray
        Array of shape :math:`(n_\\mathrm{days}, n_\\mathrm{prices/day})` representing the real market
        prices
    p_pred_1 : TYPE
        Array of shape :math:`(n_\\mathrm{days}, n_\\mathrm{prices/day})` representing the first forecast
    p_pred_2 : TYPE
        Array of shape :math:`(n_\\mathrm{days}, n_\\mathrm{prices/day})` representing the second forecast
    norm : int, optional
        Norm used to compute the loss differential series. At the moment, this value must either
        be 1 (for the norm-1) or 2 (for the norm-2).
    version : str, optional
        Version of the test as defined in
        `here <https://epftoolbox.readthedocs.io/en/latest/modules/cite.html>`_. It can have two values:
        ``'univariate'`` or ``'multivariate'``
    Returns
    -------
    float, numpy.ndarray
        The p-value after performing the test. It is a float in the case of the multivariate test
        and a numpy array with a p-value per hour for the univariate test

    Example
    -------
    >>> from epftoolbox.evaluation import GW
    >>> from epftoolbox.data import read_data
    >>> import pandas as pd
    >>> 
    >>> # Generating forecasts of multiple models
    >>> 
    >>> # Download available forecast of the NP market available in the library repository
    >>> # These forecasts accompany the original paper
    >>> forecasts = pd.read_csv('https://raw.githubusercontent.com/jeslago/epftoolbox/master/' + 
    ...                       'forecasts/Forecasts_NP_DNN_LEAR_ensembles.csv', index_col=0)
    >>> 
    >>> # Deleting the real price field as it the actual real price and not a forecast
    >>> del forecasts['Real price']
    >>> 
    >>> # Transforming indices to datetime format
    >>> forecasts.index = pd.to_datetime(forecasts.index)
    >>> 
    >>> # Extracting the real prices from the market
    >>> _, df_test = read_data(path='.', dataset='NP', begin_test_date=forecasts.index[0], 
    ...                        end_test_date=forecasts.index[-1])
    Test datasets: 2016-12-27 00:00:00 - 2018-12-24 23:00:00
    >>> 
    >>> real_price = df_test.loc[:, ['Price']]
    >>> 
    >>> # Testing the univariate GW version on an ensemble of DNN models versus an ensemble
    >>> # of LEAR models
    >>> GW(p_real=real_price.values.reshape(-1, 24), 
    ...     p_pred_1=forecasts.loc[:, 'LEAR Ensemble'].values.reshape(-1, 24), 
    ...     p_pred_2=forecasts.loc[:, 'DNN Ensemble'].values.reshape(-1, 24), 
    ...     norm=1, version='univariate')
    array([1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
           1.00000000e+00, 1.00000000e+00, 1.03217562e-01, 2.63206239e-03,
           5.23325510e-03, 5.90845414e-04, 6.55116487e-03, 9.85034605e-03,
           3.34250412e-02, 1.80798591e-02, 2.74761848e-02, 3.19436776e-02,
           8.39512169e-04, 2.11907847e-01, 5.79718600e-02, 8.73956638e-03,
           4.30521699e-01, 2.67395381e-01, 6.33448562e-01, 1.99826993e-01])
    >>> 
    >>> # Testing the multivariate GW version
    >>> GW(p_real=real_price.values.reshape(-1, 24), 
    ...     p_pred_1=forecasts.loc[:, 'LEAR Ensemble'].values.reshape(-1, 24), 
    ...     p_pred_2=forecasts.loc[:, 'DNN Ensemble'].values.reshape(-1, 24), 
    ...     norm=1, version='multivariate')
    0.017598166936843906
    """
    # Checking that all time series have the same shape
    if p_real.shape != p_pred_1.shape or p_real.shape != p_pred_2.shape:
        raise ValueError('The three time series must have the same shape')

    # Ensuring that time series have shape (n_days, n_prices_day)
    if len(p_real.shape) == 1 or (len(p_real.shape) == 2 and p_real.shape[1] == 1):
        raise ValueError('The time series must have shape (n_days, n_prices_day')

    # Computing the errors of each forecast
    loss1 = p_real - p_pred_1
    loss2 = p_real - p_pred_2
    tau = 1 # Test is only implemented for a single-step forecasts
    if norm == 1:
        d = np.abs(loss1) - np.abs(loss2)
    else:
        d = loss1**2 - loss2**2
    TT = np.max(d.shape)

    # Conditional Predictive Ability test
    if version == 'univariate':
        GWstat = np.inf * np.ones((np.min(d.shape), ))
        for h in range(24):
            instruments = np.stack([np.ones_like(d[:-tau, h]), d[:-tau, h]])
            dh = d[tau:, h]
            T = TT - tau
            
            instruments = np.array(instruments, ndmin=2)

            reg = np.ones_like(instruments) * -999
            for jj in range(instruments.shape[0]):
                reg[jj, :] = instruments[jj, :] * dh
        
            if tau == 1:
                betas = np.linalg.lstsq(reg.T, np.ones(T), rcond=None)[0]
                err = np.ones((T, 1)) - np.dot(reg.T, betas)
                r2 = 1 - np.mean(err**2)
                GWstat[h] = T * r2
            else:
                raise NotImplementedError('Only one step forecasts are implemented')

    elif version == 'multivariate':
        d = d.mean(axis=1)
        instruments = np.stack([np.ones_like(d[:-tau]), d[:-tau]])
        d = d[tau:]
        T = TT - tau
        
        instruments = np.array(instruments, ndmin=2)

        reg = np.ones_like(instruments) * -999
        for jj in range(instruments.shape[0]):
            reg[jj, :] = instruments[jj, :] * d
    
        if tau == 1:
            betas = np.linalg.lstsq(reg.T, np.ones(T), rcond=None)[0]
            err = np.ones((T, 1)) - np.dot(reg.T, betas)
            r2 = 1 - np.mean(err**2)
            GWstat = T * r2
        else:
            raise NotImplementedError('Only one step forecasts are implemented')
    
    GWstat *= np.sign(np.mean(d, axis=0))
    
    q = reg.shape[0]
    pval = 1 - scipy.stats.chi2.cdf(GWstat, q)
    return pval

def plot_multivariate_GW_test(real_price, forecasts, norm=1, title='GW test', savefig=False, path=''):
    """Plotting the results of comparing forecasts using the multivariate GW test. 
    
    The resulting plot is a heat map in a chessboard shape. It represents the p-value
    of the null hypothesis of the forecast in the y-axis being significantly more
    accurate than the forecast in the x-axis. In other words, p-values close to 0
    represent cases where the forecast in the x-axis is significantly more accurate
    than the forecast in the y-axis.
    
    Parameters
    ----------
    real_price : pandas.DataFrame
        Dataframe that contains the real prices
    forecasts : TYPE
        Dataframe that contains the forecasts of different models. The column names are the 
        forecast/model names. The number of datapoints should equal the number of datapoints
        in ``real_price``.
    norm : int, optional
        Norm used to compute the loss differential series. At the moment, this value must either
        be 1 (for the norm-1) or 2 (for the norm-2).
    title : str, optional
        Title of the generated plot
    savefig : bool, optional
        Boolean that selects whether the figure should be saved in the current folder
    path : str, optional
        Path to save the figure. Only necessary when `savefig=True`
    
    Example
    -------
    >>> from epftoolbox.evaluation import GW, plot_multivariate_GW_test
    >>> from epftoolbox.data import read_data
    >>> import pandas as pd
    >>> 
    >>> # Generating forecasts of multiple models
    >>> 
    >>> # Download available forecast of the NP market available in the library repository
    >>> # These forecasts accompany the original paper
    >>> forecasts = pd.read_csv('https://raw.githubusercontent.com/jeslago/epftoolbox/master/' + 
    ...                       'forecasts/Forecasts_NP_DNN_LEAR_ensembles.csv', index_col=0)
    >>> 
    >>> # Deleting the real price field as it the actual real price and not a forecast
    >>> del forecasts['Real price']
    >>> 
    >>> # Transforming indices to datetime format
    >>> forecasts.index = pd.to_datetime(forecasts.index)
    >>> 
    >>> # Extracting the real prices from the market
    >>> _, df_test = read_data(path='.', dataset='NP', begin_test_date=forecasts.index[0], 
    ...                        end_test_date=forecasts.index[-1])
    Test datasets: 2016-12-27 00:00:00 - 2018-12-24 23:00:00
    >>> 
    >>> real_price = df_test.loc[:, ['Price']]
    >>> 
    >>> # Generating a plot to compare the models using the multivariate GW test
    >>> plot_multivariate_GW_test(real_price=real_price, forecasts=forecasts)
    
    """

    # Computing the multivariate GW test for each forecast pair
    p_values = pd.DataFrame(index=forecasts.columns, columns=forecasts.columns) 

    for model1 in forecasts.columns:
        for model2 in forecasts.columns:
            # For the diagonal elemnts representing comparing the same model we directly set a 
            # p-value of 1
            if model1 == model2:
                p_values.loc[model1, model2] = 1
            else:
                p_values.loc[model1, model2] = GW(p_real=real_price.values.reshape(-1, 24), 
                                                  p_pred_1=forecasts.loc[:, model1].values.reshape(-1, 24), 
                                                  p_pred_2=forecasts.loc[:, model2].values.reshape(-1, 24), 
                                                  norm=norm, version='multivariate')

    # Defining color map
    red = np.concatenate([np.linspace(0, 1, 50), np.linspace(1, 0.5, 50)[1:], [0]])
    green = np.concatenate([np.linspace(0.5, 1, 50), np.zeros(50)])
    blue = np.zeros(100)
    rgb_color_map = np.concatenate([red.reshape(-1, 1), green.reshape(-1, 1), 
                                    blue.reshape(-1, 1)], axis=1)
    rgb_color_map = mpl.colors.ListedColormap(rgb_color_map)

    # Generating figure
    plt.imshow(p_values.astype(float).values, cmap=rgb_color_map, vmin=0, vmax=0.1)
    plt.xticks(range(len(forecasts.columns)), forecasts.columns, rotation=90.)
    plt.yticks(range(len(forecasts.columns)), forecasts.columns)
    plt.plot(range(p_values.shape[0]), range(p_values.shape[0]), 'wx')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()

    if savefig:
        plt.savefig(title + '.png', dpi=300)
        plt.savefig(title + '.eps')

    plt.show()
