"""
Functions to compute and plot the univariate and multivariate versions of the Diebold-Mariano (DM) test.
"""

# Author: Jesus Lago

# License: AGPL-3.0 License

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


def DM(p_real, p_pred_1, p_pred_2, norm=1, version='univariate'):
    """Function that performs the one-sided DM test in the contex of electricity price forecasting
    
    The test compares whether there is a difference in predictive accuracy between two forecast 
    ``p_pred_1`` and ``p_pred_2``. Particularly, the one-sided DM test evaluates the null hypothesis H0 
    of the forecasting errors of  ``p_pred_2`` being larger (worse) than the forecasting
    errors ``p_pred_1`` vs the alternative hypothesis H1 of the errors of ``p_pred_2`` being smaller (better).
    Hence, rejecting H0 means that the forecast ``p_pred_2`` is significantly more accurate
    that forecast ``p_pred_1``. (Note that this is an informal definition. For a formal one we refer to 
    `here <https://epftoolbox.readthedocs.io/en/latest/modules/cite.html>`_)

    Two versions of the test are possible:

        1. A univariate version with as many independent tests performed as prices per day, i.e. 24
        tests in most day-ahead electricity markets.

        2. A multivariate with the test performed jointly for all hours using the multivariate 
        loss differential series (see this 
        `article <https://epftoolbox.readthedocs.io/en/latest/modules/cite.html>`_ for details.

    
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
        ``'univariate`` or ``'multivariate``      
    Returns
    -------
    float, numpy.ndarray
        The p-value after performing the test. It is a float in the case of the multivariate test
        and a numpy array with a p-value per hour for the univariate test

    Example
    -------
    >>> from epftoolbox.evaluation import DM
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
    >>> # Testing the univariate DM version on an ensemble of DNN models versus an ensemble
    >>> # of LEAR models
    >>> DM(p_real=real_price.values.reshape(-1, 24), 
    ...     p_pred_1=forecasts.loc[:, 'LEAR Ensemble'].values.reshape(-1, 24), 
    ...     p_pred_2=forecasts.loc[:, 'DNN Ensemble'].values.reshape(-1, 24), 
    ...     norm=1, version='univariate')
    array([9.99999944e-01, 9.97562415e-01, 8.10333949e-01, 8.85201928e-01,
           9.33505978e-01, 8.78116764e-01, 1.70135981e-02, 2.37961920e-04,
           5.52337353e-04, 6.07843340e-05, 1.51249750e-03, 1.70415008e-03,
           4.22319907e-03, 2.32808010e-03, 3.55958698e-03, 4.80663621e-03,
           1.64841032e-04, 4.55829140e-02, 5.86609688e-02, 1.98878375e-03,
           1.04045731e-01, 8.71203187e-02, 2.64266732e-01, 4.06676195e-02])
    >>> 
    >>> # Testing the multivariate DM version
    >>> DM(p_real=real_price.values.reshape(-1, 24), 
    ...     p_pred_1=forecasts.loc[:, 'LEAR Ensemble'].values.reshape(-1, 24), 
    ...     p_pred_2=forecasts.loc[:, 'DNN Ensemble'].values.reshape(-1, 24), 
    ...     norm=1, version='multivariate')
    0.003005725748326471
    """

    # Checking that all time series have the same shape
    if p_real.shape != p_pred_1.shape or p_real.shape != p_pred_2.shape:
        raise ValueError('The three time series must have the same shape')

    # Ensuring that time series have shape (n_days, n_prices_day)
    if len(p_real.shape) == 1 or (len(p_real.shape) == 2 and p_real.shape[1] == 1):
        raise ValueError('The time series must have shape (n_days, n_prices_day')

    # Computing the errors of each forecast
    errors_pred_1 = p_real - p_pred_1
    errors_pred_2 = p_real - p_pred_2

    # Computing the test statistic
    if version == 'univariate':

        # Computing the loss differential series for the univariate test
        if norm == 1:
            d = np.abs(errors_pred_1) - np.abs(errors_pred_2)
        if norm == 2:
            d = errors_pred_1**2 - errors_pred_2**2

        # Computing the loss differential size
        N = d.shape[0]

        # Computing the test statistic
        mean_d = np.mean(d, axis=0)
        var_d = np.var(d, ddof=0, axis=0)
        DM_stat = mean_d / np.sqrt((1 / N) * var_d)

    elif version == 'multivariate':

        # Computing the loss differential series for the multivariate test
        if norm == 1:
            d = np.mean(np.abs(errors_pred_1), axis=1) - np.mean(np.abs(errors_pred_2), axis=1)
        if norm == 2:
            d = np.mean(errors_pred_1**2, axis=1) - np.mean(errors_pred_2**2, axis=1)

        # Computing the loss differential size
        N = d.size

        # Computing the test statistic
        mean_d = np.mean(d)
        var_d = np.var(d, ddof=0)
        DM_stat = mean_d / np.sqrt((1 / N) * var_d)
        
    p_value = 1 - stats.norm.cdf(DM_stat)

    return p_value

def plot_multivariate_DM_test(real_price, forecasts, norm=1, title='DM test', savefig=False, path=''):
    """Plotting the results of comparing forecasts using the multivariate DM test. 
    
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
    >>> from epftoolbox.evaluation import DM, plot_multivariate_DM_test
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
    >>> # Generating a plot to compare the models using the multivariate DM test
    >>> plot_multivariate_DM_test(real_price=real_price, forecasts=forecasts)
    
    """

    # Computing the multivariate DM test for each forecast pair
    p_values = pd.DataFrame(index=forecasts.columns, columns=forecasts.columns) 

    for model1 in forecasts.columns:
        for model2 in forecasts.columns:
            # For the diagonal elemnts representing comparing the same model we directly set a 
            # p-value of 1
            if model1 == model2:
                p_values.loc[model1, model2] = 1
            else:
                p_values.loc[model1, model2] = DM(p_real=real_price.values.reshape(-1, 24), 
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
