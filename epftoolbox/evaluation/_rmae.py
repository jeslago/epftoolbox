"""
Function that implements the relative mean absolute error (rMAE) metric.
"""

# Author: Jesus Lago

# License: AGPL-3.0 License

import numpy as np
from epftoolbox.evaluation._ancillary_functions import _process_inputs_for_metrics, naive_forecast, _transform_input_prices_for_naive_forecast
from epftoolbox.evaluation import MAE


def rMAE(p_real, p_pred, m=None, freq='1h'):

    """Function that computes the relative mean absolute error (rMAE) between two forecasts:
    
    .. math:: 
        \\mathrm{rMAE}_\\mathrm{m} = \\frac{1}{N}\\sum_{i=1}^N 
                         \\frac{\\bigl|p_\\mathrm{real}[i]−p_\\mathrm{pred}[i]\\bigr|}
                         {\\mathrm{MAE}(p_\\mathrm{real}, p_\\mathrm{naive})}.
    
    The numerator is the :class:`MAE` of a naive forecast ``p_naive`` that is built using the
    dataset ``p_real`` and the :class:`naive_forecast` function with a seasonality index ``m``.

    If the datasets provided are numpy.ndarray objects, the function requires a ``freq`` argument specifying
    the data frequency. The ``freq`` argument must take one of the following four values ``'1h'`` for 1 hour,
    ``'30T'`` for 30 minutes, ``'15T'`` for 15 minutes, or ``'5T'`` for 5 minutes,  (these are the 
    four standard values in day-ahead electricity markets). 
    
    Also, if the datasets provided are numpy.ndarray objects, ``m`` has to be ``'D'`` or ``'W'``, i.e. the 
    :class:`naive_forecast` cannot be the standard in electricity price forecasting because the input
    data does not have associated a day of the week.
    
    ``p_real``, ``p_pred``, and  `p_real_in`` can either be of shape 
    :math:`(n_\\mathrm{days}, n_\\mathrm{prices/day})`,
    :math:`(n_\\mathrm{prices}, 1)`, or :math:`(n_\\mathrm{prices}, )` where
    :math:`n_\\mathrm{prices} = n_\\mathrm{days} \\cdot n_\\mathrm{prices/day}`


    Parameters
    ----------
    p_real : numpy.ndarray, pandas.DataFrame
        Array/dataframe containing the real prices. 
    p_pred : numpy.ndarray, pandas.DataFrame
        Array/dataframe containing the predicted prices. 
    m : int, optional
        Index that specifies the seasonality in the :class:`naive_forecast` used to compute the normalizing
        insample :class:`MAE`. It can be be ``'D'`` for daily seasonality, ``'W'`` for weekly seasonality, or None
        for the standard naive forecast in electricity price forecasting, 
        i.e. daily seasonality for Tuesday to Friday and weekly seasonality 
        for Saturday to Monday.    
    freq : str, optional
        Frequency of the data if ``p_real``, ``p_pred``, and ``p_real_in`` are numpy.ndarray objects.
        It must take one of the following four values ``'1h'`` for 1 hour, ``'30T'`` for 30 minutes, 
        ``'15T'`` for 15 minutes, or ``'5T'`` for 5 minutes,  (these are the four standard values in 
        day-ahead electricity markets). If the shape of ``p_real`` is (ndays, n_prices_day), 
        freq should be the frequency of the columns not the daily frequency of the rows.    
    Returns
    -------
    float
        The mean absolute scaled error (MASE).

    Example
    -------
    >>> from epftoolbox.evaluation import rMAE
    >>> from epftoolbox.data import read_data
    >>> import pandas as pd
    >>> 
    >>> # Download available forecast of the NP market available in the library repository
    >>> # These forecasts accompany the original paper
    >>> forecast = pd.read_csv('https://raw.githubusercontent.com/jeslago/epftoolbox/master/' + 
    ...                       'forecasts/Forecasts_NP_DNN_LEAR_ensembles.csv', index_col=0)
    >>> 
    >>> # Transforming indices to datetime format
    >>> forecast.index = pd.to_datetime(forecast.index)
    >>> 
    >>> # Reading data from the NP market
    >>> _, df_test = read_data(path='.', dataset='NP', begin_test_date=forecast.index[0], 
    ...                        end_test_date=forecast.index[-1])
    Test datasets: 2016-12-27 00:00:00 - 2018-12-24 23:00:00
    >>> 
    >>> # Extracting forecast of DNN ensemble and display
    >>> fc_DNN_ensemble = forecast.loc[:, ['DNN Ensemble']]
    >>> 
    >>> # Extracting real price and display
    >>> real_price = df_test.loc[:, ['Price']]
    >>> 
    >>> # Building the same datasets with shape (ndays, n_prices/day) instead 
    >>> # of shape (nprices, 1) and display
    >>> fc_DNN_ensemble_2D = pd.DataFrame(fc_DNN_ensemble.values.reshape(-1, 24), 
    ...                                   index=fc_DNN_ensemble.index[::24], 
    ...                                   columns=['h' + str(hour) for hour in range(24)])
    >>> real_price_2D = pd.DataFrame(real_price.values.reshape(-1, 24), 
    ...                              index=real_price.index[::24], 
    ...                              columns=['h' + str(hour) for hour in range(24)])
    >>> fc_DNN_ensemble_2D.head()
                       h0         h1         h2  ...        h21        h22        h23
    2016-12-27  24.349676  23.127774  22.208617  ...  27.686771  27.045763  25.724071
    2016-12-28  25.453866  24.707317  24.452384  ...  29.424558  28.627130  27.321902
    2016-12-29  28.209516  27.715400  27.182692  ...  28.473288  27.926241  27.153401
    2016-12-30  28.002935  27.467572  27.028558  ...  29.086532  28.518688  27.738548
    2016-12-31  25.732282  24.668331  23.951569  ...  26.965008  26.450995  25.637346
    >>> 
    
    According to the paper, the rMAE of the DNN ensemble for the NP market is 0.403
    when ``m='W'``. Let's test the metric for different conditions
     
    >>> # Evaluating rMAE when real price and forecasts are both dataframes
    >>> rMAE(p_pred=fc_DNN_ensemble, p_real=real_price)
    0.5265639198107801
    >>> 
    >>> # Evaluating rMAE when real price and forecasts are both numpy arrays
    >>> rMAE(p_pred=fc_DNN_ensemble.values, p_real=real_price.values, m='W', freq='1h')
    0.4031805447246898
    >>> 
    >>> # Evaluating rMAE when input values are of shape (ndays, n_prices/day) instead 
    >>> # of shape (nprices, 1)
    >>> # Dataframes
    >>> rMAE(p_pred=fc_DNN_ensemble_2D, p_real=real_price_2D, m='W')
    0.4031805447246898
    >>> # Numpy arrays
    >>> rMAE(p_pred=fc_DNN_ensemble_2D.values, p_real=real_price_2D.values, m='W', freq='1h')
    0.4031805447246898
    >>> 
    >>> # Evaluating rMAE when input values are of shape (nprices,) 
    >>> # instead of shape (nprices, 1)
    >>> # Pandas Series
    >>> rMAE(p_pred=fc_DNN_ensemble.loc[:, 'DNN Ensemble'], 
    ...      p_real=real_price.loc[:, 'Price'], m='W')
    0.4031805447246898
    >>> # Numpy arrays
    >>> rMAE(p_pred=fc_DNN_ensemble.values.squeeze(), 
    ...      p_real=real_price.values.squeeze(), m='W', freq='1h')
    0.4031805447246898
    """

    # Computing the MAE of the naive forecast
    # Build copy for security

    p_real_naive = p_real.copy()
    # Pre-process prices to have the correct format
    p_real_naive = _transform_input_prices_for_naive_forecast(p_real_naive, m, freq)
    # Build naive forecast
    p_pred_naive = naive_forecast(p_real_naive, m=m)
    # Select common time indices
    p_real_naive = p_real_naive.loc[p_pred_naive.index]
    # Computing naive MAE
    MAE_naive_train = MAE(p_real_naive, p_pred_naive)
    
    # Checking if standard inputs are compatible
    p_real, p_pred = _process_inputs_for_metrics(p_real, p_pred)

    return np.mean(np.abs(p_real - p_pred) / MAE_naive_train)

