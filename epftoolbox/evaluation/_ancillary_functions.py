"""
Ancillary functions to compute accuracy metrics and statistical tests in the context of electricity price
forecasting
"""

import numpy as np
import pandas as pd

def _process_inputs_for_metrics(p_real, p_pred):
    """Function that checks that the two standard inputs of the metric functions satisfy some requirements
    
    
    Parameters
    ----------
    p_real : numpy.ndarray, pandas.DataFrame, pandas.Series
        Array/dataframe containing the real prices
    p_pred : numpy.ndarray, pandas.DataFrame, pandas.Series
        Array/dataframe containing the predicted prices
    
    Returns
    -------
    np.ndarray, np.ndarray
        The p_real and p_pred as numpy.ndarray objects after checking that they satisfy requirements 
    
    """
    
    # Checking that both arrays are of the same type
    if type(p_real) != type(p_pred):
        raise TypeError('p_real and p_pred must be of the same type. p_real is of type {}'.format(type(p_real)) +
            ' and p_pred of type {}'.format(type(p_pred)))

    # Checking that arrays are of the allowed types
    if type(p_real) != pd.DataFrame and \
       type(p_real) != pd.Series and \
       type(p_real) != np.ndarray:
        raise TypeError('p_real and p_pred must be either a pandas.DataFrame, a pandas.Serie, or ' +
        ' a numpy.aray. They are of type {}'.format(type(p_real)))

    # Transforming dataset if it is a pandas.Series to pandas.DataFrame
    if type(p_real) == pd.Series:
        p_real = p_real.to_frame()
        p_pred = p_pred.to_frame()
    
    # Checking that both datasets share the same indices
    if type(p_real) == pd.DataFrame:
        if not (p_real.index == p_pred.index).all():
            raise ValueError('p_real and p_pred must have the same indices')

        # Extracting their values as numpy.ndarrays
        p_real = p_real.values.squeeze()
        p_pred = p_pred.values.squeeze()

    return p_real, p_pred

def naive_forecast(p_real, m=None, n_prices_day=24):
    """Function to buil the naive forecast for electricity price forecasting
    
    The function is used to compute the accuracy metrics MASE and RMAE
        
    Parameters
    ----------
    p_real : pandas.DataFrame
        Dataframe containing the real prices. It must be of shape :math:`(n_\\mathrm{prices}, 1)`,
    m : int, optional
        Index that specifies the seasonality in the naive forecast. It can
        be ``'D'`` for daily seasonality, ``'W'`` for weekly seasonality, or ``None``
        for the standard naive forecast in electricity price forecasting, 
        i.e. daily seasonality for Tuesday to Friday and weekly seasonality 
        for Saturday to Monday.
    n_prices_day : int, optional
        Number of prices in a day. Usually this value is 24 for most day-ahead markets
    
    Returns
    -------
    pandas.DataFrame
        Dataframe containing the predictions of the naive forecast.
    """

    # Init the naive forecast
    if m is None or m == 'W':
        index = p_real.index[n_prices_day * 7:]
        Y_pred = pd.DataFrame(index=index, columns=p_real.columns)
    else:
        index = p_real.index[n_prices_day:]
        Y_pred = pd.DataFrame(index=index, columns=p_real.columns)

    # If m is none the standard naive for EPF is built
    if m is None:

        # Monday we have a naive forecast using daily seasonality
        indices_mon = Y_pred.index[Y_pred.index.dayofweek == 0]
        Y_pred.loc[indices_mon, :] = p_real.loc[indices_mon - pd.Timedelta(days=7), :].values

        # Tuesdays we have a naive forecast using daily seasonality
        indices_tue = Y_pred.index[Y_pred.index.dayofweek == 1]
        Y_pred.loc[indices_tue, :] = p_real.loc[indices_tue - pd.Timedelta(days=1), :].values

        # Wednesday we have a naive forecast using daily seasonality
        indices_wed = Y_pred.index[Y_pred.index.dayofweek == 2]
        Y_pred.loc[indices_wed, :] = p_real.loc[indices_wed - pd.Timedelta(days=1), :].values

        # Thursday we have a naive forecast using daily seasonality
        indices_thu = Y_pred.index[Y_pred.index.dayofweek == 3]
        Y_pred.loc[indices_thu, :] = p_real.loc[indices_thu - pd.Timedelta(days=1), :].values

        # Friday we have a naive forecast using daily seasonality
        indices_fri = Y_pred.index[Y_pred.index.dayofweek == 4]
        Y_pred.loc[indices_fri, :] = p_real.loc[indices_fri - pd.Timedelta(days=1), :].values

        # Saturday we have a naive forecast using weekly seasonality
        indices_sat = Y_pred.index[Y_pred.index.dayofweek == 5]
        Y_pred.loc[indices_sat, :] = p_real.loc[indices_sat - pd.Timedelta(days=7), :].values

        # Sunday we have a naive forecast using weekly seasonality
        indices_sun = Y_pred.index[Y_pred.index.dayofweek == 6]
        Y_pred.loc[indices_sun, :] = p_real.loc[indices_sun - pd.Timedelta(days=7), :].values

    # If m is either 24 or 168 naive forecast simply built using a seasonal naive forecast
    elif m == 'D':
        Y_pred.loc[:, :] = p_real.loc[Y_pred.index - pd.Timedelta(days=1)].values
    elif m == 'W':
        Y_pred.loc[:, :] = p_real.loc[Y_pred.index - pd.Timedelta(days=7)].values

    return Y_pred

def _transform_input_prices_for_naive_forecast(p_real, m, freq):
    """Function that ensures that the input of the naive forecast has the right format
    
    Parameters
    ----------
    p_real : numpy.ndarray, pandas.DataFrame, pandas.Series
        Array/dataframe containing the real prices
    m : int, optional
        Index that specifies the seasonality in the naive forecast. It can
        be ``'D'`` for daily seasonality, ``'W'`` for weekly seasonality, or None
        for the standard naive forecast in electricity price forecasting, 
        i.e. daily seasonality for Tuesday to Friday and weekly seasonality 
        for Saturday to Monday.
    freq : str
        Frequency of the data if ``p_real`` are numpy.ndarray objects.
        It must take one of the following four values ``'1h'`` for 1 hour, ``'30T'`` for 30 minutes, 
        ``'15T'`` for 15 minutes, or ``'5T'`` for 5 minutes,  (these are the four standard values in 
        day-ahead electricity markets). If the shape of ``p_real`` is (ndays, n_prices_day), 
        freq should be the frequency of the columns not the daily frequency of the rows.    
    Returns
    -------
    pandas.DataFrame
        ``p_real`` as a pandas.DataFrame that can be used for the naive forecast 
    """

    # Ensure that m value is correct
    if m not in ['D', 'W', None]: 
        raise ValueError('m argument has to be D, W, or None. Current values is {}'.format(m))    

    # Check that input data is not numpy.ndarray and naive forecast is standard
    if m is None and type(p_real) != pd.DataFrame and type(p_real) != pd.Series:
        raise TypeError('To use the standard naive forecast, i.e. m=None, the input' +
            ' data has to be pandas.DataFrame object.')

    # Defining number of prices per day depending on frequency
    n_prices_day = {'1h': 24, '30T': 48, '15T': 96, '5T': 288, '1T': 1440}[freq]

    # If numpy arrays are used, ensure that there is integer number of days in the dataset
    if type(p_real) == np.ndarray and p_real.size % n_prices_day != 0:
        raise ValueError('If numpy arrays are used, the size of p_real, i.e. the number of prices it '
            + 'contains, should be a multiple number of {}, i.e. of the number of '.format(n_prices_day)
            + ' prices per day. Current values is {}'.format(p_real.size))    
    
    # If pandas.Series are used, convert to DataFrame
    if type(p_real) == pd.Series:
        p_real = p_real.to_frame()

    # If input data is numpy.ndarray, transform to pandas.DataFrame
    if type(p_real) == np.ndarray:
        # Transforming p_real to correct shape, i.e. (nprices, 1)
        p_real = p_real.reshape(-1, 1)
        # Building time indices for DataFrame
        indices = pd.date_range(start='2013-01-01', periods=p_real.shape[0], freq=freq)        
        # Building DataFrame
        p_real = pd.DataFrame(p_real, index=indices)
    
    # If input data is pandas-based, make sure it is in correct shape
    elif type(p_real) == pd.DataFrame:
        # Making sure that index is of datetime format
        p_real.index = pd.to_datetime(p_real.index)

        # Raising error if frequency cannot be inferred
        if p_real.index.inferred_freq is None:
            raise ValueError('The frequency/time periodicity of the data could not be inferred. '
                + 'Ensure that the indices of the dataframe have a correct format and are equally separated.')

        # If shape (ndays, n_prices_day), ensure that frequency of index is daily        
        if p_real.shape[1] > 1 and p_real.index.inferred_freq != 'D':
            raise ValueError('If pandas dataframes are used with arrays with shape ' 
                + '(ndays, n_prices_day), the frequency of the time indices should be 1 day. '
                + 'At the moment it is {}.'.format(p_real.index.inferred_freq))

        # Reshaping dataframe if shape (ndays, n_prices_day)
        if p_real.shape[1] > 1:
            # Inferring frequency within a day
            frequency_seconds = 24 * 60 * 60 / p_real.shape[1]
            # Infering last date in the dataset based on the frequency of points within a day
            last_date = p_real.index[-1] + (p_real.shape[1] - 1) * pd.Timedelta(seconds=frequency_seconds)
            # Inferring indices
            indices = pd.date_range(start=p_real.index[0], end=last_date, periods=p_real.size)
            # Reshaping prices
            p_real = pd.DataFrame(data=p_real.values.reshape(-1, 1), columns=['Prices'], index=indices)

    # Raising error if p_real not of specified type
    else:
        raise TypeError('Input should be of type numpy.ndarray, pandas.DataFrame, or pandas.Series' +
            ' but it is of type {}'.format(type(p_real)))

    return p_real

