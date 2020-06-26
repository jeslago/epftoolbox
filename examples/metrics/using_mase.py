from epftoolbox.evaluation import MASE
from epftoolbox.data import read_data
import pandas as pd

# Download available forecast of the NP market available in the library repository
# These forecasts accompany the original paper
forecast = pd.read_csv('https://raw.githubusercontent.com/jeslago/epftoolbox/master/' + 
                      'forecasts/Forecasts_NP_DNN_LEAR_ensembles.csv', index_col=0)

# Transforming indices to datetime format
forecast.index = pd.to_datetime(forecast.index)

# Reading data from the NP market
df_train, df_test = read_data(path='.', dataset='NP', begin_test_date=forecast.index[0], 
                       end_test_date=forecast.index[-1])

# Extracting forecast of DNN ensemble and display
fc_DNN_ensemble = forecast.loc[:, ['DNN Ensemble']]

# Extracting real price and display
real_price = df_test.loc[:, ['Price']]
real_price_insample = df_train.loc[:, ['Price']]

# Building the same datasets with shape (ndays, n_prices/day) instead 
# of shape (nprices, 1) and display
fc_DNN_ensemble_2D = pd.DataFrame(fc_DNN_ensemble.values.reshape(-1, 24), 
                                  index=fc_DNN_ensemble.index[::24], 
                                  columns=['h' + str(hour) for hour in range(24)])
real_price_2D = pd.DataFrame(real_price.values.reshape(-1, 24), 
                             index=real_price.index[::24], 
                             columns=['h' + str(hour) for hour in range(24)])
real_price_insample_2D = pd.DataFrame(real_price_insample.values.reshape(-1, 24), 
                             index=real_price_insample.index[::24], 
                             columns=['h' + str(hour) for hour in range(24)])

fc_DNN_ensemble_2D.head()


# According to the paper, the MASE of the DNN ensemble for the NP market is 0.403
# when m='W'. Let's test the metric for different conditions

# Evaluating MASE when real price and forecasts are both dataframes
MASE(p_pred=fc_DNN_ensemble, p_real=real_price, 
     p_real_in=real_price_insample, m='W')

# Evaluating MASE when real price and forecasts are both numpy arrays
MASE(p_pred=fc_DNN_ensemble.values, p_real=real_price.values, 
     p_real_in=real_price_insample.values, m='W', freq='1H')

# Evaluating MASE when input values are of shape (ndays, n_prices/day) instead 
# of shape (nprices, 1)
# Dataframes
MASE(p_pred=fc_DNN_ensemble_2D, p_real=real_price_2D, 
     p_real_in=real_price_insample_2D, m='W')
# Numpy arrays
MASE(p_pred=fc_DNN_ensemble_2D.values, p_real=real_price_2D.values, 
     p_real_in=real_price_insample_2D.values, m='W', freq='1H')

# Evaluating MASE when input values are of shape (nprices,) 
# instead of shape (nprices, 1)
# Pandas Series
MASE(p_pred=fc_DNN_ensemble.loc[:, 'DNN Ensemble'], 
     p_real=real_price.loc[:, 'Price'],
     p_real_in=real_price_insample.loc[:, 'Price'], m='W')
# Numpy arrays
MASE(p_pred=fc_DNN_ensemble.values.squeeze(), 
     p_real=real_price.values.squeeze(), 
     p_real_in=real_price_insample.values.squeeze(), m='W', freq='1H')


# We can also test situations where the MASE will display errors

# Evaluating MASE when real price and forecasts are of different type (numpy.ndarray and pandas.DataFrame)
MASE(p_pred=fc_DNN_ensemble.values, p_real=real_price, m='W', freq='1H')

# Evaluating MASE when real price and forecasts are of different type (pandas.Series and pandas.DataFrame)
MASE(p_pred=fc_DNN_ensemble, p_real=real_price.loc[:, 'Price'])

# Evaluating MASE when real price and forecasts are both numpy arrays of different size
MASE(p_pred=fc_DNN_ensemble.values[1:], p_real=real_price.values, m='W')

# Evaluating MASE when real price and forecasts are both dataframes are of different size
MASE(p_pred=fc_DNN_ensemble.iloc[1:, :], p_real=real_price)

# Evaluating MASE when real price are not multiple of 1 day
MASE(p_pred=fc_DNN_ensemble.values[1:], p_real=real_price.values[1:], m='W', freq='1H')