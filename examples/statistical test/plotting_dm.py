from epftoolbox.evaluation import DM, plot_multivariate_DM_test
from epftoolbox.data import read_data
import pandas as pd

# Generating forecasts of multiple models

# Download available forecast of the NP market available in the library repository
# These forecasts accompany the original paper
forecasts = pd.read_csv('https://raw.githubusercontent.com/jeslago/epftoolbox/master/' + 
                      'forecasts/Forecasts_NP_DNN_LEAR_ensembles.csv', index_col=0)

# Deleting the real price field as it the actual real price and not a forecast
del forecasts['Real price']

# Transforming indices to datetime format
forecasts.index = pd.to_datetime(forecasts.index)

# Extracting the real prices from the market
_, df_test = read_data(path='.', dataset='NP', begin_test_date=forecasts.index[0], 
                       end_test_date=forecasts.index[-1])

real_price = df_test.loc[:, ['Price']]

# Generating a plot to compare the models using the multivariate DM test
plot_multivariate_DM_test(real_price=real_price, forecasts=forecasts)