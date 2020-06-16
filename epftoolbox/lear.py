
import numpy as np
import pandas as pd
from statsmodels.robust import mad

from sklearn.linear_model import LassoLarsIC, Lasso
from epftoolbox.wrangling import scaling
from epftoolbox.datasets import read_data
from epftoolbox.metrics import MAE, sMAPE

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

class LEAR(object):

    def __init__(self, calibration_window=364 * 3):

        # Calibration window in hours
        self.calibration_window = calibration_window

    # Ignore convergence warnings from scikit-learn LASSO module
    @ignore_warnings(category=ConvergenceWarning)
    def recalibrate(self, Xtrain, Ytrain, Xtest):

        # Predefining predicted prices
        Yp = np.zeros(24)

        # # Applying Invariant, aka asinh-median transformation to the prices
        [Ytrain], self.scaler = scaling([Ytrain], 'Invariant')

        # # Rescaling all inputs except dummies (7 last features)
        [Xtrain_no_dummies, Xtest_no_dummies], _ = scaling([Xtrain[:, :-7], Xtest[:, :-7]], 'Invariant')
        
        Xtrain[:, :-7] = Xtrain_no_dummies
        Xtest[:, :-7] = Xtest_no_dummies

        for h in range(24):

            # Estimating lambda hyperparameter using LARS
            param_model = LassoLarsIC(criterion='aic', max_iter=2500)
            param = param_model.fit(Xtrain, Ytrain[:, h]).alpha_

            # Re-calibrating LEAR using standard LASSO estimation technique
            model = Lasso(max_iter=2500, alpha=param)
            model.fit(Xtrain, Ytrain[:, h])

            # Predicting test dataset and saving
            Yp[h] = model.predict(Xtest)
        
        Yp = self.scaler.inverse_transform(Yp.reshape(1, -1))

        return Yp

    def build_and_split_XYs(self, df_train, df_test=None, date_test=None):
        
        """
        Method that generates the X,Y arrays for training and testing based on
         the pandas train and test datasets
        
        Args:
            df_train (TYPE): Pandas dataframe containing the training data
            df_test (TYPE): Pandas dataframe containing the test data
            date_test (None, optional): If given, then the test dataset is only built for that date
        Returns:
            TYPE: Description
        """

        # Checking that the first index in the dataframes corresponds with the hour 00:00 
        if df_train.index[0].hour != 0 or df_test.index[0].hour != 0:
            print('Problem with the index')

        # 
        # Defining the number of Exogenous inputs
        n_exogenous_inputs = len(df_train.columns) - 1

        # 96 prices + n_exogenous * (24 * 3 exogeneous) + 7 weekday dummies
        # Price lags: D-1, D-2, D-3, D-7
        # Exogeneous inputs lags: D, D-1, D-7
        n_features = 96 + 7 + n_exogenous_inputs * 72


        # Extracting the predicted dates for testing and training. We leave the first week of data
        # out of the prediction as we the maximum lag can be one week
        
        # We define the potential time indexes that have to be forecasted in training
        # and testing
        indexTrain = df_train.loc[df_train.index[0] + pd.Timedelta(weeks=1):].index

        # For testing, the test dataset is different whether depending on whether a specific test
        # dataset is provided
        if date_test is None:
            indexTest = df_test.loc[df_test.index[0] + pd.Timedelta(weeks=1):].index
        else:
            indexTest = df_test.loc[date_test:date_test + pd.Timedelta(hours=23)].index

        # We extract the prediction dates/days.
        predDatesTrain = indexTrain.round('1H')[::24]                
        predDatesTest = indexTest.round('1H')[::24]

        # We create two dataframe to build XY.
        # These dataframes have as indices the first hour of the day (00:00)
        # and the columns represent the 23 possible horizons/dates along a day
        indexTrain = pd.DataFrame(index=predDatesTrain, columns=['h' + str(hour) for hour in range(24)])
        indexTest = pd.DataFrame(index=predDatesTest, columns=['h' + str(hour) for hour in range(24)])
        for hour in range(24):
            indexTrain.loc[:, 'h' + str(hour)] = indexTrain.index + pd.Timedelta(hours=hour)
            indexTest.loc[:, 'h' + str(hour)] = indexTest.index + pd.Timedelta(hours=hour)

        
        # Preallocating in memory the X and Y arrays          
        Xtrain = np.zeros([indexTrain.shape[0], n_features])
        Xtest = np.zeros([indexTest.shape[0], n_features])
        Ytrain = np.zeros([indexTrain.shape[0], 24])

        # Index that 
        feature_index = 0
        
        #
        # Adding the historial prices during days D-1, D-2, D-3, and D-7
        #

        # For each hour of a day
        for hour in range(24):
            # For each possible past day where prices can be included
            for past_day in [1, 2, 3, 7]:

                # We define the corresponding past time indexs using the auxiliary dataframses 
                pastIndexTrain = pd.to_datetime(indexTrain.loc[:, 'h' + str(hour)].values) - \
                    pd.Timedelta(hours=24 * past_day)
                pastIndexTest = pd.to_datetime(indexTest.loc[:, 'h' + str(hour)].values) - \
                    pd.Timedelta(hours=24 * past_day)

                # We include the historical prices at day D-past_day and hour "h" 
                Xtrain[:, feature_index] = df_train.loc[pastIndexTrain, 'Price']
                Xtest[:, feature_index] = df_test.loc[pastIndexTest, 'Price']
                feature_index += 1

        #
        # Adding the exogenous inputs during days D, D-1,  D-7
        #
        # For each hour of a day
        for hour in range(24):
            # For each possible past day where exogenous inputs can be included
            for past_day in [1, 2]:
                # For each of the exogenous input
                for exog in range(1, n_exogenous_inputs + 1):

                    # Definying the corresponding past time indexs using the auxiliary dataframses 
                    pastIndexTrain = pd.to_datetime(indexTrain.loc[:, 'h' + str(hour)].values) - \
                        pd.Timedelta(hours=24 * past_day)
                    pastIndexTest = pd.to_datetime(indexTest.loc[:, 'h' + str(hour)].values) - \
                        pd.Timedelta(hours=24 * past_day)

                    # Including the exogenous input at day D-past_day and hour "h" 
                    Xtrain[:, feature_index] = df_train.loc[pastIndexTrain, 'Exogenous ' + str(exog)]                    
                    Xtest[:, feature_index] = df_test.loc[pastIndexTest, 'Exogenous ' + str(exog)]
                    feature_index += 1

            # For each of the exogenous inputs we include feature if feature selection indicates it
            for exog in range(1, n_exogenous_inputs + 1):
                
                # Definying the corresponding future time indexs using the auxiliary dataframses 
                futureIndexTrain = pd.to_datetime(indexTrain.loc[:, 'h' + str(hour)].values)
                futureIndexTest = pd.to_datetime(indexTest.loc[:, 'h' + str(hour)].values)

                # Including the exogenous input at day D and hour "h" 
                Xtrain[:, feature_index] = df_train.loc[futureIndexTrain, 'Exogenous ' + str(exog)]        
                Xtest[:, feature_index] = df_test.loc[futureIndexTest, 'Exogenous ' + str(exog)] 
                feature_index += 1

        #
        # Adding the dummy variables that depend on the day of the week. Monday is 0 and Sunday is 6
        #
        # For each day of the week
        for dayofweek in range(7):
            Xtrain[indexTrain.index.dayofweek == dayofweek, feature_index] = 1
            Xtest[indexTest.index.dayofweek == dayofweek, feature_index] = 1
            feature_index += 1

        # Extracting the predicted values Y
        for hour in range(24):
            # Defining time index at hour h
            futureIndexTrain = pd.to_datetime(indexTrain.loc[:, 'h' + str(hour)].values)
            futureIndexTest = pd.to_datetime(indexTest.loc[:, 'h' + str(hour)].values)

            # Extracting Y value based on time indexs
            Ytrain[:, hour] = df_train.loc[futureIndexTrain, 'Price']        

        return Xtrain, Ytrain, Xtest


    def recalibrate_and_forecast_next_day(self, df, calibration_window, next_day_date):
            
        # We define the new training dataset and test datasets 
        df_train = df.loc[:next_day_date - pd.Timedelta(hours=1)]
        # Limiting the training dataset to the calibration window
        df_train = df_train.iloc[-self.calibration_window * 24:]
    
        # We define the test dataset as the next day (they day of interest) plus the last two weeks
        # in order to be able to build the necessary input features. 
        df_test = df.loc[next_day_date - pd.Timedelta(weeks=2):, :]


        # Generating X,Y pairs for predicting prices
        Xtrain, Ytrain, Xtest, = self.build_and_split_XYs(
            df_train=df_train, df_test=df_test, date_test=next_day_date)

        # Recalibrating the LEAR model and extracting the prediction
        Yp = self.recalibrate(Xtrain=Xtrain, Ytrain=Ytrain, Xtest=Xtest)

        return Yp


def evaluate_lear_in_test_dataset(path_datasets='./datasets/', 
                                  path_recalibration_files='./experimental_files/',
                                  dataset='PJM', years_test=2, calibration_window=364 * 3, 
                                  begin_test_date=None, end_test_date=None):

        
    # Defining train and testing data
    df_train, df_test = read_data(dataset=dataset, years_test=years_test, path=path_datasets,
                                  begin_test_date=begin_test_date, end_test_date=end_test_date)

    # Defining unique name to save the forecast
    forecast_file_name = path_recalibration_files + 'fc_nl' + '_dat' + str(dataset) + \
                       '_YT' + str(years_test) + '_CW' + str(calibration_window)

    # Defining empty forecast array and the real values to be predicted in a more friendly format
    forecast = pd.DataFrame(index=df_test.index[::24], columns=['h' + str(k) for k in range(24)])
    real_values = df_test.loc[:, ['Price']].values.reshape(-1, 24)
    real_values = pd.DataFrame(real_values, index=forecast.index, columns=forecast.columns)

    forecast_dates = forecast.index

    model = LEAR(calibration_window=calibration_window)

    # For loop over the recalibration dates
    for date in forecast_dates:

        # For simulation purposes, we assume that the available data is
        # the data up to current date where the prices of current date are not known
        data_available = pd.concat([df_train, df_test.loc[:date + pd.Timedelta(hours=23), :]], axis=0)

        # We set the real prices for current date to NaN in the dataframe of available data
        data_available.loc[date:date + pd.Timedelta(hours=23), 'Price'] = np.NaN

        # Recalibrating the model with the most up-to-date available data and making a prediction
        # for the next day
        Yp = model.recalibrate_and_forecast_next_day(df=data_available, next_day_date=date, 
                                                     calibration_window=calibration_window)
        # Saving the current prediction
        forecast.loc[date, :] = Yp
        import ipdb
        ipdb.set_trace()

        # Computing metrics up-to-current-date
        mae = np.mean(MAE(forecast.loc[:date].values.squeeze(), real_values.loc[:date].values)) 
        smape = np.mean(sMAPE(forecast.loc[:date].values.squeeze(), real_values.loc[:date].values)) * 100

        # Pringint information
        print('{} - sMAPE: {:.2f}%  |  MAE: {:.2f}'.format(str(date)[:10], smape, mae))

        # Saving forecast
        forecast.to_csv(forecast_file_name + '.csv')
