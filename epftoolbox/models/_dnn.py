import numpy as np
import pandas as pd
import time
import pickle as pc
import os

import tensorflow.keras as kr
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, AlphaDropout, BatchNormalization
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.layers import LeakyReLU, PReLU
import tensorflow.keras.backend as K

from epftoolbox.evaluation import MAE, sMAPE
from epftoolbox.data import scaling
from epftoolbox.data import read_data


class DNNModel(object):

    def __init__(self, neurons, Nfeatures, outputShape=24, dropout=0, BN=False, lr=None,
                 printOut=False, maxEpochsWOImprovement=40, scaler=None, loss='mae',
                 optimizer='adam', activation='relu', initializer='glorot_uniform',
                 regularization=None, lambdaReg=0):

        self.neurons = neurons
        self.dropout = dropout
        self.BN = BN
        self.printOut = printOut
        self.maxEpochsWOImprovement = maxEpochsWOImprovement
        self.Nfeatures = Nfeatures
        self.scaler = scaler
        self.outputShape = outputShape
        self.activation = activation
        self.initializer = initializer
        self.regularization = regularization
        self.lambdaReg = lambdaReg

        self.model = self._buildAndCompileModel()

        if lr is None:
            opt = 'adam'
        else:
            if optimizer == 'adam':
                opt = kr.optimizers.Adam(lr=lr, clipvalue=10000)
            if optimizer == 'RMSprop':
                opt = kr.optimizers.RMSprop(lr=lr, clipvalue=10000)
            if optimizer == 'adagrad':
                opt = kr.optimizers.Adagrad(lr=lr, clipvalue=10000)
            if optimizer == 'adadelta':
                opt = kr.optimizers.Adadelta(lr=lr, clipvalue=10000)

        self.model.compile(loss=loss, optimizer=opt)

    def _reg(self, lambdaReg):

        if self.regularization == 'l2':
            return l2(lambdaReg)
        if self.regularization == 'l1':
            return l1(lambdaReg)
        else:
            return None

    def _buildAndCompileModel(self):

        inputShape = (None, self.Nfeatures)

        past_data = Input(batch_shape=inputShape)

        past_Dense = past_data
        if self.activation == 'selu':
            self.initializer = 'lecun_normal'

        for k, neurons in enumerate(self.neurons):

            if self.activation == 'LeakyReLU':
                past_Dense = Dense(neurons, activation='linear', batch_input_shape=inputShape,
                                   kernel_initializer=self.initializer,
                                   kernel_regularizer=self._reg(self.lambdaReg))(past_Dense)
                past_Dense = LeakyReLU(alpha=.001)(past_Dense)

            elif self.activation == 'PReLU':
                past_Dense = Dense(neurons, activation='linear', batch_input_shape=inputShape,
                                   kernel_initializer=self.initializer,
                                   kernel_regularizer=self._reg(self.lambdaReg))(past_Dense)
                past_Dense = PReLU()(past_Dense)

            else:
                past_Dense = Dense(neurons, activation=self.activation,
                                   batch_input_shape=inputShape,
                                   kernel_initializer=self.initializer,
                                   kernel_regularizer=self._reg(self.lambdaReg))(past_Dense)

            if self.BN:
                past_Dense = BatchNormalization()(past_Dense)

            if self.dropout > 0:
                if self.activation == 'selu':
                    past_Dense = AlphaDropout(self.dropout)(past_Dense)
                else:
                    past_Dense = Dropout(self.dropout)(past_Dense)

        output_layer = Dense(self.outputShape, kernel_initializer=self.initializer,
                             kernel_regularizer=self._reg(self.lambdaReg))(past_Dense)
        model = Model(inputs=[past_data], outputs=[output_layer])

        return model

    def _update_metrics(self, X, Y):

        error = self.model.evaluate(X, Y, verbose=0)
        Ybar = self.model.predict(X, verbose=0)

        if self.scaler is not None:
            if len(Y.shape) == 1:
                Y = Y.reshape(-1, 1)
                Ybar = Ybar.reshape(-1, 1)

            Y = self.scaler.inverse_transform(Y)
            Ybar = self.scaler.inverse_transform(Ybar)

        mae = MAE(Y, Ybar)

        return error, np.mean(mae)

    def fit(self, trainX, trainY, valX, valY):

        # Variables to control training improvement
        bestError = 1e20
        bestMAE = 1e20

        countNoImprovement = 0

        bestWeights = self.model.get_weights()

        for epoch in range(1000):
            start_time = time.time()

            self.model.fit(trainX, trainY, batch_size=192,
                           epochs=1, verbose=False, shuffle=True)

            # Updating epoch metrics and displaying useful information
            if self.printOut:
                print("\nEpoch {} of {} took {:.3f}s".format(epoch + 1, 1000,
                                                             time.time() - start_time))

            valError, valMAE = self._update_metrics(valX, valY)

            # print(self.model.predict(trainX))
            # Checking if current epoch is better than so far best self.self.model. If it is,
            # the self.model weights are saved to make at the end the best evaluation
            if valError < bestError:
                countNoImprovement = 0
                bestWeights = self.model.get_weights()

                bestError = valError
                bestMAE = valMAE
                if valMAE < bestMAE:
                    bestMAE = valMAE

            elif valMAE < bestMAE:
                countNoImprovement = 0
                bestWeights = self.model.get_weights()
                bestMAE = valMAE
            else:
                countNoImprovement += 1

            if self.printOut:

                print(" Best error:\t\t{:.1e}".format(bestError))
                print(" Best MAE:\t\t{:.2f}".format(bestMAE))                
                print(" Epochs without improvement:\t{}\n".format(countNoImprovement))

            if countNoImprovement >= self.maxEpochsWOImprovement:
                break

        self.model.set_weights(bestWeights)

    def predict(self, X):

        Ybar = self.model.predict(X, verbose=0)
        return Ybar

    def clear_session(self):
        K.clear_session()



class DNN(object):

    def __init__(self, experiment_id, path_hyperparameter_folder=os.path.join('.', 'experimental_files'), 
                 nlayers=2, dataset='PJM', years_test=2, shuffle_train=0, data_augmentation=0, 
                 calibration_window=4):

        """ Main function to recalibrate the DNN model
        
        Args:
            experiment_id (str): Unique identifier to read the trials file
            path_hyperparameter_folder (str, optional): Path to read trials files from hyperopt
            nlayers (int, optional): Number of layers of the DNN model
            dataset (str, optional): Market under study. If it not one of the standard ones, the file name
            has to be provided, where the file has to be a csv file
            years_test (int, optional): Number of years (a year is 364 days) in the test dataset
            shuffle_train (bool, optional): Boolean that selects whether the validation and training datasets
            are shuffled
            data_augmentation (bool, optional): Boolean that selects whether a data augmentation technique 
            for DNNs is used
        """

        # Checking if provided directories exist and if not raise exception
        self.path_hyperparameter_folder = path_hyperparameter_folder
        if not os.path.exists(self.path_hyperparameter_folder):
            raise Exception('Provided directory for hyperparameter file does not exist')

        self.experiment_id = experiment_id
        self.nlayers = nlayers
        self.years_test = years_test
        self.shuffle_train = shuffle_train
        self.data_augmentation = data_augmentation
        self.dataset = dataset
        self.calibration_window = calibration_window
        self.read_best_hyperapameters()

    def regularize_data(self, Xtrain, Xval, Xtest, Ytrain, Yval):

        # If required, datasets are scaled
        if self.best_hyperparameters['scaleX'] in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
            [Xtrain, Xval, Xtest], _ = scaling([Xtrain, Xval, Xtest], self.best_hyperparameters['scaleX'])

        if self.best_hyperparameters['scaleY'] in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
            [Ytrain, Yval], self.scaler = scaling([Ytrain, Yval], self.best_hyperparameters['scaleY'])
        else:
            self.scaler = None

        return Xtrain, Xval, Xtest, Ytrain, Yval

    def recalibrate(self, Xtrain, Ytrain, Xval, Yval):

        # Initialize model
        neurons = [int(self.best_hyperparameters['neurons' + str(k)]) for k in range(1, self.nlayers + 1)
                   if int(self.best_hyperparameters['neurons' + str(k)]) >= 50]
            
        np.random.seed(int(self.best_hyperparameters['seed']))

        self.model = DNNModel(neurons=neurons, Nfeatures=Xtrain.shape[-1], 
                              dropout=self.best_hyperparameters['dropout'], BN=self.best_hyperparameters['BN'], 
                              lr=self.best_hyperparameters['lr'], printOut=False,
                              optimizer='adam', activation=self.best_hyperparameters['activation'],
                              maxEpochsWOImprovement=20, scaler=self.scaler, loss='mae',
                              regularization=self.best_hyperparameters['reg'], 
                              lambdaReg=self.best_hyperparameters['lambdal1'],
                              initializer=self.best_hyperparameters['init'])

        self.model.fit(Xtrain, Ytrain, Xval, Yval)
        
    def recalibrate_predict(self, Xtrain, Ytrain, Xval, Yval, Xtest):

        self.recalibrate(Xtrain=Xtrain, Ytrain=Ytrain, Xval=Xval, Yval=Yval)        
        Yp = self.predict(Xtest=Xtest)

        self.model.clear_session()

        return Yp

    def predict(self, Xtest):

        # Predicting the current date using a recalibrated DNN
        Yp = self.model.predict(Xtest).squeeze()
        if self.best_hyperparameters['scaleY'] in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
            Yp = self.scaler.inverse_transform(Yp.reshape(1, -1))

        return Yp

    def read_best_hyperapameters(self):

        # Defining the trials file name used to extract the optimal hyperparameters
        trials_file_name = \
            'hyper_nl' + str(self.nlayers) + '_dat' + str(self.dataset) + '_YT' + str(self.years_test) + \
            '_SF' * (self.shuffle_train) + '_DA' * (self.data_augmentation) + \
            '_CW' + str(self.calibration_window) + '_' + str(self.experiment_id)

        trials_file_path = os.path.join(self.path_hyperparameter_folder, trials_file_name)

        # Reading and extracting the best hyperparameters
        trials = pc.load(open(trials_file_path, "rb"))
        
        self.best_hyperparameters = self.format_best_trial(trials.best_trial) 


    def format_best_trial(self, best_trial):
        """ Function to format the best_trial object from the hyperopt library to a simple
        dictionary storing the best hyperparameters
        
        Args:
            best_trial (TYPE): best_trial object from the hyperopt library
        
        Returns:
            formatted_hyperparameters (Dict): formatted dictionary with best hyperparameters
        """
        unformatted_hyperparameters = best_trial['misc']['vals']

        formatted_hyperparameters = {}

        # Removing list format
        for key, val in unformatted_hyperparameters.items():
            if len(val) > 0:
                formatted_hyperparameters[key] = val[0]
            else:
                formatted_hyperparameters[key] = 0

        # Reformatting discrete hyperparameters from integer value to keyword
        for key, val in formatted_hyperparameters.items():
            if key == 'activation':
                hyperparameter_list = ["relu", "softplus", "tanh", 'selu', 'LeakyReLU', 'PReLU', 'sigmoid']
            elif key == 'init':
                hyperparameter_list = ['Orthogonal', 'lecun_uniform', 'glorot_uniform', 'glorot_normal', 
                                       'he_uniform', 'he_normal']
            elif key == 'scaleX':
                hyperparameter_list = ['No', 'Norm', 'Norm1', 'Std', 'Median', 'Invariant']
            elif key == 'scaleY':
                hyperparameter_list = ['No', 'Norm', 'Norm1', 'Std', 'Median', 'Invariant']
            elif key == 'reg':
                hyperparameter_list = [None, 'l1']

            if key in ['activation', 'init', 'scaleX', 'scaleY', 'reg']:
                formatted_hyperparameters[key] = hyperparameter_list[val]

        return formatted_hyperparameters

    def recalibrate_and_forecast_next_day(self, df, next_day_date, calibration_window, shuffle_train,
                                          data_augmentation):
            
            # We define the new training dataset considering the last calibration_window years of data 
            df_train = df.loc[:next_day_date - pd.Timedelta(hours=1)]
            df_train = df_train.loc[next_day_date - pd.Timedelta(hours=calibration_window * 364 * 24):]

            # We define the test dataset as the next day (they day of interest) plus the last two weeks
            # in order to be able to build the necessary input features.
            df_test = df.loc[next_day_date - pd.Timedelta(weeks=2):, :]

            # Generating training, validation, and test input and outpus. For the test dataset,
            # even though the dataframe contains 15 days of data (next day + last 2 weeks),
            # we provide as parameter the date of interest so that Xtest and Ytest only reflect that
            Xtrain, Ytrain, Xval, Yval, Xtest, _, _ = \
                build_and_split_XYs(dfTrain=df_train, features=self.best_hyperparameters, 
                                    shuffle_train=shuffle_train, dfTest=df_test, date_test=next_day_date,
                                    data_augmentation=data_augmentation, 
                                    n_exogenous_inputs=len(df_train.columns) - 1)

            # Normalizing the input and outputs if needed
            # Normalizing the input and outputs if needed
            Xtrain, Xval, Xtest, Ytrain, Yval = \
                self.regularize_data(Xtrain=Xtrain, Xval=Xval, Xtest=Xtest, Ytrain=Ytrain, Yval=Yval)

            # Recalibrating the neural network and extracting the prediction
            Yp = self.recalibrate(Xtrain=Xtrain, Ytrain=Ytrain, Xval=Xval, Yval=Yval, Xtest=Xtest)

            return Yp


def build_and_split_XYs(dfTrain, features, shuffle_train, n_exogenous_inputs, dfTest=None, percentage_val=0.25,
                        date_test=None, hyperoptimization=False, data_augmentation=False):
    """
    Method that generates the X,Y arrays for training and testing based on the selected inputs
    and the pandas train and test datasets
    
    Args:
        dfTrain (TYPE): Pandas dataframe containing the training data
        features (TYPE): Dictionary of hyperparameters that contains the selected input features
        shuffle_train (TYPE): If true, validation and training are shuffled
        dfTest (TYPE): Pandas dataframe containing the test data
        percentage_val (TYPE, optional): Percentage of data to be used for validation
        date_test (None, optional): If given, then the test dataset is only built for that date
    
    Returns:
        TYPE: Description
    """

    # Checking that the first index in the dataframes corresponds with the hour 00:00 
    if dfTrain.index[0].hour != 0 or dfTest.index[0].hour != 0:
        print('Problem with the index')

        
    # Calculating the number of input features
    nFeatures = features['In: Day'] + \
        24 * features['In: Price D-1'] + 24 * features['In: Price D-2'] + \
        24 * features['In: Price D-3'] + 24 * features['In: Price D-7']

    for n_ex in range(1, n_exogenous_inputs + 1):

        nFeatures += 24 * features['In: Exog-' + str(n_ex) + ' D'] + \
                     24 * features['In: Exog-' + str(n_ex) + ' D-1'] + \
                     24 * features['In: Exog-' + str(n_ex) + ' D-7']

    # Extracting the predicted dates for testing and training. We leave the first week of data
    # out of the prediction as we the maximum lag can be one week
    # In addition, if we allow training using all possible predictions within a day, we consider
    # a indexTrain per starting hour of prediction
    
    # We define the potential time indexes that have to be forecasted in training
    # and testing
    indexTrain = dfTrain.loc[dfTrain.index[0] + pd.Timedelta(weeks=1):].index

    if date_test is None:
        indexTest = dfTest.loc[dfTest.index[0] + pd.Timedelta(weeks=1):].index
    else:
        indexTest = dfTest.loc[date_test:date_test + pd.Timedelta(hours=23)].index

    # We extract the prediction dates/days. For the regular case, 
    # it is just the index resample to 24 so we have a date per day.
    # For the multiple datapoints per day, we have as many dates as indexs
    if data_augmentation:
        predDatesTrain = indexTrain.round('1H')
    else:
        predDatesTrain = indexTrain.round('1H')[::24]            
            
    predDatesTest = indexTest.round('1H')[::24]

    # We create dataframe where the index is the time where a prediction is made
    # and the columns is the horizons of the prediction
    indexTrain = pd.DataFrame(index=predDatesTrain, columns=['h' + str(hour) for hour in range(24)])
    indexTest = pd.DataFrame(index=predDatesTest, columns=['h' + str(hour) for hour in range(24)])
    for hour in range(24):
        indexTrain.loc[:, 'h' + str(hour)] = indexTrain.index + pd.Timedelta(hours=hour)
        indexTest.loc[:, 'h' + str(hour)] = indexTest.index + pd.Timedelta(hours=hour)

    # If we consider 24 predictions per day, the last 23 indexs cannot be used as there is not data
    # for that horizon:
    if data_augmentation:
        indexTrain = indexTrain.iloc[:-23]
    
    # Preallocating in memory the X and Y arrays          
    Xtrain = np.zeros([indexTrain.shape[0], nFeatures])
    Xtest = np.zeros([indexTest.shape[0], nFeatures])
    Ytrain = np.zeros([indexTrain.shape[0], 24])
    Ytest = np.zeros([indexTest.shape[0], 24])

    # Adding the day of the week as a feature if needed
    indexFeatures = 0
    if features['In: Day']:
        # For training, we assume the day of the week is a continuous variable.
        # So monday at 00 is 1. Monday at 1h is 1.04, Tuesday at 2h is 2.08, etc.
        Xtrain[:, 0] = indexTrain.index.dayofweek + indexTrain.index.hour / 24
        Xtest[:, 0] = indexTest.index.dayofweek            
        indexFeatures += 1
    
    # For each possible horizon
    for hour in range(24):
        # For each possible past day where prices can be included
        for past_day in [1, 2, 3, 7]:

            # We define the corresponding past time indexs 
            pastIndexTrain = pd.to_datetime(indexTrain.loc[:, 'h' + str(hour)].values) - \
                pd.Timedelta(hours=24 * past_day)
            pastIndexTest = pd.to_datetime(indexTest.loc[:, 'h' + str(hour)].values) - \
                pd.Timedelta(hours=24 * past_day)

            # We include feature if feature selection indicates it
            if features['In: Price D-' + str(past_day)]:
                Xtrain[:, indexFeatures] = dfTrain.loc[pastIndexTrain, 'Price']
                Xtest[:, indexFeatures] = dfTest.loc[pastIndexTest, 'Price']
                indexFeatures += 1

    
    # For each possible horizon
    for hour in range(24):
        # For each possible past day where exogeneous can be included
        for past_day in [1, 7]:

            # We define the corresponding past time indexs 
            pastIndexTrain = pd.to_datetime(indexTrain.loc[:, 'h' + str(hour)].values) - \
                pd.Timedelta(hours=24 * past_day)
            pastIndexTest = pd.to_datetime(indexTest.loc[:, 'h' + str(hour)].values) - \
                pd.Timedelta(hours=24 * past_day)

            # For each of the exogenous inputs we include feature if feature selection indicates it
            for exog in range(1, n_exogenous_inputs + 1):
                if features['In: Exog-' + str(exog) + ' D-' + str(past_day)]:
                    Xtrain[:, indexFeatures] = dfTrain.loc[pastIndexTrain, 'Exogenous ' + str(exog)]                    
                    Xtest[:, indexFeatures] = dfTest.loc[pastIndexTest, 'Exogenous ' + str(exog)]
                    indexFeatures += 1

        # For each of the exogenous inputs we include feature if feature selection indicates it
        for exog in range(1, n_exogenous_inputs + 1):
            # Adding exogenous inputs at day D
            if features['In: Exog-' + str(exog) + ' D']:
                futureIndexTrain = pd.to_datetime(indexTrain.loc[:, 'h' + str(hour)].values)
                futureIndexTest = pd.to_datetime(indexTest.loc[:, 'h' + str(hour)].values)

                Xtrain[:, indexFeatures] = dfTrain.loc[futureIndexTrain, 'Exogenous ' + str(exog)]        
                Xtest[:, indexFeatures] = dfTest.loc[futureIndexTest, 'Exogenous ' + str(exog)] 
                indexFeatures += 1

    # Extracting the predicted values Y
    for hour in range(24):
        futureIndexTrain = pd.to_datetime(indexTrain.loc[:, 'h' + str(hour)].values)
        futureIndexTest = pd.to_datetime(indexTest.loc[:, 'h' + str(hour)].values)

        Ytrain[:, hour] = dfTrain.loc[futureIndexTrain, 'Price']        
        Ytest[:, hour] = dfTest.loc[futureIndexTest, 'Price'] 

    # Redefining indexTest to return only the dates at which a prediction is made
    indexTest = indexTest.index


    if shuffle_train:
        nVal = int(percentage_val * Xtrain.shape[0])

        if hyperoptimization:
            # We fixed the random shuffle index so that the validation dataset does not change during the
            # hyperparameter optimization process
            np.random.seed(7)

        # We shuffle the data per week to avoid data contamination
        index = np.arange(Xtrain.shape[0])
        index_week = index[::7]
        np.random.shuffle(index_week)
        index_shuffle = [ind + i for ind in index_week for i in range(7) if ind + i in index]

        Xtrain = Xtrain[index_shuffle]
        Ytrain = Ytrain[index_shuffle]

    else:
        nVal = int(percentage_val * Xtrain.shape[0])
    
    Xval = Xtrain[:nVal]
    Xtrain = Xtrain[nVal:]
    Yval = Ytrain[:nVal]
    Ytrain = Ytrain[nVal:]

    return Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, indexTest

def evaluate_dnn_in_test_dataset(experiment_id, path_datasets_folder=os.path.join('.', 'datasets'), 
                                 path_hyperparameter_folder=os.path.join('.', 'experimental_files'), 
                                 path_recalibration_folder=os.path.join('.', 'experimental_files'), 
                                 nlayers=2, dataset='PJM', years_test=2, shuffle_train=0, 
                                 data_augmentation=0, calibration_window=4, new_recalibration=0, 
                                 begin_test_date=None, end_test_date=None):


    # Checking if provided directory for recalibration exists and if not create it
    if not os.path.exists(path_recalibration_folder):
        os.makedirs(path_recalibration_folder)

    # Defining train and testing data
    df_train, df_test = read_data(dataset=dataset, years_test=years_test, path=path_datasets_folder,
                                  begin_test_date=begin_test_date, end_test_date=end_test_date)
    # Defining unique name to save the forecast

    forecast_file_name = 'DNN_forecast_nl' + str(nlayers) + '_dat' + str(dataset) + \
                         '_YT' + str(years_test) + '_SF' + str(shuffle_train) + \
                         '_DA' * data_augmentation + '_CW' + str(calibration_window) + \
                         '_' + str(experiment_id) + '.csv'

    forecast_file_path = os.path.join(path_recalibration_folder, forecast_file_name)

    # Defining empty forecast array and the real values to be predicted in a more friendly format
    forecast = pd.DataFrame(index=df_test.index[::24], columns=['h' + str(k) for k in range(24)])
    real_values = df_test.loc[:, ['Price']].values.reshape(-1, 24)
    real_values = pd.DataFrame(real_values, index=forecast.index, columns=forecast.columns)

    # If we are not starting a new recalibration but re-starting an old one, we import the
    # existing files and print metrics 
    if not new_recalibration:
        # Import existinf forecasting file
        forecast = pd.read_csv(forecast_file_path, index_col=0)
        forecast.index = pd.to_datetime(forecast.index)

        # Reading dates to still be forecasted by checking NaN values
        forecast_dates = forecast[forecast.isna().any(axis=1)].index

        # If all the dates to be forecasted have already been forecast, we print information
        # and exit the script
        if len(forecast_dates) == 0:

            mae = np.mean(MAE(forecast.values.squeeze(), real_values.values))
            smape = np.mean(sMAPE(forecast.values.squeeze(), real_values.values)) * 100
            print('{} - sMAPE: {:.2f}%  |  MAE: {:.3f}'.format('Final metrics', smape, mae))
        
    else:
        forecast_dates = forecast.index

    model = DNN(experiment_id=experiment_id, path_hyperparameter_folder=path_hyperparameter_folder,
                nlayers=nlayers, dataset=dataset, years_test=years_test, shuffle_train=shuffle_train, 
                data_augmentation=data_augmentation, calibration_window=calibration_window)


    # For loop over the recalibration dates
    for date in forecast_dates:

        # For simulation purposes, we assume that the available data is
        # the data up to current date where the prices of current date are not known
        data_available = pd.concat([df_train, df_test.loc[:date + pd.Timedelta(hours=23), :]], axis=0)

        # We set the real prices for current date to NaN in the dataframe of available data
        data_available.loc[date:date + pd.Timedelta(hours=23), 'Price'] = np.NaN

        # Recalibrating the model with the most up-to-date available data and making a prediction
        # for the next day
        Yp = model.recalibrate_and_forecast_next_day(
            df=data_available, next_day_date=date, calibration_window=calibration_window, 
            shuffle_train=shuffle_train, data_augmentation=data_augmentation)

        # Saving the current prediction
        forecast.loc[date, :] = Yp

        # Computing metrics up-to-current-date
        mae = np.mean(MAE(forecast.loc[:date].values.squeeze(), real_values.loc[:date].values)) 
        smape = np.mean(sMAPE(forecast.loc[:date].values.squeeze(), real_values.loc[:date].values)) * 100

        # Pringint information
        print('{} - sMAPE: {:.2f}%  |  MAE: {:.3f}'.format(str(date)[:10], smape, mae))

        # Saving forecast
        forecast.to_csv(forecast_file_path)

