import pandas as pd
import numpy as np
import pickle as pc
import os

from epftoolbox.models import DNN, build_and_split_XYs
from epftoolbox.data import scaling
from epftoolbox.data import read_data
from epftoolbox.evaluation import MAE, sMAPE

class DNNRecalibration(object):

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

    def recalibrate(self, Xtrain, Ytrain, Xval, Yval, Xtest):

        # Initialize model
        neurons = [int(self.best_hyperparameters['neurons' + str(k)]) for k in range(1, self.nlayers + 1)
                   if int(self.best_hyperparameters['neurons' + str(k)]) >= 50]
            
        np.random.seed(int(self.best_hyperparameters['seed']))

        model = DNN(neurons=neurons, Nfeatures=Xtrain.shape[-1], 
                         dropout=self.best_hyperparameters['dropout'], BN=self.best_hyperparameters['BN'], 
                         lr=self.best_hyperparameters['lr'], printOut=False,
                         optimizer='adam', activation=self.best_hyperparameters['activation'],
                         maxEpochsWOImprovement=20, scaler=self.scaler, loss='mae',
                         regularization=self.best_hyperparameters['reg'], 
                         lambdaReg=self.best_hyperparameters['lambdal1'],
                         initializer=self.best_hyperparameters['init'])

        model.fit(Xtrain, Ytrain, Xval, Yval)
        
        # Predicting the current date using recalibrated neural network
        Yp = model.predict(Xtest).squeeze()
        if self.best_hyperparameters['scaleY'] in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
            Yp = self.scaler.inverse_transform(Yp.reshape(1, -1))

        model.clear_session()

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

    model = DNNRecalibration(
        experiment_id=experiment_id, path_hyperparameter_folder=path_hyperparameter_folder, nlayers=nlayers, 
        dataset=dataset, years_test=years_test, shuffle_train=shuffle_train, 
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

