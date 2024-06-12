"""
Classes and functions to implement the DNN model for electricity price forecasting. The module does not
include the hyperparameter optimization functions; these are included in the module
_dnn_hyperopt.py
"""

# Author: Jesus Lago

# License: AGPL-3.0 License

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

    """Basic DNN model based on keras and tensorflow. 
    
    The model can be used standalone to train and predict a DNN using its fit/predict methods.
    However, it is intended to be used within the :class:`hyperparameter_optimizer` method
    and the :class:`DNN` class. The former obtains a set of best hyperparameter using the :class:`DNNModel` class. 
    The latter employes the set of best hyperparameters to recalibrate a :class:`DNNModel` object
    and make predictions.
    
    Parameters
    ----------
    neurons : list
        List containing the number of neurons in each hidden layer. E.g. if ``len(neurons)`` is 2,
        the DNN model has an input layer of size ``n_features``, two hidden layers, and an output 
        layer of size ``outputShape``.
    n_features : int
        Number of input features in the model. This number defines the size of the input layer.
    outputShape : int, optional
        Default number of output neurons. It is 24 as it is the default in most day-ahead markets.
    dropout : float, optional
        Number between [0, 1] that selects the percentage of dropout. A value of 0 indicates
        no dropout.
    batch_normalization : bool, optional
        Boolean that selects whether batch normalization is considered.
    lr : float, optional
        Learning rate for optimizer algorithm. If none provided, the default one is employed
        (see the `keras documentation <https://keras.io/>`_ for the default learning rates of each algorithm).
    verbose : bool, optional
        Boolean that controls the logs. If set to true, a minimum amount of information is 
        displayed.
    epochs_early_stopping : int, optional
        Number of epochs used in early stopping to stop training. When no improvement is observed
        in the validation dataset after ``epochs_early_stopping`` epochs, the training stops.
    scaler : :class:`epftoolbox.data.DataScaler`, optional
        Scaler object to invert-scale the output of the neural network if the neural network
        is trained with scaled outputs.
    loss : str, optional
        Loss to be used when training the neural network. Any of the regression losses defined in 
        keras can be used.
    optimizer : str, optional
        Name of the optimizer when training the DNN. See the `keras documentation <https://keras.io/>`_ 
        for a list of optimizers.
    activation : str, optional
        Name of the activation function in the hidden layers. See the `keras documentation <https://keras.io/>`_ for a list
        of activation function.
    initializer : str, optional
        Name of the initializer function for the weights of the neural network. See the 
        `keras documentation <https://keras.io/>`_ for a list of initializer functions.
    regularization : None, optional
        Name of the regularization technique. It can can have three values ``'l2'`` for l2-norm
        regularization, ``'l1'`` for l1-norm regularization, or ``None`` for no regularization .
    lambda_reg : int, optional
        The weight for regulization if ``regularization`` is ``'l2'`` or ``'l1'``.
    """


    
    def __init__(self, neurons, n_features, outputShape=24, dropout=0, batch_normalization=False, lr=None,
                 verbose=False, epochs_early_stopping=40, scaler=None, loss='mae',
                 optimizer='adam', activation='relu', initializer='glorot_uniform',
                 regularization=None, lambda_reg=0):

        self.neurons = neurons
        self.dropout = dropout

        if self.dropout > 1 or self.dropout < 0:
            raise ValueError('Dropout parameter must be between 0 and 1')

        self.batch_normalization = batch_normalization
        self.verbose = verbose
        self.epochs_early_stopping = epochs_early_stopping
        self.n_features = n_features
        self.scaler = scaler
        self.outputShape = outputShape
        self.activation = activation
        self.initializer = initializer
        self.regularization = regularization
        self.lambda_reg = lambda_reg

        self.model = self._build_model()

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

    def _reg(self, lambda_reg):
        """Internal method to build an l1 or l2 regularizer for the DNN
        
        Parameters
        ----------
        lambda_reg : float
            Weight of the regularization
        
        Returns
        -------
        tensorflow.keras.regularizers.L1L2
            The regularizer object
        """
        if self.regularization == 'l2':
            return l2(lambda_reg)
        if self.regularization == 'l1':
            return l1(lambda_reg)
        else:
            return None

    def _build_model(self):
        """Internal method that defines the structure of the DNN
        
        Returns
        -------
        tensorflow.keras.models.Model
            A neural network model using keras and tensorflow
        """
        inputShape = (None, self.n_features)

        past_data = Input(batch_shape=inputShape)

        past_Dense = past_data
        if self.activation == 'selu':
            self.initializer = 'lecun_normal'

        for k, neurons in enumerate(self.neurons):

            if self.activation == 'LeakyReLU':
                past_Dense = Dense(neurons, activation='linear', batch_input_shape=inputShape,
                                   kernel_initializer=self.initializer,
                                   kernel_regularizer=self._reg(self.lambda_reg))(past_Dense)
                past_Dense = LeakyReLU(alpha=.001)(past_Dense)

            elif self.activation == 'PReLU':
                past_Dense = Dense(neurons, activation='linear', batch_input_shape=inputShape,
                                   kernel_initializer=self.initializer,
                                   kernel_regularizer=self._reg(self.lambda_reg))(past_Dense)
                past_Dense = PReLU()(past_Dense)

            else:
                past_Dense = Dense(neurons, activation=self.activation,
                                   batch_input_shape=inputShape,
                                   kernel_initializer=self.initializer,
                                   kernel_regularizer=self._reg(self.lambda_reg))(past_Dense)

            if self.batch_normalization:
                past_Dense = BatchNormalization()(past_Dense)

            if self.dropout > 0:
                if self.activation == 'selu':
                    past_Dense = AlphaDropout(self.dropout)(past_Dense)
                else:
                    past_Dense = Dropout(self.dropout)(past_Dense)

        output_layer = Dense(self.outputShape, kernel_initializer=self.initializer,
                             kernel_regularizer=self._reg(self.lambda_reg))(past_Dense)
        model = Model(inputs=[past_data], outputs=[output_layer])

        return model

    def _obtain_metrics(self, X, Y):
        """Internal method to update the metrics used to train the network
        
        Parameters
        ----------
        X : numpy.array
            Input array for evaluating the model
        Y : numpy.array
            Output array for evaluating the model
        
        Returns
        -------
        list
            A list containing the metric based on the loss of the neural network and a second metric
            representing the MAE of the DNN
        """
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

    def _display_info_training(self, bestError, bestMAE, countNoImprovement):
        """Internal method that displays useful information during training
        
        Parameters
        ----------
        bestError : float
            Loss of the neural network in the validation dataset
        bestMAE : float
            MAE of the neural network in the validation dataset
        countNoImprovement : int
            Number of epochs in the validation dataset without improvements
        """
        print(" Best error:\t\t{:.1e}".format(bestError))
        print(" Best MAE:\t\t{:.2f}".format(bestMAE))                
        print(" Epochs without improvement:\t{}\n".format(countNoImprovement))


    def fit(self, trainX, trainY, valX, valY):
        """Method to estimate the DNN model.
        
        Parameters
        ----------
        trainX : numpy.array
            Inputs fo the training dataset.
        trainY : numpy.array
            Outputs fo the training dataset.
        valX : numpy.array
            Inputs fo the validation dataset used for early-stopping.
        valY : numpy.array
            Outputs fo the validation dataset used for early-stopping.
        """

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
            if self.verbose:
                print("\nEpoch {} of {} took {:.3f}s".format(epoch + 1, 1000,
                                                             time.time() - start_time))

            # Calculating relevant metrics to perform early-stopping
            valError, valMAE = self._obtain_metrics(valX, valY)

            # Early-stopping
            # Checking if current validation metrics are better than best so far metrics.
            # If the network does not improve, we stop.
            # If it improves, the optimal weights are saved
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

            if countNoImprovement >= self.epochs_early_stopping:
                if self.verbose:
                    self._display_info_training(bestError, bestMAE, countNoImprovement)
                break

            # Displaying final information
            if self.verbose:
                self._display_info_training(bestError, bestMAE, countNoImprovement)

        # After early-stopping, the best weights are set in the model
        self.model.set_weights(bestWeights)

    def predict(self, X):
        """Method to make a prediction after the DNN is trained.
        
        Parameters
        ----------
        X : numpy.array
            Input to the DNN. It has to be of size *[n, n_features]* where *n* can be any 
            integer, and *n_features* is the attribute of the DNN representing the number of
            input features.
        
        Returns
        -------
        numpy.array
            Output of the DNN after making the prediction.
        """

        Ybar = self.model.predict(X, verbose=0)
        return Ybar

    def clear_session(self):
        """Method to clear the tensorflow session. 

        It is used in the :class:`DNN` class during recalibration to avoid RAM memory leakages.
        In particular, if the DNN is retrained continuosly, at each step tensorflow slightly increases 
        the total RAM usage.

        """

        K.clear_session()


class DNN(object):

    """DNN for electricity price forecasting. 
    
    It considers a set of best hyperparameters, it recalibrates a :class:`DNNModel` based on
    these hyperparameters, and makes new predictions.
    
    The difference w.r.t. the :class:`DNNModel` class lies on the functionality. The
    :class:`DNNModel` class provides a simple interface to build a keras DNN model which
    is limited to fit and predict methods. This class extends the functionality by
    providing an interface to extract the best set of hyperparameters, and to perform recalibration
    before every prediction.
    
    Note that before using this class, a hyperparameter optimization run must be done using the
    :class:`hyperparameter_optimizer` function. Such hyperparameter optimization must be done
    using the same parameters: ``nlayers``, ``dataset``, ``years_test``, ``shuffle_train``, 
    ``data_augmentation``, and ``calibration_window``
    
    An example on how to use this class is provided :ref:`here<dnnex3>`.

    Parameters
    ----------
    experiment_id : str
        Unique identifier to read the trials file. In particular, every hyperparameter optimization 
        set has an unique identifier associated with. See :class:`hyperparameter_optimizer` for further
        details
    path_hyperparameter_folder : str, optional
        Path of the folder containing the trials file with the optimal hyperparameters
    nlayers : int, optional
        Number of layers of the DNN model
    dataset : str, optional
        Name of the dataset/market under study. If it is one one of the standard markets, 
        i.e. ``"PJM"``, ``"NP"``, ``"BE"``, ``"FR"``, or ``"DE"``, the dataset is automatically downloaded. If the name
        is different, a dataset with a csv format should be place in the ``path_datasets_folder``.
    years_test : int, optional
        Number of years (a year is 364 days) in the test dataset. This is necesary to extract the
        correct hyperparameter trials file
    shuffle_train : bool, optional
        Boolean that selects whether the validation and training datasets were shuffled when
        performing the hyperparameter optimization. Note that it does not select whether
        shuffling is used for recalibration as for recalibration the validation and the
        training datasets are always shuffled.
    data_augmentation : bool, optional
        Boolean that selects whether a data augmentation technique for electricity price forecasting
        is employed
    calibration_window : int, optional
        Number of days used in the training/validation dataset for recalibration
    
    """

    def __init__(self, experiment_id, path_hyperparameter_folder=os.path.join('.', 'experimental_files'), 
                 nlayers=2, dataset='PJM', years_test=2, shuffle_train=1, data_augmentation=0, 
                 calibration_window=4):

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
        self._read_best_hyperapameters()

    def _read_best_hyperapameters(self):
        """Internal method that reads and extracts the subset of optimal hyperparameters

        The file that is read depends on the provided input parameters to the class
        """

        # Defining the trials file name used to extract the optimal hyperparameters
        trials_file_name = \
            'DNN_hyperparameters_nl' + str(self.nlayers) + \
            '_dat' + str(self.dataset) + '_YT' + str(self.years_test) + \
            '_SF' * (self.shuffle_train) + '_DA' * (self.data_augmentation) + \
            '_CW' + str(self.calibration_window) + '_' + str(self.experiment_id)

        trials_file_path = os.path.join(self.path_hyperparameter_folder, trials_file_name)

        # Reading and extracting the best hyperparameters
        trials = pc.load(open(trials_file_path, "rb"))
        
        self.best_hyperparameters = format_best_trial(trials.best_trial)

    def _regularize_data(self, Xtrain, Xval, Xtest, Ytrain, Yval):
        """Internal method to scale the input/outputs of the DNN.

        It scales the inputs of the training, validation, and test datasets
        and the outputs of the training and validation datasets.
        
        Parameters
        ----------
        Xtrain : numpy.array
            Input of the training dataset
        Xval : numpy.array
            Input of the validation dataset
        Xtest : numpy.array
            Input of the test dataset
        Ytrain : numpy.array
            Output of the training dataset
        Yval : numpy.array
            Output of the validation dataset
        
        Returns
        -------
        list
            List containing the five arrays but scaled
        """

        # If required, datasets are scaled
        if self.best_hyperparameters['scaleX'] in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
            [Xtrain, Xval, Xtest], _ = scaling([Xtrain, Xval, Xtest], self.best_hyperparameters['scaleX'])

        if self.best_hyperparameters['scaleY'] in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
            [Ytrain, Yval], self.scaler = scaling([Ytrain, Yval], self.best_hyperparameters['scaleY'])
        else:
            self.scaler = None

        return Xtrain, Xval, Xtest, Ytrain, Yval

    def recalibrate(self, Xtrain, Ytrain, Xval, Yval):
        """Method that recalibrates the model.

        The method receives the training and validation dataset, and trains a :class:`DNNModel` model
        using the set of optimal hyperparameters that are found in ``path_hyperparameter_folder`` and
        that are defined by the class attributes: ``experiment_id``, ``nlayers``, ``dataset``, 
        ``years_test``, ``shuffle_train``, ``data_augmentation``, and ``calibration_window``
        
        Parameters
        ----------
        Xtrain : numpy.array
            Input of the training dataset
        Xval : numpy.array
            Input of the validation dataset
        Ytrain : numpy.array
            Output of the training dataset
        Yval : numpy.array
            Output of the validation dataset
        """

        # Initialize model
        neurons = [int(self.best_hyperparameters['neurons' + str(k)]) for k in range(1, self.nlayers + 1)
                   if int(self.best_hyperparameters['neurons' + str(k)]) >= 50]
            
        np.random.seed(int(self.best_hyperparameters['seed']))

        self.model = DNNModel(neurons=neurons, n_features=Xtrain.shape[-1], 
                              dropout=self.best_hyperparameters['dropout'], 
                              batch_normalization=self.best_hyperparameters['batch_normalization'], 
                              lr=self.best_hyperparameters['lr'], verbose=False,
                              optimizer='adam', activation=self.best_hyperparameters['activation'],
                              epochs_early_stopping=20, scaler=self.scaler, loss='mae',
                              regularization=self.best_hyperparameters['reg'], 
                              lambda_reg=self.best_hyperparameters['lambdal1'],
                              initializer=self.best_hyperparameters['init'])

        self.model.fit(Xtrain, Ytrain, Xval, Yval)
        
    def recalibrate_predict(self, Xtrain, Ytrain, Xval, Yval, Xtest):
        """Method that first recalibrates the DNN model and then makes a prediction.

        The method receives the training and validation dataset, and trains a :class:`DNNModel` model
        using the set of optimal hyperparameters. Then, using the inputs of the test dataset,
        it makes a new prediction.
        
        Parameters
        ----------
        Xtrain : numpy.array
            Input of the training dataset
        Xval : numpy.array
            Input of the validation dataset
        Xtest : numpy.array
            Input of the test dataset
        Ytrain : numpy.array
            Output of the training dataset
        Yval : numpy.array
            Output of the validation dataset
        
        Returns
        -------
        numpy.array
            An array containing the predictions in the test dataset
        """
        self.recalibrate(Xtrain=Xtrain, Ytrain=Ytrain, Xval=Xval, Yval=Yval)        
        Yp = self.predict(X=Xtest)

        self.model.clear_session()

        return Yp

    def predict(self, X):
        """Method that makes a prediction using some given inputs
        
        Parameters
        ----------
        X : numpy.array
            Input of the model
        
        Returns
        -------
        numpy.array
            An array containing the predictions
        """

        # Predicting the current date using a recalibrated DNN
        Yp = self.model.predict(X).squeeze()
        if self.best_hyperparameters['scaleY'] in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
            Yp = self.scaler.inverse_transform(Yp.reshape(1, -1))

        return Yp

    def recalibrate_and_forecast_next_day(self, df, next_day_date):
        """Method that builds an easy-to-use interface for daily recalibration and forecasting of the DNN model
        
        The method receives a pandas dataframe ``df`` and a day ``next_day_date``. Then, it 
        recalibrates the model using data up to the day before ``next_day_date`` and makes a prediction
        for day ``next_day_date``.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe of historical data containing prices and N exogenous inputs. The index of the 
            dataframe should be dates with hourly frequency. The columns should have the following 
            names ['Price', 'Exogenous 1', 'Exogenous 2', ...., 'Exogenous N']
        next_day_date : TYPE
            Date of the day-ahead
        
        Returns
        -------
        numpy.array
            An array containing the predictions in the provided date
        
        """

        # We define the new training dataset considering the last calibration_window years of data 
        df_train = df.loc[:next_day_date - pd.Timedelta(hours=1)]
        df_train = df_train.loc[next_day_date - pd.Timedelta(hours=self.calibration_window * 364 * 24):]

        # We define the test dataset as the next day (they day of interest) plus the last two weeks
        # in order to be able to build the necessary input features.
        df_test = df.loc[next_day_date - pd.Timedelta(weeks=2):, :]

        # Generating training, validation, and test input and outpus. For the test dataset,
        # even though the dataframe contains 15 days of data (next day + last 2 weeks),
        # we provide as parameter the date of interest so that Xtest and Ytest only reflect that
        Xtrain, Ytrain, Xval, Yval, Xtest, _, _ = \
            _build_and_split_XYs(dfTrain=df_train, features=self.best_hyperparameters, 
                                shuffle_train=True, dfTest=df_test, date_test=next_day_date,
                                data_augmentation=self.data_augmentation, 
                                n_exogenous_inputs=len(df_train.columns) - 1)

        # Normalizing the input and outputs if needed
        Xtrain, Xval, Xtest, Ytrain, Yval = \
            self._regularize_data(Xtrain=Xtrain, Xval=Xval, Xtest=Xtest, Ytrain=Ytrain, Yval=Yval)

        # Recalibrating the neural network and extracting the prediction
        Yp = self.recalibrate_predict(Xtrain=Xtrain, Ytrain=Ytrain, Xval=Xval, Yval=Yval, Xtest=Xtest)

        return Yp

def evaluate_dnn_in_test_dataset(experiment_id, path_datasets_folder=os.path.join('.', 'datasets'), 
                                 path_hyperparameter_folder=os.path.join('.', 'experimental_files'), 
                                 path_recalibration_folder=os.path.join('.', 'experimental_files'), 
                                 nlayers=2, dataset='PJM', years_test=2, shuffle_train=True, 
                                 data_augmentation=0, calibration_window=4, new_recalibration=False, 
                                 begin_test_date=None, end_test_date=None):
    """Function for easy evaluation of the DNN model in a test dataset using daily recalibration. 
    
    The test dataset is defined by a market name and the test dates dates. The function
    generates the test and training datasets, and evaluates a DNN model considering daily recalibration
    and an optimal set of hyperparameters. 
    
    Note that before using this class, a hyperparameter optimization run must be done using the
    :class:`hyperparameter_optimizer` function. Moreover, the hyperparameter optimization must be done
    using the same parameters: ``nlayers``, ``dataset``, ``shuffle_train``, 
    ``data_augmentation``, ``calibration_window``, and either the ``years_test`` or the same
    ``begin_test_date``/``end_test_date``
    
    An example on how to use this function is provided :ref:`here<dnnex2>`.

    Parameters
    ----------
    experiment_id : str
        Unique identifier to read the trials file. In particular, every hyperparameter optimization 
        set has an unique identifier associated with. See :class:`hyperparameter_optimizer` for further
        details
    path_datasets_folder : str, optional
        Path where the datasets are stored or, if they do not exist yet, the path where the datasets 
        are to be stored
    path_hyperparameter_folder : str, optional
        Path of the folder containing the trials file with the optimal hyperparameters
    path_recalibration_folder : str, optional
        Path to save the forecast of the test dataset
    nlayers : int, optional
        Number of hidden layers in the neural network
    dataset : str, optional
        Name of the dataset/market under study. If it is one one of the standard markets, 
        i.e. ``"PJM"``, ``"NP"``, ``"BE"``, ``"FR"``, or ``"DE"``, the dataset is automatically downloaded. If the name
        is different, a dataset with a csv format should be place in the ``path_datasets_folder``.
    years_test : int, optional
        Number of years (a year is 364 days) in the test dataset. It is only used if 
        the arguments begin_test_date and end_test_date are not provided.
    begin_test_date : datetime/str, optional
        Optional parameter to select the test dataset. Used in combination with the argument
        end_test_date. If either of them is not provided, the test dataset is built using the 
        years_test argument. begin_test_date should either be a string with the following 
        format d/m/Y H:M, or a datetime object
    end_test_date : datetime/str, optional
        Optional parameter to select the test dataset. Used in combination with the argument
        begin_test_date. If either of them is not provided, the test dataset is built using the 
        years_test argument. end_test_date should either be a string with the following 
        format d/m/Y H:M, or a datetime object       
    shuffle_train : bool, optional
        Boolean that selects whether the validation and training datasets were shuffled when
        performing the hyperparameter optimization. Note that it does not select whether
        shuffling is used for recalibration as for recalibration the validation and the
        training datasets are always shuffled.
    data_augmentation : bool, optional
        Boolean that selects whether a data augmentation technique for electricity price forecasting
        is employed
    calibration_window : int, optional
        Number of days used in the training/validation dataset for recalibration
    new_recalibration : bool, optional
        Boolean that selects whether a new recalibration is performed or the function re-starts an old one.
        To restart an old one, the .csv file with the forecast must exist in the 
        ``path_recalibration_folder`` folder 
    
    Returns
    -------
    pandas.DataFrame
        A dataframe with all the predictions in the test dataset. The dataframe is also
        written to the folder ``path_recalibration_folder``
    """


    # Checking if provided directory for recalibration exists and if not create it
    if not os.path.exists(path_recalibration_folder):
        os.makedirs(path_recalibration_folder)

    # Defining train and testing data
    df_train, df_test = read_data(dataset=dataset, years_test=years_test, path=path_datasets_folder,
                                  begin_test_date=begin_test_date, end_test_date=end_test_date)
    # Defining unique name to save the forecast

    forecast_file_name = 'DNN_forecast_nl' + str(nlayers) + '_dat' + str(dataset) + \
                         '_YT' + str(years_test) + '_SFH' + str(shuffle_train) + \
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
                nlayers=nlayers, dataset=dataset, years_test=years_test, 
                shuffle_train=shuffle_train, data_augmentation=data_augmentation, 
                calibration_window=calibration_window)


    # For loop over the recalibration dates
    for date in forecast_dates:

        # For simulation purposes, we assume that the available data is
        # the data up to current date where the prices of current date are not known
        data_available = pd.concat([df_train, df_test.loc[:date + pd.Timedelta(hours=23), :]], axis=0)

        # We set the real prices for current date to NaN in the dataframe of available data
        data_available.loc[date:date + pd.Timedelta(hours=23), 'Price'] = np.NaN

        # Recalibrating the model with the most up-to-date available data and making a prediction
        # for the next day
        Yp = model.recalibrate_and_forecast_next_day(df=data_available, next_day_date=date)

        # Saving the current prediction
        forecast.loc[date, :] = Yp

        # Computing metrics up-to-current-date
        mae = np.mean(MAE(forecast.loc[:date].values.squeeze(), real_values.loc[:date].values)) 
        smape = np.mean(sMAPE(forecast.loc[:date].values.squeeze(), real_values.loc[:date].values)) * 100

        # Pringint information
        print('{} - sMAPE: {:.2f}%  |  MAE: {:.3f}'.format(str(date)[:10], smape, mae))

        # Saving forecast
        forecast.to_csv(forecast_file_path)

    return forecast

def format_best_trial(best_trial):
    """Function to format the best_trial object to a python dictionary.

    The trials file used to save the hyperparameter optimization contains a lot of information. This
    function extracts from the file the values of the optimal hyperparameters and stores them in a 
    python dictionary. It receives as input a ``trial`` from a ``Trials`` object. 

    Parameters
    ----------
    best_trial : dict
        A trial dictionary as extracted from the ``Trials`` object generated by the 
        :class:`hyperparameter_optimizer`function. 
    
    Returns
    -------
    formatted_hyperparameters : Dict
        Formatted dictionary with the optimal hyperparameters
    """

    # Extracting unformatted hyperparameters
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

def _build_and_split_XYs(dfTrain, features, shuffle_train, n_exogenous_inputs, dfTest=None, percentage_val=0.25,
                        date_test=None, hyperoptimization=False, data_augmentation=False):
    """Method to buil the X,Y pairs for training/test DNN models using dataframes and a list of
    the selected inputs
    
    Parameters
    ----------
    dfTrain : pandas.DataFrame
        Pandas dataframe containing the training data
    features : dict
        Dictionary that define the selected input features. The dictionary is based on the results
        of a hyperparameter/feature optimization run using the :class:`hyperparameter_optimizer`function
    shuffle_train : bool
        If true, the validation and training datasets are shuffled
    n_exogenous_inputs : int
        Number of exogenous inputs, i.e. inputs besides historical prices
    dfTest : pandas.DataFrame
        Pandas dataframe containing the test data
    percentage_val : TYPE, optional
        Percentage of data to be used for validation
    date_test : None, optional
        If given, then the test dataset is only built for that date
    hyperoptimization : bool, optional
        Description
    data_augmentation : bool, optional
        Description
    
    Returns
    -------
    list
        A list ``[Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, indexTest]`` that contains the X, Y pairs 
        for training, validation, and testing, as well as the date index of the test dataset
    """

    # Checking that the first index in the dataframes corresponds with the hour 00:00 
    if dfTrain.index[0].hour != 0 or dfTest.index[0].hour != 0:
        print('Problem with the index')

        
    # Calculating the number of input features
    n_features = features['In: Day'] + \
        24 * features['In: Price D-1'] + 24 * features['In: Price D-2'] + \
        24 * features['In: Price D-3'] + 24 * features['In: Price D-7']

    for n_ex in range(1, n_exogenous_inputs + 1):

        n_features += 24 * features['In: Exog-' + str(n_ex) + ' D'] + \
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
        predDatesTrain = indexTrain.round('1h')
    else:
        predDatesTrain = indexTrain.round('1h')[::24]            
            
    predDatesTest = indexTest.round('1h')[::24]

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
    Xtrain = np.zeros([indexTrain.shape[0], n_features])
    Xtest = np.zeros([indexTest.shape[0], n_features])
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
    nTrain = Xtrain.shape[0] - nVal # complements nVal
    
    Xval = Xtrain[nTrain:] # last nVal obs
    Xtrain = Xtrain[:nTrain] # first nTrain obs
    Yval = Ytrain[nTrain:]
    Ytrain = Ytrain[:nTrain]

    return Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, indexTest
