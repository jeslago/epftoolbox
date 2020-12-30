"""
Classes and functions to perform hyperparameter and feature selection for the DNN model 
for electricity price forecasting
"""

# Author: Jesus Lago

# License: AGPL-3.0 License

import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import pickle as pc
from datetime import datetime
from epftoolbox.models import DNNModel
from epftoolbox.models._dnn import _build_and_split_XYs
from epftoolbox.data import scaling
from epftoolbox.data import read_data
from epftoolbox.evaluation import MAE, sMAPE
from functools import partial
import os

def _build_space(nlayer, data_augmentation, n_exogenous_inputs):
    """Function that generates the hyperparameter/feature search space 
    
    Parameters
    ----------
    nlayer : int
        Number of layers of the DNN model
    data_augmentation : bool
        Boolean that selects whether augmenting data is considered
    n_exogenous_inputs : int
        Number of exogenous inputs in the market under study
    
    Returns
    -------
    dict
        Dictionary defining the search space
    """

    # Defining the hyperparameter space. First the neural net hyperparameters,
    # later the input features
    space = {
        'batch_normalization': hp.choice('batch_normalization', [False, True]),
        'dropout': hp.uniform('dropout', 0, 1),
        'lr': hp.loguniform('lr', np.log(5e-4), np.log(0.1)),
        'seed': hp.quniform('seed', 1, 1000, 1),
        'neurons1': hp.quniform('neurons1', 50, 500, 1),
        'activation': hp.choice('activation', ["relu", "softplus", "tanh", 'selu',
                                'LeakyReLU', 'PReLU', 'sigmoid']),
        'init': hp.choice('init', ['Orthogonal', 'lecun_uniform', 'glorot_uniform',
            'glorot_normal', 'he_uniform', 'he_normal']),
        'reg': hp.choice('reg', [
            {'val': None, 'lambda': 0},
            {'val': 'l1', 'lambda': hp.loguniform('lambdal1', np.log(1e-5), np.log(1))}]),
        'scaleX': hp.choice('scaleX', ['No', 'Norm', 'Norm1', 'Std',
                                                   'Median', 'Invariant']),
        'scaleY': hp.choice('scaleY', ['No', 'Norm', 'Norm1', 'Std',
                                                   'Median', 'Invariant'])        
    }

    if nlayer >= 2:
        space['neurons2'] = hp.quniform('neurons2', 25, 400, 1)
    if nlayer >= 3:
        space['neurons3'] = hp.quniform('neurons3', 25, 300, 1)
    if nlayer >= 4:
        space['neurons4'] = hp.quniform('neurons4', 25, 200, 1)
    if nlayer >= 5:
        space['neurons5'] = hp.quniform('neurons5', 25, 200, 1)


    # Defining the possible input features as hyperparameters
    space['In: Day'] = hp.choice('In: Day', [False, True])
    space['In: Price D-1'] = hp.choice('In: Price D-1', [False, True])
    space['In: Price D-2'] = hp.choice('In: Price D-2', [False, True])
    space['In: Price D-3'] = hp.choice('In: Price D-3', [False, True])        
    space['In: Price D-7'] = hp.choice('In: Price D-7', [False, True])

    for n_ex in range(1, n_exogenous_inputs + 1):
        space['In: Exog-' + str(n_ex) + ' D'] = hp.choice('In: Exog-' + str(n_ex) + ' D', [False, True])
        space['In: Exog-' + str(n_ex) + ' D-1'] = hp.choice('In: Exog-' + str(n_ex) + ' D-1', [False, True])
        space['In: Exog-' + str(n_ex) + ' D-7'] = hp.choice('In: Exog-' + str(n_ex) + ' D-7', [False, True])
    
    return space


def _hyperopt_objective(hyperparameters, trials, trials_file_path, max_evals, nlayers, dfTrain, dfTest, 
                        shuffle_train, dataset, data_augmentation, 
                        calibration_window, n_exogenous_inputs):
    """Function that defines the hyperparameter optimization objective/loss
    
    This function receives as input a set of hyperparameters, trains a DNN using them,
    and returns the performance of the DNN for the selected hyperparameters in a validation
    dataset

    Parameters
    ----------
    hyperparameters : dict
        A dictionary provided by hyperopt indicating whether each hyperparameter/feature is selected
    trials : hyperopt.Trials
        The trials object that stores the hyperparameter optimization runs
    trials_file_path : str
        The path to store the trials object
    max_evals : int
        Maximum number of iterations for hyperparameter optimization
    nlayers : int
        Number of layers in the DNN model
    dfTrain : pandas.DataFrame
        Dataframe containing the training data
    dfTrain : pandas.DataFrame
        Dataframe containing the testing data
    shuffle_train : bool
        Boolean that selects whether the training and validation datasets are shuffled
    dataset : TYPE
        Description
    data_augmentation : TYPE
        Description
    calibration_window : TYPE
        Description
    n_exogenous_inputs : TYPE
        Description
    
    Returns
    -------
    dict
        A dictionary summarizing the result of the hyperparameter run
    """

    # Re-defining the training dataset based on the calibration window. The calibration window
    # can be given as an external parameter. If the value 0 is given, the calibration window
    # is included as a hyperparameter to optimize
    dfTrain_cw = dfTrain.loc[dfTrain.index[-1] - pd.Timedelta(weeks=52) * calibration_window +
                             pd.Timedelta(hours=1):]

    # Saving hyperoptimization state and printing message
    pc.dump(trials, open(trials_file_path, "wb"))
    if trials.losses()[0] is not None:

        MAEVal = trials.best_trial['result']['MAE Val']
        MAETest = trials.best_trial['result']['MAE Test']

        sMAPEVal = trials.best_trial['result']['sMAPE Val']
        sMAPETest = trials.best_trial['result']['sMAPE Test']
        
        print('\n\nTested {}/{} iterations.'.format(len(trials.losses()) - 1,
              max_evals))

        print('Best MAE - Validation Dataset')            
        print("  MAE: {:.1f} | sMAPE: {:.2f} %".format(MAEVal, sMAPEVal))
        print('\nBest MAE - Test Dataset')
        print("  MAE: {:.1f} | sMAPE: {:.2f} %".format(MAETest, sMAPETest))

    # Defining X,Y datasets
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, indexTest = \
        _build_and_split_XYs(dfTrain=dfTrain_cw, dfTest=dfTest, features=hyperparameters, 
                          shuffle_train=shuffle_train, hyperoptimization=True,
                          data_augmentation=data_augmentation, n_exogenous_inputs=n_exogenous_inputs)
    
    # If required, datasets are scaled
    if hyperparameters['scaleX'] in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
        [Xtrain, Xval, Xtest], _ = scaling([Xtrain, Xval, Xtest], hyperparameters['scaleX'])

    if hyperparameters['scaleY'] in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
        [Ytrain, Yval], scaler = scaling([Ytrain, Yval], hyperparameters['scaleY'])
    else:
        scaler = None

    neurons = [int(hyperparameters['neurons' + str(k)]) for k in range(1, nlayers + 1)
               if int(hyperparameters['neurons' + str(k)]) >= 50]
        
    np.random.seed(int(hyperparameters['seed']))

    # Initialize model
    forecaster = DNNModel(neurons=neurons, n_features=Xtrain.shape[-1], 
                     dropout=hyperparameters['dropout'], batch_normalization=hyperparameters['batch_normalization'], 
                     lr=hyperparameters['lr'], verbose=False,
                     optimizer='adam', activation=hyperparameters['activation'],
                     epochs_early_stopping=20, scaler=scaler, loss='mae',
                     regularization=hyperparameters['reg']['val'], 
                     lambda_reg=hyperparameters['reg']['lambda'],
                     initializer=hyperparameters['init'])

    forecaster.fit(Xtrain, Ytrain, Xval, Yval)

    Yp = forecaster.predict(Xval).squeeze()
    if hyperparameters['scaleY'] in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
        Yval = scaler.inverse_transform(Yval)
        Yp = scaler.inverse_transform(Yp)

    mae_validation = np.mean(MAE(Yval, Yp))
    smape_validation = np.mean(sMAPE(Yval, Yp)) * 100

    # If required, datasets are normalized
    Yp = forecaster.predict(Xtest).squeeze()
    if hyperparameters['scaleY'] in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
        Yp = scaler.inverse_transform(Yp).squeeze()

    maeTest = np.mean(MAE(Ytest, Yp)) 
    smape_test = np.mean(sMAPE(Ytest, Yp)) * 100

    # The test dataset is returned for directly evaluating the models without recalibration
    # while performing hyperopt. However, the hyperparameter search is performed using a validation
    # dataset
    return_values = {'loss': mae_validation, 'MAE Val': mae_validation, 'MAE Test': maeTest,
                     'sMAPE Val': smape_validation, 'sMAPE Test': smape_test, 
                     'status': STATUS_OK}
                          
    return return_values

def hyperparameter_optimizer(path_datasets_folder=os.path.join('.', 'datasets'), 
                             path_hyperparameters_folder=os.path.join('.', 'experimental_files'), 
                             new_hyperopt=1, max_evals=1500, nlayers=2, dataset='PJM', years_test=2, 
                             calibration_window=4, shuffle_train=1, data_augmentation=0,
                             experiment_id=None, begin_test_date=None, end_test_date=None):
    
    """Function to optimize the hyperparameters and input features of the DNN. An example on how to 
    use this function is provided :ref:`here<dnnex1>`.
    
    Parameters
    ----------
    path_datasets_folder : str, optional
        Path to read and store datasets.
    
    path_hyperparameters_folder : str, optional
        Path to read and store trials files from hyperopt.
    
    new_hyperopt : bool, optional
        Boolean that decides whether to start a new hyperparameter optimization or re-start an
        existing one.
    
    max_evals : int, optional
        Maximum number of iterations for hyperopt.
    
    nlayers : int, optional
        Number of layers of the DNN model.
    
    dataset : str, optional
        Name of the dataset/market under study. If it is one one of the standard markets, 
        i.e. ``"PJM"``, ``"NP"``, ``"BE"``, ``"FR"``, or ``"DE"``, the dataset is automatically downloaded. If the name
        is different, a dataset with a csv format should be place in the ``path_datasets_folder``.
    
    years_test : int, optional
        Number of years (a year is 364 days) in the test dataset. It is only used if 
        the arguments ``begin_test_date`` and ``end_test_date`` are not provided.
    
    calibration_window : int, optional
        Calibration window used for training the models.
    
    shuffle_train : bool, optional
        Boolean that selects whether the validation and training datasets
        are shuffled. Based on empirical results, this configuration does not play a role
        when selecting the hyperparameters and features. However, it is important when recalibrating
        the DNN model.
    
    data_augmentation : bool, optional
        Boolean that selects whether a data augmentation technique 
        for DNNs is used. Based on empirical results, for some markets data augmentation might
        improve forecasting accuracy at the expense of higher computational costs.
    
    experiment_id : None, optional
        Unique identifier to save/read the trials file. If not
        provided, the current date is used as identifier.
    
    begin_test_date : datetime/str, optional
        Optional parameter to select the test dataset. Used in combination with the argument
        ``end_test_date``. If either of them is not provided, the test dataset is built using the 
        ``years_test`` argument. ``begin_test_date`` should either be a string with the following 
        format ``"%d/%m/%Y %H:%M"``, or a datetime object.
    
    end_test_date : datetime/str, optional
        Optional parameter to select the test dataset. Used in combination with the argument
        ``begin_test_date``. If either of them is not provided, the test dataset is built using the 
        ``years_test`` argument. ``end_test_date`` should either be a string with the following 
        format ``"%d/%m/%Y %H:%M"``, or a datetime object.       
    
    """

    # Checking if provided directory for hyperparameter exists and if not create it
    if not os.path.exists(path_hyperparameters_folder):
        os.makedirs(path_hyperparameters_folder)

    if experiment_id is None:
        experiment_id = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    else:
        experiment_id = experiment_id

    # Defining unique trials file name (this is an unique identifier)
    trials_file_name = 'DNN_hyperparameters_nl' + str(nlayers) + '_dat' + str(dataset) + \
                       '_YT' + str(years_test) + '_SF' * (shuffle_train) + \
                       '_DA' * (data_augmentation) + '_CW' + str(calibration_window) + \
                       '_' + str(experiment_id)

    trials_file_path = os.path.join(path_hyperparameters_folder, trials_file_name)

    # If hyperparameter optimization starts from scratch, new trials object is created. If not,
    # we read existing trials object
    if new_hyperopt:
        trials = Trials()
    else:
        trials = pc.load(open(trials_file_path, "rb"))


    # Generate training and test datasets
    dfTrain, dfTest = read_data(dataset=dataset, years_test=years_test, path=path_datasets_folder,
                                begin_test_date=begin_test_date, end_test_date=end_test_date)

    n_exogenous_inputs = len(dfTrain.columns) - 1

    # Build hyperparamerter search space. This includes hyperparameter and features
    space = _build_space(nlayers, data_augmentation, n_exogenous_inputs)


    # Perform hyperparameter optimization
    fmin_objective = partial(_hyperopt_objective, trials=trials, trials_file_path=trials_file_path, 
                             max_evals=max_evals, nlayers=nlayers, dfTrain=dfTrain, dfTest=dfTest, 
                             shuffle_train=shuffle_train, dataset=dataset, 
                             data_augmentation=data_augmentation, calibration_window=calibration_window,
                             n_exogenous_inputs=n_exogenous_inputs)

    fmin(fmin_objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials, verbose=False)