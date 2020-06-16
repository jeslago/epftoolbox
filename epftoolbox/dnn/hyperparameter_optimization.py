import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import pickle as pc
from datetime import datetime
from epftoolbox.dnn.model import DNN
from epftoolbox.wrangling import scaling, createAndSplitXYs
from epftoolbox.datasets import read_data
from epftoolbox.metrics import MAE, sMAPE
from functools import partial
import os

def build_space(nlayer, data_augmentation, n_exogenous_inputs):

    # Defining the hyperparameter space. First the neural net hyperparameters,
    # later the input features

    space = {
        'BN': hp.choice('BN', [False, True]),
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

    if data_augmentation:
        # For the multiple output model, we allow as an option to use the 24 horizons in a day
        # during training, i.e. not only predict 00 to 23, but 01 to 24, 02 to 01, etc.
        # For testing the evaluation is normal
        space['24 datapoints per day'] = hp.choice('24 datapoints per day', [False, True])
    
    return space


def hyperopt_objective(hyperparameters, trials, trials_file_name, max_evals, nlayers, dfTrain, dfTest, 
                       path_experiment_files, shuffle_train, dataset, data_augmentation, 
                       calibration_window, n_exogenous_inputs):

    # Re-defining the training dataset based on the calibration window. The calibration window
    # can be given as an external parameter. If the value 0 is given, the calibration window
    # is included as a hyperparameter to optimize
    dfTrain_cw = dfTrain.loc[dfTrain.index[-1] - pd.Timedelta(weeks=52) * calibration_window +
                             pd.Timedelta(hours=1):]

    # Saving hyperoptimization state and printing message
    pc.dump(trials, open(trials_file_name, "wb"))
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
        createAndSplitXYs(dfTrain=dfTrain_cw, dfTest=dfTest, features=hyperparameters, 
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
    forecaster = DNN(neurons=neurons, Nfeatures=Xtrain.shape[-1], 
                     dropout=hyperparameters['dropout'], BN=hyperparameters['BN'], 
                     lr=hyperparameters['lr'], printOut=False,
                     optimizer='adam', activation=hyperparameters['activation'],
                     maxEpochsWOImprovement=20, scaler=scaler, loss='mae',
                     regularization=hyperparameters['reg']['val'], 
                     lambdaReg=hyperparameters['reg']['lambda'],
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


def hyperparameter_optimizer(path_datasets='./datasets/', path_experiment_files='./experimental_files/', 
                             new_hyperopt=1, max_evals=1500, nlayers=2, dataset='PJM', years_test=2, 
                             calibration_window=4, shuffle_train=1, data_augmentation=0,
                             experiment_id=None, begin_test_date=None, end_test_date=None):
    
    """ Main fucntion to perform hyperparameter optimization
    
    Args:
        path_datasets (str, optional): Path to read and store datasets
        
        path_experiment_files (str, optional): Path to read and store trials files from hyperopt
        
        new_hyperopt (bool, optional): Boolean that decides whether to start a new hyperparameter optimization
        
        max_evals (int, optional): Maximum number of iterations for hyperopt
        
        nlayers (int, optional): Number of layers of the DNN model
        
        dataset (str, optional): Market under study. If it not one of the standard ones, the file name
            has to be provided, where the file has to be a csv file
        
        years_test (int, optional): Number of years (a year is 364 days) in the test dataset
        
        calibration_window (int, optional): Calibration window used for training the models
        
        shuffle_train (bool, optional): Boolean that selects whether the validation and training datasets
            are shuffled
        
        data_augmentation (bool, optional): Boolean that selects whether a data augmentation technique 
            for DNNs is used
        
        experiment_id (None, optional): Unique identifier to save/read the trials file. If not
            provided, the current date is used as identifier

        begin_test_date (None, optional): Optional parameter for selecting the test dataset together
            with end_test_date. If either of them is not provided, the test dataset is built using the 
            years_test parameter. It should either be one of the date formats existing in python or a 
            string representing a date with the following format "%d/%m/%Y %H:%M"
        
        end_test_date (None, optional): Optional parameter for selecting the test dataset together
            with end_test_date. If either of them is not provided, the test dataset is built using the 
            years_test parameter. It should either be one of the date formats existing in python or a 
            string representing a date with the following format "%d/%m/%Y %H:%M"

    """

    # Checking if provided directories exist and if not create them
    if not os.path.exists(path_datasets):
        os.makedirs(path_datasets)
    
    if not os.path.exists(path_experiment_files):
        os.makedirs(path_experiment_files)

    if experiment_id is None:
        experiment_id = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    else:
        experiment_id = experiment_id

    # Defining unique trials file name (this is an unique identifier)
    trials_file_name = path_experiment_files + 'hyper_nl' + str(nlayers) + '_dat' + str(dataset) + \
                       '_YT' + str(years_test) + '_SF' * (shuffle_train) + \
                       '_DA' * (data_augmentation) + '_CW' + str(calibration_window) + \
                       '_' + str(experiment_id)

    # If hyperparameter optimization starts from scratch, new trials object is created. If not,
    # we read existing trials object
    if new_hyperopt:
        trials = Trials()
    else:
        trials = pc.load(open(trials_file_name, "rb"))


    # Generate training and test datasets
    dfTrain, dfTest = read_data(dataset=dataset, years_test=years_test, path=path_datasets,
                                begin_test_date=begin_test_date, end_test_date=end_test_date)

    n_exogenous_inputs = len(dfTrain.columns) - 1

    # Build hyperparamerter search space. This includes hyperparameter and features
    space = build_space(nlayers, data_augmentation, n_exogenous_inputs)


    # Perform hyperparameter optimization
    fmin_objective = partial(hyperopt_objective, trials=trials, trials_file_name=trials_file_name, 
                             max_evals=max_evals, nlayers=nlayers, dfTrain=dfTrain, dfTest=dfTest, 
                             path_experiment_files=path_experiment_files, shuffle_train=shuffle_train, 
                             dataset=dataset, data_augmentation=data_augmentation, 
                             calibration_window=calibration_window,n_exogenous_inputs=n_exogenous_inputs)

    fmin(fmin_objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)