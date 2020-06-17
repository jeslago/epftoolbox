import numpy as np
import pandas as pd
import time

import tensorflow.keras as kr
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, AlphaDropout, BatchNormalization
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.layers import LeakyReLU, PReLU
from epftoolbox.metrics import MAE
import tensorflow.keras.backend as K

class DNN(object):

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

        self.model = self.buildAndCompileModel()

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

    def reg(self, lambdaReg):

        if self.regularization == 'l2':
            return l2(lambdaReg)
        if self.regularization == 'l1':
            return l1(lambdaReg)
        else:
            return None

    def buildAndCompileModel(self):

        inputShape = (None, self.Nfeatures)

        past_data = Input(batch_shape=inputShape)

        past_Dense = past_data
        if self.activation == 'selu':
            self.initializer = 'lecun_normal'

        for k, neurons in enumerate(self.neurons):

            if self.activation == 'LeakyReLU':
                past_Dense = Dense(neurons, activation='linear', batch_input_shape=inputShape,
                                   kernel_initializer=self.initializer,
                                   kernel_regularizer=self.reg(self.lambdaReg))(past_Dense)
                past_Dense = LeakyReLU(alpha=.001)(past_Dense)

            elif self.activation == 'PReLU':
                past_Dense = Dense(neurons, activation='linear', batch_input_shape=inputShape,
                                   kernel_initializer=self.initializer,
                                   kernel_regularizer=self.reg(self.lambdaReg))(past_Dense)
                past_Dense = PReLU()(past_Dense)

            else:
                past_Dense = Dense(neurons, activation=self.activation,
                                   batch_input_shape=inputShape,
                                   kernel_initializer=self.initializer,
                                   kernel_regularizer=self.reg(self.lambdaReg))(past_Dense)

            if self.BN:
                past_Dense = BatchNormalization()(past_Dense)

            if self.dropout > 0:
                if self.activation == 'selu':
                    past_Dense = AlphaDropout(self.dropout)(past_Dense)
                else:
                    past_Dense = Dropout(self.dropout)(past_Dense)

        output_layer = Dense(self.outputShape, kernel_initializer=self.initializer,
                             kernel_regularizer=self.reg(self.lambdaReg))(past_Dense)
        model = Model(inputs=[past_data], outputs=[output_layer])

        return model

    def update_metrics(self, X, Y):

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

            valError, valMAE = self.update_metrics(valX, valY)

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
