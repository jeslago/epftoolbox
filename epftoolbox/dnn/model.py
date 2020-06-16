import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Dropout, AlphaDropout
from keras.layers.normalization import BatchNormalization
import time
import keras as kr
from keras.regularizers import l2, l1
from keras.layers.advanced_activations import LeakyReLU, PReLU
from epftoolbox.metrics import MAE


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
