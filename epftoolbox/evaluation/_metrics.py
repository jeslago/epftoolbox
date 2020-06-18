import numpy as np
import pandas as pd

def naive_forecast(Yreal, m=None):

    index = Yreal.iloc[168:, :].index
    Y_pred = pd.DataFrame(index=index, columns=['Naive'])
    
    # If m is none the standard naive for EPF is built
    if m is None:
        # Extracting days from dataset
        days = index[::24]

        # Iterating over days to create the naive forecast
        for day in days:
            end_day = day + pd.Timedelta(hours=23)

            # If it is Tuesday, Wednesday, Thursday, or Friday
            if day.dayofweek in [1, 2, 3, 4]:
                Y_pred.loc[day:end_day, :] = \
                    Yreal.loc[day - pd.Timedelta(days=1):end_day - pd.Timedelta(days=1), :].values            

            # If it is Saturday or Sunday
            elif day.dayofweek in [5, 6]:
                Y_pred.loc[day:end_day, :] = \
                    Yreal.loc[day - pd.Timedelta(days=7):end_day - pd.Timedelta(days=7), :].values            

            # If it is Saturday or Sunday
            elif day.dayofweek in [0]:
                Y_pred.loc[day:end_day, :] = \
                    Yreal.loc[day - pd.Timedelta(days=3):end_day - pd.Timedelta(days=3), :].values

    # If m is either 24 or 168 naive forecast simply built using a seasonal naive forecast
    elif m == 24:
        Y_pred.loc[:, :] = Yreal.loc[Y_pred.index - pd.Timedelta(days=1)].values

    elif m == 168:
        Y_pred.loc[:, :] = Yreal.loc[Y_pred.index - pd.Timedelta(days=7)].values

    return Y_pred


def RMSE(Yreal, Ypred):
    return np.sqrt(np.mean((Yreal - Ypred)**2, axis=0))


def rRMSE(Yreal, Ypred):
    return np.sqrt(np.mean((Yreal - Ypred)**2, axis=0)) / np.mean(np.abs(Yreal), axis=0)


def sMAPE(Yreal, Ypred):
    return np.mean(np.abs(Yreal - Ypred) / ((np.abs(Yreal) + np.abs(Ypred)) / 2), axis=0)


def MAE(Yreal, Ypred):
    return np.mean(np.abs(Yreal - Ypred), axis=0)

def MASE(Yreal, Ypred, Yrealtrain, m=None):

        Ypred_train = naive_forecast(Yrealtrain, m=m)
        Yrealtrain = Yrealtrain.loc[Ypred_train.index]
        MAE_naive_train = MAE(Yrealtrain, Ypred_train)

        return np.mean(np.abs(Yreal.values.squeeze() - Ypred.values.squeeze()) / MAE_naive_train, axis=0)

def RMAE(Yreal, Ypred, m=None):

        Ypred_naive = naive_forecast(Yreal, m=m)
        Yreal_naive = Yreal.loc[Ypred_naive.index]

        MAE_naive_train = MAE(Yreal_naive, Ypred_naive)

        return np.mean(np.abs(Yreal.values.squeeze() - Ypred.values.squeeze()) / MAE_naive_train, axis=0)
