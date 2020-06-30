"""
Classes and functions to implement data wrangling operations. At the moment, this is limited to
data scaling.
"""

# Author: Jesus Lago

# License: AGPL-3.0 License

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.robust import mad

class MedianScaler(object):

    def __init__(self):
        self.fitted = False

    def fit(self, data):

        if len(data.shape)!=2:
            raise IndexError('Error: Provide 2-D array. First dimension is datapoints and' + 
                  ' second features')
            return -1

        self.median = np.median(data, axis=0)
        self.mad = mad(data, axis=0)
        self.fitted = True
        
    def fit_transform(self, data):

        self.fit(data)
        return self.transform(data)
    
    def transform(self, data):

        if not self.fitted:
            print('Error: The scaler has not been yet fitted. Called fit or fit_transform')
            return -1

        if len(data.shape)!=2:
            raise IndexError('Error: Provide 2-D array. First dimension is datapoints and' + 
                  ' second features')

        transformed_data = np.zeros(shape=data.shape)

        for i in range(data.shape[1]):

            transformed_data[:, i] = (data[:, i] - self.median[i]) / self.mad[i]

        return transformed_data

    def inverse_transform(self, data):

        if not self.fitted:
            print('Error: The scaler has not been yet fitted. Called fit or fit_transform')
            return -1

        if len(data.shape)!=2:
            raise IndexError('Error: Provide 2-D array. First dimension is datapoints and' + 
                  ' second features')

        transformed_data = np.zeros(shape=data.shape)

        for i in range(data.shape[1]):

            transformed_data[:, i] = data[:, i] * self.mad[i] + self.median[i] 

        return transformed_data


class InvariantScaler(MedianScaler):

    def __init__(self):
        super()

    def fit(self, data):

        super().fit(data)
        
    def fit_transform(self, data):

        self.fit(data)
        return self.transform(data)
    
    def transform(self, data):

        transformed_data = super().transform(data)
        transformed_data = np.arcsinh(transformed_data)

        return transformed_data

    def inverse_transform(self, data):

        transformed_data = np.sinh(data)
        transformed_data = super().inverse_transform(transformed_data)

        return transformed_data


class DataScaler(object):

    """Class to perform data scaling operations

    The scaling technique is defined by the ``normalize`` parameter which takes one of the 
    following values: 

    - ``'Norm'`` for normalizing the data to the interval [0, 1].

    - ``'Norm1'`` for normalizing the data to the interval [-1, 1]. 

    - ``'Std'`` for standarizing the data to follow a normal distribution. 

    - ``'Median'`` for normalizing the data based on the median as defined in as defined in `here <https://doi.org/10.1109/TPWRS.2017.2734563>`_.

    - ``'Invariant'`` for scaling the data based on the asinh transformation (a variance stabilizing transformations) as defined in `here <https://doi.org/10.1109/TPWRS.2017.2734563>`_. 
    
    This class follows the same syntax of the scalers defined in the 
    `sklearn.preprocessing <https://scikit-learn.org/stable/modules/preprocessing.html>`_ module of the 
    scikit-learn library

    Parameters
    ----------
    normalize : str
        Type of scaling to be performed. Possible values are ``'Norm'``, ``'Norm1'``, ``'Std'``, 
        ``'Median'``, or ``'Invariant'``

    Example
    --------
    >>> from epftoolbox.data import read_data
    >>> from epftoolbox.data import DataScaler
    >>> df_train, df_test = read_data(path='.', dataset='PJM', begin_test_date='01-01-2016', end_test_date='01-02-2016')
    Test datasets: 2016-01-01 00:00:00 - 2016-02-01 23:00:00
    >>> df_train.tail()
                             Price  Exogenous 1  Exogenous 2
    Date                                                    
    2015-12-31 19:00:00  29.513832     100700.0      13015.0
    2015-12-31 20:00:00  28.440134      99832.0      12858.0
    2015-12-31 21:00:00  26.701700      97033.0      12626.0
    2015-12-31 22:00:00  23.262253      92022.0      12176.0
    2015-12-31 23:00:00  22.262431      86295.0      11434.0
    >>> df_test.head()
                             Price  Exogenous 1  Exogenous 2
    Date                                                    
    2016-01-01 00:00:00  20.341321      76840.0      10406.0
    2016-01-01 01:00:00  19.462741      74819.0      10075.0
    2016-01-01 02:00:00  17.172706      73182.0       9795.0
    2016-01-01 03:00:00  16.963876      72300.0       9632.0
    2016-01-01 04:00:00  17.403722      72535.0       9566.0
    >>> Xtrain = df_train.values
    >>> Xtest = df_train.values
    >>> scaler = DataScaler('Norm')
    >>> Xtrain_scaled = scaler.fit_transform(Xtrain)
    >>> Xtest_scaled = scaler.transform(Xtest)
    >>> Xtrain_inverse = scaler.inverse_transform(Xtrain_scaled)
    >>> Xtest_inverse = scaler.inverse_transform(Xtest_scaled)
    >>> Xtrain[:3,:]
    array([[2.5464211e+01, 8.5049000e+04, 1.1509000e+04],
           [2.3554578e+01, 8.2128000e+04, 1.0942000e+04],
           [2.2122277e+01, 8.0729000e+04, 1.0639000e+04]])
    >>> Xtrain_scaled[:3,:]
    array([[0.03833877, 0.2736787 , 0.28415155],
           [0.03608228, 0.24425597, 0.24633138],
           [0.03438982, 0.23016409, 0.2261206 ]])
    >>> Xtrain_inverse[:3,:]
    array([[2.5464211e+01, 8.5049000e+04, 1.1509000e+04],
           [2.3554578e+01, 8.2128000e+04, 1.0942000e+04],
           [2.2122277e+01, 8.0729000e+04, 1.0639000e+04]])
    >>> Xtest[:3,:]
    array([[2.5464211e+01, 8.5049000e+04, 1.1509000e+04],
           [2.3554578e+01, 8.2128000e+04, 1.0942000e+04],
           [2.2122277e+01, 8.0729000e+04, 1.0639000e+04]])
    >>> Xtest_scaled[:3,:]
    array([[0.03833877, 0.2736787 , 0.28415155],
           [0.03608228, 0.24425597, 0.24633138],
           [0.03438982, 0.23016409, 0.2261206 ]])
    >>> Xtest_inverse[:3,:]
    array([[2.5464211e+01, 8.5049000e+04, 1.1509000e+04],
           [2.3554578e+01, 8.2128000e+04, 1.0942000e+04],
           [2.2122277e+01, 8.0729000e+04, 1.0639000e+04]])
    """
    
    def __init__(self, normalize):

        if normalize == 'Norm':
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif normalize == 'Norm1':
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
        elif normalize == 'Std':
            self.scaler = StandardScaler()
        elif normalize == 'Median':
            self.scaler = MedianScaler()
        elif normalize == 'Invariant':
            self.scaler = InvariantScaler()        

    def fit_transform(self, dataset):
        """Method that estimates an scaler object using the data in ``dataset`` and scales the data in  ``dataset``
        
        Parameters
        ----------
        dataset : numpy.array
            Dataset used to estimate the scaler
        
        Returns
        -------
        numpy.array
            Scaled data
        """

        return self.scaler.fit_transform(dataset)

    def transform(self, dataset):
        """Method that scales the data in ``dataset``
        
        It must be called after calling the :class:`fit_transform` method for estimating the scaler
        Parameters
        ----------
        dataset : numpy.array
            Dataset to be scaled
        
        Returns
        -------
        numpy.array
            Scaled data
        """

        return self.scaler.transform(dataset)

    def inverse_transform(self, dataset):
        """Method that inverse-scale the data in ``dataset``
        
        It must be called after calling the :class:`fit_transform` method for estimating the scaler

        Parameters
        ----------
        dataset : numpy.array
            Dataset to be scaled
        
        Returns
        -------
        numpy.array
            Inverse-scaled data
        """

        return self.scaler.inverse_transform(dataset)

def scaling(datasets, normalize):
    """Function that scales data and returns the scaled data and the :class:`DataScaler` used for scaling.

    It rescales all the datasets contained in the list ``datasets`` using the first dataset as reference. 
    For example, if ``datasets=[X_1, X_2, X_3]``, the function estimates a :class:`DataScaler` object using the array ``X_1``, 
    and transform ``X_1``, ``X_2``, and ``X_3`` using the :class:`DataScaler` object.

    Each dataset must be a numpy.array and it should have the same column-dimensions. For example, if
    ``datasets=[X_1, X_2, X_3]``, ``X_1`` must be a numpy.array of size ``[n_1, m]``,
    ``X_2`` of size ``[n_2, m]``, and ``X_3`` of size ``[n_3, m]``, where ``n_1``, ``n_2``, ``n_3`` can be
    different.

    The scaling technique is defined by the ``normalize`` parameter which takes one of the 
    following values: 

    - ``'Norm'`` for normalizing the data to the interval [0, 1].

    - ``'Norm1'`` for normalizing the data to the interval [-1, 1]. 

    - ``'Std'`` for standarizing the data to follow a normal distribution. 

    - ``'Median'`` for normalizing the data based on the median as defined in as defined in `here <https://doi.org/10.1109/TPWRS.2017.2734563>`_.

    - ``'Invariant'`` for scaling the data based on the asinh transformation (a variance stabilizing transformations) as defined in `here <https://doi.org/10.1109/TPWRS.2017.2734563>`_. 


    The function returns the scaled data together with a :class:`DataScaler` object representing the scaling. 
    This object can be used to scale other dataset using the same rules or to inverse-transform the data.
    
    Parameters
    ----------
    datasets : list
        List of numpy.array objects to be scaled.
    normalize : str
        Type of scaling to be performed. Possible values are ``'Norm'``, ``'Norm1'``, ``'Std'``, 
        ``'Median'``, or ``'Invariant'``
    
    Returns
    -------
    List, :class:`DataScaler`
        List of scaled datasets and the :class:`DataScaler` object used for scaling. Each dataset in the 
        list is a numpy.array.
    
    Example
    --------
    >>> from epftoolbox.data import read_data
    >>> from epftoolbox.data import scaling
    >>> df_train, df_test = read_data(path='.', dataset='PJM', begin_test_date='01-01-2016', end_test_date='01-02-2016')
    Test datasets: 2016-01-01 00:00:00 - 2016-02-01 23:00:00
    >>> df_train.tail()
                             Price  Exogenous 1  Exogenous 2
    Date                                                    
    2015-12-31 19:00:00  29.513832     100700.0      13015.0
    2015-12-31 20:00:00  28.440134      99832.0      12858.0
    2015-12-31 21:00:00  26.701700      97033.0      12626.0
    2015-12-31 22:00:00  23.262253      92022.0      12176.0
    2015-12-31 23:00:00  22.262431      86295.0      11434.0
    >>> df_test.head()
                             Price  Exogenous 1  Exogenous 2
    Date                                                    
    2016-01-01 00:00:00  20.341321      76840.0      10406.0
    2016-01-01 01:00:00  19.462741      74819.0      10075.0
    2016-01-01 02:00:00  17.172706      73182.0       9795.0
    2016-01-01 03:00:00  16.963876      72300.0       9632.0
    2016-01-01 04:00:00  17.403722      72535.0       9566.0
    >>> Xtrain = df_train.values
    >>> Xtest = df_train.values
    >>> [Xtrain_scaled, Xtest_scaled], scaler = scaling([Xtrain,Xtest],'Norm')
    >>> Xtrain[:3,:]
    array([[2.5464211e+01, 8.5049000e+04, 1.1509000e+04],
           [2.3554578e+01, 8.2128000e+04, 1.0942000e+04],
           [2.2122277e+01, 8.0729000e+04, 1.0639000e+04]])
    >>> Xtrain_scaled[:3,:]
    array([[0.03833877, 0.2736787 , 0.28415155],
           [0.03608228, 0.24425597, 0.24633138],
           [0.03438982, 0.23016409, 0.2261206 ]])
    >>> Xtest[:3,:]
    array([[2.5464211e+01, 8.5049000e+04, 1.1509000e+04],
           [2.3554578e+01, 8.2128000e+04, 1.0942000e+04],
           [2.2122277e+01, 8.0729000e+04, 1.0639000e+04]])           
    >>> Xtest_scaled[:3,:]
    array([[0.03833877, 0.2736787 , 0.28415155],
           [0.03608228, 0.24425597, 0.24633138],
           [0.03438982, 0.23016409, 0.2261206 ]])
    >>> type(scaler)
    <class 'epftoolbox.data._wrangling.DataScaler'>

    """

    scaler = DataScaler(normalize)

    for i, dataset in enumerate(datasets):
        if i == 0:
            dataset = scaler.fit_transform(dataset)
        else:
            dataset = scaler.transform(dataset)

        datasets[i] = dataset

    return datasets, scaler