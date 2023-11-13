"""
Function to read electricity market data either locally or from an online database
"""

# Author: Jesus Lago

# License: AGPL-3.0 License


import pandas as pd
import os

def read_data(path, dataset='PJM', years_test=2, begin_test_date=None, end_test_date=None):
    """Function to read and import data from day-ahead electricity markets. 
    
    It receives a ``dataset`` name, and the ``path`` of the folder where datasets are saved. 
    It reads the file ``dataset.csv`` in the ``path`` directory and provides a split between training and
    testing dataset based on the test dates provided.

    It also names the columns of the training and testing dataset to match the requirements of the
    prediction models of the library. Namely, assuming that there are `N` exogenous inputs,
    the columns of the resulting training and testing dataframes are named
    ``['Price', 'Exogenous 1', 'Exogenous 2', ...., 'Exogenous N']``.

    If `dataset` is either ``"PJM"``, ``"NP"``, ``"BE"``, ``"FR"``, or ``"DE"``,
    the function checks whether ``dataset.csv`` exists in ``path``. If it doesn't exist,
    it downloads the data from an online database and saves it under the ``path`` directory. ``"PJM"``
    refes to the Pennsylvania-New Jersey-Maryland market, ``"NP"`` to the Nord Pool market,
    and ``"BE"``, ``"FR"``, and ``"DE"`` respectively to the EPEX-Belgium, EPEX-France, and EPEX-Germany 
    day-ahead markets.

    Note that the data available online for these five markets is limited to certain periods (see the 
    `database <https://zenodo.org/records/4624805>`_ for further details).  
    
    Parameters
    ----------
    path : str, optional
        Path where the datasets are stored or, if they do not exist yet, the path where the datasets 
        are to be stored
    nlayers : int, optional
        Number of hidden layers in the neural network
    dataset : str, optional
        Name of the dataset/market under study. If it is one one of the standard markets, 
        i.e. ``"PJM"``, ``"NP"``, ``"BE"``, ``"FR"``, or ``"DE"``, the dataset is automatically downloaded. If the name
        is different, a dataset with a csv format should be place in the ``path``.
    years_test : int, optional
        Number of years (a year is 364 days) in the test dataset. It is only used if 
        the arguments begin_test_date and end_test_date are not provided.
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
    Returns
    -------
    pandas.DataFrame, pandas.DataFrame
        Training dataset, testing dataset

    Example
    --------
    >>> from epftoolbox.data import read_data
    >>> df_train, df_test = read_data(path='.', dataset='PJM', begin_test_date='01-01-2016', 
    ...                               end_test_date='01-02-2016')
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
    >>> df_test.tail()
                             Price  Exogenous 1  Exogenous 2
    Date                                                    
    2016-02-01 19:00:00  28.056729      99400.0      12680.0
    2016-02-01 20:00:00  26.916456      97553.0      12495.0
    2016-02-01 21:00:00  24.041505      93983.0      12267.0
    2016-02-01 22:00:00  22.044896      88535.0      11747.0
    2016-02-01 23:00:00  20.593339      82900.0      10974.0

    """

    # Checking if provided directory exist and if not create it
    if not os.path.exists(path):
        os.makedirs(path)

    # If dataset is one of the existing open-access ones,
    # they are imported if they exist locally or download from 
    # the repository if they do not
    if dataset in ['PJM', 'NP', 'FR', 'BE', 'DE']:
        file_path = os.path.join(path, dataset + '.csv')

        # The first time this function is called, the datasets
        # are downloaded and saved in a local folder
        # After the first called they are imported from the local
        # folder
        if os.path.exists(file_path):
            data = pd.read_csv(file_path, index_col=0)
        else:
            url_dir = 'https://zenodo.org/records/4624805/files/'
            data = pd.read_csv(url_dir + dataset + '.csv', index_col=0)
            data.to_csv(file_path)
    else:
        try:
            file_path = os.path.join(path, dataset + '.csv')
            data = pd.read_csv(file_path, index_col=0)
        except IOError as e:
            raise IOError("%s: %s" % (path, e.strerror))

    data.index = pd.to_datetime(data.index)

    columns = ['Price']
    n_exogeneous_inputs = len(data.columns) - 1

    for n_ex in range(1, n_exogeneous_inputs + 1):
        columns.append('Exogenous ' + str(n_ex))
        
    data.columns = columns

    # The training and test datasets can be defined by providing a number of years for testing
    # or by providing the init and end date of the test period
    if begin_test_date is None and end_test_date is None:
        number_datapoints = len(data.index)
        number_training_datapoints = number_datapoints - 24 * 364 * years_test

        # We consider that a year is 52 weeks (364 days) instead of the traditional 365
        df_train = data.loc[:data.index[0] + pd.Timedelta(hours=number_training_datapoints - 1), :]
        df_test = data.loc[data.index[0] + pd.Timedelta(hours=number_training_datapoints):, :]
    
    else:
        try:
            begin_test_date = pd.to_datetime(begin_test_date, dayfirst=True)
            end_test_date = pd.to_datetime(end_test_date, dayfirst=True)
        except ValueError:
            print("Provided values for dates are not valid")

        if begin_test_date.hour != 0:
            raise Exception("Starting date for test dataset should be midnight") 
        if end_test_date.hour != 23:
            if end_test_date.hour == 0:
                end_test_date = end_test_date + pd.Timedelta(hours=23)
            else:
                raise Exception("End date for test dataset should be at 0h or 23h") 

        print('Test datasets: {} - {}'.format(begin_test_date, end_test_date))
        df_train = data.loc[:begin_test_date - pd.Timedelta(hours=1), :]
        df_test = data.loc[begin_test_date:end_test_date, :]

    return df_train, df_test
