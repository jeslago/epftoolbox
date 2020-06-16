import pandas as pd
import os

def read_data(path, dataset='PJM', years_test=2, begin_test_date=None, end_test_date=None):
    """ Function to read and import new day-ahead market datasets. If one of the five existing datasets
        is request but it does not exist locally, it is download the first time 
    
    Args:
        path (TYPE): path to save/read the datasets
        
        dataset (str, optional): name of the dataset file (if exists) or the name of the file to be used
            to save the dataset

        years_test (int, optional): # Number of years (a year is 364 days) in the test dataset. Used when
            the begin_test_date and end_test_date parameters are not provided
        
        begin_test_date (None, optional): Optional parameter for selecting the test dataset together
            with end_test_date. If either of them is not provided, the test dataset is built using the 
            years_test parameter. It should either be one of the date formats existing in python or a 
            string representing a date with the following format "%d/%m/%Y %H:%M"
        
        end_test_date (None, optional): Optional parameter for selecting the test dataset together
            with end_test_date. If either of them is not provided, the test dataset is built using the 
            years_test parameter. It should either be one of the date formats existing in python or a 
            string representing a date with the following format "%d/%m/%Y %H:%M"
    Returns:
        df_train, df_test: Test and training dataframes of the requested dataset

    """

    # If dataset is one of the existing open-access ones,
    # they are imported if they exist locally or download from 
    # the repository if they do not
    if dataset in ['PJM', 'NP', 'FR', 'BE', 'DE']:
        file_path = path + dataset + 'csv'

        # The first time this function is called, the datasets
        # are downloaded and saved in a local folder
        # After the first called they are imported from the local
        # folder
        if os.path.exists(file_path):
            data = pd.read_csv(file_path, index_col=0)
        else:
            url_dir = 'https://sandbox.zenodo.org/api/files/da5b2c6f-8418-4550-a7d0-7f2497b40f1b/'
            data = pd.read_csv(url_dir + dataset + '.csv', index_col=0)
            data.to_csv(path + dataset + '.csv')
    else:
        try:
            data = pd.read_csv(path + dataset + '.csv', index_col=0)
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

        if begin_test_date.hours != 0:
            raise Exception("Starting date for test dataset should be midnight") 
        if end_test_date.hours != 23:
            raise Exception("End date for test dataset should be at 23h") 

        df_train = data.loc[:begin_test_date - pd.Timedelta(hours=1), :]
        df_test = data.loc[begin_test_date:end_test_date, :]

    return df_train, df_test
