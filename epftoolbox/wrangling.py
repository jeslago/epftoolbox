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

def scaling(datasets, normalize):
    """Summary
    Scaling datasets
    Args:
        datasets (TYPE): list of datasets to be scaled
        normalize (TYPE): type of scaling to be performed. Possible values are
        Norm, Norm1, or Std, Median, or 'Median', 'Invariant'
    Returns:
        TYPE: Description
    """

    if normalize == 'Norm':
        scaler = MinMaxScaler(feature_range=(0, 1))
    elif normalize == 'Norm1':
        scaler = MinMaxScaler(feature_range=(-1, 1))
    elif normalize == 'Std':
        scaler = StandardScaler()
    elif normalize == 'Median':
        scaler = MedianScaler()
    elif normalize == 'Invariant':
        scaler = InvariantScaler()        

    for i, dataset in enumerate(datasets):
        if i == 0:
            dataset = scaler.fit_transform(dataset)
        else:
            dataset = scaler.transform(dataset)

        datasets[i] = dataset

    return datasets, scaler