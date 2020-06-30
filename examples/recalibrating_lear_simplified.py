"""
Simplified example for using the LEAR model for forecasting prices with daily recalibration
"""

# Author: Jesus Lago

# License: AGPL-3.0 License

from epftoolbox.models import evaluate_lear_in_test_dataset
import os

# Market under study. If it not one of the standard ones, the file name
# has to be provided, where the file has to be a csv file
dataset = 'PJM'

# Number of years (a year is 364 days) in the test dataset.
years_test = 2

# Number of days used in the training dataset for recalibration
calibration_window = 364 * 4

# Optional parameters for selecting the test dataset, if either of them is not provided, 
# the test dataset is built using the years_test parameter. They should either be one of
# the date formats existing in python or a string with the following format
# "%d/%m/%Y %H:%M"
begin_test_date = None
end_test_date = None

path_datasets_folder = os.path.join('.', 'datasets')
path_recalibration_folder = os.path.join('.', 'experimental_files')
    
evaluate_lear_in_test_dataset(path_recalibration_folder=path_recalibration_folder, 
                             path_datasets_folder=path_datasets_folder, dataset=dataset, years_test=years_test, 
                             calibration_window=calibration_window, begin_test_date=begin_test_date, 
                             end_test_date=end_test_date)
