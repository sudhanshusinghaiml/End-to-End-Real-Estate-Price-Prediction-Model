import re
import pandas as pd
import numpy as np
from scipy import stats
import os
import pickle

"""
    This file will store all the utility functions that are used in preprocessing or model piepline
"""


# Function to read the data
def read_data(file_path, **kwargs):
    try:
        raw_data = pd.read_excel(file_path, **kwargs)
    except Exception as e:
        print(e)
    else:
        return raw_data


# Function to dump python objects
def pickle_dump(data, filename):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(e)


def get_interval(train_actual_values, train_predicted_values, pi=.60):
    """
    Get a prediction interval for the regression model.

    INPUTS:
        - actual_values (y_train)
        - predicted_values (prediction from x_train)
        - Prediction interval threshold (default = .95)
    OUTPUT:
        - Interval estimate
    """
    try:
        print("Started Executing function utils.get_interval")
        # get standard deviation of prediction on the train dataset
        sum_of_square_error = np.sum((train_actual_values - train_predicted_values) ** 2)
        stddev = np.sqrt(sum_of_square_error / (len(train_actual_values) - 1))

        # get interval from standard deviation
        one_minus_pi = 1 - pi
        ppf_lookup = 1 - (
                one_minus_pi / 2)  # If we need to calculate a 'Two-tail test' (i.e. We're concerned with values both greater and less than our mean) then we need to split the significance (i.e. our alpha value) because we're still using a calculation method for one-tail. The split in half symbolizes the significance level being appropriated to both tails. A 95% significance level has a 5% alpha; splitting the 5% alpha across both tails returns 2.5%. Taking 2.5% from 100% returns 97.5% as an input for the significance level.
        z_score = stats.norm.ppf(
            ppf_lookup)  # This will return a value (that functions as a 'standard-deviation multiplier') marking where 95% (pi%) of data points would be contained if our data is a normal distribution.
        interval_value = z_score * stddev
        print("Started Executing function utils.get_average_area")
    except Exception as e:
        print('Error in get_interval function', e)
    else:
        return interval_value


def get_average_area(x):
    try:
        print("Started Executing function utils.get_average_area")
        regx_numbers = re.compile(r"[-+]?(\d*\.\d+|\d+)")
        x = regx_numbers.findall(x)
    except Exception as e:
        print('Error in get_average_area function', e)
    else:
        if len(x) == 1:
            return np.float(x[0])
        elif len(x) == 2:
            return (np.float(x[0]) + np.float(x[1])) / 2
        else:
            return -99


def get_prediction_interval(y_predicted_value, interval_value):
    try:
        print("Started Executing function utils.get_prediction_interval")
        # generate prediction interval lower and upper bound cs_24
        lower, upper = y_predicted_value - interval_value, y_predicted_value + interval_value
        print("Completed Executing function utils.get_prediction_interval")
    except Exception as e:
        print('Error in get_prediction_interval function', e)
    else:
        return lower, upper
