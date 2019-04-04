import re
import os
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
import pandas as pd
import math
import numpy as np
from geopy.distance import vincenty

# The replacement patterns used in cleaning the raw text data
replacement_patterns = [
    (r"won\'t", "will not"),
    (r"[^A-Za-z0-9^,!.\/'+-=]", " "),
    (r"can\'t", "cannot"),
    (r"I\'m", "I am"),
    (r"ain\'t", 'is not'),
    (r"(\d+)(k)", r"\g<1>000"),
    # \g<1> are using back-references to capture part of the matched pattern
    # \g means referencing group content in the previous pattern. <1> means the first group. In the following case, the first group is w+
    (r"(\w+)\'ll", "\g<1> will"),
    (r"(\w+)n\'t", "\g<1> not"),
    (r"(\w+)\'ve", "\g<1> have"),
    (r"(\w+)\'s", "\g<1> is"),
    (r"(\w+)\'re", "\g<1> are"),
    (r"(\w+)\'d", "\g<1> would")
]


# A RegexpReplacer to clean some texts based on specified patterns
class RegexpReplacer(object):
    def __init__(self, patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in replacement_patterns]

    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            s = re.sub(pattern=pattern, repl=repl, string=s)  # subn returns the times of replacement
        return s


# Check if there is blank list in pandas dataframe text
def check_nan(text):
    nan_list = []
    if len(text) == 0:
        nan_list.append(1)
    else:
        nan_list.append(0)
    return nan_list


def value_count(data_frame):
    result = data_frame['text'].apply(check_nan)
    return result.value_counts()


# Use this function to select the MTR-related tweets
def find_tweet(keywords, tweet):
    result = ''
    for word in tweet:
        if word in keywords:
            result = True
        else:
            result = False
    return result


# Get the ROC and AUC for evaluations
def compute_auc(y_true, y_pred):
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true, y_pred)
    auc_keras = auc(fpr_keras, tpr_keras)
    return auc_keras


# Plot the ROC curve
def plot_roc(fpr_keras, tpr_keras, auc_keras, model_name: str):
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras '+ model_name + '(area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve: '+model_name)
    plt.legend(loc='best')
    plt.show()


# Calculate the haversine distance between two points based on latitude and longitude
# More about the haversine distance: https://en.wikipedia.org/wiki/Haversine_formula
def distance_calc(row, station_lat, station_lon):
    start = (row['lat'], row['lon'])
    stop = (station_lat, station_lon)
    return vincenty(start, stop).meters

"""
An instance of using haversine_distance to calculate the distance of two points
lat1 = 52.2296756
lon1 = 21.0122287
Whampoa_lat = 22.3051
Whampoa_lon = 114.1895
Ho_Man_Tin_lat = 22.3094
Ho_Man_Tin_lon = 114.1827

print(distance(lat1, lon1, Whampoa_lat, Whampoa_lon))
"""


def select_data_based_on_location(row, station_lat, station_lon):
    if distance_calc(row, station_lat, station_lon) < 500:
        result = 'TRUE'
    else:
        result = 'FALSE'
    return result


def read_file_from_multi_csvs(path):
    all_csv_files = os.listdir(path)
    dataframes = []
    for file in all_csv_files:
        dataframe = pd.read_csv(os.path.join(path, file), encoding='latin-1', na_values=['nan',''])
    combined_dataframes = pd.concat(dataframes)
    return combined_dataframes







