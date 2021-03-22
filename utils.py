import re
import os
import numpy as np
import pandas as pd
import csv
import pytz
from datetime import datetime
from collections import Counter

# For plots
from matplotlib import pyplot as plt
import seaborn as sns

import data_paths

time_zone_hk = pytz.timezone('Asia/Shanghai')

# For instance, if we want to compare the sentiment and activity level before and after the
# openning date of the Whampoa MTR railway station in Hong Kong, since the station is opened on 23 Oct 2016,
# we could specify the openning date using datatime package and output before and after dataframes
october_1_start = datetime(2016, 10, 1, 0, 0, 0, tzinfo=time_zone_hk)
october_31_end = datetime(2016, 10, 31, 23, 59, 59, tzinfo=time_zone_hk)
december_1_start = datetime(2016, 12, 1, 0, 0, 0, tzinfo=time_zone_hk)
december_31_end = datetime(2016, 12, 31, 23, 59, 59, tzinfo=time_zone_hk)
start_date = datetime(2016, 5, 7, 0, 0, 0, tzinfo=time_zone_hk)
end_date = datetime(2018, 12, 18, 23, 59, 59, tzinfo=time_zone_hk)

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


def weighted_average(group: pd.DataFrame, value_col: str, weight_col: str) -> float:
    """
    Compute the weighted average based on a pandas dataframe
    :param group: a pandas dataframe that we want to compute the weighted average
    :param value_col: the column name that saves the values of weighted average
    :param weight_col: the column name that saves the weights of weighted average
    :return: the weighted average
    """
    weights = group[value_col]
    values = group[weight_col]
    return np.average(values, weights=weights)


def read_local_csv_file(path, filename, dtype_str=True):
    if dtype_str:
        dataframe = pd.read_csv(os.path.join(path, filename), encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC,
                                dtype='str', index_col=0)
    else:
        dataframe = pd.read_csv(os.path.join(path, filename), encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC,
                                index_col=0)
    return dataframe


def transform_string_time_to_datetime(string):
    """
    :param string: the string which records the time of the posted tweets(this string's timezone is HK time)
    :return: a datetime object which could get access to the year, month, day easily
    """
    datetime_object = datetime.strptime(string, '%Y-%m-%d %H:%M:%S+08:00')
    final_time_object = datetime_object.replace(tzinfo=time_zone_hk)
    return final_time_object


def number_of_tweet_user(df: pd.DataFrame, print_values=False):
    """
    Output the number of tweets and number of unique social media users for a tweet dataframe
    :param df: a tweet dataframe
    :param print_values: whether print the results or not
    :return: A description or the results of the number of tweets and number of unique social media users
    """
    user_num = len(set(df['user_id_str']))
    tweet_num = df.shape[0]
    if print_values:
        print('Total number of tweet is: {}; Total number of user is {}'.format(tweet_num, user_num))
    else:
        return tweet_num, user_num


# Use this function to select the MTR-related tweets
def find_tweet(keywords, tweet):
    result = ''
    for word in tweet:
        if word in keywords:
            result = True
        else:
            result = False
    return result


# get the hk_time column based on the created_at column
def get_hk_time(df):
    changed_time_list = []
    for _, row in df.iterrows():
        time_to_change = datetime.strptime(row['created_at'], '%a %b %d %H:%M:%S %z %Y')
        # get the hk time
        changed_time = time_to_change.astimezone(time_zone_hk)
        changed_time_list.append(changed_time)
    df['hk_time'] = changed_time_list
    return df


# get the year, month, day information of based on any tweet dataframe
def get_year_month_day(df):
    df_copy = df.copy()
    df_copy['year'] = df_copy.apply(lambda row: row['hk_time'].year, axis=1)
    df_copy['month'] = df_copy.apply(lambda row: row['hk_time'].month, axis=1)
    df_copy['day'] = df_copy.apply(lambda row: row['hk_time'].day, axis=1)
    return df_copy


def read_text_from_multi_csvs(path: str) -> pd.DataFrame:
    """
    Combine csv files in a local path
    :param path: a path containing many csv files
    :return: a combined pandas dataframe
    """
    all_csv_files = os.listdir(path)
    dataframes = []
    for file in all_csv_files:
        dataframe = pd.read_csv(os.path.join(path, file), encoding='latin-1', dtype='str',
                                quoting=csv.QUOTE_NONNUMERIC)
        dataframes.append(dataframe)
    combined_dataframes = pd.concat(dataframes, sort=True)
    return combined_dataframes


def build_dataframe_for_urban_rate(source_dataframe):
    result_dataframe = pd.DataFrame(columns=['Year', 'US', 'China', 'World'])
    China_dataframe = source_dataframe.loc[source_dataframe['Country Name'] == 'China']
    us_dataframe = source_dataframe.loc[source_dataframe['Country Name'] == 'United States']
    World_dataframe = source_dataframe.loc[source_dataframe['Country Name'] == 'World']
    year_list = list(range(1960, 2019, 1))
    result_dataframe['Year'] = year_list
    result_dataframe['US'] = us_dataframe.values[0][4:]
    result_dataframe['China'] = China_dataframe.values[0][4:]
    result_dataframe['World'] = World_dataframe.values[0][4:]
    return result_dataframe


def build_line_graph_urban_rate(dataframe):
    x = list(dataframe['Year'])
    y_china = list(dataframe['China'])
    y_us = list(dataframe['US'])
    y_world = list(dataframe['World'])

    figure, ax = plt.subplots(1, 1, figsize=(20, 10))
    lns1 = ax.plot(x, y_world, 'k-', label='World', linestyle='--', marker='o')
    lns2 = ax.plot(x, y_china, 'y-', label='China', linestyle='--', marker='^')
    lns3 = ax.plot(x, y_us, 'b-', label='US', linestyle='--', marker='^')

    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs)

    ax.set_xlabel('Time')
    ax.set_ylabel('Urban Population Rate %')
    ax.set_title('Urban Population Rate for US, China and World from 1960 to 2018')
    plt.savefig(os.path.join(data_paths.plot_path_2017, 'urban_rate_plot.png'))
    plt.show()


def build_bar_plot_distribution_comparison(**key_list_dict: dict):
    """
    Build the bar plot showing the sentiment distribution
    :param key_list_dict: a dictionary containing the number of tweets for each sentiment category
    :return:
    """
    name_list = list(key_list_dict.keys())
    if len(name_list) == 1:
        value_list = key_list_dict[name_list[0]]
    else:
        return 'Not Worked Out'
    sentiment_tag = ['Positive', 'Neutral', 'Negative']
    x_values_for_plot = list(range(len(sentiment_tag)))
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.bar(x_values_for_plot, value_list, color='red')
    ax.set_xticks(x_values_for_plot)
    ax.set_xticklabels(sentiment_tag)
    filename = name_list[0] + '_distribution'
    fig.savefig(os.path.join(data_paths.human_review_result_path, filename))
    plt.show()


def classifiers_performance_compare(filename):
    result_dataframe = pd.DataFrame(columns=['metrics', 'performance', 'Classifiers'])

    accuracy_list = [0.64, 0.70, 0.70, 0.67]
    precision_list = [0.48, 0.47, 0.51, 0.52]
    recall_list = [0.50, 0.48, 0.51, 0.54]
    f1_list = [0.47, 0.47, 0.51, 0.50]

    performance_list = accuracy_list + precision_list + recall_list + f1_list
    metrics_list = ['Accuracy'] * 4 + ['Precision'] * 4 + ['Recall'] * 4 + ['F1 Score'] * 4
    classifier_list = ['Decision Tree', 'Random Forest', 'SVM', 'Neural Net'] * 4

    result_dataframe['metrics'] = metrics_list
    result_dataframe['performance'] = performance_list
    result_dataframe['Classifiers'] = classifier_list

    fig_classifier_compare, ax = plt.subplots(1, 1, figsize=(10, 8))
    # qualitative_colors = sns.color_palette("Set1", 4)
    # sns.set_palette(qualitative_colors)
    sns.barplot(x="metrics", y="performance", hue="Classifiers", data=result_dataframe, ax=ax,
                palette=["#6553FF", "#E8417D", "#FFAC42", '#A5FF47'])
    fig_classifier_compare.savefig(os.path.join(data_paths.human_review_result_path, filename))
    plt.show()


def draw_urban_rate_main(dataframe):
    data_for_plot = build_dataframe_for_urban_rate(dataframe)
    build_line_graph_urban_rate(dataframe=data_for_plot)


def general_info_of_tweet_dataset(df, study_area: str):
    """
    Get the general info of a tweet dataframe
    :param df: a pandas tweet dataframe having been sorted by time
    :param study_area: a string describing the study area
    :return: None. A short description of the tweet dataframe is given, including user number, tweet number,
    average number of tweets per day, language distribution and sentiment distribution
    """
    user_number = len(set(list(df['user_id_str'])))
    tweet_number = df.shape[0]
    starting_time = list(df['hk_time'])[0]
    ending_time = list(df['hk_time'])[-1]
    daily_tweet_count = df.shape[0] / (ending_time - starting_time).days
    language_dist_dict = Counter(df['lang'])
    sentiment_dist_dict = Counter(df['sentiment'])
    print('For {}, number of users: {}; number of tweets: {}; average daily number of tweets: {}; '
          'language distribution: {}; sentiment distribution: {}'.format(study_area, user_number,
                                                                         tweet_number,
                                                                         daily_tweet_count, language_dist_dict,
                                                                         sentiment_dist_dict))


def get_tweets_before_after(df: pd.DataFrame, oct_open: bool,
                            start_time=datetime(2016, 5, 1, 0, 0, 0, tzinfo=time_zone_hk),
                            end_time=datetime(2018, 12, 31, 23, 59, 59, tzinfo=time_zone_hk)):
    """
    Get the tweets posted before and after the introduction of MTR stations
    :param df: a pandas tweet dataframe
    :param oct_open: whether the station in opened on October, 2016 or not
    :param start_time: the considered start time of the tweets
    :param end_time: the considered end time of the tweets
    :return: two pandas dataframe, one is for the 'before' period and another is for the 'after' period
    """
    df_copy = df.copy()
    if isinstance(list(df['hk_time'])[0], str):
        df_copy['hk_time'] = df_copy.apply(lambda row: transform_string_time_to_datetime(row['hk_time']), axis=1)
    else:
        pass
    # Set the user id to int
    df_copy['user_id_str'] = df_copy['user_id_str'].astype(float)
    df_copy['user_id_str'] = df_copy['user_id_str'].astype(np.int64)
    df_copy_sorted = df_copy.sort_values(by='hk_time')
    if oct_open:
        before_time_mask = (df_copy_sorted['hk_time'] < october_1_start) & (start_time <= df_copy_sorted['hk_time'])
        after_time_mask = (df_copy_sorted['hk_time'] > october_31_end) & (end_time > df_copy_sorted['hk_time'])
    else:
        before_time_mask = (df_copy_sorted['hk_time'] < december_1_start) & (start_time <= df_copy_sorted['hk_time'])
        after_time_mask = (df_copy_sorted['hk_time'] > december_31_end) & (end_time > df_copy_sorted['hk_time'])
    df_before = df_copy_sorted.loc[before_time_mask]
    df_after = df_copy_sorted.loc[after_time_mask]
    return df_before, df_after


if __name__ == '__main__':
    urban_rate_dataframe = pd.read_csv(os.path.join(data_paths.datasets, 'urban_rate.csv'), encoding='latin-1',
                                       dtype=str)
    draw_urban_rate_main(urban_rate_dataframe)

    # draw the barplot which shows the distribution of sentiment label
    build_bar_plot_distribution_comparison(**{'total_sentiment_label_comparison': [1942, 2920, 137]})

    # draw bar plot which show the performance of various algorithms
    classifiers_performance_compare(filename='classifier_performance_compare.png')

    # Output general information of the dataframes involved in the longitudinal study
    treatment_control_saving_path = os.path.join(data_paths.transit_non_transit_comparison_before_after,
                                                 'three_areas_longitudinal_analysis')
    kwun_tong_line_treatment_dataframe = read_local_csv_file(filename='kwun_tong_line_treatment.csv',
                                                             path=treatment_control_saving_path, dtype_str=False)
    kwun_tong_line_control_dataframe = read_local_csv_file(filename='kwun_tong_line_control_1000.csv',
                                                           path=treatment_control_saving_path, dtype_str=False)
    south_horizons_treatment_dataframe = read_local_csv_file(filename='south_horizons_lei_tung_treatment.csv',
                                                             path=treatment_control_saving_path, dtype_str=False)
    south_horizons_control_dataframe = read_local_csv_file(filename='south_horizons_lei_tung_control_1500.csv',
                                                           path=treatment_control_saving_path, dtype_str=False)
    ocean_park_treatment_dataframe = read_local_csv_file(filename='ocean_park_wong_chuk_hang_treatment.csv',
                                                         path=treatment_control_saving_path, dtype_str=False)
    ocean_park_control_dataframe = read_local_csv_file(filename='ocean_park_wong_chuk_hang_control_1500.csv',
                                                       path=treatment_control_saving_path, dtype_str=False)
