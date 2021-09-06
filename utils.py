# basics
import re
import os
import numpy as np
import pandas as pd
import pytz
from datetime import datetime
from collections import Counter

# Specify the timezone of Hong Kong
# No time difference between Shanghai and Hong Kong
time_zone_hk = pytz.timezone('Asia/Shanghai')

# For instance, if we want to compare the sentiment and activity level before and after the
# opening date of the Whampoa MTR railway station in Hong Kong, since the station is opened on 23 Oct 2016,
# we could specify the opening date using datetime package and output before and after dataframes
october_1_start = datetime(2016, 10, 1, 0, 0, 0, tzinfo=time_zone_hk)
october_31_end = datetime(2016, 10, 31, 23, 59, 59, tzinfo=time_zone_hk)
december_1_start = datetime(2016, 12, 1, 0, 0, 0, tzinfo=time_zone_hk)
december_31_end = datetime(2016, 12, 31, 23, 59, 59, tzinfo=time_zone_hk)
start_date = datetime(2016, 5, 7, 0, 0, 0, tzinfo=time_zone_hk)
end_date = datetime(2018, 12, 18, 23, 59, 59, tzinfo=time_zone_hk)


# A RegexpReplacer to clean some texts based on specified patterns
class RegexpReplacer(object):

    """
    Replace strings like "I'm, We won't" to "I am and we will not"
    """

    # The replacement patterns used in cleaning the raw text data
    replacement_patterns = [
        (r"won\'t", "will not"),
        (r"[^A-Za-z0-9^,!.\/'+-=]", " "),
        (r"can\'t", "cannot"),
        (r"I\'m", "I am"),
        (r"ain\'t", 'is not'),
        (r"(\d+)(k)", r"\g<1>000"),
        # \g<1> are using back-references to capture part of the matched pattern
        # \g means referencing group content in the previous pattern. <1> means the first group.
        # In the following case, the first group is w+
        (r"(\w+)\'ll", "\g<1> will"),
        (r"(\w+)n\'t", "\g<1> not"),
        (r"(\w+)\'ve", "\g<1> have"),
        (r"(\w+)\'s", "\g<1> is"),
        (r"(\w+)\'re", "\g<1> are"),
        (r"(\w+)\'d", "\g<1> would")
    ]

    def __init__(self):
        self.patterns = [(re.compile(regex), replace) for (regex, replace) in RegexpReplacer.replacement_patterns]

    def replace(self, text):
        """
        Replace the text given the regex expression
        :param text: a text string
        :return: the replaced text
        """
        s = text
        for (pattern, replace) in self.patterns:
            s = re.sub(pattern=pattern, repl=replace, string=s)  # subn returns the times of replacement
        return s


def weighted_average(group: pd.DataFrame, value_col: str, weight_col: str) -> float:
    """
    Compute the weighted average based on a pandas dataframe
    :param group: a pandas dataframe that we want to compute the weighted average
    :param value_col: the column name that saves the values of weighted average
    :param weight_col: the column name that saves the weights of weighted average
    :return: the weighted average
    """
    weights = group[weight_col]
    values = group[value_col]
    return np.average(values, weights=weights)


def read_local_csv_file(path, filename, dtype_str=True):
    """
    Read local csv file
    :param path: a path containing the desired csv file
    :param filename: the name of the desired csv file
    :param dtype_str: whether read each column as string or not
    :return: a loaded pandas dataframe
    """
    if dtype_str:
        dataframe = pd.read_csv(open(os.path.join(path, filename), errors='ignore', encoding='utf-8'), index_col=0)
    else:
        dataframe = pd.read_csv(open(os.path.join(path, filename), errors='ignore', encoding='utf-8'), index_col=0,
                                dtype='str')
    return dataframe


def transform_string_time_to_datetime(string):
    """
    Transform the string like "2018-05-30 15:30:26+8:00" to a datetime object
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


def get_year_month_day(df):
    """
    Get the year, month, and day information
    :param df: a pandas dataframe saving the tweets
    :return: a tweet dataframe with year, month, and day information
    """
    assert 'hk_time' in df, "The dataframe should have a column named hk_time"
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
        dataframe = pd.read_csv(open(os.path.join(path, file), errors='ignore', encoding='utf-8'), index_col=0,
                                dtype='str')
        dataframes.append(dataframe)
    combined_dataframes = pd.concat(dataframes, sort=True)
    combined_dataframes_reindex = combined_dataframes.reset_index(drop=True)
    return combined_dataframes_reindex


def general_info_of_tweet_dataset(df, study_area: str):
    """
    Get the general info of a tweet dataframe
    :param df: a pandas tweet dataframe
    :param study_area: a string describing the study area
    :return: None. A short description of the tweet dataframe is given, including user number, tweet number,
    average number of tweets per day, language distribution and sentiment distribution
    """
    assert 'hk_time' in df, "Make sure that the dataframe has a column named hk_time"
    df_sorted = df.sort_values(by='hk_time')
    user_number = len(set(list(df_sorted['user_id_str'])))
    tweet_number = df_sorted.shape[0]
    starting_time = list(df_sorted['hk_time'])[0]
    ending_time = list(df_sorted['hk_time'])[-1]
    daily_tweet_count = df_sorted.shape[0] / (ending_time - starting_time).days
    language_dist_dict = Counter(df_sorted['lang'])
    sentiment_dist_dict = Counter(df_sorted['sentiment'])
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
