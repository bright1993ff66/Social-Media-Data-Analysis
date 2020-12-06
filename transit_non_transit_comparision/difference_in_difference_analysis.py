import pandas as pd
import numpy as np
import csv
import os
from collections import Counter

import pytz
from datetime import datetime
from dateutil.relativedelta import relativedelta

import data_paths
import before_and_after_final_tpu
import utils

# statistics
import statsmodels.api as sm
import statsmodels.formula.api as smf

from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import rc
rc('mathtext', default='regular')

# Hong Kong and Shanghai share the same time zone.
# Hence, we transform the utc time in our dataset into Shanghai time
time_zone_hk = pytz.timezone('Asia/Shanghai')
october_1_start = datetime(2016, 10, 1, 0, 0, 0, tzinfo=time_zone_hk)
october_31_end = datetime(2016, 10, 31, 23, 59, 59, tzinfo=time_zone_hk)
december_1_start = datetime(2016, 12, 1, 0, 0, 0, tzinfo=time_zone_hk)
december_31_end = datetime(2016, 12, 31, 23, 59, 59, tzinfo=time_zone_hk)


def transform_string_time_to_datetime(string):
    """
    :param string: the string which records the time of the posted tweets
    :return: a datetime object which could get access to the year, month, day easily
    """
    assert isinstance(string, str)
    datetime_object = datetime.strptime(string, '%Y-%m-%d %H:%M:%S+08:00')
    final_time_object = datetime_object.replace(tzinfo=time_zone_hk)
    return final_time_object


def add_post_variable(string, opening_start_date, opening_end_date, check_window=0):
    """
    Add the value of the POST variable in the DID analysis
    :param string: the time string
    :param opening_start_date: the opening date of the studied station
    :param opening_end_date: the closing date of the studied station
    :param check_window: the month window size used to check the temporal effect of the studied station
    :return: the post variable based on the time of one tweet
    """
    time_object = transform_string_time_to_datetime(string)
    if check_window == 0:
        if time_object > opening_end_date:
            return 1
        elif time_object < opening_start_date:
            return 0
        else:
            return 'not considered'
    else:
        left_time_range = opening_start_date - relativedelta(months=check_window)
        right_time_range = opening_end_date + relativedelta(months=check_window)
        if left_time_range <= time_object < opening_start_date:
            return 0
        elif opening_end_date < time_object <= right_time_range:
            return 1
        else:
            return 'not considered'


def build_regress_datafrane_for_one_newly_built_station(treatment_dataframe, control_dataframe,
                                                        station_open_start_date, station_open_end_date,
                                                        open_year_plus_month, check_window_value=0):
    """
    build the regression model for a specified area
    :param treatment_dataframe: the tweet dataframe of the treatment area
    :param control_dataframe: the tweet dataframe of the control area
    :param station_open_month_start: the opening month date of the studied MTR station
    :param station_open_month_end: the ending month date of the studied MTR station
    :param open_year_plus_month: a string which contains open year plus month. For instance, 2016_10
    :return: a dataframe for DID analysis
    """
    # check the date
    assert open_year_plus_month in ['2016_10', '2016_12']
    result_dataframe = pd.DataFrame(columns=['Time', 'T_i_t', 'Post', 'Sentiment', 'Activity'])
    # build the T_i_t variable
    ones_list = [1] * treatment_dataframe.shape[0]
    treatment_dataframe['T_i_t'] = ones_list
    zeros_list = [0] * control_dataframe.shape[0]
    control_dataframe['T_i_t'] = zeros_list
    # build the post variable
    treatment_dataframe['Post'] = treatment_dataframe.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=station_open_start_date,
                                      opening_end_date=station_open_end_date, check_window=check_window_value), axis=1)
    print('Check the post variable distribution of treatment group: {}'.format(
        Counter(treatment_dataframe['Post'])))
    print('Check the T_i_t variable distribution of treatment group: {}'.format(
        Counter(treatment_dataframe['T_i_t'])))
    control_dataframe['Post'] = control_dataframe.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=station_open_start_date,
                                      opening_end_date=station_open_end_date, check_window=check_window_value), axis=1)
    print('Check the post variable distribution of control group: {}'.format(
        Counter(control_dataframe['Post'])))
    print('Check the T_i_t variable distribution of control group: {}'.format(
        Counter(control_dataframe['T_i_t'])))
    combined_dataframe = pd.concat([treatment_dataframe, control_dataframe], axis=0, sort=True)
    combined_dataframe = combined_dataframe.reset_index(drop=True)
    # We don't consider the tweets posted on the open date
    combined_dataframe_without_not_considered = \
        combined_dataframe.loc[combined_dataframe['Post'] != 'not considered']
    combined_data_copy = combined_dataframe_without_not_considered.copy()
    combined_data_copy['month_plus_year'] = combined_data_copy.apply(
        lambda row: str(int(float(row['year']))) + '_' + str(int(float(row['month']))), axis=1)
    sentiment_dict = {}
    activity_dict = {}
    for _, dataframe in combined_data_copy.groupby(['month_plus_year', 'T_i_t', 'Post']):
        time = str(list(dataframe['month_plus_year'])[0])
        t_i_t = str(list(dataframe['T_i_t'])[0])
        post = str(list(dataframe['Post'])[0])
        sentiment_dict[time + '_' + t_i_t + '_' + post] = before_and_after_final_tpu.pos_percent_minus_neg_percent(
            dataframe)
        activity_dict[time + '_' + t_i_t + '_' + post] = np.log(dataframe.shape[0])
    result_dataframe_copy = result_dataframe.copy()
    time_list = []
    t_i_t_list = []
    post_list = []
    sentiment_list = []
    activity_list = []
    for key in list(sentiment_dict.keys()):
        # don't consider the tweets posted in 2016_10(for Whampoa and Ho Man Tin) or 2016_12(for other stations)
        if key[:7] != open_year_plus_month:
            time_list.append(key[:-4])
            t_i_t_list.append(int(key[-3]))
            post_list.append(int(key[-1]))
            sentiment_list.append(sentiment_dict[key])
            activity_list.append(activity_dict[key])
        else:
            pass
    result_dataframe_copy['Time'] = time_list
    result_dataframe_copy['T_i_t'] = t_i_t_list
    result_dataframe_copy['Post'] = post_list
    result_dataframe_copy['Sentiment'] = sentiment_list
    result_dataframe_copy['Activity'] = activity_list
    return result_dataframe_copy


def build_regress_dataframe_for_combined_areas(kwun_tong_treatment, kwun_tong_control, south_horizons_treatment,
                                               south_horizons_control, ocean_park_treatment, ocean_park_control,
                                               check_window_value=0, sentiment_did = False):
    """
    Build dataframes for the combined DID analysis based on treatment & control dataframes of each station
    :param kwun_tong_treatment: the dataframe saving tweets for kwun tong treatment area
    :param kwun_tong_control: the dataframe saving tweets for kwun tong control area
    :param south_horizons_treatment: the dataframe saving tweets for south horizons treatment area
    :param south_horizons_control: the dataframe saving tweets for south horizons control area
    :param ocean_park_treatment: the dataframe saving tweets for ocean park treatment area
    :param ocean_park_control: the dataframe saving tweets for ocean park control area
    :param check_window_value: the month window we consider when doing the DID analysis
    :return: a combined dataframe which could be used for combined DID analysis
    """
    result_dataframe = pd.DataFrame(columns=['Time', 'T_i_t', 'Post', 'Sentiment', 'Activity'])
    # build the treatment control binary variable
    kwun_tong_treatment['T_i_t'] = [1] * kwun_tong_treatment.shape[0]
    kwun_tong_control['T_i_t'] = [0] * kwun_tong_control.shape[0]
    south_horizons_treatment['T_i_t'] = [1] * south_horizons_treatment.shape[0]
    south_horizons_control['T_i_t'] = [0] * south_horizons_control.shape[0]
    ocean_park_treatment['T_i_t'] = [1] * ocean_park_treatment.shape[0]
    ocean_park_control['T_i_t'] = [0] * ocean_park_control.shape[0]
    # add the post variable
    kwun_tong_treatment['Post'] = kwun_tong_treatment.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=october_1_start,
                                      opening_end_date=october_31_end, check_window=check_window_value), axis=1)
    kwun_tong_control['Post'] = kwun_tong_control.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=october_1_start,
                                      opening_end_date=october_31_end, check_window=check_window_value), axis=1)
    south_horizons_treatment['Post'] = south_horizons_treatment.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=december_1_start,
                                      opening_end_date=december_31_end, check_window=check_window_value), axis=1)
    south_horizons_control['Post'] = south_horizons_control.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=december_1_start,
                                      opening_end_date=december_31_end, check_window=check_window_value), axis=1)
    ocean_park_treatment['Post'] = ocean_park_treatment.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=december_1_start,
                                      opening_end_date=december_31_end, check_window=check_window_value), axis=1)
    ocean_park_control['Post'] = ocean_park_control.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=december_1_start,
                                      opening_end_date=december_31_end, check_window=check_window_value), axis=1)

    dataframe_list = [kwun_tong_treatment, kwun_tong_control, south_horizons_treatment,
                          south_horizons_control, ocean_park_treatment, ocean_park_control]
    combined_dataframe = pd.concat(dataframe_list, axis=0, sort=True)
    combined_dataframe = combined_dataframe.reset_index(drop=True)
    # We don't consider the tweets posted on the open date
    combined_dataframe_without_not_considered = \
        combined_dataframe.loc[combined_dataframe['Post'] != 'not considered']
    combined_data_copy = combined_dataframe_without_not_considered.copy()
    combined_data_copy['month_plus_year'] = combined_data_copy.apply(
        lambda row: str(int(float(row['year']))) + '_' + str(int(float(row['month']))), axis=1)

    result_dataframe_copy = result_dataframe.copy()
    if sentiment_did:
        time_list = []
        t_i_t_list = []
        post_list = []
        sentiment_list = []
        sentiment_dict = {}
        for _, dataframe in combined_data_copy.groupby(['month_plus_year', 'T_i_t', 'Post']):
            time = str(list(dataframe['month_plus_year'])[0])
            t_i_t = str(list(dataframe['T_i_t'])[0])
            post = str(list(dataframe['Post'])[0])
            sentiment_dict[time + '_' + t_i_t + '_' + post] = before_and_after_final_tpu.pos_percent_minus_neg_percent(
                dataframe)
        for key in list(sentiment_dict.keys()):
            # don't consider the tweets posted in 2016_10(for Whampoa and Ho Man Tin) or 2016_12(for other stations)
            if key[:7] not in ['2016_10', '2016_12']:
                time_list.append(key[:-4])
                t_i_t_list.append(int(key[-3]))
                post_list.append(int(key[-1]))
                sentiment_list.append(sentiment_dict[key])
            else:
                pass
        result_dataframe_copy['Time'] = time_list
        result_dataframe_copy['T_i_t'] = t_i_t_list
        result_dataframe_copy['Post'] = post_list
        result_dataframe_copy['Sentiment'] = sentiment_list
    else:
        time_list = []
        t_i_t_list = []
        post_list = []
        activity_list = []
        activity_dict = {}
        for _, dataframe in combined_data_copy.groupby(['month_plus_year', 'T_i_t', 'Post']):
            time = str(list(dataframe['month_plus_year'])[0])
            t_i_t = str(list(dataframe['T_i_t'])[0])
            post = str(list(dataframe['Post'])[0])
            activity_dict[time + '_' + t_i_t + '_' + post] = np.log(dataframe.shape[0])
        for key in list(activity_dict.keys()):
            # don't consider the tweets posted in 2016_10(for Whampoa and Ho Man Tin) or 2016_12(for other stations)
            if key[:7] not in ['2016_10', '2016_12']:
                time_list.append(key[:-4])
                t_i_t_list.append(int(key[-3]))
                post_list.append(int(key[-1]))
                activity_list.append(activity_dict[key])
            else:
                pass
        result_dataframe_copy['Time'] = time_list
        result_dataframe_copy['T_i_t'] = t_i_t_list
        result_dataframe_copy['Post'] = post_list
        result_dataframe_copy['Activity'] = activity_list
    return result_dataframe_copy


def build_regress_dataframe_for_one_newly_built_station(treatment_dataframe, control_dataframe,
                                                        station_open_month_start, station_open_month_end,
                                                        open_year_plus_month, check_window_value=0):
    """
    Build the dataframe for one influenced area
    :param treatment_dataframe: the tweet dataframe for treatment area
    :param control_dataframe: the tweet dataframe for control area
    :param station_open_month_start: the starting time of the month when the studied station opens
    :param station_open_month_end: the ending time of the month when the studied station opens
    :param open_year_plus_month: the month plus year information
    :param check_window_value: the window size for DID analysis
    :return: a pandas dataframe which could be used for the following DID analysis
    """
    # check the date
    assert open_year_plus_month in ['2016_10', '2016_12']
    result_dataframe = pd.DataFrame(columns=['Time', 'T_i_t', 'Post', 'Sentiment', 'Activity', 'Activity_log'])
    # build the T_i_t variable
    ones_list = [1] * treatment_dataframe.shape[0]
    treatment_dataframe['T_i_t'] = ones_list
    zeros_list = [0] * control_dataframe.shape[0]
    control_dataframe['T_i_t'] = zeros_list
    # build the post variable
    treatment_dataframe['Post'] = treatment_dataframe.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=station_open_month_start,
                                      opening_end_date=station_open_month_end, check_window=check_window_value), axis=1)
    print('Check the post variable distribution of treatment group: {}'.format(
        Counter(treatment_dataframe['Post'])))
    print('Check the T_i_t variable distribution of treatment group: {}'.format(
        Counter(treatment_dataframe['T_i_t'])))
    control_dataframe['Post'] = control_dataframe.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=station_open_month_start,
                                      opening_end_date=station_open_month_end, check_window=check_window_value), axis=1)
    print('Check the post variable distribution of control group: {}'.format(
        Counter(control_dataframe['Post'])))
    print('Check the T_i_t variable distribution of control group: {}'.format(
        Counter(control_dataframe['T_i_t'])))
    combined_dataframe = pd.concat([treatment_dataframe, control_dataframe], axis=0, sort=True)
    combined_dataframe = combined_dataframe.reset_index(drop=True)
    # We don't consider the tweets posted on the open date
    combined_dataframe_without_not_considered = \
        combined_dataframe.loc[combined_dataframe['Post'] != 'not considered']
    combined_data_copy = combined_dataframe_without_not_considered.copy()
    combined_data_copy['month_plus_year'] = combined_data_copy.apply(
        lambda row: str(int(float(row['year']))) + '_' + str(int(float(row['month']))), axis=1)
    sentiment_dict = {}
    activity_dict = {}
    activity_dict_log = {}
    for _, dataframe in combined_data_copy.groupby(['month_plus_year', 'T_i_t', 'Post']):
        time = str(list(dataframe['month_plus_year'])[0])
        t_i_t = str(list(dataframe['T_i_t'])[0])
        post = str(list(dataframe['Post'])[0])
        sentiment_dict[time + '_' + t_i_t + '_' + post] = before_and_after_final_tpu.pos_percent_minus_neg_percent(
            dataframe)
        activity_dict[time + '_' + t_i_t + '_' + post] = dataframe.shape[0]
        activity_dict_log[time + '_' + t_i_t + '_' + post] = np.log(dataframe.shape[0])
    result_dataframe_copy = result_dataframe.copy()
    time_list = []
    t_i_t_list = []
    post_list = []
    sentiment_list = []
    activity_list = []
    activity_log_list = []
    for key in list(sentiment_dict.keys()):
        # don't consider the tweets posted in 2016_10(for Whampoa and Ho Man Tin) or 2016_12(for other stations)
        if key[:7] != open_year_plus_month:
            time_list.append(key[:-4])
            t_i_t_list.append(int(key[-3]))
            post_list.append(int(key[-1]))
            sentiment_list.append(sentiment_dict[key])
            activity_list.append(activity_dict[key])
            activity_log_list.append(activity_dict_log[key])
        else:
            pass
    result_dataframe_copy['Time'] = time_list
    result_dataframe_copy['T_i_t'] = t_i_t_list
    result_dataframe_copy['Post'] = post_list
    result_dataframe_copy['Sentiment'] = sentiment_list
    result_dataframe_copy['Activity'] = activity_list
    result_dataframe_copy['Activity_log'] = activity_log_list
    return result_dataframe_copy


def conduct_combined_did_analysis(kwun_tong_treatment_dataframe, kwun_tong_control_dataframe,
                                  south_horizons_treatment_dataframe, south_horizons_control_dataframe,
                                  ocean_park_treatment_dataframe, ocean_park_control_dataframe,
                                  dataframe_saving_path, filename, check_window_value=0, for_sentiment=False):

    longitudinal_dataframe = build_regress_dataframe_for_combined_areas(
        kwun_tong_treatment=kwun_tong_treatment_dataframe,
        kwun_tong_control=kwun_tong_control_dataframe,
        south_horizons_treatment=south_horizons_treatment_dataframe,
        south_horizons_control=south_horizons_control_dataframe, ocean_park_treatment=ocean_park_treatment_dataframe,
        ocean_park_control=ocean_park_control_dataframe, check_window_value=check_window_value,
        sentiment_did=for_sentiment)
    longitudinal_dataframe.to_csv(os.path.join(dataframe_saving_path, filename))

    if for_sentiment:
        reg_combined_sentiment = smf.ols('Sentiment ~ T_i_t+Post+T_i_t:Post', longitudinal_dataframe).fit()
        print('----The sentiment did result-----')
        print(reg_combined_sentiment.summary())
    else:
        reg_combined_activity = smf.ols('Activity ~ T_i_t+Post+T_i_t:Post', longitudinal_dataframe).fit()
        print('----The activity did result-----')
        print(reg_combined_activity.summary())


def conduct_did_analysis_one_area(treatment_considered_dataframe, control_considered_dataframe, opening_start_date,
                                  opening_end_date, open_year_month, window_size_value, file_path, filename):
    constructed_dataframe = build_regress_dataframe_for_one_newly_built_station(
        treatment_dataframe=treatment_considered_dataframe,
        control_dataframe=control_considered_dataframe,
        station_open_month_start=opening_start_date,
        station_open_month_end=opening_end_date, open_year_plus_month=open_year_month,
        check_window_value=window_size_value)
    # constructed_dataframe_selected = constructed_dataframe.loc[constructed_dataframe['Activity'] >= 10]
    constructed_dataframe.to_csv(os.path.join(file_path, filename), encoding='utf-8')
    combined_sentiment = smf.ols('Sentiment ~ T_i_t+Post+T_i_t:Post', constructed_dataframe).fit()
    combined_activity = smf.ols('Activity_log ~ T_i_t+Post+Post:T_i_t:Post', constructed_dataframe).fit()
    print('----The sentiment did result-----')
    print(combined_sentiment.summary())
    print('----The activity did result-----')
    print(combined_activity.summary())
    print('-------------------------------------------------------\n')


def build_dataframe_based_on_set(datapath, tpu_set):
    tpu_name_list = []
    dataframe_list = []
    for tpu in tpu_set:
        tpu_name_list.append(tpu)
        dataframe = pd.read_csv(os.path.join(datapath, tpu, tpu+'_data.csv'), encoding='utf-8', dtype='str',
                                quoting=csv.QUOTE_NONNUMERIC)
        tweet_num, user_num = utils.number_of_tweet_user(dataframe, print_values=False)
        # print('For TPU {}, the general info of tweet data is: tweet num: {}; unique tweet user: {}'.format(
        #     tpu, tweet_num, user_num))
        dataframe_list.append(dataframe)
    combined_dataframe = pd.concat(dataframe_list, axis=0)
    return combined_dataframe


if __name__ == '__main__':

    path = os.path.join(data_paths.tweet_combined_path, 'longitudinal_tpus')

    kwun_tong_line_treatment_tpu_set = {'243', '245', '236', '213'}
    kwun_tong_line_control_tpu_set = {'247', '234', '242', '212', '235'}
    south_horizons_lei_tung_treatment_tpu_set = {'174'}
    south_horizons_lei_tung_control_tpu_set = {'172', '182'}
    ocean_park_wong_chuk_hang_treatment_tpu_set = {'175'}
    ocean_park_wong_chuk_hang_control_tpu_set = {'184', '183', '182'}
    south_island_treatment_tpu_set = {'174', '175'}
    south_island_control_tpu_set = {'172', '182', '183', '184'}
    # tpu_213_set, tpu_236_set, tpu_243_set, tpu_245_set = {'213'}, {'236'}, {'243'}, {'245'}

    print('Treatment & Control Data Overview...')
    kwun_tong_line_treatment_dataframe = build_dataframe_based_on_set(datapath=path,
                                                                      tpu_set=kwun_tong_line_treatment_tpu_set)
    print('For Kwun Tong Line treatment...')
    utils.number_of_tweet_user(kwun_tong_line_treatment_dataframe)
    kwun_tong_line_control_dataframe = build_dataframe_based_on_set(datapath=path,
                                                                tpu_set=kwun_tong_line_control_tpu_set)
    print('For Kwun Tong Line control sentiment...')
    utils.number_of_tweet_user(kwun_tong_line_control_dataframe)

    south_horizons_lei_tung_treatment_dataframe = build_dataframe_based_on_set(datapath=path,
                                                                tpu_set=south_horizons_lei_tung_treatment_tpu_set)

    south_horizons_lei_tung_control_dataframe = build_dataframe_based_on_set(datapath=path,
                                                                tpu_set=south_horizons_lei_tung_control_tpu_set)

    ocean_park_wong_chuk_hang_treatment_dataframe = build_dataframe_based_on_set(datapath=path,
                                                                tpu_set=ocean_park_wong_chuk_hang_treatment_tpu_set)

    ocean_park_wong_chuk_hang_control_dataframe = build_dataframe_based_on_set(datapath=path,
                                                                tpu_set=ocean_park_wong_chuk_hang_control_tpu_set)

    print('************************DID Analysis Starts....************************')
    dataframe_saving_path = os.path.join(data_paths.tweet_combined_path, 'longitudinal_did_analysis_dataframes')

    print('Overall Treatment and Control Comparison for sentiment(3 months)...')
    conduct_combined_did_analysis(kwun_tong_treatment_dataframe=kwun_tong_line_treatment_dataframe,
                                  kwun_tong_control_dataframe=kwun_tong_line_control_dataframe,
                                  south_horizons_treatment_dataframe=south_horizons_lei_tung_treatment_dataframe,
                                  south_horizons_control_dataframe=south_horizons_lei_tung_control_dataframe,
                                  ocean_park_treatment_dataframe=ocean_park_wong_chuk_hang_treatment_dataframe,
                                  ocean_park_control_dataframe=ocean_park_wong_chuk_hang_control_dataframe,
                                  dataframe_saving_path=dataframe_saving_path,
                                  filename='longitudinal_did_dataframe_3_months_sentiment.csv', check_window_value=3,
                                  for_sentiment=True)

    print('Overall Treatment and Control Comparison for activity(3 months)...')
    conduct_combined_did_analysis(kwun_tong_treatment_dataframe=kwun_tong_line_treatment_dataframe,
                                  kwun_tong_control_dataframe=kwun_tong_line_control_dataframe,
                                  south_horizons_treatment_dataframe=south_horizons_lei_tung_treatment_dataframe,
                                  south_horizons_control_dataframe=south_horizons_lei_tung_control_dataframe,
                                  ocean_park_treatment_dataframe=ocean_park_wong_chuk_hang_treatment_dataframe,
                                  ocean_park_control_dataframe=ocean_park_wong_chuk_hang_control_dataframe,
                                  dataframe_saving_path=dataframe_saving_path,
                                  filename='longitudinal_did_dataframe_3_months_activity.csv', check_window_value=3,
                                  for_sentiment=False)

    print('Overall Treatment and Control Comparison for sentiment(6 months)...')
    conduct_combined_did_analysis(kwun_tong_treatment_dataframe=kwun_tong_line_treatment_dataframe,
                                  kwun_tong_control_dataframe=kwun_tong_line_control_dataframe,
                                  south_horizons_treatment_dataframe=south_horizons_lei_tung_treatment_dataframe,
                                  south_horizons_control_dataframe=south_horizons_lei_tung_control_dataframe,
                                  ocean_park_treatment_dataframe=ocean_park_wong_chuk_hang_treatment_dataframe,
                                  ocean_park_control_dataframe=ocean_park_wong_chuk_hang_control_dataframe,
                                  dataframe_saving_path=dataframe_saving_path,
                                  filename='longitudinal_did_dataframe_6_months_sentiment.csv',
                                  check_window_value=6, for_sentiment=True)

    print('Overall Treatment and Control Comparison for activity(6 months)...')
    conduct_combined_did_analysis(kwun_tong_treatment_dataframe=kwun_tong_line_treatment_dataframe,
                                  kwun_tong_control_dataframe=kwun_tong_line_control_dataframe,
                                  south_horizons_treatment_dataframe=south_horizons_lei_tung_treatment_dataframe,
                                  south_horizons_control_dataframe=south_horizons_lei_tung_control_dataframe,
                                  ocean_park_treatment_dataframe=ocean_park_wong_chuk_hang_treatment_dataframe,
                                  ocean_park_control_dataframe=ocean_park_wong_chuk_hang_control_dataframe,
                                  dataframe_saving_path=dataframe_saving_path,
                                  filename='longitudinal_did_dataframe_6_months_activity.csv',
                                  check_window_value=6, for_sentiment=False)

    print('Overall Treatment and Control Comparison for sentiment(12 months)...')
    conduct_combined_did_analysis(kwun_tong_treatment_dataframe=kwun_tong_line_treatment_dataframe,
                                  kwun_tong_control_dataframe=kwun_tong_line_control_dataframe,
                                  south_horizons_treatment_dataframe=south_horizons_lei_tung_treatment_dataframe,
                                  south_horizons_control_dataframe=south_horizons_lei_tung_control_dataframe,
                                  ocean_park_treatment_dataframe=ocean_park_wong_chuk_hang_treatment_dataframe,
                                  ocean_park_control_dataframe=ocean_park_wong_chuk_hang_control_dataframe,
                                  dataframe_saving_path=dataframe_saving_path,
                                  filename='longitudinal_did_dataframe_12_months_sentiment.csv',
                                  check_window_value=12, for_sentiment=True)

    print('Overall Treatment and Control Comparison for activity(12 months)...')
    conduct_combined_did_analysis(kwun_tong_treatment_dataframe=kwun_tong_line_treatment_dataframe,
                                  kwun_tong_control_dataframe=kwun_tong_line_control_dataframe,
                                  south_horizons_treatment_dataframe=south_horizons_lei_tung_treatment_dataframe,
                                  south_horizons_control_dataframe=south_horizons_lei_tung_control_dataframe,
                                  ocean_park_treatment_dataframe=ocean_park_wong_chuk_hang_treatment_dataframe,
                                  ocean_park_control_dataframe=ocean_park_wong_chuk_hang_control_dataframe,
                                  dataframe_saving_path=dataframe_saving_path,
                                  filename='longitudinal_did_dataframe_12_months_activity.csv',
                                  check_window_value=12, for_sentiment=False)

    print('Overall Treatment and Control Comparison(sentiment)...')
    conduct_combined_did_analysis(kwun_tong_treatment_dataframe=kwun_tong_line_treatment_dataframe,
                                  kwun_tong_control_dataframe=kwun_tong_line_control_dataframe,
                                  south_horizons_treatment_dataframe=south_horizons_lei_tung_treatment_dataframe,
                                  south_horizons_control_dataframe=south_horizons_lei_tung_control_dataframe,
                                  ocean_park_treatment_dataframe=ocean_park_wong_chuk_hang_treatment_dataframe,
                                  ocean_park_control_dataframe=ocean_park_wong_chuk_hang_control_dataframe,
                                  dataframe_saving_path=dataframe_saving_path,
                                  filename='longitudinal_did_dataframe_all_sentiment.csv', check_window_value=0,
                                  for_sentiment=True)
    print('Overall Treatment and Control Comparison(activity)...')
    conduct_combined_did_analysis(kwun_tong_treatment_dataframe=kwun_tong_line_treatment_dataframe,
                                  kwun_tong_control_dataframe=kwun_tong_line_control_dataframe,
                                  south_horizons_treatment_dataframe=south_horizons_lei_tung_treatment_dataframe,
                                  south_horizons_control_dataframe=south_horizons_lei_tung_control_dataframe,
                                  ocean_park_treatment_dataframe=ocean_park_wong_chuk_hang_treatment_dataframe,
                                  ocean_park_control_dataframe=ocean_park_wong_chuk_hang_control_dataframe,
                                  dataframe_saving_path=dataframe_saving_path,
                                  filename='longitudinal_did_dataframe_all_activity.csv', check_window_value=0,
                                  for_sentiment=False)

    print('Cope with the three areas seperately...')
    print('---------------------Kwun Tong Line---------------------------')

    print('For 3 months....')
    conduct_did_analysis_one_area(treatment_considered_dataframe=kwun_tong_line_treatment_dataframe,
                                  control_considered_dataframe=kwun_tong_line_control_dataframe,
                                  opening_start_date=october_1_start, opening_end_date=october_31_end,
                                  open_year_month='2016_10', window_size_value=3, file_path=dataframe_saving_path,
                                  filename='kwun_tong_did_3_months.csv')

    print('For 6 months....')
    conduct_did_analysis_one_area(treatment_considered_dataframe=kwun_tong_line_treatment_dataframe,
                                  control_considered_dataframe=kwun_tong_line_control_dataframe,
                                  opening_start_date=october_1_start, opening_end_date=october_31_end,
                                  open_year_month='2016_10', window_size_value=6, file_path=dataframe_saving_path,
                                  filename='kwun_tong_did_6_months.csv')

    print('For 12 months....')
    conduct_did_analysis_one_area(treatment_considered_dataframe=kwun_tong_line_treatment_dataframe,
                                  control_considered_dataframe=kwun_tong_line_control_dataframe,
                                  opening_start_date=october_1_start, opening_end_date=october_31_end,
                                  open_year_month='2016_10', window_size_value=12, file_path=dataframe_saving_path,
                                  filename='kwun_tong_did_12_months.csv')

    print('For all combined did analysis....')
    conduct_did_analysis_one_area(treatment_considered_dataframe=kwun_tong_line_treatment_dataframe,
                                  control_considered_dataframe=kwun_tong_line_control_dataframe,
                                  opening_start_date=october_1_start, opening_end_date=october_31_end,
                                  open_year_month='2016_10', window_size_value=0, file_path=dataframe_saving_path,
                                  filename='kwun_tong_did_all_months.csv')
    print('-------------------------------------------------------\n')

    print('---------------------South Horizons & Lei Tung---------------------------')
    print('For 3 months....')
    conduct_did_analysis_one_area(treatment_considered_dataframe=south_horizons_lei_tung_treatment_dataframe,
                                  control_considered_dataframe=south_horizons_lei_tung_control_dataframe,
                                  opening_start_date=december_1_start, opening_end_date=december_31_end,
                                  open_year_month='2016_12', window_size_value=3, file_path=dataframe_saving_path,
                                  filename='south_horizons_did_3_months.csv')
    print('For 6 months....')
    conduct_did_analysis_one_area(treatment_considered_dataframe=south_horizons_lei_tung_treatment_dataframe,
                                  control_considered_dataframe=south_horizons_lei_tung_control_dataframe,
                                  opening_start_date=december_1_start, opening_end_date=december_31_end,
                                  open_year_month='2016_12', window_size_value=6, file_path=dataframe_saving_path,
                                  filename='south_horizons_did_6_months.csv')
    print('For 12 months....')
    conduct_did_analysis_one_area(treatment_considered_dataframe=south_horizons_lei_tung_treatment_dataframe,
                                  control_considered_dataframe=south_horizons_lei_tung_control_dataframe,
                                  opening_start_date=december_1_start, opening_end_date=december_31_end,
                                  open_year_month='2016_12', window_size_value=12, file_path=dataframe_saving_path,
                                  filename='south_horizons_did_12_months.csv')
    print('For all combined did analysis....')
    conduct_did_analysis_one_area(treatment_considered_dataframe=south_horizons_lei_tung_treatment_dataframe,
                                  control_considered_dataframe=south_horizons_lei_tung_control_dataframe,
                                  opening_start_date=december_1_start, opening_end_date=december_31_end,
                                  open_year_month='2016_12', window_size_value=0, file_path=dataframe_saving_path,
                                  filename='south_horizons_did_all.csv')
    print('-------------------------------------------------------\n')

    print('---------------------Ocean Park & Wong Chuk Hang---------------------------')
    print('For 3 months....')
    conduct_did_analysis_one_area(treatment_considered_dataframe=ocean_park_wong_chuk_hang_treatment_dataframe,
                                  control_considered_dataframe=ocean_park_wong_chuk_hang_control_dataframe,
                                  opening_start_date=december_1_start, opening_end_date=december_31_end,
                                  open_year_month='2016_12', window_size_value=3, file_path=dataframe_saving_path,
                                  filename='ocean_park_did_3_months.csv')
    print('For 6 months....')
    conduct_did_analysis_one_area(treatment_considered_dataframe=ocean_park_wong_chuk_hang_treatment_dataframe,
                                  control_considered_dataframe=ocean_park_wong_chuk_hang_control_dataframe,
                                  opening_start_date=december_1_start, opening_end_date=december_31_end,
                                  open_year_month='2016_12', window_size_value=6, file_path=dataframe_saving_path,
                                  filename='ocean_park_did_6_months.csv')
    print('For 12 months....')
    conduct_did_analysis_one_area(treatment_considered_dataframe=ocean_park_wong_chuk_hang_treatment_dataframe,
                                  control_considered_dataframe=ocean_park_wong_chuk_hang_control_dataframe,
                                  opening_start_date=december_1_start, opening_end_date=december_31_end,
                                  open_year_month='2016_12', window_size_value=12, file_path=dataframe_saving_path,
                                  filename='ocean_park_did_12_months.csv')
    print('For all combined did analysis....')
    conduct_did_analysis_one_area(treatment_considered_dataframe=ocean_park_wong_chuk_hang_treatment_dataframe,
                                  control_considered_dataframe=ocean_park_wong_chuk_hang_control_dataframe,
                                  opening_start_date=december_1_start, opening_end_date=december_31_end,
                                  open_year_month='2016_12', window_size_value=0, file_path=dataframe_saving_path,
                                  filename='ocean_park_did_all.csv')
    print('-------------------------------------------------------\n')
