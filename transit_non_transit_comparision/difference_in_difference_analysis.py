import pandas as pd
import numpy as np
import csv
import os
from collections import Counter

import pytz
from datetime import datetime
from dateutil.relativedelta import relativedelta

import read_data
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
october_23_start = datetime(2016, 10, 23, 0, 0, 0, tzinfo=time_zone_hk)
october_23_end = datetime(2016, 10, 23, 23, 59, 59, tzinfo=time_zone_hk)
december_28_start = datetime(2016, 12, 28, 0, 0, 0, tzinfo=time_zone_hk)
december_28_end = datetime(2016, 12, 28, 23, 59, 59, tzinfo=time_zone_hk)

before_after_stations = ['Whampoa', 'Ho Man Tin', 'South Horizons', 'Wong Chuk Hang', 'Ocean Park',
                             'Lei Tung']


def transform_string_time_to_datetime(string):
    """
    :param string: the string which records the time of the posted tweets
    :return: a datetime object which could get access to the year, month, day easily
    """
    datetime_object = datetime.strptime(string, '%Y-%m-%d %H:%M:%S+08:00')
    final_time_object = datetime_object.replace(tzinfo=time_zone_hk)
    return final_time_object


def get_tweets_based_on_date(file_path:str, station_name:str, start_date, end_date, buffer_radius=500):
    """
    :param file_path: path which saves the folders of each TN
    :param station_name: the name of MTR station in each TN
    :param start_date: the start date of the time range we consider
    :param end_date: the end date of the time range we consider
    :return: a filtered dataframe which contains tweets in a specific time range
    """
    combined_dataframe = pd.read_csv(os.path.join(file_path, station_name, station_name+'_{}m_tn_tweets.csv'.format(buffer_radius)),
                                     encoding='latin-1')
    combined_dataframe['hk_time'] = combined_dataframe.apply(
        lambda row: transform_string_time_to_datetime(row['hk_time']), axis=1)
    combined_dataframe['year'] = combined_dataframe.apply(
        lambda row: row['hk_time'].year, axis=1
    )
    combined_dataframe['month'] = combined_dataframe.apply(
        lambda row: row['hk_time'].month, axis=1
    )
    combined_dataframe['day'] = combined_dataframe.apply(
        lambda row: row['hk_time'].day, axis=1
    )
    # Only consider the tweets posted in a specific time range
    time_mask = (combined_dataframe['hk_time'] >= start_date) & (combined_dataframe['hk_time'] <= end_date)
    filtered_dataframe = combined_dataframe.loc[time_mask]
    # Fix the column name of the cleaned_text
    filtered_dataframe.rename(columns = {'cleaned_te': 'cleaned_text', 'user_id_st':'user_id_str'},
                              inplace=True)
    return filtered_dataframe


def get_nontn_tweets(station_name, folder_path):
    data_path = os.path.join(os.path.join(folder_path, station_name, station_name+'_tweets_annulus'))
    non_tn_tweets = pd.read_csv(os.path.join(data_path, station_name+'_1000_erase_500.csv'), encoding='latin-1')
    return non_tn_tweets


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
        right_time_range = opening_start_date + relativedelta(months=check_window)
        # Here, for the dec 28 2016 case, if we set window_size = 5, the starting date would be May 28 2016.
        # We should delete the May 2016 in the DID analysis in this case. Apply to all the did analysis
        final_left_year = left_time_range.year
        final_left_month = (left_time_range + relativedelta(months=1)).month
        final_left_range = datetime(final_left_year, final_left_month, 1, 0, 0, 0, tzinfo=time_zone_hk)
        if final_left_range < time_object < opening_start_date:
            return 0
        elif opening_end_date < time_object < right_time_range:
            return 1
        else:
            return 'not considered'


def build_regress_dataframe_for_combined_areas(kwun_tong_treatment, kwun_tong_control, south_horizons_treatment,
                                               south_horizons_control, ocean_park_treatment, ocean_park_control,
                                               check_window_value=0):
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
        lambda row: add_post_variable(row['hk_time'], opening_start_date=october_23_start,
                                      opening_end_date=october_23_end, check_window=check_window_value), axis=1)
    kwun_tong_control['Post'] = kwun_tong_control.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=october_23_start,
                                      opening_end_date=october_23_end, check_window=check_window_value), axis=1)
    south_horizons_treatment['Post'] = south_horizons_treatment.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=december_28_start,
                                      opening_end_date=december_28_end, check_window=check_window_value), axis=1)
    south_horizons_control['Post'] = south_horizons_control.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=december_28_start,
                                      opening_end_date=december_28_end, check_window=check_window_value), axis=1)
    ocean_park_treatment['Post'] = ocean_park_treatment.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=december_28_start,
                                      opening_end_date=december_28_end, check_window=check_window_value), axis=1)
    ocean_park_control['Post'] = ocean_park_control.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=december_28_start,
                                      opening_end_date=december_28_end, check_window=check_window_value), axis=1)
    dataframe_list = [kwun_tong_treatment, kwun_tong_control, south_horizons_treatment, south_horizons_control,
                      ocean_park_treatment, ocean_park_control]
    combined_dataframe = pd.concat(dataframe_list, axis=0, sort=True)
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
        if key[:7] not in ['2016_10', '2016_12']:
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


def build_regress_dataframe_for_one_newly_built_station(treatment_dataframe, control_dataframe,
                                                        station_open_start_date, station_open_end_date,
                                                        open_year_plus_month, check_window_value=0):
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
        sentiment_dict[time + '_' + t_i_t + '_' + post] = before_and_after_final_tpu.pos_percent_minus_neg_percent(dataframe)
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


def conduct_combined_did_analysis(kwun_tong_treatment_dataframe, kwun_tong_control_dataframe,
                                  south_horizons_treatment_dataframe, south_horizons_control_dataframe,
                                  ocean_park_treatment_dataframe, ocean_park_control_dataframe,
                                  dataframe_saving_path, filename, check_window_value=0):

    longitudinal_dataframe = build_regress_dataframe_for_combined_areas(
        kwun_tong_treatment=kwun_tong_treatment_dataframe, kwun_tong_control=kwun_tong_control_dataframe,
        south_horizons_treatment=south_horizons_treatment_dataframe,
        south_horizons_control=south_horizons_control_dataframe, ocean_park_treatment=ocean_park_treatment_dataframe,
        ocean_park_control=ocean_park_control_dataframe, check_window_value=check_window_value)
    longitudinal_dataframe.to_csv(os.path.join(dataframe_saving_path, filename))

    reg_combined_sentiment = smf.ols('Sentiment ~ T_i_t+Post+T_i_t:Post', longitudinal_dataframe).fit()
    reg_combined_activity = smf.ols('Activity ~ T_i_t+Post+T_i_t:Post', longitudinal_dataframe).fit()
    print('----The sentiment did result-----')
    print(reg_combined_sentiment.summary())
    print('----The activity did result-----')
    print(reg_combined_activity.summary())
    print()


def conduct_did_analysis_one_area(treatment_considered_dataframe, control_considered_dataframe, opening_start_date,
                                  opening_end_date, open_year_month, window_size_value, file_path, filename):
    constructed_dataframe = build_regress_dataframe_for_one_newly_built_station(
        treatment_dataframe=treatment_considered_dataframe,
        control_dataframe=control_considered_dataframe,
        station_open_start_date=opening_start_date,
        station_open_end_date=opening_end_date, open_year_plus_month=open_year_month,
        check_window_value=window_size_value)
    constructed_dataframe.to_csv(os.path.join(file_path, filename), encoding='utf-8')
    combined_sentiment = smf.ols('Sentiment ~ T_i_t+Post+T_i_t:Post', constructed_dataframe).fit()
    combined_activity = smf.ols('Activity ~ T_i_t+Post+Post:T_i_t:Post', constructed_dataframe).fit()
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

    path = os.path.join(read_data.tweet_combined_path, 'longitudinal_tpus')

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
    print('For Kwun Tong Line control...')
    utils.number_of_tweet_user(kwun_tong_line_control_dataframe)
    south_horizons_lei_tung_treatment_dataframe = build_dataframe_based_on_set(datapath=path,
                                                                               tpu_set=south_horizons_lei_tung_treatment_tpu_set)

    south_horizons_lei_tung_control_dataframe = build_dataframe_based_on_set(datapath=path,
                                                                             tpu_set=south_horizons_lei_tung_control_tpu_set)

    ocean_park_wong_chuk_hang_treatment_dataframe = build_dataframe_based_on_set(datapath=path,
                                                                                 tpu_set=ocean_park_wong_chuk_hang_treatment_tpu_set)

    ocean_park_wong_chuk_hang_control_dataframe = build_dataframe_based_on_set(datapath=path,
                                                                               tpu_set=ocean_park_wong_chuk_hang_control_tpu_set)

    # south_island_treatment_dataframe = build_dataframe_based_on_set(datapath=path,
    #                                                                 tpu_set=south_island_treatment_tpu_set)
    # print('For South Island treatment area...')
    # utils.number_of_tweet_user(south_island_treatment_dataframe)
    # south_island_control_dataframe = build_dataframe_based_on_set(datapath=path,
    #                                                                 tpu_set=south_island_control_tpu_set)
    # print('For South Island control area...')
    # utils.number_of_tweet_user(south_island_control_dataframe)

    # tpu_213_treatment_dataframe = build_dataframe_based_on_set(datapath=path, tpu_set=tpu_213_set)
    # tpu_236_treatment_dataframe = build_dataframe_based_on_set(datapath=path, tpu_set=tpu_236_set)
    # tpu_243_treatment_dataframe = build_dataframe_based_on_set(datapath=path, tpu_set=tpu_243_set)
    # tpu_245_treatment_dataframe = build_dataframe_based_on_set(datapath=path, tpu_set=tpu_245_set)

    print('************************DID Analysis Starts....************************')
    dataframe_saving_path = os.path.join(read_data.tweet_combined_path, 'longitudinal_did_analysis_dataframes')
    print('Overall Treatment and Control Comparison...')
    conduct_combined_did_analysis(kwun_tong_treatment_dataframe=kwun_tong_line_treatment_dataframe,
                                  kwun_tong_control_dataframe=kwun_tong_line_control_dataframe,
                                  south_horizons_treatment_dataframe=south_horizons_lei_tung_treatment_dataframe,
                                  south_horizons_control_dataframe=south_horizons_lei_tung_control_dataframe,
                                  ocean_park_treatment_dataframe=ocean_park_wong_chuk_hang_treatment_dataframe,
                                  ocean_park_control_dataframe=ocean_park_wong_chuk_hang_control_dataframe,
                                  dataframe_saving_path=dataframe_saving_path,
                                  filename='longitudinal_did_dataframe_all.csv', check_window_value=0)

    print('Overall Treatment and Control Comparison(3 months)...')
    conduct_combined_did_analysis(kwun_tong_treatment_dataframe=kwun_tong_line_treatment_dataframe,
                                  kwun_tong_control_dataframe=kwun_tong_line_control_dataframe,
                                  south_horizons_treatment_dataframe=south_horizons_lei_tung_treatment_dataframe,
                                  south_horizons_control_dataframe=south_horizons_lei_tung_control_dataframe,
                                  ocean_park_treatment_dataframe=ocean_park_wong_chuk_hang_treatment_dataframe,
                                  ocean_park_control_dataframe=ocean_park_wong_chuk_hang_control_dataframe,
                                  dataframe_saving_path=dataframe_saving_path,
                                  filename='longitudinal_did_dataframe_3_months.csv', check_window_value=3)

    print('Overall Treatment and Control Comparison(4 months)...')
    conduct_combined_did_analysis(kwun_tong_treatment_dataframe=kwun_tong_line_treatment_dataframe,
                                  kwun_tong_control_dataframe=kwun_tong_line_control_dataframe,
                                  south_horizons_treatment_dataframe=south_horizons_lei_tung_treatment_dataframe,
                                  south_horizons_control_dataframe=south_horizons_lei_tung_control_dataframe,
                                  ocean_park_treatment_dataframe=ocean_park_wong_chuk_hang_treatment_dataframe,
                                  ocean_park_control_dataframe=ocean_park_wong_chuk_hang_control_dataframe,
                                  dataframe_saving_path=dataframe_saving_path,
                                  filename='longitudinal_did_dataframe_4_months.csv', check_window_value=4)

    print('Overall Treatment and Control Comparison(5 months)...')
    conduct_combined_did_analysis(kwun_tong_treatment_dataframe=kwun_tong_line_treatment_dataframe,
                                  kwun_tong_control_dataframe=kwun_tong_line_control_dataframe,
                                  south_horizons_treatment_dataframe=south_horizons_lei_tung_treatment_dataframe,
                                  south_horizons_control_dataframe=south_horizons_lei_tung_control_dataframe,
                                  ocean_park_treatment_dataframe=ocean_park_wong_chuk_hang_treatment_dataframe,
                                  ocean_park_control_dataframe=ocean_park_wong_chuk_hang_control_dataframe,
                                  dataframe_saving_path=dataframe_saving_path,
                                  filename='longitudinal_did_dataframe_5_months.csv', check_window_value=5)

    print('Overall Treatment and Control Comparison(6 months)...')
    conduct_combined_did_analysis(kwun_tong_treatment_dataframe=kwun_tong_line_treatment_dataframe,
                                  kwun_tong_control_dataframe=kwun_tong_line_control_dataframe,
                                  south_horizons_treatment_dataframe=south_horizons_lei_tung_treatment_dataframe,
                                  south_horizons_control_dataframe=south_horizons_lei_tung_control_dataframe,
                                  ocean_park_treatment_dataframe=ocean_park_wong_chuk_hang_treatment_dataframe,
                                  ocean_park_control_dataframe=ocean_park_wong_chuk_hang_control_dataframe,
                                  dataframe_saving_path=dataframe_saving_path,
                                  filename='longitudinal_did_dataframe_6_months.csv', check_window_value=6)

    print('Cope with the three areas seperately...')
    print('---------------------Kwun Tong Line---------------------------')
    print('For 3 months....')
    conduct_did_analysis_one_area(treatment_considered_dataframe=kwun_tong_line_treatment_dataframe,
                                  control_considered_dataframe=kwun_tong_line_control_dataframe,
                                  opening_start_date=october_23_start, opening_end_date=october_23_end,
                                  open_year_month='2016_10', window_size_value=3, file_path=dataframe_saving_path,
                                  filename='kwun_tong_line_did_3_months.csv')

    print('For 4 months....')
    conduct_did_analysis_one_area(treatment_considered_dataframe=kwun_tong_line_treatment_dataframe,
                                  control_considered_dataframe=kwun_tong_line_control_dataframe,
                                  opening_start_date=october_23_start, opening_end_date=october_23_end,
                                  open_year_month='2016_10', window_size_value=4, file_path=dataframe_saving_path,
                                  filename='kwun_tong_line_did_4_months.csv')

    print('For 5 months....')
    conduct_did_analysis_one_area(treatment_considered_dataframe=kwun_tong_line_treatment_dataframe,
                                  control_considered_dataframe=kwun_tong_line_control_dataframe,
                                  opening_start_date=october_23_start, opening_end_date=october_23_end,
                                  open_year_month='2016_10', window_size_value=5, file_path=dataframe_saving_path,
                                  filename='kwun_tong_line_did_5_months.csv')

    print('For 6 months....')
    conduct_did_analysis_one_area(treatment_considered_dataframe=kwun_tong_line_treatment_dataframe,
                                  control_considered_dataframe=kwun_tong_line_control_dataframe,
                                  opening_start_date=october_23_start, opening_end_date=october_23_end,
                                  open_year_month='2016_10', window_size_value=6, file_path=dataframe_saving_path,
                                  filename='kwun_tong_line_did_6_months.csv')

    print('For all combined did analysis....')
    conduct_did_analysis_one_area(treatment_considered_dataframe=kwun_tong_line_treatment_dataframe,
                                  control_considered_dataframe=kwun_tong_line_control_dataframe,
                                  opening_start_date=october_23_start, opening_end_date=october_23_end,
                                  open_year_month='2016_10', window_size_value=0, file_path=dataframe_saving_path,
                                  filename='kwun_tong_line_did_all.csv')
    print('-------------------------------------------------------\n')

    print('---------------------South Horizons & Lei Tung---------------------------')
    print('For 3 months....')
    conduct_did_analysis_one_area(treatment_considered_dataframe=south_horizons_lei_tung_treatment_dataframe,
                                  control_considered_dataframe=south_horizons_lei_tung_control_dataframe,
                                  opening_start_date=december_28_start, opening_end_date=december_28_end,
                                  open_year_month='2016_12', window_size_value=3, file_path=dataframe_saving_path,
                                  filename='south_horizons_did_3_months.csv')
    print('For 4 months....')
    conduct_did_analysis_one_area(treatment_considered_dataframe=south_horizons_lei_tung_treatment_dataframe,
                                  control_considered_dataframe=south_horizons_lei_tung_control_dataframe,
                                  opening_start_date=december_28_start, opening_end_date=december_28_end,
                                  open_year_month='2016_12', window_size_value=4, file_path=dataframe_saving_path,
                                  filename='south_horizons_did_4_months.csv')
    print('For 5 months....')
    conduct_did_analysis_one_area(treatment_considered_dataframe=south_horizons_lei_tung_treatment_dataframe,
                                  control_considered_dataframe=south_horizons_lei_tung_control_dataframe,
                                  opening_start_date=december_28_start, opening_end_date=december_28_end,
                                  open_year_month='2016_12', window_size_value=5, file_path=dataframe_saving_path,
                                  filename='south_horizons_did_5_months.csv')
    print('For 6 months....')
    conduct_did_analysis_one_area(treatment_considered_dataframe=south_horizons_lei_tung_treatment_dataframe,
                                  control_considered_dataframe=south_horizons_lei_tung_control_dataframe,
                                  opening_start_date=december_28_start, opening_end_date=december_28_end,
                                  open_year_month='2016_12', window_size_value=6, file_path=dataframe_saving_path,
                                  filename='south_horizons_did_6_months.csv')
    print('For all combined did analysis....')
    conduct_did_analysis_one_area(treatment_considered_dataframe=south_horizons_lei_tung_treatment_dataframe,
                                  control_considered_dataframe=south_horizons_lei_tung_control_dataframe,
                                  opening_start_date=december_28_start, opening_end_date=december_28_end,
                                  open_year_month='2016_12', window_size_value=0, file_path=dataframe_saving_path,
                                  filename='south_horizons_did_all.csv')
    print('-------------------------------------------------------\n')

    print('---------------------Ocean Park & Wong Chuk Hang---------------------------')
    print('For 3 months....')
    conduct_did_analysis_one_area(treatment_considered_dataframe=ocean_park_wong_chuk_hang_treatment_dataframe,
                                  control_considered_dataframe=ocean_park_wong_chuk_hang_control_dataframe,
                                  opening_start_date=december_28_start, opening_end_date=december_28_end,
                                  open_year_month='2016_12', window_size_value=3, file_path=dataframe_saving_path,
                                  filename='ocean_park_did_3_months.csv')
    print('For 4 months....')
    conduct_did_analysis_one_area(treatment_considered_dataframe=ocean_park_wong_chuk_hang_treatment_dataframe,
                                  control_considered_dataframe=ocean_park_wong_chuk_hang_control_dataframe,
                                  opening_start_date=december_28_start, opening_end_date=december_28_end,
                                  open_year_month='2016_12', window_size_value=4, file_path=dataframe_saving_path,
                                  filename='ocean_park_did_4_months.csv')
    print('For 5 months....')
    conduct_did_analysis_one_area(treatment_considered_dataframe=ocean_park_wong_chuk_hang_treatment_dataframe,
                                  control_considered_dataframe=ocean_park_wong_chuk_hang_control_dataframe,
                                  opening_start_date=december_28_start, opening_end_date=december_28_end,
                                  open_year_month='2016_12', window_size_value=5, file_path=dataframe_saving_path,
                                  filename='ocean_park_did_5_months.csv')
    print('For 6 months....')
    conduct_did_analysis_one_area(treatment_considered_dataframe=ocean_park_wong_chuk_hang_treatment_dataframe,
                                  control_considered_dataframe=ocean_park_wong_chuk_hang_control_dataframe,
                                  opening_start_date=december_28_start, opening_end_date=december_28_end,
                                  open_year_month='2016_12', window_size_value=6, file_path=dataframe_saving_path,
                                  filename='ocean_park_did_6_months.csv')
    print('For all combined did analysis....')
    conduct_did_analysis_one_area(treatment_considered_dataframe=ocean_park_wong_chuk_hang_treatment_dataframe,
                                  control_considered_dataframe=ocean_park_wong_chuk_hang_control_dataframe,
                                  opening_start_date=december_28_start, opening_end_date=december_28_end,
                                  open_year_month='2016_12', window_size_value=0, file_path=dataframe_saving_path,
                                  filename='ocean_park_did_all.csv')
    print('-------------------------------------------------------\n')
