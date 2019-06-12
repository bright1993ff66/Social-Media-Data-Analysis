import pandas as pd
import numpy as np
import os
import pytz
from datetime import datetime

import read_data
import before_and_after

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
studied_stations = before_and_after.TransitNeighborhood_before_after.before_after_stations
october_23_start = datetime(2016, 10, 23, 0, 0, 0, tzinfo=time_zone_hk)
october_23_end = datetime(2016, 10, 23, 23, 59, 59, tzinfo=time_zone_hk)
december_28_start = datetime(2016, 12, 28, 0, 0, 0, tzinfo=time_zone_hk)
december_28_end = datetime(2016, 12, 28, 23, 59, 59, tzinfo=time_zone_hk)
start_date = datetime(2016, 5, 7, tzinfo=time_zone_hk)
end_date = datetime(2017, 12, 31, tzinfo=time_zone_hk)


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


def add_post_variable(string, opening_start_date, opening_end_date):
    time_object = transform_string_time_to_datetime(string)
    if time_object > opening_end_date:
        return 1
    elif time_object < opening_start_date:
        return 0
    else:
        return 'not considered'


def build_regress_dataframe_one_station(station_name, treatment_dict, control_dict, station_open_start_date,
                                        station_open_end_date):
    result_dataframe = pd.DataFrame(columns=['Time', 'T_i_t', 'Post', 'Sentiment'])
    buffer_dataframe = treatment_dict[station_name]
    annulus_dataframe = control_dict[station_name]
    # build the T_i_t variable
    ones_list = [1] * buffer_dataframe.shape[0]
    buffer_dataframe['T_i_t'] = ones_list
    zeros_list = [0] * annulus_dataframe.shape[0]
    annulus_dataframe['T_i_t'] = zeros_list
    # build the post variable
    buffer_dataframe['Post'] = buffer_dataframe.apply(
    lambda row: add_post_variable(row['hk_time'], opening_start_date=station_open_start_date,
                                  opening_end_date=station_open_end_date), axis=1)
    annulus_dataframe['Post'] = annulus_dataframe.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=station_open_start_date,
                                  opening_end_date=station_open_end_date), axis=1)
    combined_dataframe = pd.concat([buffer_dataframe, annulus_dataframe])
    combined_dataframe = combined_dataframe.reset_index(drop=True)
    combined_dataframe = combined_dataframe.drop(['FID', 'FID_1', 'Field1', 'FID_2', 'FID_1_1',
                                                  'merge_Nums'], axis=1)
    combined_dataframe['tweet_number'] = list(combined_dataframe.index)
    combined_dataframe_without_not_considered = combined_dataframe.loc[combined_dataframe['Post'] != 'not considered']
    combined_dataframe_without_not_considered.rename(columns={'user_id_st': 'user_id_str'}, inplace=True)
    combined_dataframe_without_not_considered['month_plus_year'] = combined_dataframe_without_not_considered.apply(
        lambda row: str(int(row['year'])) + '_' + str(int(row['month'])), axis=1
    )
    sentiment_dict = {}
    for _, dataframe in combined_dataframe_without_not_considered.groupby(['month_plus_year', 'T_i_t', 'Post']):
        time = str(list(dataframe['month_plus_year'])[0])
        t_i_t = str(list(dataframe['T_i_t'])[0])
        post = str(list(dataframe['Post'])[0])
        sentiment_dict[time + '_' + t_i_t + '_' + post] = before_and_after.pos_percent_minus_neg_percent(dataframe)

    time_list = []
    t_i_t_list = []
    post_list = []
    sentiment_list = []
    for key in list(sentiment_dict.keys()):
        time_list.append(key)
        t_i_t_list.append(int(key[-3]))
        post_list.append(int(key[-1]))
        sentiment_list.append(sentiment_dict[key])

    result_dataframe['Time'] = time_list
    result_dataframe['T_i_t'] = t_i_t_list
    result_dataframe['Post'] = post_list
    result_dataframe['Sentiment'] = sentiment_list

    return result_dataframe


if __name__ == '__main__':

    path = read_data.longitudinal_data_path
    before_after_dataframe_dict = {}
    control_group_dataframe_dict = {}
    station_names = ['Whampoa', 'Ho Man Tin', 'South Horizons', 'Wong Chuk Hang', 'Ocean Park']
    for file in os.listdir(path):
        data_path = os.path.join(path, file)
        dataframe = pd.read_csv(os.path.join(data_path, file + '_500m_tn_tweets.csv'), encoding='latin-1')
        non_tn_datapath = os.path.join(path, file, file + '_tweets_annulus')
        non_tn_dataframe = pd.read_csv(os.path.join(non_tn_datapath, file + '_1000_erase_500.csv'), encoding='latin-1')
        before_after_dataframe_dict[file] = dataframe
        control_group_dataframe_dict[file] = non_tn_dataframe

    print('--------------------------------------------------')
    print('Stations considered in the DID model are...')
    print(before_after_dataframe_dict.keys())
    print('--------------------------------------------------')

    print('DID Analysis Starts....')
    print('---------------------Whampoa---------------------------')
    whampoa_result_dataframe = build_regress_dataframe_one_station(station_name='Whampoa',
                                                                   treatment_dict=before_after_dataframe_dict,
                                                                   control_dict=control_group_dataframe_dict,
                                                                   station_open_start_date=october_23_start,
                                                                   station_open_end_date=october_23_end)
    reg_whampoa = smf.ols('Sentiment ~ T_i_t+Post+Post:T_i_t', whampoa_result_dataframe).fit()
    print(reg_whampoa.summary())
    print('-------------------------------------------------------\n')

    print('---------------------Ho Man Tin---------------------------')
    ho_man_tin_result_dataframe = build_regress_dataframe_one_station(station_name='Ho Man Tin',
                                                                   treatment_dict=before_after_dataframe_dict,
                                                                   control_dict=control_group_dataframe_dict,
                                                                   station_open_start_date=october_23_start,
                                                                   station_open_end_date=october_23_end)
    reg_ho_man_tin = smf.ols('Sentiment ~ T_i_t+Post+Post:T_i_t', ho_man_tin_result_dataframe).fit()
    print(reg_ho_man_tin.summary())
    print('-------------------------------------------------------\n')

    print('---------------------South Horizons---------------------------')
    south_horizons_result_dataframe = build_regress_dataframe_one_station(station_name='South Horizons',
                                                                      treatment_dict=before_after_dataframe_dict,
                                                                      control_dict=control_group_dataframe_dict,
                                                                      station_open_start_date=december_28_start,
                                                                      station_open_end_date=december_28_end)
    reg_south_horizons = smf.ols('Sentiment ~ T_i_t+Post+Post:T_i_t', south_horizons_result_dataframe).fit()
    print(reg_south_horizons.summary())
    print('-------------------------------------------------------\n')

    print('---------------------Wong Chuk Hang---------------------------')
    wong_chuk_hang_result_dataframe = build_regress_dataframe_one_station(station_name='Wong Chuk Hang',
                                                                          treatment_dict=before_after_dataframe_dict,
                                                                          control_dict=control_group_dataframe_dict,
                                                                          station_open_start_date=december_28_start,
                                                                          station_open_end_date=december_28_end)
    reg_wong_chuk_hang = smf.ols('Sentiment ~ T_i_t+Post+Post:T_i_t', wong_chuk_hang_result_dataframe).fit()
    print(reg_wong_chuk_hang.summary())
    print('-------------------------------------------------------\n')

    print('---------------------Ocean Park---------------------------')
    ocean_park_result_dataframe = build_regress_dataframe_one_station(station_name='Ocean Park',
                                                                          treatment_dict=before_after_dataframe_dict,
                                                                          control_dict=control_group_dataframe_dict,
                                                                          station_open_start_date=december_28_start,
                                                                          station_open_end_date=december_28_end)
    reg_ocean_park = smf.ols('Sentiment ~ T_i_t+Post+Post:T_i_t', ocean_park_result_dataframe).fit()
    print(reg_ocean_park.summary())
    print('-------------------------------------------------------\n')