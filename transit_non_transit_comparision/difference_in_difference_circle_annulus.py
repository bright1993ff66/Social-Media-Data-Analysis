import pandas as pd
import numpy as np
import csv
import os
import pytz
from datetime import datetime
from collections import Counter

import read_data
import before_and_after_final_tpu

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
studied_stations = before_and_after_final_tpu.TransitNeighborhood_Before_After.before_after_stations
october_23_start = datetime(2016, 10, 23, 0, 0, 0, tzinfo=time_zone_hk)
october_23_end = datetime(2016, 10, 23, 23, 59, 59, tzinfo=time_zone_hk)
december_28_start = datetime(2016, 12, 28, 0, 0, 0, tzinfo=time_zone_hk)
december_28_end = datetime(2016, 12, 28, 23, 59, 59, tzinfo=time_zone_hk)
start_date = datetime(2016, 5, 7, tzinfo=time_zone_hk)
end_date = datetime(2017, 12, 31, tzinfo=time_zone_hk)

before_after_stations = ['Whampoa', 'Ho Man Tin', 'South Horizons', 'Wong Chuk Hang', 'Ocean Park',
                             'Lei Tung']


def transform_string_time_to_datetime(string):
    """
    :param string: the string which records the time of the posted tweets
    :return: a datetime object which could get access to the year, month, day easily
    """
    assert isinstance(string, str)
    datetime_object = datetime.strptime(string, '%Y-%m-%d %H:%M:%S+08:00')
    final_time_object = datetime_object.replace(tzinfo=time_zone_hk)
    return final_time_object


def add_post_variable(string, opening_start_date, opening_end_date):
    """
    based on two time, create value for the post variable
    :param string:
    :param opening_start_date:
    :param opening_end_date:
    :return:
    """
    time_object = transform_string_time_to_datetime(string)
    if time_object > opening_end_date:
        return 1
    elif time_object < opening_start_date:
        return 0
    else:
        return 'not considered'


def build_regress_datafrane_for_one_newly_built_station(treatment_dataframe, control_dataframe,
                                                        station_open_start_date, station_open_end_date,
                                                        open_year_plus_month):
    """
    build the regression model for a specified area
    :param treatment_dataframe: the tweet dataframe of the treatment area
    :param control_dataframe: the tweet dataframe of the control area
    :param station_open_start_date: the opening time of the studied MTR station
    :param station_open_end_date: the ending time of the studied MTR station
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
                                      opening_end_date=station_open_end_date), axis=1)
    print('Check the post variable distribution of treatment group: {}'.format(
        Counter(treatment_dataframe['Post'])))
    print('Check the T_i_t variable distribution of treatment group: {}'.format(
        Counter(treatment_dataframe['T_i_t'])))
    control_dataframe['Post'] = control_dataframe.apply(
        lambda row: add_post_variable(row['hk_time'], opening_start_date=station_open_start_date,
                                      opening_end_date=station_open_end_date), axis=1)
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


def build_regres_dataframe_for_combined_ares(treatment_dataframe, control_dataframe, open_date1, open_date2):
    assert open_date1, open_date2 in ['2016_10', '2016_12']
    result_dataframe = pd.DataFrame(columns=['Time', 'T_i_t', 'Post', 'Sentiment', 'Activity'])
    # build the T_i_t variable
    ones_list = [1] * treatment_dataframe.shape[0]
    treatment_dataframe['T_i_t'] = ones_list
    zeros_list = [0] * control_dataframe.shape[0]
    control_dataframe['T_i_t'] = zeros_list


if __name__ == '__main__':

    # Load the treatment and control dataframes for three areas
    dataframe_path = os.path.join(read_data.transit_non_transit_comparison_before_after, 'circle_annulus',
                                  'tweet_for_three_areas', 'dataframes')
    kwun_tong_line_treatment_dataframe = pd.read_csv(os.path.join(dataframe_path, 'whampoa_treatment.csv'),
                                                     encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    kwun_tong_line_control_1500_dataframe = pd.read_csv(
        os.path.join(dataframe_path, 'whampoa_control_1500_annulus.csv'),
        encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    south_horizons_treatment_dataframe = pd.read_csv(os.path.join(dataframe_path, 'south_horizons_treatment.csv'),
                                                     encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    south_horizons_control_1500_dataframe = pd.read_csv(os.path.join(dataframe_path,
                                                                     'south_horizons_control_1500_annulus.csv'),
                                                        encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    ocean_park_treatment_dataframe = pd.read_csv(os.path.join(dataframe_path, 'ocean_park_treatment.csv'),
                                                 encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    ocean_park_control_1500_dataframe = pd.read_csv(os.path.join(dataframe_path, 'ocean_park_control_1500_annulus.csv'),
                                                    encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)

    whole_treatment_dataframe = pd.read_csv(os.path.join(dataframe_path, 'whole_treatment.csv'), encoding='utf-8',
                                            quoting=csv.QUOTE_NONNUMERIC)
    whole_control_dataframe = pd.read_csv(os.path.join(dataframe_path, 'whole_control.csv'), encoding='utf-8',
                                            quoting=csv.QUOTE_NONNUMERIC)


    print('************************DID Analysis Starts....************************')
    print('Overall Treatment and Control Comparison...')
    print('---------------------Kwun Tong Line---------------------------')
    kwun_tong_line_result_dataframe = build_regress_datafrane_for_one_newly_built_station(
        treatment_dataframe=kwun_tong_line_treatment_dataframe,
        control_dataframe=kwun_tong_line_control_1500_dataframe,
        station_open_start_date=october_23_start,
        station_open_end_date=october_23_end,
        open_year_plus_month='2016_10')
    reg_kwun_tong_line_sentiment = smf.ols('Sentiment ~ T_i_t+Post+Post:T_i_t', kwun_tong_line_result_dataframe).fit()
    reg_kwun_tong_line_activity = smf.ols('Activity ~ T_i_t+Post+Post:T_i_t', kwun_tong_line_result_dataframe).fit()
    print('----The sentiment did result-----')
    print(reg_kwun_tong_line_sentiment.summary())
    print('----The activity did result-----')
    print(reg_kwun_tong_line_activity.summary())
    print('-------------------------------------------------------\n')

    print('---------------------South Horizons & Lei Tung---------------------------')
    south_horizons_lei_tung_result_dataframe = build_regress_datafrane_for_one_newly_built_station(
        treatment_dataframe=south_horizons_treatment_dataframe,
        control_dataframe=south_horizons_control_1500_dataframe,
        station_open_start_date=december_28_start,
        station_open_end_date=december_28_end,
        open_year_plus_month='2016_12')
    reg_south_horizons_lei_tung_sentiment = smf.ols('Sentiment ~ T_i_t+Post+Post:T_i_t',
                                          south_horizons_lei_tung_result_dataframe).fit()
    reg_south_horizons_lei_tung_activity = smf.ols('Activity ~ T_i_t+Post+Post:T_i_t',
                                          south_horizons_lei_tung_result_dataframe).fit()
    print('----The sentiment did result-----')
    print(reg_south_horizons_lei_tung_sentiment.summary())
    print('----The activity did result-----')
    print(reg_south_horizons_lei_tung_activity.summary())
    print('-------------------------------------------------------\n')

    print('---------------------Ocean Park & Wong Chuk Hang---------------------------')
    ocean_park_wong_chuk_hang_result_dataframe = build_regress_datafrane_for_one_newly_built_station(
        treatment_dataframe=ocean_park_treatment_dataframe,
        control_dataframe=ocean_park_control_1500_dataframe,
        station_open_start_date=december_28_start,
        station_open_end_date=december_28_end,
        open_year_plus_month='2016_12')
    reg_ocean_park_wong_chuk_hang_sentiment = smf.ols('Sentiment ~ T_i_t+Post+Post:T_i_t',
                                          ocean_park_wong_chuk_hang_result_dataframe).fit()
    reg_ocean_park_wong_chuk_hang_activity = smf.ols('Activity ~ T_i_t+Post+Post:T_i_t',
                                                      ocean_park_wong_chuk_hang_result_dataframe).fit()
    print('----The sentiment did result-----')
    print(reg_ocean_park_wong_chuk_hang_sentiment.summary())
    print('----The activity did result-----')
    print(reg_ocean_park_wong_chuk_hang_activity.summary())
    print('-------------------------------------------------------\n')
