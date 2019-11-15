import pandas as pd
import os
import numpy as np
import csv
from datetime import datetime
from collections import Counter
import pytz

import matplotlib.pyplot as plt

import read_data
import utils

time_zone_hk = pytz.timezone('Asia/Shanghai')
october_23_start = datetime(2016, 10, 23, 0, 0, 0, tzinfo=time_zone_hk)
october_23_end = datetime(2016, 10, 23, 23, 59, 59, tzinfo=time_zone_hk)
december_28_start = datetime(2016, 12, 28, 0, 0, 0, tzinfo=time_zone_hk)
december_28_end = datetime(2016, 12, 28, 23, 59, 59, tzinfo=time_zone_hk)
start_date = datetime(2016, 5, 7, tzinfo=time_zone_hk)
end_date = datetime(2017, 12, 31, tzinfo=time_zone_hk)


def get_tweets_before_after(df, studied_area:str, saving_path, oct_open=True):
    """
    Create the before & after tweet dataframe based on the opening date of a station
    :param df: a dataframe which saves all the tweets posted in one TPU unit
    :param studied_area: the studied area: used to save the file
    :param saving_path: saving path
    :param oct_open: whether the studied station opened in Oct 2016 or not
    :return: the 'before' tweet dataframe and the 'after' tweet dataframe
    """
    if oct_open:
        time_mask_before = (df['hk_time'] < october_23_start)
        time_mask_after = (df['hk_time'] > october_23_end)
    else:
        time_mask_before = (df['hk_time'] < december_28_start)
        time_mask_after = (df['hk_time'] > december_28_end)
    df_before = df.loc[time_mask_before]
    df_after = df.loc[time_mask_after]
    df_before.to_csv(os.path.join(saving_path, studied_area+'_before_posted_by_night_users.csv'), encoding='utf-8')
    df_after.to_csv(os.path.join(saving_path, studied_area+'_after_posted_by_night_users.csv'), encoding='utf-8')
    return df_before, df_after


def find_residents_of_tpu(total_dataframe, tpu_list):
    """
    Find the residents in a specified tpu
    :param total_dataframe: the total filtered tweet dataframe
    :param tpu_list: a list which saves the tpu units we consider
    :return: a user list which saves the residents of considered tpu units in the tpu_list
    """

    # Select tweets which are posted between 12am and 6am
    # Use the same time range in the paper: Identifying tourists and analyzing spatial patterns of their destinations
    # from location-based social media data
    dataframe_copy_selected = total_dataframe.loc[(total_dataframe['hour'] >= 0) & (total_dataframe['hour'] < 6)]

    # Output the users which could be thought of as residents of tpu
    user_set_list = list(set(dataframe_copy_selected['user_id_str']))
    tpu_resident_list = []
    for user in user_set_list:
        data_for_this_user = dataframe_copy_selected.loc[dataframe_copy_selected['user_id_str'] == user]
        user_post_tpu = data_for_this_user.loc[data_for_this_user['TPU_longitudinal'].isin(tpu_list)]
        user_not_post_tpu = data_for_this_user.loc[~data_for_this_user['TPU_longitudinal'].isin(tpu_list)]
        if user_post_tpu.shape[0] > user_not_post_tpu.shape[0]:
            tpu_resident_list.append(user)
        else:
            pass
    return tpu_resident_list


def get_daytime_footprints(dataframe, studied_area: str, before_or_not=True):
    assert type(list(dataframe['hour'])[0]) == int

    time_mask = (dataframe['hour'] >= 9) & (dataframe['hour'] < 18)
    seleted_time_dataframe = dataframe.loc[time_mask]
    if before_or_not:
        seleted_time_dataframe.to_csv(os.path.join(read_data.footprint_analysis,
                                                   '{}_before_day_footprint.csv'.format(studied_area)),
                                      encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    else:
        seleted_time_dataframe.to_csv(os.path.join(read_data.footprint_analysis,
                                                   '{}_after_day_footprint.csv'.format(studied_area)),
                                      encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)


if __name__ == '__main__':

    # Load all the geocoded tweets
    tweet_2016_2017_2018 = utils.read_local_csv_file(path=read_data.tweet_combined_path,
                                                     filename='tweet_combined_sentiment_without_bots.csv',
                                                     dtype_str=True)
    all_geocoded_data = tweet_2016_2017_2018.copy()
    all_geocoded_data['hk_time'] = all_geocoded_data.apply(
        lambda row: utils.transform_string_time_to_datetime(row['hk_time']), axis=1)
    # get the hour and minute columns
    all_geocoded_data['hour'] = all_geocoded_data.apply(lambda row: row['hk_time'].hour, axis=1)
    all_geocoded_data['minutes'] = all_geocoded_data.apply(lambda row: row['hk_time'].minute, axis=1)
    print(all_geocoded_data.columns)

    # Find the residents user id based on the treatment TPU list
    kwun_tong_residents = find_residents_of_tpu(all_geocoded_data, tpu_list=['236', '245', '213', '243'])
    south_horizons_residents = find_residents_of_tpu(all_geocoded_data, tpu_list=['174'])
    ocean_park_residents = find_residents_of_tpu(all_geocoded_data, tpu_list=['175'])
    print(kwun_tong_residents[:10])

    # Find all the tweets posted by these residents
    print('Get an overview about the number of tweets posted by residents in the studied tpus...')
    kwun_tong_resident_tweets = all_geocoded_data.loc[all_geocoded_data['user_id_str'].isin(kwun_tong_residents)]
    print('For areas near Whampoa & Ho Man Tin, {}'.format(utils.number_of_tweet_user(kwun_tong_resident_tweets)))
    south_horizons_resident_tweets = all_geocoded_data.loc[all_geocoded_data['user_id_str'].isin(
        south_horizons_residents)]
    print('For areas near South Horizons & Lei Tung, {}'.format(utils.number_of_tweet_user(
        south_horizons_resident_tweets)))
    ocean_park_resident_tweets = all_geocoded_data.loc[all_geocoded_data['user_id_str'].isin(ocean_park_residents)]
    print('For areas near Ocean Park & Wong Chuk Hang, {}'.format(utils.number_of_tweet_user(
        ocean_park_resident_tweets)))

    # Get the posted tweets before and after the introduction of corresponding MTR stations
    kwun_tong_before_df, kwun_tong_after_df = get_tweets_before_after(kwun_tong_resident_tweets,
                                                                      saving_path=read_data.footprint_analysis,
                                                                      oct_open=True, studied_area='kwun_tong')

    south_horizons_before_df, south_horizons_after_df = get_tweets_before_after(south_horizons_resident_tweets,
                                                                      saving_path=read_data.footprint_analysis,
                                                                      oct_open=False, studied_area='south_horizons')

    ocean_park_before_df, ocean_park_after_df = get_tweets_before_after(ocean_park_resident_tweets,
                                                                      saving_path=read_data.footprint_analysis,
                                                                      oct_open=False, studied_area='ocean_park')

    # Get the tweets posted during the daytime
    get_daytime_footprints(kwun_tong_before_df, studied_area='kwun_tong', before_or_not=True)
    get_daytime_footprints(kwun_tong_after_df, studied_area='kwun_tong', before_or_not=False)
    get_daytime_footprints(south_horizons_before_df, studied_area='south_horizons', before_or_not=True)
    get_daytime_footprints(south_horizons_after_df, studied_area='south_horizons', before_or_not=False)
    get_daytime_footprints(ocean_park_before_df, studied_area='ocean_park', before_or_not=True)
    get_daytime_footprints(ocean_park_after_df, studied_area='ocean_park', before_or_not=False)

