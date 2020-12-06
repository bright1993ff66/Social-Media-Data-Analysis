import pandas as pd
import os
import numpy as np
import csv

import data_paths


def created_dataframe_from_index(tweet_dataframe, index_num_npy, saving_path, filename):
    result_dataframe = tweet_dataframe.loc[tweet_dataframe['index_num'].isin(index_num_npy)]
    result_dataframe.to_csv(os.path.join(saving_path, filename), encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    return result_dataframe


def number_of_tweet_user(df):
    user_num = len(set(df['user_id_str']))
    tweet_num = df.shape[0]
    print('Total number of tweet is: {}; Total number of user is {}'.format(
        tweet_num, user_num))

def build_data_for_cross_sectional_study(tweet_data_path, saving_path, tpu_dataframe, only_2017_2018=True):
    """
    construct tweet dataframe for each TPU unit
    :param tweet_data_path: path which is used to save all the filtered tweets
    :param saving_path: path which is used to save the tweets posted in each TPU
    :return:
    """
    all_tweet_data = pd.read_csv(os.path.join(tweet_data_path, 'tweet_combined_with_sentiment.csv'),
                                 encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    if only_2017_2018:
        # Only consider tweets posted in 2017
        assert 2017.0 in list(all_tweet_data['year'])
        assert 2018.0 in list(all_tweet_data['year'])
        tweet_2017_2018 = all_tweet_data.loc[all_tweet_data['year'].isin([2017.0, 2018.0])]
        # Change the name of the column
        tpu_set = set(tpu_dataframe['TPU Names'])
        for tpu in tpu_set:
            try:
                os.mkdir(os.path.join(saving_path, tpu))
            except WindowsError:
                pass
            # Use the TPU_cross_sectional column
            dataframe = tweet_2017_2018.loc[tweet_2017_2018['TPU_cross_sectional'] == tpu]
            dataframe.to_csv(os.path.join(saving_path, tpu, tpu+'_data.csv'), encoding='utf-8',
                             quoting=csv.QUOTE_NONNUMERIC)
    else:
        tpu_set = set(tpu_dataframe['TPU Names'])
        for tpu in tpu_set:
            try:
                os.mkdir(os.path.join(saving_path, tpu))
            except WindowsError:
                pass
            dataframe = all_tweet_data.loc[all_tweet_data['TPU_cross_sectional'] == tpu]
            dataframe.to_csv(os.path.join(saving_path, tpu, tpu+'_data.csv'), encoding='utf-8',
                             quoting=csv.QUOTE_NONNUMERIC)


if __name__ == '__main__':

    # Load the tweet 2016, 2017 dataframe
    path = r'F:\CityU\Datasets\tweet_filtering'
    tweet_2016_2017_2018 = pd.read_csv(os.path.join(data_paths.tweet_combined_path,
                                                    'tweet_combined_sentiment_without_bots.csv'),
                                 encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, dtype='str')
    print(tweet_2016_2017_2018.columns)
    # tweet_2016_2017['index_num'] = list(range(0, tweet_2016_2017.shape[0]))

    print('For the tpu setting...')
    with open(os.path.join(data_paths.tweet_combined_path, 'tpu_longitudinal_names.txt'), encoding='utf-8') as f:
        header = f.readline()
        lines = f.readlines()
        tpu_name_list = []
        for line in lines:
            line_list = line.split(',')
            tpu_name_list.append(line_list[1])

    print(tpu_name_list)

    for name in tpu_name_list:
        try:
            os.mkdir(os.path.join(data_paths.tweet_combined_path, 'longitudinal_tpus', name))
        except WindowsError:
            pass

    for tpu_name in tpu_name_list:
        # Use the TPU_cross_sectional column
        print('Coping with the tpu: {}'.format(tpu_name))
        dataframe = tweet_2016_2017_2018.loc[tweet_2016_2017_2018['TPU_longitudinal'] == tpu_name]
        dataframe.to_csv(os.path.join(data_paths.tweet_combined_path, 'longitudinal_tpus', tpu_name,
                                      tpu_name + '_data.csv'), encoding='utf-8',
                         quoting=csv.QUOTE_NONNUMERIC)