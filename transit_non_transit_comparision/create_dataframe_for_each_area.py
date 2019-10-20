import pandas as pd
import os
import numpy as np
import csv

import read_data


def created_dataframe_from_index(tweet_dataframe, index_num_npy, saving_path, filename):
    result_dataframe = tweet_dataframe.loc[tweet_dataframe['index_num'].isin(index_num_npy)]
    result_dataframe.to_csv(os.path.join(saving_path, filename), encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    return result_dataframe


def number_of_tweet_user(df):
    user_num = len(set(df['user_id_str']))
    tweet_num = df.shape[0]
    print('Total number of tweet is: {}; Total number of user is {}'.format(
        tweet_num, user_num))


if __name__ == '__main__':

    # Load the tweet 2016, 2017 dataframe
    path = r'F:\CityU\Datasets\tweet_filtering'
    tweet_2016_2017 = pd.read_csv(os.path.join(path, 'tweet_2016_2017_more_tweets_with_visitors.csv'),
                                    encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index_col=0)
    tweet_2016_2017['index_num'] = list(range(0, tweet_2016_2017.shape[0]))

    # Load the corresponding index num list
    index_num_list_path = os.path.join(read_data.transit_non_transit_comparison_before_after, 'circle_annulus',
                                       'tweet_for_three_areas', 'index_num_list')
    whampoa_treatment = np.load(os.path.join(index_num_list_path, '500_whampoa_ho_man_tin_dissolve_index_list.npy'))
    south_horizons_treatment = np.load(os.path.join(index_num_list_path,
                                                    '500_south_horizons_lei_tung_dissolve_index_list.npy'))
    ocean_park_treatment = np.load(os.path.join(index_num_list_path,
                                                    '500_ocean_park_wong_chuk_hang_dissolve_index_list.npy'))
    whampoa_1000_control = np.load(os.path.join(index_num_list_path, '1000_minus_500_whampoa_dissolve_index_list.npy'))
    south_horizons_1000_control = np.load(os.path.join(index_num_list_path,
                                                    '1000_minus_500_south_horizons_dissolve_index_list.npy'))
    ocean_park_1000_control = np.load(os.path.join(index_num_list_path,
                                                    '1000_minus_500_ocean_park_dissolve_index_list.npy'))
    whampoa_1500_control = np.load(os.path.join(index_num_list_path, '1500_minus_500_whampoa_dissolve_index_list.npy'))
    south_horizons_1500_control = np.load(os.path.join(index_num_list_path,
                                                       '1500_minus_500_south_horizons_dissolve_index_list.npy'))
    ocean_park_1500_control = np.load(os.path.join(index_num_list_path,
                                                   '1500_minus_500_ocean_park_dissolve_index_list.npy'))

    # create corresponding dataframe based on the loaded index
    dataframe_saving_path = os.path.join(read_data.transit_non_transit_comparison_before_after, 'circle_annulus',
                                         'tweet_for_three_areas', 'dataframes')
    whampoa_treatment_tweet = created_dataframe_from_index(tweet_dataframe=tweet_2016_2017,
                                                           index_num_npy=whampoa_treatment,
                                                           saving_path=dataframe_saving_path,
                                                           filename='whampoa_treatment.csv')
    whampoa_control_1000_tweet = created_dataframe_from_index(tweet_dataframe=tweet_2016_2017,
                                                              index_num_npy=whampoa_1000_control,
                                                              saving_path=dataframe_saving_path,
                                                              filename='whampoa_control_1000_annulus.csv')
    whampoa_control_1500_tweet = created_dataframe_from_index(tweet_dataframe=tweet_2016_2017,
                                                              index_num_npy=whampoa_1500_control,
                                                              saving_path=dataframe_saving_path,
                                                              filename='whampoa_control_1500_annulus.csv')
    south_horizons_treatment_tweet = created_dataframe_from_index(tweet_dataframe=tweet_2016_2017,
                                                                  index_num_npy=south_horizons_treatment,
                                                                  saving_path=dataframe_saving_path,
                                                                  filename='south_horizons_treatment.csv')
    south_horizons_control_1000_tweet = created_dataframe_from_index(tweet_dataframe=tweet_2016_2017,
                                                                  index_num_npy=south_horizons_1000_control,
                                                                  saving_path=dataframe_saving_path,
                                                                  filename='south_horizons_control_1000_annulus.csv')
    south_horizons_control_1500_tweet = created_dataframe_from_index(tweet_dataframe=tweet_2016_2017,
                                                                  index_num_npy=south_horizons_1500_control,
                                                                  saving_path=dataframe_saving_path,
                                                                  filename='south_horizons_control_1500_annulus.csv')
    ocean_park_treatment_tweet = created_dataframe_from_index(tweet_dataframe=tweet_2016_2017,
                                                                  index_num_npy=ocean_park_treatment,
                                                                  saving_path=dataframe_saving_path,
                                                                  filename='ocean_park_treatment.csv')
    ocean_park_control_1000_tweet = created_dataframe_from_index(tweet_dataframe=tweet_2016_2017,
                                                                     index_num_npy=ocean_park_1000_control,
                                                                     saving_path=dataframe_saving_path,
                                                                     filename='ocean_park_control_1000_annulus.csv')
    ocean_park_control_1500_tweet = created_dataframe_from_index(tweet_dataframe=tweet_2016_2017,
                                                                     index_num_npy=ocean_park_1500_control,
                                                                     saving_path=dataframe_saving_path,
                                                                     filename='ocean_park_control_1500_annulus.csv')
    
    # Create the combined dataframe
    # For treatment
    whole_treatment = np.concatenate([whampoa_treatment, south_horizons_treatment, ocean_park_treatment])
    whole_1500_control = np.concatenate([whampoa_1500_control, south_horizons_1500_control, ocean_park_1500_control])
    whole_treatment_tweet = created_dataframe_from_index(tweet_dataframe=tweet_2016_2017,
                                                         index_num_npy=whole_treatment,
                                                         saving_path=dataframe_saving_path,
                                                         filename='whole_treatment.csv')
    whole_control_tweet = created_dataframe_from_index(tweet_dataframe=tweet_2016_2017,
                                                         index_num_npy=set(whole_1500_control),
                                                         saving_path=dataframe_saving_path,
                                                         filename='whole_control.csv')


    print('----------------------------------------------------------------')
    print('The general information of the generated dataframes is....')
    print('Whampoa treatment area...')
    number_of_tweet_user(whampoa_treatment_tweet)
    print('Whampoa control 1000 annulus area...')
    number_of_tweet_user(whampoa_control_1000_tweet)
    print('Whampoa control 1500 annulus area...')
    number_of_tweet_user(whampoa_control_1500_tweet)
    print('South Horizons treatment area...')
    number_of_tweet_user(south_horizons_treatment_tweet)
    print('South Horizons control 1000 annulus area...')
    number_of_tweet_user(south_horizons_control_1000_tweet)
    print('South Horizons control 1500 annulus area...')
    number_of_tweet_user(south_horizons_control_1500_tweet)
    print('Ocean Park treatment area...')
    number_of_tweet_user(ocean_park_treatment_tweet)
    print('Ocean Park control 1000 annulus area...')
    number_of_tweet_user(ocean_park_control_1000_tweet)
    print('Ocean Park control 1500 annulus area...')
    number_of_tweet_user(ocean_park_control_1500_tweet)
    print('----------------------------------------------------------------')
    print('For whole.....')
    number_of_tweet_user(whole_treatment_tweet)
    number_of_tweet_user(whole_control_tweet)