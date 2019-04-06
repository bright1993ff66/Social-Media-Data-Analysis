import pandas as pd
import os

import read_data
import utils

stations_location_dict = read_data.location_stations


def station_relatd_dataframes(df, station_name, path):
    df[station_name] = df.apply(
        lambda row: utils.select_data_based_on_location(row,stations_location_dict[station_name][0],
                                                        stations_location_dict[station_name][1]), axis=1)
    new_dataframes = df[df[station_name] == 'TRUE']
    new_dataframes.to_pickle(os.path.join(path, station_name+'.pkl'))
    return new_dataframes


if __name__ == '__main__':
    # Select the tweets near specific stations

    # For 2017
    final_cleaned = pd.read_pickle(os.path.join(read_data.tweet_2017, 'final_zh_en_for_paper_hk_time_2017.pkl'))
    station_names = list(stations_location_dict.keys())
    # neutral1 means that the sentiment of a Tweet will be neutral if one reviewer labels neutral
    station_related_zh_en_cleaned = read_data.station_related_path_zh_en 
    for name in station_names:
        station_relatd_dataframes(final_cleaned, name, station_related_zh_en_cleaned)
		
    # For 2016
    start_time = time.time()
    final_cleaned = pd.read_pickle(os.path.join(read_data.tweet_2016, 'tweet_2016_compare_with_Yao_with_sentiment.pkl'))
	# It is very strange that the geoinformation of two tweets in tweet 2016 are not latitudes and longitudes, but website links
	# Find these tweets
    wrong_list = find_unuseful_geoinformation(final_cleaned)
    filtered_final_cleaned = final_cleaned.loc[~final_cleaned['lat'].isin(wrong_list)]
    print(filtered_final_cleaned.shape)
    station_names = list(stations_location_dict.keys())
    # neutral1 means that the sentiment of a Tweet will be neutral if one reviewer labels neutral
    station_related_zh_en_cleaned_2016 = \
        r'F:\CityU\Datasets\Hong Kong Tweets 2016\station_zh_en_cleaned'
    for name in station_names:
        station_relatd_dataframes(filtered_final_cleaned, name, station_related_zh_en_cleaned_2016)
    end_time = time.time()
    print('Total time is: ', end_time-start_time)