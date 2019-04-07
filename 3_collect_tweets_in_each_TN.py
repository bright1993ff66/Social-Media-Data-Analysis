import pandas as pd
import os

import read_data
import utils

stations_location_dict = read_data.location_stations


# Find tweets based on the geoinformation of each station
def station_relatd_dataframes(df, station_name, path):
    df[station_name] = df.apply(
        lambda row: utils.select_data_based_on_location(row, float(stations_location_dict[station_name][0]),
                                                        float(stations_location_dict[station_name][1])), axis=1)
    new_dataframes = df[df[station_name] == 'TRUE']
    new_dataframes.to_pickle(os.path.join(path, station_name+'.pkl'))
    return new_dataframes
	

# Some latitude is nonsense - we find two tweets in 2016 of which the latitude is a website link	
def find_unuseful_geoinformation(df):
    wrong_list = []
    for _, row in df.iterrows():
        try:
            float(row['lat'])
        except:
            wrong_list.append(row['lat'])
    return wrong_list


if __name__ == '__main__':

    # Select the tweets near specific stations
    # For 2016
	# For the 2017 data, just change the path and the dataset
    start_time = time.time()
    final_cleaned = pd.read_pickle(os.path.join(read_data.tweet_2016, 'final_zh_en_for_paper_hk_time_2016_with_sentiment.pkl'))
	# It is very strange that the geoinformation of two tweets in tweet 2016 are not latitudes and longitudes, but website links
	# Find these tweets
    wrong_list_2016 = find_unuseful_geoinformation(final_cleaned)
    filtered_final_cleaned = final_cleaned.loc[~final_cleaned['lat'].isin(wrong_list_2016)]
    print(filtered_final_cleaned.shape)
    station_names = list(stations_location_dict.keys())
    # neutral1 means that the sentiment of a Tweet will be neutral if one reviewer labels neutral
    station_related_zh_en_cleaned_2016 = \
        r'F:\CityU\Datasets\Hong Kong Tweets 2016\station_zh_en_cleaned'
    for name in station_names:
        station_relatd_dataframes(filtered_final_cleaned, name, station_related_zh_en_cleaned_2016)
    end_time = time.time()
    print('Total time is: ', end_time-start_time)