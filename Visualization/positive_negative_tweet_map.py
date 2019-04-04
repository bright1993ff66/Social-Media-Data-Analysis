import plotly
import plotly.plotly as py
import pandas as pd
import os
import read_data
from collections import Counter
import numpy as np

# Load the plotly with username and api_key
plotly.tools.set_credentials_file(username='XXXXX', api_key='XXXXX')

# Packages used to draw the map
import matplotlib
from geopy.geocoders import Bing
from geopy.point import Point
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import folium


def count_positive_percentage(dataframe_path, df_name):
    dataframe = pd.read_pickle(os.path.join(dataframe_path, df_name))
    positive_rows = dataframe.loc[dataframe['sentiment']==2]
    positive_percentage = positive_rows.shape[0]/dataframe.shape[0]
    return positive_percentage


def count_negative_percentage(dataframe_path, df_name):
    dataframe = pd.read_pickle(os.path.join(dataframe_path, df_name))
    negative_rows = dataframe.loc[dataframe['sentiment']==0]
    negative_percentage = negative_rows.shape[0]/dataframe.shape[0]
    return negative_percentage


def count_tweets(dataframe_path, df_name):
    dataframe = pd.read_pickle(os.path.join(dataframe_path, df_name))
    return dataframe.shape[0]


def draw_the_map_for_tweets(dataframe_for_map):
    folium_map_for_positive = folium.Map(location=[22.322051, 114.172604],
                                         zoom_start=13,
                                         tiles="CartoDB positron",
                                         attr='<a href=https://github.com/bright1993ff66/>Hong Kong Positive Tweets</a>')
    folium_map_for_negative = folium.Map(location=[22.322051, 114.172604],
                                         zoom_start=13,
                                         tiles="CartoDB positron",
                                         attr='<a href=https://github.com/bright1993ff66/>Hong Kong Positive Tweets</a>')
    for index, row in dataframe_for_map.iterrows():
        positive_percents = row['Positive Percentage']
        # change the radius if needed
        radius_positive = (positive_percents + 0.1) * 30
        folium.CircleMarker(location=(row["lat"],
                                      row["lon"]),
                            radius=radius_positive,
                            color='#008000',
                            fill=True,
                            popup=folium.Popup(row['Name'])).add_to(folium_map_for_positive)
    for index, row in dataframe_for_map.iterrows():
        negative_percents = row['Negative Percentage']
        # change the radius if needed
        radius_negative = (negative_percents + 0.1) * 30
        folium.CircleMarker(location=(row["lat"],
                                      row["lon"]),
                            radius=radius_negative,
                            color='#FF0000',
                            fill=True,
                            popup=folium.Popup(row['Name'])).add_to(folium_map_for_negative)
    folium_map_for_negative.save(os.path.join(read_data.lda_plot_path, "station_location_negative.html"))
    folium_map_for_positive.save(os.path.join(read_data.lda_plot_path, "station_location_positive.html"))


if __name__ == '__main__':
    station_location = pd.read_csv(os.path.join(read_data.tweet_2017, 'station_location.csv'))
    location_dict = {}

    for _, row in station_location.iterrows():
        location_dict[row['Name']] = (row['lat'], row['lon'])

    # Get the name of all the TNs
    TN_name_list = []
    positive_percent_list = []
    negative_percent_list = []
    activity_list = []

    for file in os.listdir(read_data.station_related_path_zh_en):
        TN_name_list.append(file[:-4])
        positive_percent_list.append(count_positive_percentage(read_data.station_related_path_zh_en, file))
        negative_percent_list.append(count_negative_percentage(read_data.station_related_path_zh_en, file))
        activity_list.append(count_tweets(read_data.station_related_path_zh_en, file))

    dataframe_for_map = pd.DataFrame({'Name': TN_name_list, 'Positive Percentage': positive_percent_list,
                                      'Negative Percentage': negative_percent_list,
                                      'Tweet Activity': activity_list},
                                     columns=['Name', 'Positive Percentage',
                                              'Negative Percentage', 'Tweet Activity'])

    combined_frame = pd.merge(station_location, dataframe_for_map, on='Name')

    draw_the_map_for_tweets(combined_frame)