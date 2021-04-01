import pandas as pd
import numpy as np
import os
import pytz
import csv
from collections import Counter
from scipy.stats import linregress
from datetime import datetime
import time
import warnings

from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import gensim

import data_paths
import utils
from Visualization import topic_model_tweets, wordcloud_tweets

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.font_manager as font_manageer
import matplotlib.ticker as mtick
import geopandas as gpd
import seaborn as sns

# Ignore the warnings produced by the gensim package
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

# Specify the font usage in the matplotlib
font = {'family': 'serif',
        'size': 25}
matplotlib.rc('font', **font)

# Some time attributes
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
time_list = ['2016_5', '2016_6', '2016_7', '2016_8', '2016_9', '2016_10', '2016_11', '2016_12', '2017_1',
             '2017_2', '2017_3', '2017_4', '2017_5', '2017_6', '2017_7', '2017_8', '2017_9', '2017_10',
             '2017_11', '2017_12', '2018_1', '2018_2', '2018_3', '2018_4', '2018_5', '2018_6', '2018_7',
             '2018_8', '2018_9', '2018_10', '2018_11', '2018_12']

# Hong Kong and Shanghai share the same time zone.
# Hence, we transform the utc time in our dataset into Shanghai time
time_zone_hk = pytz.timezone('Asia/Shanghai')


class TransitNeighborhood_Before_After(object):

    before_after_stations = ['Whampoa', 'Ho Man Tin', 'South Horizons', 'Wong Chuk Hang', 'Ocean Park',
                             'Lei Tung']

    def __init__(self, name, tn_dataframe, non_tn_dataframe, poi_dataframe,
                 oct_open: bool, before_and_after: bool, compute_positive: bool, compute_negative: bool):
        """
        :param name: the name of the studied area
        :param tn_dataframe: the dataframe which records all the tweets posted in the TN
        :param non_tn_dataframe: the dataframe which records all the tweets posted in corresponding non_tn
        :param treatment_not_considered_dataframe: the dataframe which records all the tweets posted in the not
               considered TN
        :param poi_dataframe: a pandas dataframe containing the pois in the treatment group of a study area
        :param oct_open: check whether the station is opened on oct 23, 2016
        :param before_and_after: only True if the MTR station in this TN is built recently(in 2016)
        :param compute_positive: True if use positive percent as the sentiment metric
        :param compute_negative: True if use negative percent as the sentiment metric
        """
        self.name = name
        self.tn_dataframe = tn_dataframe
        self.non_tn_dataframe = non_tn_dataframe
        self.poi_dataframe = poi_dataframe
        self.oct_open = oct_open
        self.before_and_after = before_and_after
        self.compute_positive = compute_positive
        self.compute_negative = compute_negative

    # Based on the dataframes for treatment, not considerd treatment, and control group, output dataframes which
    # record the sentiment and activity over the study period
    def output_sent_act_dataframe(self):
        # Construct the monthly sentiment dictionary for TN datafrane
        result_dict_tn = sentiment_by_month(self.tn_dataframe, compute_positive_percent=self.compute_positive,
                                            compute_negative_percent=self.compute_negative)
        # Compute the monthly sentiment dictionary for non-TN dataframe
        result_dict_non_tn = sentiment_by_month(self.non_tn_dataframe, compute_positive_percent=self.compute_positive,
                                                compute_negative_percent=self.compute_negative)
        result_dataframe_tn = pd.DataFrame(list(result_dict_tn.items()), columns=['Date', 'Value'])
        result_dataframe_non_tn = pd.DataFrame(list(result_dict_non_tn.items()), columns=['Date', 'Value'])
        return result_dataframe_tn, result_dataframe_non_tn

    def compute_abs_coeff_difference(self):
        # Compute the coefficient difference of sentiment against time before the treatment
        treatment_group_dataframe, control_group_dataframe = self.output_sent_act_dataframe()
        tn_dataframe_with_sentiment_activity = treatment_group_dataframe.set_index('Date')
        tn_dataframe_for_plot = tn_dataframe_with_sentiment_activity.loc[time_list]
        non_tn_dataframe_with_sentiment_activity = control_group_dataframe.set_index('Date')
        non_tn_dataframe_for_plot = non_tn_dataframe_with_sentiment_activity.loc[time_list]
        if self.oct_open:
            tn_dataframe_compute_diff = tn_dataframe_for_plot.head(5)
            non_tn_dataframe_compute_diff = non_tn_dataframe_for_plot.head(5)
            time_tokens_list = list(range(1, 6))
            tn_dataframe_compute_diff_copy = tn_dataframe_compute_diff.copy()
            non_tn_dataframe_compute_diff_copy = non_tn_dataframe_compute_diff.copy()
            tn_dataframe_compute_diff_copy['time'] = time_tokens_list
            non_tn_dataframe_compute_diff_copy['time'] = time_tokens_list
        else:
            tn_dataframe_compute_diff = tn_dataframe_for_plot.head(7)
            non_tn_dataframe_compute_diff = non_tn_dataframe_for_plot.head(7)
            time_tokens_list = list(range(1, 8))
            tn_dataframe_compute_diff_copy = tn_dataframe_compute_diff.copy()
            non_tn_dataframe_compute_diff_copy = non_tn_dataframe_compute_diff.copy()
            tn_dataframe_compute_diff_copy['time'] = time_tokens_list
            non_tn_dataframe_compute_diff_copy['time'] = time_tokens_list
        tn_dataframe_compute_diff_copy['sentiment'] = tn_dataframe_compute_diff_copy.apply(
            lambda row: row['Value'][0], axis=1)
        non_tn_dataframe_compute_diff_copy['sentiment'] = non_tn_dataframe_compute_diff_copy.apply(
            lambda row: row['Value'][0], axis=1)
        tn_dataframe_compute_diff_copy['activity'] = tn_dataframe_compute_diff_copy.apply(
            lambda row: row['Value'][1], axis=1)
        non_tn_dataframe_compute_diff_copy['activity'] = non_tn_dataframe_compute_diff_copy.apply(
            lambda row: row['Value'][1], axis=1)
        selected_time_value = list(tn_dataframe_compute_diff_copy['time'])
        tn_dataframe_sentiment_list = list(tn_dataframe_compute_diff_copy['sentiment'])
        non_tn_dataframe_sentiment_list = list(non_tn_dataframe_compute_diff_copy['sentiment'])
        tn_dataframe_activity_list = list(tn_dataframe_compute_diff_copy['activity'])
        non_tn_dataframe_activity_list = list(non_tn_dataframe_compute_diff_copy['activity'])
        slope_tn_sentiment, _, _, _, _ = linregress(selected_time_value, tn_dataframe_sentiment_list)
        slop_non_tn_sentiment, _, _, _, _ = linregress(selected_time_value, non_tn_dataframe_sentiment_list)
        slope_tn_activity, _, _, _, _ = linregress(selected_time_value, tn_dataframe_activity_list)
        slope_non_tn_activity, _, _, _, _ = linregress(selected_time_value, non_tn_dataframe_activity_list)
        print('For sentiment: The slope value for tn is: {} while for non-tn the value is: {}'.format(
            slope_tn_sentiment, slop_non_tn_sentiment))
        print('For activity: The slope value for tn is: {} while for non-tn the value is: {}'.format(
            slope_tn_activity, slope_non_tn_activity))
        return abs(slope_tn_sentiment - slop_non_tn_sentiment), abs(slope_tn_activity - slope_non_tn_activity)

    def find_station_influenced_other_users(self, start_time=datetime(2016, 5, 1, 0, 0, 0, tzinfo=time_zone_hk),
                                            end_time=datetime(2018, 12, 31, 23, 59, 59, tzinfo=time_zone_hk)):
        """
        Find the station influenced users
        :param start_time: the considered start time of the tweet dataframe
        :param end_time: the considered end time of the tweet dataframe
        :return: a python set containing the id of users who visited the treatment group both before and after
        the introduction of MTR stations
        """
        assert 'hk_time' in self.tn_dataframe, 'The tweet dataframe should contain hk_time...'
        if isinstance(list(self.tn_dataframe['hk_time'])[0], str):
            dataframe_copy = self.tn_dataframe.copy()
            dataframe_copy['hk_time'] = dataframe_copy.apply(
                lambda row: TransitNeighborhood_Before_After.transform_string_time_to_datetime(row['hk_time']), axis=1)
        else:
            dataframe_copy = self.tn_dataframe.copy()
        dataframe_copy['user_id_str'] = dataframe_copy['user_id_str'].astype(float)
        dataframe_copy['user_id_str'] = dataframe_copy['user_id_str'].astype(np.int64)
        if self.oct_open:
            before_time_mask = (dataframe_copy['hk_time'] < october_1_start) & (start_time <= dataframe_copy['hk_time'])
            after_time_mask = (dataframe_copy['hk_time'] > october_31_end) & (end_time > dataframe_copy['hk_time'])
        else:
            before_time_mask = (dataframe_copy['hk_time'] < december_1_start) & (
                    start_time <= dataframe_copy['hk_time'])
            after_time_mask = (dataframe_copy['hk_time'] > december_31_end) & (end_time > dataframe_copy['hk_time'])
        df_before = dataframe_copy.loc[before_time_mask]
        df_after = dataframe_copy.loc[after_time_mask]
        user_before = set(df_before['user_id_str'])
        user_after = set(df_after['user_id_str'])
        station_influenced_users = user_before.intersection(user_after)
        other_users = user_before.symmetric_difference(user_after)
        print('For {}, we find {} station influenced users, {} other users.'.format(
            self.name, len(station_influenced_users), len(other_users)))
        return station_influenced_users, other_users

    def plot_footprints(self, tweet_dataframe, hk_shape):
        """
        Plot the footprints of station influenced users
        :param tweet_dataframe: the pandas dataframe containing all the tweets posted in HK
        :param hk_shape: the shapefile of the whole Hong Kong
        :return: None. The created figure is saved to local
        """

        if self.oct_open:
            station_influenced_users, _ = self.find_station_influenced_other_users(
                start_time=datetime(2016, 5, 1, 0, 0, 0,
                                    tzinfo=time_zone_hk),
                end_time=datetime(2017, 5, 1, 0, 0, 0,
                                  tzinfo=time_zone_hk))
            before_df_plot, after_df_plot = utils.get_tweets_before_after(tweet_dataframe, oct_open=self.oct_open,
                                                                          start_time=datetime(2016, 5, 1, 0, 0, 0,
                                                                                              tzinfo=time_zone_hk),
                                                                          end_time=datetime(2017, 5, 1, 0, 0, 0,
                                                                                            tzinfo=time_zone_hk))
        else:
            station_influenced_users, _ = self.find_station_influenced_other_users(
                start_time=datetime(2016, 6, 1, 0, 0, 0,
                                    tzinfo=time_zone_hk),
                end_time=datetime(2017, 7, 1, 0, 0, 0,
                                  tzinfo=time_zone_hk))
            before_df_plot, after_df_plot = utils.get_tweets_before_after(tweet_dataframe, oct_open=self.oct_open,
                                                                          start_time=datetime(2016, 6, 1, 0, 0, 0,
                                                                                              tzinfo=time_zone_hk),
                                                                          end_time=datetime(2017, 7, 1, 0, 0, 0,
                                                                                            tzinfo=time_zone_hk))
        before_df_plot = before_df_plot.loc[before_df_plot['user_id_str'].isin(station_influenced_users)]
        after_df_plot = after_df_plot.loc[after_df_plot['user_id_str'].isin(station_influenced_users)]
        before_gpd = gpd.GeoDataFrame(before_df_plot,
                                      geometry=gpd.points_from_xy(before_df_plot.lon, before_df_plot.lat))
        before_gpd_proj = before_gpd.set_crs(epsg=4326).to_crs(2326)
        after_gpd = gpd.GeoDataFrame(after_df_plot,
                                     geometry=gpd.points_from_xy(after_df_plot.lon, after_df_plot.lat))
        after_gpd_proj = after_gpd.set_crs(epsg=4326).to_crs(2326)

        assert before_gpd_proj.crs == hk_shape.crs, 'The coordinate systems do not match!'
        assert after_gpd_proj.crs == hk_shape.crs, 'The coordinate systems do not match!'

        figure, axes = plt.subplots(1, 2, figsize=(10, 16), dpi=300)

        axes[0].set_aspect('equal')
        hk_shape.plot(ax=axes[0], color='white', edgecolor='black', linewidth=0.2)
        before_gpd_proj.plot(ax=axes[0], marker='o', color='red', markersize=1, alpha=0.5)
        axes[1].set_aspect('equal')
        hk_shape.plot(ax=axes[1], color='white', edgecolor='black', linewidth=0.2)
        after_gpd_proj.plot(ax=axes[1], marker='o', color='red', markersize=1, alpha=0.5)

        # Specify the 'before' and 'after' & Number of tweets
        for index, ax in enumerate(axes):
            if not index & 1:
                ax.text(805000, 840000, 'before', color='black', size=25)
                ax.text(820000, 800500, 'Number of tweets: {}'.format(before_df_plot.shape[0]),
                        color='black', size=25)
            else:
                ax.text(805000, 840000, 'after', size=25)
                ax.text(820000, 800500, 'Number of tweets: {}'.format(after_df_plot.shape[0]),
                        color='black', size=25)

        # Turn off the axes
        for ax in axes:
            ax.axis('off')

        # Tight the space between subplots
        figure.tight_layout(pad=0.5)

        # Save the figure to local
        figure.savefig(os.path.join(data_paths.plot_path, self.name+'_users_footprints.tif'), bbox_inches='tight')

    def plot_sentiment_comparison(self):
        """
        Plot the sentiment comparison figures for the station influenced users and other users
        :return: the sentiment comparison plot for one study area
        """
        if self.oct_open:
            station_influenced_users, other_users = self.find_station_influenced_other_users(
                start_time=datetime(2016, 5, 1, 0, 0, 0,
                                    tzinfo=time_zone_hk),
                end_time=datetime(2017, 5, 1, 0, 0, 0,
                                  tzinfo=time_zone_hk))
            considered_start_time = datetime(2016, 5, 1, 0, 0, 0, tzinfo=time_zone_hk)
            considered_end_time = datetime(2017, 5, 1, 0, 0, tzinfo=time_zone_hk)
        else:
            station_influenced_users, other_users = self.find_station_influenced_other_users(
                start_time=datetime(2016, 6, 1, 0, 0, 0,
                                    tzinfo=time_zone_hk),
                end_time=datetime(2017, 7, 1, 0, 0, 0,
                                  tzinfo=time_zone_hk))
            considered_start_time = datetime(2016, 6, 1, 0, 0, 0, tzinfo=time_zone_hk)
            considered_end_time = datetime(2017, 7, 1, 0, 0, 0, tzinfo=time_zone_hk)
        before_df, after_df = utils.get_tweets_before_after(self.tn_dataframe, oct_open=self.oct_open,
                                                            start_time=considered_start_time,
                                                            end_time=considered_end_time)
        before_df['sentiment_vader_percent'] = before_df['sentiment_vader_percent'].astype(np.int)
        after_df['sentiment_vader_percent'] = after_df['sentiment_vader_percent'].astype(np.int)
        before_station_users_df = before_df.loc[before_df['user_id_str'].isin(station_influenced_users)]
        before_other_users_df = before_df.loc[before_df['user_id_str'].isin(other_users)]
        after_station_users_df = after_df.loc[after_df['user_id_str'].isin(station_influenced_users)]
        after_other_users_df = after_df.loc[after_df['user_id_str'].isin(other_users)]
        before_station_users_counter = Counter(before_station_users_df['sentiment_vader_percent'])
        before_station_users_sum = sum(before_station_users_counter.values())
        before_other_users_counter = Counter(before_other_users_df['sentiment_vader_percent'])
        before_other_users_sum = sum(before_other_users_counter.values())
        after_station_users_counter = Counter(after_station_users_df['sentiment_vader_percent'])
        after_station_users_sum = sum(after_station_users_counter.values())
        after_other_users_counter = Counter(after_other_users_df['sentiment_vader_percent'])
        after_other_users_sum = sum(after_other_users_counter.values())
        # Positive, Neutral, Negative
        # Values and Proportions
        sentiment_keys = np.array([2, 1, 0])
        before_station_users_vals = np.array([before_station_users_counter.get(key, 0) for key in sentiment_keys])
        before_other_users_vals = np.array([before_other_users_counter.get(key, 0) for key in sentiment_keys])
        after_station_users_vals = np.array([after_station_users_counter.get(key, 0) for key in sentiment_keys])
        after_other_users_vals = np.array([after_other_users_counter.get(key, 0) for key in sentiment_keys])
        before_station_users_props = np.array(
            [before_station_users_counter.get(key, 0) / before_station_users_sum for key in sentiment_keys])
        before_other_users_props = np.array(
            [before_other_users_counter.get(key, 0) / before_other_users_sum for key in sentiment_keys])
        after_station_users_props = np.array(
            [after_station_users_counter.get(key, 0) / after_station_users_sum for key in sentiment_keys])
        after_other_users_props = np.array(
            [after_other_users_counter.get(key, 0) / after_other_users_sum for key in sentiment_keys])

        sent_ticks = ['Positive', 'Neutral', 'Negative']
        pos_arrays = np.arange(len(sent_ticks))

        figure, axes = plt.subplots(1, 2, figsize=(20, 8), dpi=300)

        # font = font_manageer.FontProperties(size=25)

        axes[0].bar(pos_arrays - 0.1, before_station_users_props, width=0.2, color='#5B9BD5', align='center',
                    label='before: {}'.format(np.sum(before_station_users_vals)))
        axes[0].bar(pos_arrays + 0.1, after_station_users_props, width=0.2, color='#ED7D31', align='center',
                    label='after: {}'.format(np.sum(after_station_users_vals)))
        axes[0].set_ylabel('Percentage (%)')
        axes[0].set_xticks(pos_arrays)
        axes[0].set_xticklabels(['Positive', 'Neutral', 'Negative'], size=20)
        axes[0].set_title('Station Influenced Users\n(Number of Users: {})'.format(len(station_influenced_users)),
                          size=23)
        axes[0].spines['right'].set_visible(False)
        axes[0].spines['top'].set_visible(False)

        axes[1].bar(pos_arrays - 0.1, before_other_users_props, width=0.2, color='#5B9BD5', align='center',
                    label='before: {}'.format(np.sum(before_other_users_vals)))
        axes[1].bar(pos_arrays + 0.1, after_other_users_props, width=0.2, color='#ED7D31', align='center',
                    label='after: {}'.format(np.sum(after_other_users_vals)))
        axes[1].set_ylabel('Percentage (%)')
        axes[1].set_xticks(pos_arrays)
        axes[1].set_xticklabels(['Positive', 'Neutral', 'Negative'], size=20)
        axes[1].set_title('Other Users\n(Number of Users: {})'.format(len(other_users)), size=23)
        axes[1].spines['right'].set_visible(False)
        axes[1].spines['top'].set_visible(False)

        # Set the legend size
        for axis in axes:
            axis.legend(fontsize=18)

        # Set the yticks and labels
        ytick_values = np.arange(0, 1.2, 0.2)
        axes[0].set_yticks(ytick_values)
        axes[0].set_yticklabels(['0', '20', '40', '60', '80', '100'])
        axes[1].set_yticks(ytick_values)
        axes[1].set_yticklabels(['0', '20', '40', '60', '80', '100'])

        figure.savefig(os.path.join(data_paths.plot_path, self.name+'_sentiment_compare.tif'), bbox_inches='tight')

    # Function used to create plot for one TN and the control group
    def line_map_comparison(self, fig, ax, line_labels: tuple, ylabel: str, plot_title_name: str,
                            draw_sentiment: bool = True):
        """
        :param fig: the matplotlib figure
        :param ax: the matplotlib axis
        :param line_labels: a tuple which records the line labels in the line graph
        :param ylabel: the ylabel of the final plot
        :param plot_title_name: the title of the final plot
        :param draw_sentiment: if True we draw sentiment comparison plot; Otherwise we draw activity comparison plot
        :return: the sentiment/activity comparison plot
        """
        tn_dataframe_sent_act, non_tn_dataframe_sent_act = self.output_sent_act_dataframe()
        # Set Date as the index and reorder rows based on time list
        tn_dataframe_with_sentiment_activity = tn_dataframe_sent_act.set_index('Date')
        tn_dataframe_for_plot = tn_dataframe_with_sentiment_activity.loc[time_list]
        non_tn_dataframe_with_sentiment_activity = non_tn_dataframe_sent_act.set_index('Date')
        non_tn_dataframe_for_plot = non_tn_dataframe_with_sentiment_activity.loc[time_list]
        # x is used in plot to record time in x axis
        x = np.arange(0, len(list(tn_dataframe_for_plot.index)), 1)
        tn_act_val_array = np.array([value[1] for value in list(tn_dataframe_for_plot['Value'])])
        if draw_sentiment:  # draw the sentiment comparison plot: y1: TN-TPUs; y2: non-TN-TPUs
            y1 = np.array([value[0] for value in list(tn_dataframe_for_plot['Value'])])
            # y2 = [value[0] for value in list(tn_not_considered_dataframe_for_plot['Value'])]
            y3 = np.array([value[0] for value in list(non_tn_dataframe_for_plot['Value'])])
        else:
            if 'Kwun Tong' in plot_title_name:
                y1 = np.array([value[1] / 1152801 for value in list(tn_dataframe_for_plot['Value'])])
                # y2 = [value[1]/2573440.774 for value in list(tn_not_considered_dataframe_for_plot['Value'])]
                y3 = np.array([value[1] / 1602812 for value in list(non_tn_dataframe_for_plot['Value'])])
            elif 'South Horizons' in plot_title_name:
                y1 = np.array([value[1] / 1398085 for value in list(tn_dataframe_for_plot['Value'])])
                # y2 = [value[1]/1810822.755 for value in list(tn_not_considered_dataframe_for_plot['Value'])]
                y3 = np.array([value[1] / 7186657 for value in list(non_tn_dataframe_for_plot['Value'])])
            elif 'Ocean Park' in plot_title_name:
                y1 = np.array([value[1] / 3817811 for value in list(tn_dataframe_for_plot['Value'])])
                # y2 = [value[1]/3807838.975 for value in list(tn_not_considered_dataframe_for_plot['Value'])]
                y3 = np.array([value[1] / 9002772 for value in list(non_tn_dataframe_for_plot['Value'])])
            else:
                wrong_message = 'You should set a proper title name!'
                return wrong_message

        # Specify the line marker in the line graphs
        if 'Kwun Tong' in plot_title_name:
            line_marker = "o"
        elif 'South Horizons' in plot_title_name:
            line_marker = "^"
        else:
            line_marker = "s"

        # Don't consider the months where the number of tweets is less than 10
        y1_copy, y3_copy = y1.copy(), y3.copy()
        y1_copy[tn_act_val_array < 10] = None
        y3_copy[tn_act_val_array < 10] = None

        # Draw the line graph
        if draw_sentiment:
            lns1 = ax.plot(x, y1_copy, '#4659FF', label=line_labels[0], linestyle='--', marker=line_marker, linewidth=3,
                           markersize=8)
            # lns2 = ax.plot(x, y2, 'y-', label=line_labels[1], linestyle='--', marker='o')
            lns3 = ax.plot(x, y3_copy, '#FFA238', label=line_labels[2], linestyle='--', marker=line_marker, linewidth=3,
                           markersize=8)
        else:
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
            lns1 = ax.plot(x, y1_copy, '#4659FF', label=line_labels[0], linestyle='--', marker=line_marker, linewidth=3,
                           markersize=8)
            # lns2 = ax.plot(x, y2, 'y-', label=line_labels[1], linestyle='--', marker='o')
            lns3 = ax.plot(x, y3_copy, '#FFA238', label=line_labels[2], linestyle='--', marker=line_marker, linewidth=3,
                           markersize=8)

        # Whether to draw the vertical line that indicates the open date
        if self.oct_open:
            ax.axvline(5, color='black')
        else:
            ax.axvline(7, color='black')

        # Add the legend
        # lns = lns1 + lns2 + lns3
        lns = lns1 + lns3
        labs = [l.get_label() for l in lns]
        legend_font = font_manageer.FontProperties(size=30, family='serif')
        ax.legend(lns, labs, prop=legend_font)

        # Draw the average of the sentiment level
        # This average sentiment level means the average sentiment of 93 500-meter TBs
        if draw_sentiment:
            ax.axhline(y=0.07, color='r', linestyle='solid')
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.text(3, 0.12, 'Average Sentiment: 0.07', horizontalalignment='center', color='r', size=20)
            ax.set_ylim(-0.3, 0.3)
        else:  # here I want to make sure that all three areas share the same y axis
            ax.set_ylim(0, 0.00032)

        # Set the yticks
        if draw_sentiment:
            ytick_vals= np.arange(-0.3, 0.4, 0.1)
            ax.set_yticks(ytick_vals)
            ax.set_yticklabels(['-30', '-20', '-10', '0', '10', '20', '30'])
        else:
            ytick_vals = np.arange(0, 3.5e-04, 5e-05)
            ax.set_yticks(ytick_vals)
            ax.set_yticklabels(['0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0'])

        # ax.set_ylabel(ylabel, color='k')  # color='k' means black
        ax.set_xticks(x)
        ax.set_xticklabels(time_list, rotation='vertical', size=25)

    def plot_wordclouds(self):
        """
        Plot the wordcloud comparison for the station influenced users and other users before & after the
        introduction of MTR stations
        :return: None. The wordclouds have saved to the local directory
        """
        if self.oct_open:
            station_influenced_users, other_users = self.find_station_influenced_other_users(
                start_time=datetime(2016, 5, 1, 0, 0, 0,
                                    tzinfo=time_zone_hk),
                end_time=datetime(2017, 5, 1, 0, 0, 0,
                                  tzinfo=time_zone_hk))
            considered_start_time = datetime(2016, 5, 1, 0, 0, 0, tzinfo=time_zone_hk)
            considered_end_time = datetime(2017, 5, 1, 0, 0, tzinfo=time_zone_hk)
        else:
            station_influenced_users, other_users = self.find_station_influenced_other_users(
                start_time=datetime(2016, 6, 1, 0, 0, 0,
                                    tzinfo=time_zone_hk),
                end_time=datetime(2017, 7, 1, 0, 0, 0,
                                  tzinfo=time_zone_hk))
            considered_start_time = datetime(2016, 6, 1, 0, 0, 0, tzinfo=time_zone_hk)
            considered_end_time = datetime(2017, 7, 1, 0, 0, 0, tzinfo=time_zone_hk)
        self.tn_dataframe['user_id_str'] = self.tn_dataframe['user_id_str'].astype(float)
        self.tn_dataframe['user_id_str'] = self.tn_dataframe['user_id_str'].astype(np.int64)
        self.tn_dataframe['sentiment_vader_percent'] = self.tn_dataframe['sentiment_vader_percent'].astype(np.int)
        data_selected = utils.get_tweets_in_time_range(self.tn_dataframe, start_time=considered_start_time,
                                                       end_time=considered_end_time)
        station_users_df = data_selected.loc[data_selected['user_id_str'].isin(station_influenced_users)]
        other_users_df = data_selected.loc[data_selected['user_id_str'].isin(other_users)]
        station_users_pos = station_users_df.loc[station_users_df['sentiment_vader_percent'] == 2]
        other_users_pos = other_users_df.loc[other_users_df['sentiment_vader_percent'] == 2]
        before_text_station_users, after_text_station_users = build_text_for_wordcloud_topic_model(
            df=station_users_pos, oct_open=self.oct_open, build_wordcloud=True, save_raw_text=True,
            saving_path=os.path.join(data_paths.tweet_combined_path, 'longitudinal_plots'),
            filename_before='{}_station_users_before_text_pos.npy'.format(self.name),
            filename_after='{}_station_users_after_text_pos.npy'.format(self.name))
        before_text_other_users, after_text_other_users = build_text_for_wordcloud_topic_model(
            df=other_users_pos, oct_open=self.oct_open, build_wordcloud=True, save_raw_text=True,
            saving_path=os.path.join(data_paths.tweet_combined_path, 'longitudinal_plots'),
            filename_before='{}_other_users_before_text_pos.npy'.format(self.name),
            filename_after='{}_other_users_after_text_pos.npy'.format(self.name))
        generate_wordcloud(before_text_station_users, after_text_station_users, mask=wordcloud_tweets.circle_mask,
                           file_name_before='before_{}_station_users_wordcloud_pos'.format(self.name),
                           file_name_after="after_{}_station_users_wordcloud_pos".format(self.name),
                           color_func=wordcloud_tweets.green_func,
                           figure_saving_path=data_paths.plot_path)
        generate_wordcloud(before_text_other_users, after_text_other_users, mask=wordcloud_tweets.circle_mask,
                           file_name_before='before_{}_other_users_wordcloud_pos'.format(self.name),
                           file_name_after="after_{}_other_users_wordcloud_pos".format(self.name),
                           color_func=wordcloud_tweets.green_func,
                           figure_saving_path=data_paths.plot_path)

    def plot_poi_distribution(self):
        """
        Plot the poi type distribution for the treatment group of a study area
        :return: None. The created figure is saved to a local directory
        """
        count_dict = Counter(self.poi_dataframe['fclass'])
        keys = list(count_dict.keys())
        vals = [count_dict[key] for key in keys]
        plot_dataframe = pd.DataFrame()
        plot_dataframe['place_type'] = keys
        plot_dataframe['count'] = vals
        plot_dataframe_sorted = plot_dataframe.sort_values(by='count', ascending=False).reset_index(drop=True)
        plot_dataframe_sorted_head = plot_dataframe_sorted.head(8)

        figure, axis = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
        x = list(range(plot_dataframe_sorted_head.shape[0]))
        axis.bar(x, list(plot_dataframe_sorted_head['count']), color='blue')
        # axis.set_ylabel('Count', size=18)
        axis.set_xticks(x)
        axis.set_xticklabels(list(plot_dataframe_sorted_head['place_type']), rotation='vertical', size=18)
        axis.set_ylabel('# of POIs')
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        figure.savefig(os.path.join(data_paths.plot_path, '{}_poi_types.tif'.format(self.name)), bbox_inches='tight')

    @staticmethod
    def find_residents_of_tpu(total_dataframe, tpu_list):
        if isinstance(list(total_dataframe['hk_time'])[0], str):
            dataframe_copy = total_dataframe.copy()
            dataframe_copy['hk_time'] = dataframe_copy.apply(
                lambda row: TransitNeighborhood_Before_After.transform_string_time_to_datetime(row['hk_time']), axis=1)
        else:
            dataframe_copy = total_dataframe.copy()

        # get the hour and minute columns
        dataframe_copy['hour'] = dataframe_copy.apply(lambda row: row['hk_time'].hour, axis=1)
        dataframe_copy['minutes'] = dataframe_copy.apply(lambda row: row['hk_time'].minute, axis=1)

        # Select tweets which are posted between 12am and 6am
        dataframe_copy_selected = dataframe_copy.loc[(dataframe_copy['hour'] >= 0) & (dataframe_copy['hour'] < 6)]
        print(utils.number_of_tweet_user(dataframe_copy_selected))

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

    @staticmethod
    def find_max_tweet_days_tweet_count(dataframe: pd.DataFrame):
        """
        Compute the time gap between the first tweet and last tweet of each Twitter user
        And count the number of tweets posted by each user for each year
        :param dataframe: The whole tweet dataframe
        :return: a pandas dataframe saving a summary of tweet posting time gap and tweet count
        """
        user_set_list = list(set(dataframe['user_id_str']))
        print('We have {} users'.format(len(user_set_list)))
        days_list, tweet_count_list = [], []
        count_2016, count_2017, count_2018 = [], [], []
        days_2016_list, days_2017_list, days_2018_list = [], [], []
        result_data = pd.DataFrame()
        user_counter = 0
        for user in user_set_list:
            data_select = dataframe.loc[dataframe['user_id_str'] == user]
            data_select_sort = data_select.sort_values(by='hk_time').reset_index(drop=True)
            data_select_2016 = data_select_sort.loc[data_select_sort['year'] == 2016]
            data_select_2017 = data_select_sort.loc[data_select_sort['year'] == 2017]
            data_select_2018 = data_select_sort.loc[data_select_sort['year'] == 2018]
            year_counter = Counter(data_select_sort['year'])

            # Compute the time range and tweet count
            start = pd.to_datetime(data_select_sort.head(1)['hk_time'].values[0])
            end = pd.to_datetime(data_select_sort.tail(1)['hk_time'].values[0])
            if data_select_2016.shape[0] > 0:
                start_2016 = pd.to_datetime(data_select_2016.head(1)['hk_time'].values[0])
                end_2016 = pd.to_datetime(data_select_2016.tail(1)['hk_time'].values[0])
                days_2016_list.append((end_2016 - start_2016).days)
            else:
                days_2016_list.append(0)
            if data_select_2017.shape[0] > 0:
                start_2017 = pd.to_datetime(data_select_2017.head(1)['hk_time'].values[0])
                end_2017 = pd.to_datetime(data_select_2017.tail(1)['hk_time'].values[0])
                days_2017_list.append((end_2017 - start_2017).days)
            else:
                days_2017_list.append(0)
            if data_select_2018.shape[0] > 0:
                start_2018 = pd.to_datetime(data_select_2018.head(1)['hk_time'].values[0])
                end_2018 = pd.to_datetime(data_select_2018.tail(1)['hk_time'].values[0])
                days_2018_list.append((end_2018 - start_2018).days)
            else:
                days_2018_list.append(0)

            days_list.append((end - start).days)
            tweet_count_list.append(data_select_sort.shape[0])
            count_2016.append(year_counter[2016])
            count_2017.append(year_counter[2017])
            count_2018.append(year_counter[2018])
            user_counter += 1

            if not user_counter % 1000:
                print('We have processed 1000 users...')

        result_data['user_id'] = user_set_list # user id
        result_data['max_days'] = days_list # The time between the first tweet and the last tweet
        result_data['max_days_2016'] = days_2016_list # The time between the first tweet and the last tweet in 2016
        result_data['max_days_2017'] = days_2017_list # The time between the first tweet and the last tweet in 2017
        result_data['max_days_2018'] = days_2018_list # The time between the first tweet and the last tweet in 2018
        result_data['tweet_count'] = tweet_count_list # Total number of tweets posted by this user
        result_data['year_2016'] = count_2016 # Total number of tweets posted by this user in 2016
        result_data['year_2017'] = count_2017 # Total number of tweets posted by this user in 2017
        result_data['year_2018'] = count_2018 # Total number of tweets posted by this user in 2018

        result_2016_more_7 = result_data.loc[result_data['max_days_2016'] > 7]
        result_2017_more_7 = result_data.loc[result_data['max_days_2017'] > 7]
        result_2018_more_7 = result_data.loc[result_data['max_days_2018'] > 7]
        select_mask = (result_data['max_days'] > 7)
        result_all_selected = result_data.loc[select_mask]

        print('In 2016, {}% of users are likely to be influenced by new transit stations'.format(
            round(result_2016_more_7.shape[0] * 100 /result_data.shape[0], 6)))
        print('In 2017, {}% of users are likely to be influenced by new transit stations'.format(
            round(result_2017_more_7.shape[0] * 100/ result_data.shape[0], 6)))
        print('In 2018, {}% of users who are likely to be influenced by new transit stations'.format(
            round(result_2018_more_7.shape[0] * 100 / result_data.shape[0], 6)))
        all_selected_users = set(result_all_selected['user_id'])
        select_dataframe = dataframe.loc[dataframe['user_id_str'].isin(all_selected_users)].reset_index(drop=True)
        select_dataframe.to_csv(os.path.join(data_paths.tweets_data, 'hk_tweets_filtered_final.csv'), encoding='utf-8')
        print('Applying 7 days rule to the whole data...')
        all_percent_user = round(len(all_selected_users)/len(set(result_data['user_id'])), 6) * 100
        all_tweet_percent = round(select_dataframe.shape[0] / dataframe.shape[0], 6) * 100
        print('We get {}% users who are likely to be influenced by new transit stations'.format(all_percent_user))
        print('They posted {}% of tweets'.format(all_tweet_percent))
        print('Average sentiment: {}'.format(pos_percent_minus_neg_percent(select_dataframe)))
        utils.number_of_tweet_user(select_dataframe, print_values=True)
        result_data.to_csv('tweet_time_count_summary.csv', encoding='utf-8')
        return all_selected_users

    @staticmethod
    def count_tweet_in_tpus_monthly(tweet_combined_dataframe, user_ids, interested_tpus: list):
        """
        Count the number of tweets found in some TPUs
        :param tweet_combined_dataframe: the filtered tweet dataframe
        :param interested_tpus: the list of interested TPU names
        :return: a dataframe showing the number of tweets posted in each month
        """
        result_dataframe_list = []
        result_dataframe_list.append(pd.DataFrame(time_list, columns=['time']))
        filtered_data = tweet_combined_dataframe.loc[tweet_combined_dataframe['user_id_str'].isin(user_ids)]
        for tpu in interested_tpus:
            select_dataframe = filtered_data.loc[filtered_data['TPU_cross_sectional'] == tpu]
            count_list = []
            for time in time_list:
                select_dataframe_date = select_dataframe.loc[select_dataframe['month_plus_year'] == time]
                count_list.append(select_dataframe_date.shape[0])
            count_dataframe = pd.DataFrame()
            count_dataframe[tpu] = count_list
            result_dataframe_list.append(count_dataframe)
        result_dataframe = pd.concat(result_dataframe_list, axis=1)
        result_dataframe.to_excel(os.path.join(data_paths.transit_non_transit_compare_code_path,
                                               'tweets_in_tpus_count.xlsx'))


    @staticmethod
    def plot_tweet_time_count_distribution(tweet_count_dataframe, timespan_check=90):

        """
        Plot the tweet time and count distribution
        :param tweet_count_dataframe: a dataframe saving the tweet count and time span information
        :param percentile_check: a specific percentile for check: 1 < percentile_check <=100
        :return: None. The histograms are saved to a local directory
        """

        tweet_count_select = tweet_count_dataframe.loc[tweet_count_dataframe['max_days'] <= timespan_check].reset_index(
            drop=True)
        xtick_values_all = np.arange(0, 1140, 140)
        xtick_values_time_span = np.arange(0, timespan_check + 7, 7)

        fig = plt.figure(figsize=(20, 16), dpi=300)

        sub1 = fig.add_subplot(2, 2, 1)  # two rows, two columns, fist cell
        sub1.hist(tweet_count_select['max_days'], bins=len(xtick_values_time_span) - 1, color='orange')
        sub1.spines['right'].set_visible(False)
        sub1.spines['top'].set_visible(False)
        sub1.set_xlabel('Days')
        sub1.set_ylabel('# of Users')
        sub1.axvline(7, color='black', linewidth=2, linestyle='--')
        sub1.set_xticks(xtick_values_time_span)
        sub1.set_xticklabels(xtick_values_time_span, size=20)

        sub2 = fig.add_subplot(2, 2, 2)
        _, _, patches_time_span = sub2.hist(tweet_count_select['max_days'], cumulative=True, color='green',
                                            histtype='step', density=True,
                                            bins=len(xtick_values_time_span) - 1)
        patches_time_span[0].set_xy(patches_time_span[0].get_xy()[:-1])
        sub2.spines['right'].set_visible(False)
        sub2.spines['top'].set_visible(False)
        sub2.set_xlabel('Days')
        sub2.set_ylabel('Cumulative Percentage (%)')
        sub2.axvline(7, color='black', linewidth=2, linestyle='--')
        sub2.set_xticks(xtick_values_time_span)
        sub2.set_xticklabels(xtick_values_time_span, size=20)
        sub2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        sub2.set_yticklabels(['0', '20', '40', '60', '80', '100'])

        sub3 = fig.add_subplot(2, 2, 3)  # two rows, two colums, combined third and fourth cell
        sub3.hist(tweet_count_dataframe['max_days'], bins=len(xtick_values_all) - 1, color='darkorchid', alpha=0.7)
        sub3.axvline(0, color='orange')
        sub3.axvline(timespan_check, color='orange')
        sub3.spines['right'].set_visible(False)
        sub3.spines['top'].set_visible(False)
        sub3.set_xlabel('Days')
        sub3.set_ylabel('# of Users')

        sub4 = fig.add_subplot(2, 2, 4)
        sub4.fill_between((0, timespan_check), 0, 1, facecolor='green', alpha=0.2)
        _, _, patches_all = sub4.hist(tweet_count_dataframe['max_days'], cumulative=True, color='darkorchid',
                                      histtype='step', density=True,
                                      bins=len(xtick_values_all) - 1)
        patches_all[0].set_xy(patches_all[0].get_xy()[:-1])
        sub4.spines['right'].set_visible(False)
        sub4.spines['top'].set_visible(False)
        sub4.set_xlabel('Days')
        sub4.set_ylabel('Cumulative Percentage (%)')
        sub4.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        sub4.set_yticklabels(['0', '20', '40', '60', '80', '100'])

        fig.savefig(os.path.join(data_paths.plot_path, 'max_days_dist.tif'), bbox_inches='tight')

    @staticmethod
    # Transform the string time to the datetime object
    def transform_string_time_to_datetime(string):
        """
        :param string: the string which records the time of the posted tweets(this string's timezone is HK time)
        :return: a datetime object which could get access to the year, month, day easily
        """
        datetime_object = datetime.strptime(string, '%Y-%m-%d %H:%M:%S+08:00')
        final_time_object = datetime_object.replace(tzinfo=time_zone_hk)
        return final_time_object


# compute the percentage of positive Tweets: 2 is positive
def positive_percent(df):
    positive = 0
    for sentiment in list(df['sentiment_vader_percent']):
        if int(float(sentiment)) == 2:
            positive += 1
        else:
            pass
    return positive / df.shape[0]


# compute the percentage of positive Tweets: 0 is negative
def negative_percent(df):
    negative = 0
    for sentiment in list(df['sentiment_vader_percent']):
        if int(float(sentiment)) == 0:
            negative += 1
        else:
            pass
    return negative / df.shape[0]


# compute positive percentage minus negative percentage: metric used to evaluate the sentiment of an area
# https://www.sciencedirect.com/science/article/pii/S0040162515002024
def pos_percent_minus_neg_percent(df):
    pos_percent = positive_percent(df)
    neg_percent = negative_percent(df)
    return pos_percent - neg_percent


# compute the sentiment level for each month
def sentiment_by_month(df, compute_positive_percent=False, compute_negative_percent=False):
    # check whether the value in the hk_time attribute is string or not
    if isinstance(list(df['hk_time'])[0], str):
        df['hk_time'] = df.apply(
            lambda row: TransitNeighborhood_Before_After.transform_string_time_to_datetime(row['hk_time']), axis=1)
    else:
        pass
    # Check whether one dataframe has the year and the month columns. If not, generate them!
    assert 'month_plus_year' in df, 'The dataframe should contain the month_plus_year column!'
    dataframe_dict = {}
    # Iterate over the pandas dataframe based on the month_plus_year column
    for time, dataframe_by_time in df.groupby('month_plus_year'):
        dataframe_dict[time] = dataframe_by_time
    # time_list = list(dataframe_dict.keys())
    tweet_month_sentiment = {}
    for time in time_list:
        if compute_positive_percent:
            # At any given month, we record both the sentiment and activity
            tweet_month_sentiment[time] = (positive_percent(dataframe_dict[time]), dataframe_dict[time].shape[0])
        elif compute_negative_percent:
            tweet_month_sentiment[time] = (negative_percent(dataframe_dict[time]), dataframe_dict[time].shape[0])
        else:
            tweet_month_sentiment[time] = (pos_percent_minus_neg_percent(dataframe_dict[time]),
                                           dataframe_dict[time].shape[0])
    return tweet_month_sentiment


def build_text_for_wordcloud_topic_model(df, oct_open=True, build_wordcloud=True, save_raw_text=False, saving_path=None,
                                         filename_before=None, filename_after=None):
    """
    :param df: the whole dataframe for before and after study
    :param oct_open: if the station is opened in October or not
    :param build_wordcloud: whether for drawing wordcloud or for topic modelling
    :return: text or dataframes which would be used to generate word cloud or build topic model
    """
    if oct_open:
        open_date_start = october_1_start
        open_date_end = october_31_end
        df_copy = df.copy()
        if isinstance(list(df_copy['hk_time'])[0], str):
            df_copy['hk_time'] = df_copy.apply(
                lambda row: TransitNeighborhood_Before_After.transform_string_time_to_datetime(row['hk_time']), axis=1)
        else:
            pass
        df_before = df_copy.loc[df_copy['hk_time'] < open_date_start]
        df_after = df_copy.loc[df_copy['hk_time'] > open_date_end]
    else:
        open_date_start = december_1_start
        open_date_end = december_31_end
        df_copy = df.copy()
        if isinstance(list(df_copy['hk_time'])[0], str):
            df_copy['hk_time'] = df_copy.apply(
                lambda row: TransitNeighborhood_Before_After.transform_string_time_to_datetime(row['hk_time']), axis=1)
        else:
            pass
        df_before = df_copy.loc[df_copy['hk_time'] < open_date_start]
        df_after = df_copy.loc[df_copy['hk_time'] > open_date_end]
    if build_wordcloud:  # return a string, for wordcloud creation
        before_text = wordcloud_tweets.create_text_for_wordcloud(df_before)
        after_text = wordcloud_tweets.create_text_for_wordcloud(df_after)
        return before_text, after_text
    else:  # if not, return a dataframe
        return df_before, df_after


def generate_wordcloud(words_before, words_after, mask, file_name_before, file_name_after, color_func,
                       figure_saving_path):
    """
    :param words_before: words before the openning date of a station
    :param words_after: words after the openning date of a station
    :param mask: shape mask used to draw the plot
    :param file_name_before: the name of the saved file before the MTR station starts operation
    :param file_name_after: the name of the saved file after the MTR station starts operation
    :param color_func: color function
    :param figure_saving_path: the saving path of the created figures
    """
    # stopwords argument in word_cloud: specify the words we neglect when outputing the wordcloud
    word_cloud_before = WordCloud(width=520, height=520, background_color='white',
                                  font_path=wordcloud_tweets.symbola_font_path,
                                  mask=mask, max_words=800, collocations=False).generate(words_before)
    word_cloud_after = WordCloud(width=520, height=520, background_color='white',
                                 font_path=wordcloud_tweets.symbola_font_path,
                                 mask=mask, max_words=800, collocations=False).generate((words_after))
    fig_before = plt.figure(figsize=(15, 13), facecolor='white', edgecolor='black', dpi=200)
    plt.imshow(word_cloud_before.recolor(color_func=color_func, random_state=3), interpolation="bilinear")
    plt.axis('off')
    plt.tight_layout(pad=0)
    fig_before.savefig(os.path.join(figure_saving_path, file_name_before))
    fig_after = plt.figure(figsize=(15, 13), facecolor='white', edgecolor='black', dpi=200)
    plt.imshow(word_cloud_after.recolor(color_func=color_func, random_state=3), interpolation="bilinear")
    plt.axis('off')
    plt.tight_layout(pad=0)
    fig_after.savefig(os.path.join(figure_saving_path, file_name_after), bbox_inches='tight')


def draw_word_count_histogram(df, station_name, saved_file_name):
    """
    :param df: the dataframe which contains the cleaned posted tweets
    :param saved_file_name: the saved picture file name
    """
    text_list = list(df['cleaned_text'])
    tokenized_text_list = [word_tokenize(text) for text in text_list]
    bigram_phrases = gensim.models.phrases.Phrases(tokenized_text_list, min_count=2, threshold=10)

    bigram_mod = gensim.models.phrases.Phraser(bigram_phrases)

    trigram_phrases = gensim.models.phrases.Phrases(bigram_mod[tokenized_text_list])

    trigram_mod = gensim.models.phrases.Phraser(trigram_phrases)

    data_ready = topic_model_tweets.process_words(tokenized_text_list,
                                                  stop_words=topic_model_tweets.unuseful_terms_set,
                                                  bigram_mod=bigram_mod,
                                                  trigram_mod=trigram_mod)
    # save the processed text
    np.save(os.path.join(data_paths.transit_non_transit_comparison_before_after, station_name + '_text.npy'),
            data_ready)
    text_count_list = [len(text) for text in data_ready]

    fig_word_count, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.distplot(text_count_list)
    # check whether tweet count=7 is appropriate
    ax.axvline(np.median(text_count_list), color='#EB1B52', label='50% Percentile')
    ax.set_xlim((0, 100))
    ax.set_ylim((0, 700))
    # Check if it is appropriate to set the number of keywords as 7 in this dataframe
    ax.set_xticks(list(plt.xticks()[0]) + [np.median(text_count_list)])
    ax.set_title(station_name + ': Tweet Word Count Histogram')
    ax.legend()
    fig_word_count.savefig(os.path.join(data_paths.longitudinal_plot_path, saved_file_name))
    # plt.show()


# Set the hyperparameter: the number of the topics
topic_modelling_search_params = {'n_components': [5, 6, 7, 8, 9, 10]}


def build_topic_model(df, keyword_file_name, topic_number, topic_predict_file_name, saving_path):
    """
    :param df: the dataframe which contains the posted tweets
    :param keyword_file_name: the name of the saved file which contains the keyword for each topic
    :param topic_number: the number of topics we set for the topic modelling
    :param topic_predict_file_name: the name of the saved file which contains the topic prediction for each tweet
    :param saving_path: the saving path
    """
    text_list = list(df['cleaned_text'])
    tokenized_text_list = [word_tokenize(text) for text in text_list]
    bigram_phrases = gensim.models.phrases.Phrases(tokenized_text_list, min_count=5, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram_phrases)
    trigram_phrases = gensim.models.phrases.Phrases(bigram_mod[tokenized_text_list])
    trigram_mod = gensim.models.phrases.Phraser(trigram_phrases)
    data_ready = topic_model_tweets.process_words(tokenized_text_list,
                                                  stop_words=topic_model_tweets.unuseful_terms_set,
                                                  bigram_mod=bigram_mod, trigram_mod=trigram_mod)
    # np.save(os.path.join(read_data.desktop, 'saving_path', keyword_file_name[:-12]+'_text_topic.pkl'), data_ready)
    # Draw the distribution of the length of the tweet: waiting to be changed tomorrow
    data_sentence_in_one_list = [' '.join(text) for text in data_ready]
    # Get the median of number of phrases
    count_list = [len(tweet) for tweet in data_ready]
    print('The number of keywords we use is {}'.format(np.median(count_list)))
    topic_model_tweets.get_lda_model(data_sentence_in_one_list,
                                     grid_search_params=topic_modelling_search_params,
                                     number_of_keywords=int(np.median(count_list)),
                                     keywords_file=keyword_file_name,
                                     topic_predict_file=topic_predict_file_name,
                                     saving_path=saving_path, grid_search_or_not=True,
                                     topic_number=topic_number)


def build_treatment_control_tpu_compare_for_one_line(treatment_csv, control_1000_csv,
                                                     control_1500_csv):
    """
    :param treatment_csv: a csv file which records the tpus that intersect with the 500-meter buffers
    :param control_1000_csv: a csv file which records the tpus that intersect with 1000-meter buffers
    :param control_1500_csv: a csv file which records the tpus that intersect with 1500-meter buffers
    :return: tpu names for the treatment group and control groups
    """
    datapath = os.path.join(data_paths.transit_non_transit_comparison_before_after,
                            'tpu_based_longitudinal_analysis')
    treatment_data = pd.read_csv(os.path.join(datapath, treatment_csv), encoding='utf-8')
    control_1000_data = pd.read_csv(os.path.join(datapath, control_1000_csv), encoding='utf-8')
    control_1500_data = pd.read_csv(os.path.join(datapath, control_1500_csv), encoding='utf-8')
    treatment_tpus_set = set(list(treatment_data['SmallTPU']))
    control_1000_set = set(list(control_1000_data['SmallTPU'])) - treatment_tpus_set
    control_1500_set = set(list(control_1500_data['SmallTPU'])) - treatment_tpus_set
    return treatment_tpus_set, control_1000_set, control_1500_set


def select_dataframe_for_treatment_control(treatment_set, control_set, datapath, select_user_set,
                                           return_dataframe=False):
    """
    Create the treatment & control dataframe based on tweets posted in each TPU
    :param treatment_set: a set containing the treatment TPU name strings
    :param control_set: a set containing the control TPU name strings
    :param datapath: the datapath used to save the tweets posted in each TPU
    :param select_user_set: a set containing the users we are interested in
    :param return_dataframe: whether return a pandas dataframe or not
    :return:
    """
    treatment_dataframe_list, control_dataframe_list = [], []
    for tpu_name in treatment_set:
        dataframe_treatment = pd.read_csv(os.path.join(datapath, tpu_name, tpu_name + '_data.csv'), encoding='utf-8',
                                          quoting=csv.QUOTE_NONNUMERIC, dtype='str', index_col=0)
        dataframe_treatment['month_plus_year'] = dataframe_treatment.apply(
            lambda row: str(int(float(row['year']))) + '_' + str(int(float(row['month']))), axis=1)
        dataframe_treatment['user_id_str'] = dataframe_treatment.apply(
            lambda row: np.int64(float(row['user_id_str'])), axis=1)
        treatment_dataframe_list.append(dataframe_treatment)
    for tpu_name in control_set:
        dataframe_control = pd.read_csv(os.path.join(datapath, tpu_name, tpu_name + '_data.csv'), encoding='utf-8',
                                        quoting=csv.QUOTE_NONNUMERIC, dtype='str')
        dataframe_control['month_plus_year'] = dataframe_control.apply(
            lambda row: str(int(float(row['year']))) + '_' + str(int(float(row['month']))), axis=1)
        dataframe_control['user_id_str'] = dataframe_control.apply(
            lambda row: np.int64(float(row['user_id_str'])), axis=1)
        control_dataframe_list.append(dataframe_control)
    combined_treatment = pd.concat(treatment_dataframe_list, axis=0)
    combined_control = pd.concat(control_dataframe_list, axis=0)
    # Select the users we are interested in
    combined_treatment_final = combined_treatment.loc[combined_treatment['user_id_str'].isin(select_user_set)]
    combined_control_final = combined_control.loc[combined_control['user_id_str'].isin(select_user_set)]
    print('The size of the treatment group {}; The size of the control group {}'.format(
        combined_treatment_final.shape, combined_control_final.shape))
    if return_dataframe:
        return combined_treatment_final, combined_control_final
    else:
        treatment_sent_act_dict = sentiment_by_month(combined_treatment_final)
        control_sent_act_dict = sentiment_by_month(combined_control_final)
        return treatment_sent_act_dict, control_sent_act_dict


def sort_data_based_on_date(df):
    """
    sort the monthly sentiment dataframe based on the predefined month list
    :param df: a dataframe which contains the sentiment information. The first column is month and the second column
    stores the monthly tweet sentiment
    :return: a sorted sentiment dataframe
    """
    df_time_index = df.set_index('Date')
    df_for_plot = df_time_index.loc[time_list]
    df_for_plot['Date'] = time_list
    final_df = df_for_plot.reset_index(drop=True)
    final_df_copy = final_df.copy()
    final_df_copy['sentiment_vader_percent'] = final_df_copy.apply(lambda row: row['Value'][0], axis=1)
    final_df_copy['activity'] = final_df_copy.apply(lambda row: row['Value'][1], axis=1)
    return final_df_copy


if __name__ == '__main__':
    starting_time = time.time()

    # For instance, if we want to compare the sentiment and activity level before and after the
    # opening date of the Whampoa MTR railway station in Hong Kong, since the station is opened on 23 Oct 2016,
    # we could specify the openning date using datatime package and output before and after dataframes
    october_1_start = datetime(2016, 10, 1, 0, 0, 0, tzinfo=time_zone_hk)
    october_31_end = datetime(2016, 10, 31, 23, 59, 59, tzinfo=time_zone_hk)
    december_1_start = datetime(2016, 12, 1, 0, 0, 0, tzinfo=time_zone_hk)
    december_31_end = datetime(2016, 12, 31, 23, 59, 59, tzinfo=time_zone_hk)
    start_date = datetime(2016, 5, 7, 0, 0, 0, tzinfo=time_zone_hk)
    end_date = datetime(2018, 12, 18, 23, 59, 59, tzinfo=time_zone_hk)

    # Load the combined tweet dataframe
    print('Load the tweet data and shapefile...')
    tweet_combined = pd.read_csv(os.path.join(data_paths.tweets_data, 'tweets_with_chinese_vader.csv'),
                                 encoding='utf-8', index_col=0)
    tweet_combined['user_id_str'] = tweet_combined.apply(lambda row: np.int64(float(row['user_id_str'])), axis=1)
    tweet_combined['TPU_cross_sectional'] = tweet_combined.apply(lambda row: str(row['TPU_cross_sectional']), axis=1)
    # users_not_visitors = TransitNeighborhood_Before_After.find_max_tweet_days_tweet_count(tweet_combined)
    # np.save(os.path.join(data_paths.transit_non_transit_compare_code_path, 'users_not_visitors.npy'),
    #         users_not_visitors)
    hk_shapefile = gpd.read_file(os.path.join(data_paths.shapefile_path, 'hk_tpu_project.shp'))
    users_not_visitors = np.load(os.path.join(
        data_paths.transit_non_transit_compare_code_path, 'users_not_visitors.npy'), allow_pickle=True).item()
    TransitNeighborhood_Before_After.count_tweet_in_tpus_monthly(tweet_combined, user_ids=users_not_visitors,
                                                                 interested_tpus=['212', '234', '235', '236', '242',
                                                                                  '243', '245', '247', '172', '174',
                                                                                  '181 - 182', '175 - 176', '183 - 184'])

    # List the TPUs in the treatment group and control
    kwun_tong_line_treatment_selected = {'243', '245', '236'}
    kwun_tong_line_treatment = {'213', '215', '217', '226', '236', '237', '241', '243', '244', '245'}
    kwun_tong_line_treatment_not_considered = kwun_tong_line_treatment - kwun_tong_line_treatment_selected
    kwun_tong_line_control_1000 = {'212', '213', '215', '217', '226', '234', '235', '236', '237', '241', '242',
                                   '243', '244', '245', '247'} - kwun_tong_line_treatment
    kwun_tong_line_control_1500 = {'211', '212', '213', '214', '215', '217', '220', '225', '226', '227',
                                   '228', '229', '232', '233', '234', '235', '236', '237', '241', '242',
                                   '243', '244', '245', '246', '247'} - kwun_tong_line_treatment
    print('----------------------------------------------------------------------------------')
    print('For Kwun Tong Line Extension: the treatment group is: {}; '
          'the control group 1000-meter is: {}; the control group 1500-meter is: {}'.format(
        kwun_tong_line_treatment_selected, kwun_tong_line_control_1000, kwun_tong_line_control_1500))
    print('----------------------------------------------------------------------------------\n')

    south_horizons_lei_tung_treatment_selected = {'174'}
    south_horizons_lei_tung_treatment = {'173', '174'}
    south_horizons_lei_tung_treatment_not_considered = south_horizons_lei_tung_treatment - \
                                                       south_horizons_lei_tung_treatment_selected
    south_horizons_lei_tung_control_1000 = \
        {'172', '173', '174', '175', '176'} - south_horizons_lei_tung_treatment - {'175', '176'}
    south_horizons_lei_tung_control_1500 = {'172', '173', '174', '175', '176', '181', '182'} - \
                                           south_horizons_lei_tung_treatment - {'175', '176'}
    print('----------------------------------------------------------------------------------')
    print('For Souths Horizons&Lei Tung Line Extension: the treatment group is: {}; '
          'the control group 1000-meter is: {}; the control group 1500-meter is: {}'.format(
        south_horizons_lei_tung_treatment_selected, south_horizons_lei_tung_control_1000,
        south_horizons_lei_tung_control_1500))
    print('----------------------------------------------------------------------------------\n')
    ocean_park_wong_chuk_hang_treatment_selected = {'175', '176'}
    ocean_park_wong_chuk_hang_treatment = {'175', '176', '191'}
    ocean_park_wong_chuk_hang_treatment_not_considered = ocean_park_wong_chuk_hang_treatment - \
                                                         ocean_park_wong_chuk_hang_treatment_selected
    ocean_park_wong_chuk_hang_control_1000 = {'173', '174', '175', '176', '183', '184', '191'} - \
                                             ocean_park_wong_chuk_hang_treatment - south_horizons_lei_tung_treatment
    ocean_park_wong_chuk_hang_control_1500 = {'173', '174', '175', '176', '181', '182', '183', '184', '191'} - \
                                             ocean_park_wong_chuk_hang_treatment - south_horizons_lei_tung_treatment
    print('----------------------------------------------------------------------------------')
    print('For Wong Chuk Hang & Ocean Park Line Extension: the treatment group is: {}; '
          'the control group 1000-meter is: {}; the control group 1500-meter is: {}'.format(
        ocean_park_wong_chuk_hang_treatment_selected, ocean_park_wong_chuk_hang_control_1000,
        ocean_park_wong_chuk_hang_control_1500))
    print('----------------------------------------------------------------------------------\n')

    # Get the dataframe for treatment group and control group
    print('For Kwun Tong Line extension...')
    print('treatment vs 1000-meter control group')
    kwun_tong_line_treatment_dataframe, kwun_tong_line_control_1000_dataframe = \
        select_dataframe_for_treatment_control(treatment_set=kwun_tong_line_treatment_selected,
                                               control_set=kwun_tong_line_control_1000,
                                               select_user_set=users_not_visitors,
                                               datapath=os.path.join(data_paths.tweet_combined_path,
                                                                     'longitudinal_tpus'),
                                               return_dataframe=True)
    print('treatment vs 1500-meter control group')
    _, kwun_tong_line_control_1500_dataframe = \
        select_dataframe_for_treatment_control(treatment_set=kwun_tong_line_treatment_selected,
                                               control_set=kwun_tong_line_control_1500,
                                               select_user_set=users_not_visitors,
                                               datapath=os.path.join(data_paths.tweet_combined_path,
                                                                     'longitudinal_tpus'), return_dataframe=True)
    print('For South Horizons & Lei Tung...')
    print('treatment vs 1000-meter control group')
    south_horizons_lei_tung_treatment_dataframe, south_horizons_lei_tung_control_1000_dataframe = \
        select_dataframe_for_treatment_control(treatment_set=south_horizons_lei_tung_treatment_selected,
                                               control_set=south_horizons_lei_tung_control_1000,
                                               select_user_set=users_not_visitors,
                                               datapath=os.path.join(data_paths.tweet_combined_path,
                                                                     'longitudinal_tpus'),
                                               return_dataframe=True)
    print('treatment vs 1500-meter control group')
    _, south_horizons_lei_tung_control_1500_dataframe = \
        select_dataframe_for_treatment_control(treatment_set=south_horizons_lei_tung_treatment_selected,
                                               control_set=south_horizons_lei_tung_control_1500,
                                               select_user_set=users_not_visitors,
                                               datapath=os.path.join(data_paths.tweet_combined_path,
                                                                     'longitudinal_tpus'),
                                               return_dataframe=True)
    print('For Ocean Park & Wong Chuk Hang...')
    print('treatment vs 1000-meter control group')
    ocean_park_wong_chuk_hang_treatment_dataframe, ocean_park_wong_chuk_hang_control_1000_dataframe = \
        select_dataframe_for_treatment_control(treatment_set=ocean_park_wong_chuk_hang_treatment_selected,
                                               control_set=ocean_park_wong_chuk_hang_control_1000,
                                               select_user_set=users_not_visitors,
                                               datapath=os.path.join(data_paths.tweet_combined_path,
                                                                     'longitudinal_tpus'),
                                               return_dataframe=True
                                               )
    print('treatment vs 1500-meter control group')
    _, ocean_park_wong_chuk_hang_control_1500_dataframe = \
        select_dataframe_for_treatment_control(treatment_set=ocean_park_wong_chuk_hang_treatment_selected,
                                               control_set=ocean_park_wong_chuk_hang_control_1500,
                                               select_user_set=users_not_visitors,
                                               datapath=os.path.join(data_paths.tweet_combined_path,
                                                                     'longitudinal_tpus'),
                                               return_dataframe=True
                                               )

    # Load the poi information
    kwun_tong_poi_dataframe = pd.read_csv(os.path.join(data_paths.land_use_poi_path, 'kwun_tong_pois.txt'),
                                          encoding='utf-8')
    south_horizons_poi_dataframe = pd.read_csv(os.path.join(data_paths.land_use_poi_path, 'south_horizons_pois.txt'),
                                          encoding='utf-8')
    ocean_park_poi_dataframe = pd.read_csv(os.path.join(data_paths.land_use_poi_path, 'ocean_park_pois.txt'),
                                          encoding='utf-8')

    # Create the study area objects
    kwun_tong_line_extension_1000_control = TransitNeighborhood_Before_After(name='Kwun_Tong_Line',
                                                                             tn_dataframe=kwun_tong_line_treatment_dataframe,
                                                                             non_tn_dataframe=kwun_tong_line_control_1000_dataframe,
                                                                             poi_dataframe=kwun_tong_poi_dataframe,
                                                                             oct_open=True, before_and_after=True,
                                                                             compute_positive=False,
                                                                             compute_negative=False)
    kwun_tong_line_extension_1500_control = TransitNeighborhood_Before_After(name='Kwun_Tong_Line',
                                                                             tn_dataframe=kwun_tong_line_treatment_dataframe,
                                                                             non_tn_dataframe=kwun_tong_line_control_1500_dataframe,
                                                                             poi_dataframe=kwun_tong_poi_dataframe,
                                                                             oct_open=True, before_and_after=True,
                                                                             compute_positive=False,
                                                                             compute_negative=False)
    south_horizons_lei_tung_1000_control = TransitNeighborhood_Before_After(name='South_Horizons_Lei_Tung',
                                                                            tn_dataframe=south_horizons_lei_tung_treatment_dataframe,
                                                                            non_tn_dataframe=south_horizons_lei_tung_control_1000_dataframe,
                                                                            poi_dataframe=south_horizons_poi_dataframe,
                                                                            oct_open=False, before_and_after=True,
                                                                            compute_positive=False,
                                                                            compute_negative=False)
    south_horizons_lei_tung_1500_control = TransitNeighborhood_Before_After(name='South_Horizons_Lei_Tung',
                                                                            tn_dataframe=south_horizons_lei_tung_treatment_dataframe,
                                                                            non_tn_dataframe=south_horizons_lei_tung_control_1500_dataframe,
                                                                            poi_dataframe=south_horizons_poi_dataframe,
                                                                            oct_open=False, before_and_after=True,
                                                                            compute_positive=False,
                                                                            compute_negative=False)
    ocean_park_wong_chuk_hang_1000_control = TransitNeighborhood_Before_After(name='Ocean_Park_Wong_Chuk_Hang',
                                                                              tn_dataframe=ocean_park_wong_chuk_hang_treatment_dataframe,
                                                                              non_tn_dataframe=ocean_park_wong_chuk_hang_control_1000_dataframe,
                                                                              poi_dataframe=ocean_park_poi_dataframe,
                                                                              oct_open=False, before_and_after=True,
                                                                              compute_positive=False,
                                                                              compute_negative=False)
    ocean_park_wong_chuk_hang_1500_control = TransitNeighborhood_Before_After(name='Ocean_Park_Wong_Chuk_Hang',
                                                                              tn_dataframe=ocean_park_wong_chuk_hang_treatment_dataframe,
                                                                              non_tn_dataframe=ocean_park_wong_chuk_hang_control_1500_dataframe,
                                                                              poi_dataframe=ocean_park_poi_dataframe,
                                                                              oct_open=False, before_and_after=True,
                                                                              compute_positive=False,
                                                                              compute_negative=False)

    # Kwun Tong Line - Overall Comparison between treatment and control group
    fig_sent, axes_sent = plt.subplots(3, 1, figsize=(26, 24), dpi=300, sharex=True)
    kwun_tong_line_extension_1000_control.line_map_comparison(
        fig=fig_sent, ax=axes_sent[0],
        line_labels=('Whampoa Treatment', 'Treatment Group Not Considered', 'Whampoa Control'),
        ylabel='Percentage of Positive Tweets Minus Percentage of Negative Tweets',
        draw_sentiment=True,
        plot_title_name='Kwun Tong Line Treatment Group and Control Group Comparison')
    south_horizons_lei_tung_1500_control.line_map_comparison(
        fig=fig_sent, ax=axes_sent[1],
        line_labels=('South Horizons Treatment', 'Treatment Group Not Considered', 'South Horizons Control'),
        ylabel='Percentage of Positive Tweets Minus Percentage of Negative Tweets',
        draw_sentiment=True,
        plot_title_name='South Horizons & Lei Tung Treatment Group and Control Group Comparison')
    ocean_park_wong_chuk_hang_1500_control.line_map_comparison(
        fig=fig_sent, ax=axes_sent[2],
        line_labels=('Ocean Park Treatment', 'Treatment Group Not Considered', 'Ocean Park Control'),
        ylabel='Percentage of Positive Tweets Minus Percentage of Negative Tweets',
        draw_sentiment=True,
        plot_title_name='Ocean Park & Wong Chuk Hang Treatment Group and Control Group Comparison')
    fig_sent.savefig(os.path.join(data_paths.plot_path, 'sentiment_combined.tif'), bbox_inches='tight')

    fig_act, axes_act = plt.subplots(3, 1, figsize=(26, 24), dpi=300, sharex=True)
    kwun_tong_line_extension_1000_control.line_map_comparison(
        fig=fig_act, ax=axes_act[0],
        line_labels=('Whampoa Treatment', 'Treatment Group Not Considered', 'Whampoa Control'),
        ylabel='Number of Posted Tweets Per Square Meter',
        draw_sentiment=False,
        plot_title_name='Kwun Tong Line Treatment Group and Control Group Comparison')
    south_horizons_lei_tung_1500_control.line_map_comparison(
        fig=fig_act, ax=axes_act[1],
        line_labels=('South Horizons Treatment', 'Treatment Group Not Considered', 'South Horizons Control'),
        ylabel='Number of Posted Tweets Per Square Meter',
        draw_sentiment=False,
        plot_title_name='South Horizons & Lei Tung Treatment Group and Control Group Comparison')
    ocean_park_wong_chuk_hang_1500_control.line_map_comparison(
        fig=fig_act, ax=axes_act[2],
        line_labels=('Ocean Park Treatment', 'Treatment Group Not Considered', 'Ocean Park Control'),
        ylabel='Number of Tweets Per Square Meter',
        draw_sentiment=False,
        plot_title_name='Ocean Park & Wong Chuk Hang Treatment Group and Control Group Comparison')
    fig_act.savefig(os.path.join(data_paths.plot_path, 'acvitiy_combined.tif'), bbox_inches='tight')

    # Save the treatment, treatment not considered and control dataframes
    treatment_control_saving_path = os.path.join(data_paths.tweet_combined_path, 'longitudinal_involved_tpus')
    kwun_tong_line_treatment_dataframe.to_csv(
        os.path.join(treatment_control_saving_path, 'kwun_tong_line_treatment.csv'),
        encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index=False)
    kwun_tong_line_control_1000_dataframe.to_csv(
        os.path.join(treatment_control_saving_path, 'kwun_tong_line_control_1000.csv'),
        encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index=False)
    kwun_tong_line_control_1500_dataframe.to_csv(
        os.path.join(treatment_control_saving_path, 'kwun_tong_line_control_1500.csv'),
        encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index=False)
    south_horizons_lei_tung_treatment_dataframe.to_csv(
        os.path.join(treatment_control_saving_path, 'south_horizons_lei_tung_treatment.csv'),
        encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index=False)
    south_horizons_lei_tung_control_1000_dataframe.to_csv(
        os.path.join(treatment_control_saving_path, 'south_horizons_lei_tung_control_1000.csv'),
        encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index=False)
    south_horizons_lei_tung_control_1500_dataframe.to_csv(
        os.path.join(treatment_control_saving_path, 'south_horizons_lei_tung_control_1500.csv'),
        encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index=False)
    ocean_park_wong_chuk_hang_treatment_dataframe.to_csv(
        os.path.join(treatment_control_saving_path, 'ocean_park_wong_chuk_hang_treatment.csv'),
        encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index=False)
    ocean_park_wong_chuk_hang_control_1000_dataframe.to_csv(
        os.path.join(treatment_control_saving_path, 'ocean_park_wong_chuk_hang_control_1000.csv'),
        encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index=False)
    ocean_park_wong_chuk_hang_control_1500_dataframe.to_csv(
        os.path.join(treatment_control_saving_path, 'ocean_park_wong_chuk_hang_control_1500.csv'),
        encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index=False)

    # Compare the coefficient difference in the kwun tong line extension
    print('------------------------Kwun Tong Line Extension-------------------------------')
    print('***treatment vs 1000 control****')
    kwun_tong_line_extension_1000_control.compute_abs_coeff_difference()
    print('***treatment vs 1500 control****')
    kwun_tong_line_extension_1500_control.compute_abs_coeff_difference()
    print('-------------------------------------------------------------------------------')

    print('--------------------South Horizons & Lei Tung Line Extension-------------------')
    print('***treatment vs 1000 control****')
    south_horizons_lei_tung_1000_control.compute_abs_coeff_difference()
    print('***treatment vs 1500 control****')
    south_horizons_lei_tung_1500_control.compute_abs_coeff_difference()
    print('-------------------------------------------------------------------------------')

    print('--------------------Ocean Park & Wong Chuk Hang Line Extension-------------------')
    print('***treatment vs 1000 control****')
    # the size of the control group is too small
    # ocean_park_wong_chuk_hang_1000_control.compute_abs_coeff_difference()
    print('***treatment vs 1500 control****')
    ocean_park_wong_chuk_hang_1500_control.compute_abs_coeff_difference()
    print('-------------------------------------------------------------------------------')

    # # =========================================Build the wordcloud============================================
    # print('Generating the wordcloud comparison plots...')
    # kwun_tong_line_extension_1000_control.plot_wordclouds()
    # south_horizons_lei_tung_1500_control.plot_wordclouds()
    # ocean_park_wong_chuk_hang_1500_control.plot_wordclouds()
    # # =========================================================================================================
    #
    # # ====================================Footprint Comparison=================================================
    # print('Generating the footprints comparison plots...')
    # kwun_tong_line_extension_1000_control.plot_footprints(tweet_dataframe=tweet_combined, hk_shape=hk_shapefile)
    # south_horizons_lei_tung_1500_control.plot_footprints(tweet_dataframe=tweet_combined, hk_shape=hk_shapefile)
    # ocean_park_wong_chuk_hang_1500_control.plot_footprints(tweet_dataframe=tweet_combined, hk_shape=hk_shapefile)
    # # =========================================================================================================
    #
    # # ====================================Sentiment Comparison=================================================
    # print('Generating the sentiment comparison plots...')
    # kwun_tong_line_extension_1000_control.plot_sentiment_comparison()
    # south_horizons_lei_tung_1500_control.plot_sentiment_comparison()
    # ocean_park_wong_chuk_hang_1500_control.plot_sentiment_comparison()
    # # ======================================================================================================
    #
    # # ===================================Poi distribution comparison==========================================
    # kwun_tong_line_extension_1000_control.plot_poi_distribution()
    # south_horizons_lei_tung_1500_control.plot_poi_distribution()
    # ocean_park_wong_chuk_hang_1500_control.plot_poi_distribution()
    # # =========================================================================================================
    #
    # # ============================Plot tweet time span & count histograms======================================
    # tweet_time_count_summary = pd.read_csv('tweet_time_count_summary.csv', index_col=0, encoding='utf-8')
    # TransitNeighborhood_Before_After.plot_tweet_time_count_distribution(tweet_time_count_summary, timespan_check=90)
    # # =========================================================================================================

    ending_time = time.time()

    print('Total time for running this code is: {}'.format(ending_time - starting_time))
