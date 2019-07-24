import pandas as pd
import numpy as np
import os
import pytz
import csv
from datetime import datetime
from collections import Counter

from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import gensim

import read_data
import wordcloud_generate
import Topic_Modelling_for_tweets
import cross_sectional

from matplotlib import pyplot as plt
from matplotlib import rc
rc('mathtext', default='regular')
import seaborn as sns

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
time_list = ['2016_5', '2016_6','2016_7', '2016_8', '2016_9', '2016_10', '2016_11', '2016_12', '2017_1',
             '2017_2', '2017_3', '2017_4', '2017_5', '2017_6', '2017_7', '2017_8', '2017_9', '2017_10',
             '2017_11', '2017_12']

# Hong Kong and Shanghai share the same time zone.
# Hence, we transform the utc time in our dataset into Shanghai time
time_zone_hk = pytz.timezone('Asia/Shanghai')


class TransitNeighborhood_before_after(object):

    before_after_stations = ['Whampoa', 'Ho Man Tin', 'South Horizons', 'Wong Chuk Hang', 'Ocean Park',
                             'Lei Tung']

    def __init__(self, tn_dataframe, non_tn_dataframe, oct_open:bool, before_and_after:bool, compute_positive:bool,
                 compute_negative:bool):
        """
        :param tn_dataframe: the dataframe which records all the tweets posted in this TN
        :param non_tn_dataframe: the dataframe which records all the tweets posted in corresponding non_tn
        :param oct_open: check whether the station is opened on oct 23, 2016
        :param before_and_after: only True if the MTR station in this TN is built recently(in 2016)
        :param compute_positive: True if use positive percent as the sentiment metric
        :param compute_negative: True if use negative percent as the sentiment metric
        """
        self.tn_dataframe = tn_dataframe
        self.non_tn_dataframe = non_tn_dataframe
        self.oct_open = oct_open
        self.before_and_after = before_and_after
        self.compute_positive = compute_positive
        self.compute_negative = compute_negative

    # Based on the dataframes for treatment and control group, output dataframes which record the sentiment and
    # activity over the study period
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

    # Function used to create plot for one TN and the control group
    def line_map_comparison(self, line_labels:tuple, ylabel:str, plot_title_name:str,
                            saving_file_name:str, draw_sentiment:bool=True):
        """
        :param line_labels: a tuple which records the line labels in the line graph
        :param ylabel: the ylabel of the final plot
        :param plot_title_name: the title of the final plot
        :param saving_file_name: the name of the saved file
        :param draw_sentiment: if True we draw sentiment comparison plot; Otherwise we draw activity comparison plot
        :return: the sentiment/activity comparison plot
        """
        tn_dataframe_sent_act, non_tn_dataframe_sent_act = self.output_sent_act_dataframe()
        # Set one column as the index
        tn_dataframe_with_sentiment_activity = tn_dataframe_sent_act.set_index('Date')
        # So that we could reorder it based on an ordered time list
        tn_dataframe_for_plot = tn_dataframe_with_sentiment_activity.loc[time_list]
        non_tn_dataframe_with_sentiment_activity = non_tn_dataframe_sent_act.set_index('Date')
        non_tn_dataframe_for_plot = non_tn_dataframe_with_sentiment_activity.loc[time_list]
        # x is used in plot to record time in x axis
        x = np.arange(0, len(list(tn_dataframe_for_plot.index)), 1)
        if draw_sentiment:  # draw the sentiment comparison plot: y1: TN-buffer; y2: non-TN-buffer
            y1 = [value[0] for value in list(tn_dataframe_for_plot['Value'])]
            y2 = [value[0] for value in list(non_tn_dataframe_for_plot['Value'])]
        else:  # draw the activity comparison plot. Use log10(num of tweets) instead
            y1 = [np.log10(value[1]) for value in list(tn_dataframe_for_plot['Value'])]
            y2 = [np.log10(value[1]) for value in list(non_tn_dataframe_for_plot['Value'])]

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        lns1 = ax.plot(x, y1, 'g-', label=line_labels[0], linestyle='--', marker='o')
        lns2 = ax.plot(x, y2, 'y-', label=line_labels[1], linestyle='--', marker='^')
        # Whether to draw the vertical line that indicates the open date
        if self.before_and_after:
            if self.oct_open:
                # draw a verticle showing the openning date of the station
                plt.axvline(5.77, color='black')
                if draw_sentiment:  # the ylim of sentiment and activity plots are different
                    ax.text(2.8, 0, 'Opening Date: \nOct 23, 2016', horizontalalignment='center', color='black')
                else:
                    ax.text(2.8, 3.0, 'Opening Date: \nOct 23, 2016', horizontalalignment='center', color='black')
            else:
                # draw a verticle showing the openning date of the station
                plt.axvline(7.95, color='black')
                if draw_sentiment:  # the ylim of sentiment and activity plots are different
                    ax.text(5, 0, 'Opening Date: \nDec 28, 2016', horizontalalignment='center', color='black')
                else:
                    ax.text(5, 3.0, 'Opening Date: \nDec 28, 2016', horizontalalignment='center', color='black')
        else:
            pass

        # Add the legend
        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs)

        # Draw the average of the sentiment level
        # This average sentiment level means the average sentiment of 93 500-meter TBs
        if draw_sentiment:
            ax.axhline(y=0.40, color='r', linestyle='solid')
            ax.text(3, 0.43, 'Average Sentiment Level: 0.40', horizontalalignment='center', color='r')
            # The ylim is set as (-1, 1)
            ax.set_ylim(-1, 1)
        else:  # here I don't want to draw the horizontal activity line as the activity level varies greatly between TNs
            pass

        ax.set_xlabel('Time')
        ax.set_ylabel(ylabel, color='k')  # color='k' means black
        ax.set_xticks(x)
        ax.set_xticklabels(time_list, rotation='vertical')
        plt.title(plot_title_name)
        fig.savefig(os.path.join(read_data.transit_non_transit_comparison_before_after, saving_file_name))
        # plt.show()

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

    @staticmethod
    # Draw the line graph comparison between whampoa, ho_man_tin, south_horizons, wong_chuk_hang, ocean_park
    # and control group comparison
    def line_map_comparison_between_treatment_and_control(whampoa, ho_man_tin, south_horizons, wong_chuk_hang,
                                                          ocean_park, control_dataframe, line_labels, ylabel,
                                                          plot_title_name,
                                                          draw_sentiment=True):
        """
        :param whampoa: dataframe for 500-meter whampoa TN
        :param ho_man_tin: dataframe for 500-meter ho_man_tin TN
        :param south_horizons: dataframe for 500-meter south_horizons TN
        :param wong_chuk_hang: dataframe for 500-meter wong_chuk_hang TN
        :param ocean_park: dataframe for 500-meter ocean_park TN
        :param control_dataframe: dataframe for the other 87 500-meter TNs
        :param line_labels: the line labels of the line graph
        :param ylabel: the label of the generated plot
        :param plot_title_name: the title of the generated plot
        :param draw_sentiment: draw sentiment line graph or not
        :return: a plot for sentiment or activity in the study period
        """
        # Get the sentiment activity dictionary for each considered station and control group
        whampoa_sent_act_dict = sentiment_by_month(whampoa)
        ho_man_tin_sent_act_dict = sentiment_by_month(ho_man_tin)
        south_horizons_sent_act_dict = sentiment_by_month(south_horizons)
        wong_chuk_hang_sent_act_dict = sentiment_by_month(wong_chuk_hang)
        ocean_park_sent_act_dict = sentiment_by_month(ocean_park)
        control_sent_act_dict = sentiment_by_month(control_dataframe)
        # Build the dataframe: time and (sentiment, activity)
        result_dataframe_whampoa = pd.DataFrame(list(whampoa_sent_act_dict.items()), columns=['Date', 'Value'])
        result_dataframe_ho_man_tin = pd.DataFrame(list(ho_man_tin_sent_act_dict.items()), columns=['Date', 'Value'])
        result_dataframe_south_horizons = pd.DataFrame(list(south_horizons_sent_act_dict.items()), columns=['Date', 'Value'])
        result_dataframe_wong_chuk_hang = pd.DataFrame(list(wong_chuk_hang_sent_act_dict.items()), columns=['Date', 'Value'])
        result_dataframe_ocean_park = pd.DataFrame(list(ocean_park_sent_act_dict.items()), columns=['Date', 'Value'])
        result_dataframe_control = pd.DataFrame(list(control_sent_act_dict.items()), columns=['Date', 'Value'])
        # Reorder the result based on the time
        result_dataframe_whampoa_time_index = result_dataframe_whampoa.set_index('Date')
        result_dataframe_whampoa_for_plot = result_dataframe_whampoa_time_index.loc[time_list]
        result_dataframe_ho_man_tin_time_index = result_dataframe_ho_man_tin.set_index('Date')
        result_dataframe_ho_man_tin_for_plot = result_dataframe_ho_man_tin_time_index.loc[time_list]
        result_dataframe_south_horizons_time_index = result_dataframe_south_horizons.set_index('Date')
        result_dataframe_south_horizons_for_plot = result_dataframe_south_horizons_time_index.loc[time_list]
        result_dataframe_wong_chuk_hang_time_index = result_dataframe_wong_chuk_hang.set_index('Date')
        result_dataframe_wong_chuk_hang_for_plot = result_dataframe_wong_chuk_hang_time_index.loc[time_list]
        result_dataframe_ocean_park_time_index = result_dataframe_ocean_park.set_index('Date')
        result_dataframe_ocean_park_for_plot = result_dataframe_ocean_park_time_index.loc[time_list]
        result_dataframe_control_time_index = result_dataframe_control.set_index('Date')
        result_dataframe_control_for_plot = result_dataframe_control_time_index.loc[time_list]

        x = np.arange(0, len(time_list), 1)
        if draw_sentiment:
            y1 = [value[0] for value in list(result_dataframe_whampoa_for_plot['Value'])]
            y2 = [value[0] for value in list(result_dataframe_ho_man_tin_for_plot['Value'])]
            y3 = [value[0] for value in list(result_dataframe_south_horizons_for_plot['Value'])]
            y4 = [value[0] for value in list(result_dataframe_wong_chuk_hang_for_plot['Value'])]
            y5 = [value[0] for value in list(result_dataframe_ocean_park_for_plot['Value'])]
            y6 = [value[0] for value in list(result_dataframe_control_for_plot['Value'])]
        else:  # draw the activity comparison plot. Use log10(num of tweets) instead
            y1 = [np.log10(value[1]) for value in list(result_dataframe_whampoa_for_plot['Value'])]
            y2 = [np.log10(value[1]) for value in list(result_dataframe_ho_man_tin_for_plot['Value'])]
            y3 = [np.log10(value[1]) for value in list(result_dataframe_south_horizons_for_plot['Value'])]
            y4 = [np.log10(value[1]) for value in list(result_dataframe_wong_chuk_hang_for_plot['Value'])]
            y5 = [np.log10(value[1]) for value in list(result_dataframe_ocean_park_for_plot['Value'])]
            y6 = [np.log10(value[1]) for value in list(result_dataframe_control_for_plot['Value'])]

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        lns1 = ax.plot(x, y1, 'b-', label=line_labels[0], linestyle='--', marker='o')
        lns2 = ax.plot(x, y2, 'g-', label=line_labels[1], linestyle='--', marker='o')
        lns3 = ax.plot(x, y3, 'c-', label=line_labels[2], linestyle='--', marker='o')
        lns4 = ax.plot(x, y4, 'm-', label=line_labels[3], linestyle='--', marker='o')
        lns5 = ax.plot(x, y5, 'r-', label=line_labels[4], linestyle='--', marker='o')
        lns6 = ax.plot(x, y6, 'y-', label=line_labels[5], linestyle='--', marker='^')
        # Whether to draw the vertical line that in
        plt.axvline(5.77, color='black')
        plt.axvline(7.95, color='black')
        if draw_sentiment:  # the ylim of sentiment and activity plots are different
            ax.text(3.5, 0, 'Opening Date(WHA,HOM): \nOct 23, 2016',
                    horizontalalignment='center', color='black')
            ax.text(11, 0, 'Opening Date(SOH,WCH,OCP): \nDec 28, 2016',
                    horizontalalignment='center', color='black')
        else:
            ax.text(3.5, 3.0, 'Opening Date(WHA,HOM): \nOct 23, 2016',
                    horizontalalignment='center', color='black')
            ax.text(11, 3.0,  'Opening Date(SOH,WCH,OCP): \nDec 28, 2016',
                    horizontalalignment='center', color='black')

        # Add the legend
        lns = lns1 + lns2 + lns3 + lns4 + lns5 + lns6
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs)

        # Draw the average of the sentiment level
        # This average sentiment level means the average sentiment of 93 500-meter TBs
        if draw_sentiment:
            ax.axhline(y=0.40, color='r', linestyle='solid')
            ax.text(3, 0.43, 'Average Sentiment Level: 0.40', horizontalalignment='center', color='r')
            # The ylim is set as (-1, 1)
            ax.set_ylim(-1, 1)
        else:  # here I don't want to draw the horizontal activity line as the activity level varies greatly between TNs
            pass

        ax.set_xlabel('Time')
        ax.set_ylabel(ylabel, color='k')  # color='k' means black
        ax.set_xticks(x)
        ax.set_xticklabels(time_list, rotation='vertical')
        plt.title(plot_title_name)
        if draw_sentiment:
            file_name = 'Sentiment_treatment_control_compare.png'
            fig.savefig(os.path.join(read_data.transit_non_transit_comparison_before_after, file_name))
            plt.show()
        else:
            file_name = 'Activity_treatment_control_compare.png'
            fig.savefig(os.path.join(read_data.transit_non_transit_comparison_before_after, file_name))
            plt.show()

    @staticmethod
    # Draw the station-level treatment and control comparison
    def draw_treatment_control_compare_station_level(path, ylabel, plot_title_name, line_tags=('Treatment', 'Control'),
                                                     draw_sentiment:bool=True):
        """
        :param path: path for storing the data in the longitudinal study(500-meter buffers)
        :param ylabel: the ylabel of the plot
        :param plot_title_name: the title of the plot
        :param line_tags: the line label of the line graph
        :param draw_sentiment: boolean variable: whether we draw the sentiment plot or not
        :return: a plot which shows the sentiment and activity variation through time from May 2016 to Dec 2017
        """
        dataframe_dict = {}
        for file in os.listdir(path):
            dataframe = pd.read_pickle(os.path.join(path, file))
            # Only consider the TPUs in which more than 100 tweets are posted
            if dataframe.shape[0] > 100:
                dataframe_dict[file[:-4]] = dataframe
            else:
                pass
        station_name_list = list(dataframe_dict.keys())
        dataframe_for_plot_list = []
        selected_station_name_list = []
        for station_name in station_name_list:
            try:  # some 500-meter transit neighborhood don't contain tweets in some months
                station_sentiment_sent_act_dict = sentiment_by_month(dataframe_dict[station_name])
                selected_station_name_list.append(station_name)
                result_dataframe = pd.DataFrame(list(station_sentiment_sent_act_dict.items()), columns=['Date', 'Value'])
                result_dataframe_time_index = result_dataframe.set_index('Date')
                result_dataframe_for_plot = result_dataframe_time_index.loc[time_list]
                dataframe_for_plot_list.append(result_dataframe_for_plot)
            except:
                pass
        # Get the numeber of 500-meters buffers we consider
        print('The number of stations we considered is: {}'.format(len(selected_station_name_list)))
        x = np.arange(0, len(time_list), 1)
        if draw_sentiment:
            y_list = []
            for dataframe_plot in dataframe_for_plot_list:
                plot_list = [value[0] for value in list(dataframe_plot['Value'])]
                y_list.append(plot_list)
        else:
            y_list = []
            for dataframe_plot in dataframe_for_plot_list:
                plot_list = [np.log10(value[1]) for value in list(dataframe_plot['Value'])]
                y_list.append(plot_list)

        fig, ax = plt.subplots(1, 1, figsize=(20, 16))

        lns_list = []
        for station, y_value in zip(selected_station_name_list, y_list):
            if station in TransitNeighborhood_before_after.before_after_stations:
                lns = ax.plot(x, y_value, 'r-', label=line_tags[0], linestyle='--', marker='o')
                lns_list.append(lns)
            else:
                lns = ax.plot(x, y_value, 'g-', label=line_tags[1], linestyle='--', marker='^')
                lns_list.append(lns)

        # lns_sum = 0
        # for lns_item in lns_list:
        #     lns_sum += lns_item
        #
        # labs = [l.get_label() for l in lns_sum]

        # Draw the average of the sentiment level
        # This average sentiment level means the average sentiment of 93 500-meter TBs
        if draw_sentiment:
            ax.axhline(y=0.40, color='r', linestyle='solid')
            ax.text(3, 0.43, 'Average Sentiment Level: 0.40', horizontalalignment='center', color='r')
            # The ylim is set as (-1, 1)
            ax.set_ylim(-1, 1)
        else:  # here I don't want to draw the horizontal activity line as the activity level varies greatly between TNs
            pass

        ax.set_xlabel('Time')
        ax.set_ylabel(ylabel, color='k')  # color='k' means black
        ax.set_xticks(x)
        ax.set_xticklabels(time_list, rotation='vertical')
        plt.title(plot_title_name)
        if draw_sentiment:
            file_name = 'Sentiment_treatment_control_compare_station_level.png'
            fig.savefig(os.path.join(read_data.transit_non_transit_comparison_before_after, file_name))
            plt.show()
        else:
            file_name = 'Activity_treatment_control_compare_station_level.png'
            fig.savefig(os.path.join(read_data.transit_non_transit_comparison_before_after, file_name))
            plt.show()


# Get the number of tweets and the number of unique users in a tweet dataframe
def number_of_tweet_user(df, station_name):
    user_num = len(set(df['user_id_str']))
    tweet_num = df.shape[0]
    print('For the {}, Total number of tweet is: {}; Total number of user is {}'.format(
        station_name, tweet_num, user_num))


# compute the percentage of positive Tweets: 2 is positive
def positive_percent(df):
    positive = 0
    for sentiment in df['sentiment']:
        if sentiment=='2':
            positive+=1
        else:
            pass
    return positive/df.shape[0]


# compute the percentage of positive Tweets: 0 is negative
def negative_percent(df):
    negative = 0
    for sentiment in df['sentiment']:
        if sentiment=='0':
            negative+=1
        else:
            pass
    return negative/df.shape[0]


# compute positive percentage minus negative percentage: metric used to evaluate the sentiment of an area
# https://www.sciencedirect.com/science/article/pii/S0040162515002024
def pos_percent_minus_neg_percent(df):
    pos_percent = positive_percent(df)
    neg_percent = negative_percent(df)
    return pos_percent - neg_percent


# compute the number of positive Tweets/number of negative Tweets
def positive_tweets_divide_negative_tweets(df):
    positive = 0
    negative = 0
    for sentiment in df['sentiment']:
        if sentiment == 2:
            positive += 1
        elif sentiment == 0:
            negative += 1
        else:
            pass
    try:
        result = positive/negative
    except:
        print('This file does not have negative Tweets')
        if positive==0 and negative==0:
            result = 1
        else:
            result = 150 # 150 hear means infinity-just for easy plotting
    return result


# compute the sentiment level for each month
def sentiment_by_month(df, compute_positive_percent=False, compute_negative_percent=False):
    # check whether the value in the hk_time attribute is string or not
    if isinstance(list(df['hk_time'])[0], str):
        df['hk_time'] = df.apply(
            lambda row: TransitNeighborhood_before_after.transform_string_time_to_datetime(row['hk_time']), axis=1)
    else:
        pass
    # Check whether one dataframe has the year and the month columns. If not, generate them!
    try:
        df['month_plus_year'] = df.apply(lambda row: str(int(row['year']))+'_'+str(int(row['month'])),
                                         axis=1)
    except KeyError:
        df['month_plus_year'] = df.apply(lambda row: str(row['hk_time'].year) + '_' + str(row['hk_time'].month),
                                         axis=1)
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


def get_tweets_based_on_date(file_path:str, station_name:str, start_date, end_date):
    """
    :param file_path: path which saves the folders of each TN
    :param station_name: the name of MTR station in each TN
    :param start_date: the start date of the time range we consider
    :param end_date: the end date of the time range we consider
    :return: a dataframe which contains tweets in a specific time range
    """
    combined_dataframe = pd.read_pickle(os.path.join(file_path, station_name+'.pkl'))
    print('---------------------------------------------------------------------------------------')
    print('The distribution of the tweet sentiment is: {}'.format(Counter(combined_dataframe['sentiment'])))
    combined_dataframe['hk_time'] = combined_dataframe.apply(
        lambda row: TransitNeighborhood_before_after.transform_string_time_to_datetime(row['hk_time']), axis=1)
    # combined_dataframe['year'] = combined_dataframe.apply(
    #     lambda row: row['hk_time'].year, axis=1
    # )
    # combined_dataframe['month'] = combined_dataframe.apply(
    #     lambda row: row['hk_time'].month, axis=1
    # )
    # combined_dataframe['day'] = combined_dataframe.apply(
    #     lambda row: row['hk_time'].day, axis=1
    # )
    # Only consider the tweets posted in a specific time range
    # Use the time mask to find rows in a specific time range
    time_mask = (combined_dataframe['hk_time'] >= start_date) & (combined_dataframe['hk_time'] <= end_date)
    filtered_dataframe = combined_dataframe.loc[time_mask]
    return filtered_dataframe


def build_text_for_wordcloud_topic_model(df, oct_open=True, build_wordcloud=True):
    """
    :param df: the whole dataframe for before and after study
    :param oct_open: if the station is opened in October or not
    :param build_wordcloud: whether for drawing wordcloud or for topic modelling
    :return: text or dataframes which would be used to generate word cloud or build topic model
    """
    if oct_open:
        open_date_start = october_23_start
        open_date_end = october_23_end
        df_before = df.loc[df['hk_time'] < open_date_start]
        df_after = df.loc[df['hk_time'] > open_date_end]
    else:
        open_date_start = december_28_start
        open_date_end = december_28_end
        df_before = df.loc[df['hk_time'] < open_date_start]
        df_after = df.loc[df['hk_time'] > open_date_end]
    if build_wordcloud:
        before_text = wordcloud_generate.create_text_for_wordcloud(df_before)
        after_text = wordcloud_generate.create_text_for_wordcloud(df_after)
        return before_text, after_text
    else:
        return df_before, df_after


def generate_wordcloud(words_before, words_after, mask, file_name_before, file_name_after, color_func):
    """
    :param words_before: words before the openning date of a station
    :param words_after: words after the openning date of a station
    :param mask: shape mask used to draw the plot
    :param file_name_before: the name of the saved file before the MTR station starts operation
    :param file_name_after: the name of the saved file after the MTR station starts operation
    :param color_func: color function
    """
    # stopwords argument in word_cloud: specify the words we neglect when outputing the wordcloud
    word_cloud_before = WordCloud(width = 520, height = 520, background_color='white',
                           font_path=wordcloud_generate.symbola_font_path,
                                  mask=mask, max_words=800).generate(words_before)
    word_cloud_after = WordCloud(width = 520, height = 520, background_color='white',
                           font_path=wordcloud_generate.symbola_font_path,
                                  mask=mask, max_words=800).generate((words_after))
    fig_before = plt.figure(figsize=(15,13), facecolor = 'white', edgecolor='black')
    plt.imshow(word_cloud_before.recolor(color_func=color_func, random_state=3), interpolation="bilinear")
    plt.axis('off')
    plt.tight_layout(pad=0)
    fig_before.savefig(os.path.join(read_data.transit_non_transit_comparison_before_after, file_name_before))
    plt.show()
    fig_after = plt.figure(figsize=(15, 13), facecolor='white', edgecolor='black')
    plt.imshow(word_cloud_after.recolor(color_func=color_func, random_state=3), interpolation="bilinear")
    plt.axis('off')
    plt.tight_layout(pad=0)
    fig_after.savefig(os.path.join(read_data.transit_non_transit_comparison_before_after, file_name_after))
    plt.show()


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

    data_ready = Topic_Modelling_for_tweets.process_words(tokenized_text_list,
                                                          stop_words=Topic_Modelling_for_tweets.unuseful_terms_set,
                                                          bigram_mod=bigram_mod,
                                                          trigram_mod=trigram_mod)
    text_count_list = [len(text) for text in data_ready]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.distplot(text_count_list, kde=False, hist=True)
    ax.axvline(7, color='black')
    plt.xlim((0, 100))
    plt.ylim((0, 600))
    # Check if it is appropriate to set the number of keywords as 7 in this dataframe
    plt.xticks(list(plt.xticks()[0]) + [7])
    plt.title(station_name+': Tweet Word Count Histogram')
    plt.savefig(os.path.join(read_data.transit_non_transit_comparison_before_after, saved_file_name))
    plt.show()


# Set the hyperparameter: the number of the topics
topic_modelling_search_params = {'n_components': [5, 6, 7]}


def build_topic_model(df, keyword_file_name, topic_predict_file_name, saving_path):
    """
    :param df: the dataframe which contains the posted tweets
    :param keyword_file_name: the name of the saved file which contains the keyword for each topic
    :param topic_predict_file_name: the name of the saved file which contains the topic prediction for each tweet
    :param saving_path: the saving path
    """
    text_list = list(df['cleaned_text'])
    tokenized_text_list = [word_tokenize(text) for text in text_list]
    bigram_phrases = gensim.models.phrases.Phrases(tokenized_text_list, min_count=2, threshold=10)
    bigram_mod = gensim.models.phrases.Phraser(bigram_phrases)
    trigram_phrases = gensim.models.phrases.Phrases(bigram_mod[tokenized_text_list])
    trigram_mod = gensim.models.phrases.Phraser(trigram_phrases)
    data_ready = Topic_Modelling_for_tweets.process_words(tokenized_text_list,
                                                          stop_words=Topic_Modelling_for_tweets.unuseful_terms_set,
                                                          bigram_mod=bigram_mod, trigram_mod=trigram_mod)
    # Draw the distribution of the length of the tweet: waiting to be changed tomorrow
    data_sentence_in_one_list = [' '.join(text) for text in data_ready]
    Topic_Modelling_for_tweets.get_lda_model(data_sentence_in_one_list,
                                             grid_search_params=topic_modelling_search_params,
                                             number_of_keywords=7,
                                             keywords_file=keyword_file_name,
                                             topic_predict_file=topic_predict_file_name,
                                             saving_path=saving_path)


if __name__ == '__main__':
    # For instance, if we want to compare the sentiment and activity level before and after the
    # openning date of the Whampoa MTR railway station in Hong Kong, since the station is opened on 23 Oct 2016,
    # we could specify the openning date using datatime package and output before and after dataframes
    october_23_start = datetime(2016, 10, 23, 0, 0, 0, tzinfo=time_zone_hk)
    october_23_end = datetime(2016, 10, 23, 23, 59, 59, tzinfo=time_zone_hk)
    december_28_start = datetime(2016, 12, 28, 0, 0, 0, tzinfo=time_zone_hk)
    december_28_end = datetime(2016, 12, 28, 23, 59, 59, tzinfo=time_zone_hk)
    start_date = datetime(2016, 5, 7, 0, 0, 0, tzinfo=time_zone_hk)
    end_date = datetime(2017, 12, 31,  23, 59, 59, tzinfo=time_zone_hk)

    longitudinal_data_path = os.path.join(read_data.datasets, 'station_related_frames')
    print('The path we currently use is: {}'.format(longitudinal_data_path))
    print('------------------------------------------------------------------------')
    print('The general information of the stations considered in the longitudinal study....')
    whampoa_dataframe = get_tweets_based_on_date(longitudinal_data_path, 'Whampoa', start_date,
                                                  end_date)
    number_of_tweet_user(whampoa_dataframe, station_name='Whampoa')
    print('------------------------------------------------------------------------')
    ho_man_tin_dataframe = get_tweets_based_on_date(longitudinal_data_path,
                                                    'Ho Man Tin', start_date, end_date)
    number_of_tweet_user(ho_man_tin_dataframe, station_name='Ho Man Tin')
    print('------------------------------------------------------------------------')
    south_horizons_dataframe = get_tweets_based_on_date(longitudinal_data_path,
                                                        'South Horizons', start_date, end_date)
    number_of_tweet_user(south_horizons_dataframe, station_name='South Horizons')
    print('------------------------------------------------------------------------')
    lei_tung_dataframe = get_tweets_based_on_date(longitudinal_data_path,
                                                  'Lei Tung', start_date, end_date)
    number_of_tweet_user(lei_tung_dataframe, station_name='Lei Tung')
    print('------------------------------------------------------------------------')
    wong_chuk_hang_dataframe = get_tweets_based_on_date(longitudinal_data_path,
                                                        'Wong Chuk Hang', start_date, end_date)
    number_of_tweet_user(wong_chuk_hang_dataframe, station_name='Wong Chuk Hang')
    print('------------------------------------------------------------------------')
    ocean_park_dataframe = get_tweets_based_on_date(longitudinal_data_path,
                                                    'Ocean Park', start_date, end_date)
    number_of_tweet_user(ocean_park_dataframe, station_name='Ocean Park')
    print('------------------------------------------------------------------------\n')
    # Draw the histogram of the number of words in each TNs
    draw_word_count_histogram(whampoa_dataframe, saved_file_name='whampoa_word_count_hist.png',
                              station_name='Whampoa')
    draw_word_count_histogram(ho_man_tin_dataframe, saved_file_name='ho_man_tin_word_count_hist.png',
                              station_name='Ho Man Tin')
    draw_word_count_histogram(south_horizons_dataframe, saved_file_name='south_horizons_word_count_hist.png',
                              station_name='South Horizons')
    draw_word_count_histogram(lei_tung_dataframe, saved_file_name='lei_tung_word_count_hist.png',
                              station_name='Lei Tung')
    draw_word_count_histogram(wong_chuk_hang_dataframe, saved_file_name='wung_chuk_hang_word_count_hist.png',
                              station_name='Wong Chuk Hang')
    draw_word_count_histogram(ocean_park_dataframe, saved_file_name='ocean_park_word_count_hist.png',
                              station_name='Ocean Park')
    #
    # ================================='Activity and Sentiment Comparison'==========================================
    # Here we use the other 87 stations as the control group
    selected_columns = ['user_id_str', 'lat', 'lon', 'url', 'lang', 'hk_time', 'created_at', 'year',
                        'month','text', 'SmallTPU', 'cleaned_text', 'sentiment']
    nontn_dataframe_list = []
    for file in os.listdir(longitudinal_data_path):
        station_name = file[:-4]
        if station_name not in TransitNeighborhood_before_after.before_after_stations:
            dataframe = pd.read_pickle(os.path.join(longitudinal_data_path, station_name+'.pkl'))
            dataframe_seleted_columns = dataframe[selected_columns]
            nontn_dataframe_list.append(dataframe_seleted_columns)
        else:
            pass
    non_tn_dataframe = pd.concat(nontn_dataframe_list, axis=0)
    non_tn_dataframe.to_pickle(os.path.join(read_data.transit_non_transit_comparison_before_after,
                                            'nontn_dataframe.pkl'))
    print('-----------------------------------------')
    print('The shape of the non_tn_dataframe is: ')
    print(non_tn_dataframe.shape)
    print('-----------------------------------------')

    line_labels_treatment_control_comparison = ('Whampoa', 'Ho Man Tin', 'South Horizons', 'Wong Chuk Hang', 'Ocean Park',
                   'Control Group')
    TransitNeighborhood_before_after.line_map_comparison_between_treatment_and_control(whampoa=whampoa_dataframe,
                                                                                       ho_man_tin=ho_man_tin_dataframe,
                                                                                       south_horizons=south_horizons_dataframe,
                                                                                       wong_chuk_hang=wong_chuk_hang_dataframe,
                                                                                       ocean_park=ocean_park_dataframe,
                                                                                       control_dataframe=non_tn_dataframe,
                                                                                       line_labels=line_labels_treatment_control_comparison,
                                                                                       ylabel='Percentage of Positive Tweets Minus Percentage of Negative Tweets',
                                                                                       plot_title_name='Treatment Group and Control Group Comparison - Monthly Tweet Sentiment',
                                                                                       draw_sentiment=True)
    TransitNeighborhood_before_after.line_map_comparison_between_treatment_and_control(whampoa=whampoa_dataframe,
                                                                                       ho_man_tin=ho_man_tin_dataframe,
                                                                                       south_horizons=south_horizons_dataframe,
                                                                                       wong_chuk_hang=wong_chuk_hang_dataframe,
                                                                                       ocean_park=ocean_park_dataframe,
                                                                                       control_dataframe=non_tn_dataframe,
                                                                                       line_labels=line_labels_treatment_control_comparison,
                                                                                       ylabel='Number of Tweets(log10)',
                                                                                       plot_title_name='Treatment Group and Control Group Comparison - Number of Posted Tweets(log10)',
                                                                                       draw_sentiment=False)

    TransitNeighborhood_before_after.draw_treatment_control_compare_station_level(path=r'F:\CityU\Datasets\station_related_frames',
                                                                                  ylabel='Percentage of Positive Tweets Minus Percentage of Negative Tweets',
                                                                                  plot_title_name='Treatment Group and Control Group Comparison - Monthly Tweet Sentiment')
    TransitNeighborhood_before_after.draw_treatment_control_compare_station_level(
        path=r'F:\CityU\Datasets\station_related_frames',
        ylabel='Number of Posted Tweets(log10)',
        plot_title_name='Treatment Group and Control Group Comparison(Station Level) - Number of Posted Tweets(log10)',
        draw_sentiment=False)

    # Build the instances of TNs in the treatment group
    whampoa_tn = TransitNeighborhood_before_after(tn_dataframe=whampoa_dataframe,
                                                  non_tn_dataframe=non_tn_dataframe,
                                                  before_and_after=True, oct_open=True, compute_positive=False,
                                                  compute_negative=False)
    ho_man_tin_tn = TransitNeighborhood_before_after(tn_dataframe=ho_man_tin_dataframe,
                                                     non_tn_dataframe=non_tn_dataframe,
                                                     before_and_after=True, oct_open=True, compute_positive=False,
                                                     compute_negative=False)
    south_horizons_tn = TransitNeighborhood_before_after(tn_dataframe=south_horizons_dataframe,
                                                         non_tn_dataframe=non_tn_dataframe,
                                                         before_and_after=True, oct_open=False,
                                                         compute_positive=False,
                                                         compute_negative=False)
    wong_chuk_hang_tn = TransitNeighborhood_before_after(tn_dataframe=wong_chuk_hang_dataframe,
                                                         non_tn_dataframe=non_tn_dataframe,
                                                         before_and_after=True, oct_open=False,
                                                         compute_positive=False,
                                                         compute_negative=False)
    ocean_park_tn = TransitNeighborhood_before_after(tn_dataframe=ocean_park_dataframe,
                                                     non_tn_dataframe=non_tn_dataframe,
                                                     before_and_after=True, oct_open=False, compute_positive=False,
                                                     compute_negative=False)
    lei_tung_tn = TransitNeighborhood_before_after(tn_dataframe=lei_tung_dataframe,
                                                   non_tn_dataframe=non_tn_dataframe,
                                                   before_and_after=True, oct_open=False, compute_positive=False,
                                                   compute_negative=False)
    # Draw the sentiment and activity comparison plots
    whampoa_tn.line_map_comparison(line_labels=('Sentiment Level of Whampoa TN', 'Sentiment Level of Control Group'),
                                   ylabel='Percentage of Positive Tweets Minus Percentage of Negative Tweets',
                                   plot_title_name='Sentiment Before and After Study: Whampoa',
                                   saving_file_name='Whampoa_sent_compare.png', draw_sentiment=True)
    whampoa_tn.line_map_comparison(line_labels=('Activity Level of Whampoa TN', 'Activity Level of Control Group'),
                                   ylabel='Number of Tweets(log10)',
                                   plot_title_name='Activity Before and After Study: Whampoa',
                                   saving_file_name='Whampoa_act_compare.png', draw_sentiment=False)
    ho_man_tin_tn.line_map_comparison(line_labels=('Sentiment Level of Ho Man Tin TN', 'Sentiment Level of Control Group'),
                                      ylabel='Percentage of Positive Tweets Minus Percentage of Negative Tweets',
                                      plot_title_name='Sentiment Before and After Study: Ho Man Tin',
                                      saving_file_name='Ho_Man_Tin_sent_compare.png', draw_sentiment=True)
    ho_man_tin_tn.line_map_comparison(line_labels=('Activity Level of Ho Man Tin TN', 'Activity Level of Control Group'),
                                      ylabel='Number of Tweets(log10)',
                                      plot_title_name='Activity Before and After Study: Ho Man Tin',
                                      saving_file_name='Ho_Man_Tin_act_compare.png', draw_sentiment=False)
    wong_chuk_hang_tn.line_map_comparison(line_labels=('Sentiment Level of Wong Chuk Hang TN', 'Sentiment Level of Control Group'),
                                      ylabel='Percentage of Positive Tweets Minus Percentage of Negative Tweets',
                                      plot_title_name='Sentiment Before and After Study: Wong Chuk Hang',
                                      saving_file_name='Wong_Chuk_Hang_sent_compare.png', draw_sentiment=True)
    wong_chuk_hang_tn.line_map_comparison(line_labels=('Activity Level of Wong Chuk Hang TN', 'Activity Level of Control Group'),
                                      ylabel='Number of Tweets(log10)',
                                      plot_title_name='Activity Before and After Study: Wong Chuk Hang',
                                      saving_file_name='Wong_Chuk_Hang_act_compare.png', draw_sentiment=False)
    south_horizons_tn.line_map_comparison(
        line_labels=('Sentiment Level of South Horizons TN', 'Sentiment Level of Control Group'),
        ylabel='Percentage of Positive Tweets Minus Percentage of Negative Tweets',
        plot_title_name='Sentiment Before and After Study: South Horizons',
        saving_file_name='South_Horizons_sent_compare.png', draw_sentiment=True)
    south_horizons_tn.line_map_comparison(
        line_labels=('Activity Level of South Horizons TN', 'Activity Level of Control Group'),
        ylabel='Number of Tweets(log10)',
        plot_title_name='Activity Before and After Study: South Horizons',
        saving_file_name='South_Horizons_act_compare.png', draw_sentiment=False)
    ocean_park_tn.line_map_comparison(
        line_labels=('Sentiment Level of Ocean Park TN', 'Sentiment Level of Control Group'),
        ylabel='Percentage of Positive Tweets Minus Percentage of Negative Tweets',
        plot_title_name='Sentiment Before and After Study: Ocean Park',
        saving_file_name='Ocean_Park_sent_compare.png', draw_sentiment=True)
    ocean_park_tn.line_map_comparison(
        line_labels=('Activity Level of Ocean Park TN', 'Activity Level of Control Group'),
        ylabel='Number of Tweets(log10)',
        plot_title_name='Activity Before and After Study: Ocean Park',
        saving_file_name='Ocean_Park_act_compare.png', draw_sentiment=False)
    lei_tung_tn.line_map_comparison(line_labels=('Sentiment Level of Lei Tung TN', 'Sentiment Level of Control Group'),
        ylabel='Percentage of Positive Tweets Minus Percentage of Negative Tweets',
        plot_title_name='Sentiment Before and After Study: Lei Tung',
        saving_file_name='Lei_Tung_sent_compare.png', draw_sentiment=True)
    lei_tung_tn.line_map_comparison(line_labels=('Activity Level of Lei Tung TN', 'Activity Level of Control Group'),
                                    ylabel='Number of Tweets(log10)',
                                    plot_title_name='Activity Before and After Study: Lei Tung',
                                    saving_file_name='Lei_Tung_act_compare.png', draw_sentiment=False)

    # ===================================================================================================

    # ======================================Wordcloud comparison=========================================
    before_text_whampoa, after_text_whampoa = build_text_for_wordcloud_topic_model(whampoa_dataframe, oct_open=True)
    before_text_ho_man_tin, after_text_ho_man_tin = build_text_for_wordcloud_topic_model(ho_man_tin_dataframe,
                                                                                         oct_open=True)
    before_text_south_horizons, after_text_south_horizons = \
        build_text_for_wordcloud_topic_model(south_horizons_dataframe, oct_open=False)
    before_text_lei_tung, after_text_lei_tung = build_text_for_wordcloud_topic_model(lei_tung_dataframe,
                                                                                     oct_open=False)
    before_text_wong_chuk_hang, after_text_wong_chuk_hang = \
        build_text_for_wordcloud_topic_model(wong_chuk_hang_dataframe, oct_open=False)
    before_text_ocean_park, after_text_ocean_park = build_text_for_wordcloud_topic_model(ocean_park_dataframe,
                                                                                         oct_open=False)

    generate_wordcloud(before_text_whampoa, after_text_whampoa, mask=wordcloud_generate.circle_mask,
                       file_name_before='before_whampoa_wordcloud', file_name_after="after_whampoa_wordcloud",
                       color_func=wordcloud_generate.green_func)
    generate_wordcloud(before_text_ho_man_tin, after_text_ho_man_tin, mask=wordcloud_generate.circle_mask,
                       file_name_before="before_ho_man_tin_wordcloud",
                       file_name_after="after_ho_man_tin_wordcloud", color_func=wordcloud_generate.green_func)
    generate_wordcloud(before_text_south_horizons, after_text_south_horizons, mask=wordcloud_generate.circle_mask,
                       file_name_before="before_south_horizons_wordcloud",
                       file_name_after="after_south_horizons_wordcloud", color_func=wordcloud_generate.green_func)
    generate_wordcloud(before_text_lei_tung, after_text_lei_tung, mask=wordcloud_generate.circle_mask,
                       file_name_before="before_lei_tung_wordcloud",
                       file_name_after="after_lei_tung_wordcloud", color_func=wordcloud_generate.green_func)
    generate_wordcloud(before_text_wong_chuk_hang, after_text_wong_chuk_hang, mask=wordcloud_generate.circle_mask,
                       file_name_before="before_wong_chuk_hang_wordcloud",
                       file_name_after="after_wong_chuk_hang_wordcloud", color_func=wordcloud_generate.green_func)
    generate_wordcloud(before_text_ocean_park, after_text_ocean_park, mask=wordcloud_generate.circle_mask,
                       file_name_before="before_ocean_park_wordcloud",
                       file_name_after="after_ocean_park_wordcloud", color_func=wordcloud_generate.green_func)
    # # ================================================================================================================

    # =======================================Topic Modelling Comparison================================================
    before_dataframe_whampoa, after_dataframe_whampoa = \
        build_text_for_wordcloud_topic_model(whampoa_dataframe, oct_open=True, build_wordcloud=False)
    before_dataframe_ho_man_tin, after_dataframe_ho_man_tin = \
        build_text_for_wordcloud_topic_model(ho_man_tin_dataframe, oct_open=True, build_wordcloud=False)
    before_dataframe_south_horizons, after_dataframe_south_horizons = \
        build_text_for_wordcloud_topic_model(south_horizons_dataframe, oct_open=False, build_wordcloud=False)
    before_dataframe_lei_tung, after_datarame_lei_tung = \
        build_text_for_wordcloud_topic_model(lei_tung_dataframe, oct_open=False, build_wordcloud=False)
    before_dataframe_wong_chuk_hang, after_dataframe_wong_chuk_hang = \
        build_text_for_wordcloud_topic_model(wong_chuk_hang_dataframe, oct_open=False, build_wordcloud=False)
    before_dataframe_ocean_park, after_dataframe_ocean_park = \
        build_text_for_wordcloud_topic_model(ocean_park_dataframe, oct_open=False, build_wordcloud=False)

    before_and_after_dataframes_list = [before_dataframe_whampoa, after_dataframe_whampoa, before_dataframe_ho_man_tin,
                                        after_dataframe_ho_man_tin, before_dataframe_south_horizons,
                                        after_dataframe_south_horizons, before_dataframe_lei_tung,
                                        after_datarame_lei_tung, before_dataframe_wong_chuk_hang,
                                        after_dataframe_wong_chuk_hang, before_dataframe_ocean_park,
                                        after_dataframe_ocean_park]
    name_list = ['before_whampoa', 'after_whampoa', 'before_ho_man_tin', 'after_ho_man_tin',
                 'before_south_horizons', 'after_south_horizons', 'before_lei_tung', 'after_lei_tung',
                 'before_wong_chuk_hang', 'after_wong_chuk_hang', 'before_ocean_park', 'after_ocean_park']

    for dataframe, file_name in zip(before_and_after_dataframes_list, name_list):
        print('-------------------'+file_name+' starts--------------------------')
        build_topic_model(df=dataframe, keyword_file_name=file_name+'_keyword.pkl',
                          topic_predict_file_name=file_name+'_tweet_topic.pkl',
                          saving_path=read_data.before_and_after_topic_modelling_compare)
        print('------------------'+file_name+' ends-----------------------------')
    # =================================================================================================================

