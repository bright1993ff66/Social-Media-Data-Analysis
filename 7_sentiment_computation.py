# Commonly used
import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates
import os
import time
import re
import string
import read_data
from collections import Counter

# A package which could deal with emojis
import emoji

# keras is used to construct the neural nets
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, load_model
from keras.utils import to_categorical
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Flatten, Input, BatchNormalization, Dropout, Conv1D, MaxPooling1D

# Draw the plot
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# Adjust text
from adjustText import adjust_text


# Load all the necessary paths
station_related_path_zh_en_cleaned = read_data.station_related_2017_zh_en_cleaned
station_related_without_same_geo = \
    read_data.station_related_2017_without_same_geo
plot_path = read_data.plot_path

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']



# compute the percentage of positive Tweets: 2 is positive
def positive_percent(df):
    positive = 0
    for sentiment in df['sentiment']:
        if sentiment==2:
            positive+=1
        else:
            pass
    return positive/df.shape[0]


# compute the percentage of positive Tweets: 0 is negative
def negative_percent(df):
    negative = 0
    for sentiment in df['sentiment']:
        if sentiment==0:
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
def sentiment_by_month(df, compute_positive_percent=False):
    Jan = df.loc[df['month']=='Jan']
    Feb = df.loc[df['month']=='Feb']
    Mar = df.loc[df['month'] == 'Mar']
    Apr = df.loc[df['month'] == 'Apr']
    May = df.loc[df['month'] == 'May']
    Jun = df.loc[df['month'] == 'Jun']
    Jul = df.loc[df['month'] == 'Jul']
    Aug = df.loc[df['month'] == 'Aug']
    Sep = df.loc[df['month'] == 'Sep']
    Oct = df.loc[df['month'] == 'Oct']
    Nov = df.loc[df['month'] == 'Nov']
    Dec = df.loc[df['month'] == 'Dec']
    month_list = [Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec]
    tweet_month_sentiment = {}
    for month in month_list:
        if compute_positive_percent:
            tweet_month_sentiment[list(month['month'])[0]] = positive_percent(month)
        else:
            tweet_month_sentiment[list(month['month'])[0]] = pos_percent_minus_neg_percent(month)
    return tweet_month_sentiment


# compute tweet activity for one station
def compute_tweet_activity_for_all_stations(for_human_review=False, for_review=True,
                                            on_monthly_basis=False):
    if for_human_review:
        if for_review: # path for the human reviewed data
            input_path = r'F:\CityU\Datasets\Hong Kong Tweets 2017\human review\station_review'
        else: # path for the predictions generated from algorithms based on the human review data
            input_path = r'F:\CityU\Datasets\Hong Kong Tweets 2017\human review\stations_algo'
    else:
        input_path = \
            r'F:\CityU\Datasets\Hong Kong Tweets 2017\station_related_without_accounts_have_same_geo'
    if not on_monthly_basis:
        tweet_activity_for_each_station = {}
        for file in os.listdir(input_path):
            station_name = file[0:-4] # The file name is like Admiralty.pkl. Use file[:-4] get the station name
            df = pd.read_pickle(os.path.join(input_path, file))
            activity = df.shape[0]
            tweet_activity_for_each_station[station_name] = activity
        result_dict = tweet_activity_for_each_station
    else:
        tweet_activity_for_each_station = {}
        for file in os.listdir(input_path):
            activity_dict = {}
            station_name = file[0:-4]
            df = pd.read_pickle(os.path.join(input_path, file))
            for month in months:
                activity_dict[month] = df.loc[df['month']==month].shape[0]
            tweet_activity_for_each_station[station_name] = activity_dict
        # Only consider the transit neighborhoods whose Tweet acitivity is bigger than 50 in each month
        selected_stations = []
        for index, items in tweet_activity_for_each_station.items():
            if min(list(items.values())) > 30:
                selected_stations.append(index)
            else:
                pass
        result_dict = {station:tweet_activity_for_each_station[station] for station in selected_stations}
    return result_dict


def compute_sentiment_for_one_station_ffnn(df_name, input_path, human_review=False,
                                           for_review=True, output_path=None, output_dataframe=False,
                                           sentiment_for_each_month=False, compute_positive_percent=False):
    """
    df_name: the name of the dataframe
    input_path: the path of our df
    human_review: if we want the human review result
    for_review: just see the sentiment result based on human review. Otherwise, see how the algo performs on the
    human reviewed data
    output_path: the output path which contains sentiment column
    output_dataframe: whether output the dataframe contains the sentiment column
    sentiment_for_each_month: whether compute the sentiment for each month or not(the output would be a dictionary)
    compute_positive_percent: compute the positive percent or compute the positive percent minus negative percent

    """
    df = pd.read_pickle(os.path.join(input_path, df_name))
    if human_review:
        if for_review:
            classes = df['final sentiment_2']
        else:
            classes = df['predictions']
    else:
        classes = df['sentiment']
    if sentiment_for_each_month:
        # For some stations with small number of tweets, no tweet was posted in a particular month
        try:
            result = sentiment_by_month(df, compute_positive_percent)
        except:
            result = {month:0 for month in months }
        result_tuple = result
    else:
        # 2 here means positive
        # result1: percentage of positive tweets
        # result2: percentage of negative tweets
        # result3: number of positive tweets/number of negative tweets
        result1, result2, result3 = [0,0,0]
        try:
            # Compute positive percentage
            result1 = Counter(classes)[2]/sum(Counter(classes).values())
        except:
            print('Result1-The file divided by zero is: ', df_name)
            result1 = float('inf')
        try:
            # Compute negative percentage
            result2 = Counter(classes)[0]/sum(Counter(classes).values())
        except:
            print('Result2-The file divided by zero is: ', df_name)
            result2 = float('inf')
        try:
            # Compute positive percentage minus negative percentage
            result3 = (Counter(classes)[2]-Counter(classes)[0])/sum(Counter(classes).values())
        except:
            print('Result3-The file divided by zero is: ', df_name)
            result3 = float('inf')
        result_tuple = (result1, result2, result3)
    if output_dataframe:
        df.to_pickle(os.path.join(output_path, df_name[:-4]+'_with_sentiment.pkl'))
    return result_tuple


def select_stations_for_overall_sentiment_plot(sentiment_dict, activity_dict, human_review=False,
                                               positive_percent=False, negative_percent=False):
    stations_list = []
    activity_list = []
    sentiment_list = []
    if human_review: # here we only consider the stations when activiy is bigger than 0
        for station, acitivy in activity_dict.items():
            if activity_dict[station] != 0:
                stations_list.append(station)
                activity_list.append(activity_dict[station])
            else:
                pass
        for station in stations_list:
            sentiment_list.append(sentiment_dict[station])
    else: # here we only consider the stations when activiy is bigger than 100
        for station, acitivy in activity_dict.items():
            # For the whole dataset, we only consider the transit neighborhoods whose activity > 100
            if activity_dict[station] > 100:
                stations_list.append(station)
                activity_list.append(activity_dict[station])
            else:
                pass
        for station in stations_list:
            sentiment_list.append(sentiment_dict[station])
    # Here each value of a sentiment list is (posisitve percent, negative percent, pos percent-neg percent)
    pos_minus_neg_list = [value[2] for value in sentiment_list]
    pos_percent_list = [value[0] for value in sentiment_list]
    neg_percent_list = [value[1] for value in sentiment_list]
    if positive_percent:
        df = pd.DataFrame({'Station': stations_list, 'Activity': activity_list, 'Sentiment': pos_percent_list})
    elif negative_percent:
        df = pd.DataFrame({'Station': stations_list, 'Activity': activity_list, 'Sentiment': neg_percent_list})
    else:
        df = pd.DataFrame({'Station': stations_list, 'Activity': activity_list, 'Sentiment': pos_minus_neg_list})
    df['Activity_log10'] = df.apply(lambda row: np.log10(row['Activity']), axis=1)
    df['Activity_log_e'] = df.apply(lambda row: np.log(row['Activity']), axis=1)
    tweet_with_abbreviation = pd.read_csv(os.path.join(read_data.tweet_2017,
                                                       'station_with_abbreviations.csv'))
    name_abbre = tweet_with_abbreviation[['Station', 'Station abbreviations']]
    result = pd.merge(df, name_abbre, on='Station')
    result = result.reset_index(drop=True)
    return result


def plot_overall_sentiment_for_whole_tweets(df, y_label_name, figure_title=None, saved_file_name=None,
                                            without_outlier = False):
    fig, ax = plt.subplots(figsize=(10,10))
    if without_outlier:
        # outliers: these transit neighborhoods have very high pos/neg
        neglected_stations = ['WAC', 'STW', 'CKT', 'TWH']
        df = df.loc[~df['Station abbreviations'].isin(neglected_stations)]
    else:
        pass
    x = list(df['Activity_log10'])
    y = list(df['Sentiment'])
    plt.xlabel('Tweet Activity(log10)')
    plt.ylabel(y_label_name)
    plt.title(figure_title)
    stations_abbreviations_for_annotations = list(df['Station abbreviations'])

    p1 = plt.scatter(x, y, color='red', marker=".")

    texts = []
    for x,y,s in zip(x, y, stations_abbreviations_for_annotations):
        texts.append(plt.text(x,y,s))

    # font = matplotlib.font_manager.FontProperties(family='Tahoma', weight='extra bold', size=8)

    adjust_text(texts, only_move={'points':'y', 'text':'y'},
                arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    fig.savefig(os.path.join(plot_path, saved_file_name), dpi=fig.dpi, bbox_inches='tight')
    plt.show()


def plot_heatmap(df, y_label, file_name):
    # plot the heatmap of sentiment or activity level of the selected transit neighborhooods on a monthly basis
    plt.rcParams['xtick.top'] = True
    df = df.reindex(df.mean(axis=1).sort_values(ascending=False).index)
    sentiment_values = df.values
    stations_we_consider = df.index
    fig, ax = plt.subplots(figsize=(200, 100))
    im = ax.imshow(sentiment_values, cmap = plt.cm.get_cmap('RdBu'))

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(months)))
    ax.set_yticks(np.arange(len(stations_we_consider)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(months, fontsize=7)
    ax.set_yticklabels(stations_we_consider, fontsize=7)

    # Loop over data dimensions and create text annotations.
    for i in range(len(stations_we_consider)):
        for j in range(len(months)):
            text = ax.text(j, i, str(round(sentiment_values[i, j], 2)),
                           ha="center", va="center", color="black", fontsize=6)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.3)
    cbar.ax.set_ylabel(y_label, rotation=-90, va="bottom")

    # ax.set_title("Sentiment Level of Transit Neighborhoods - On a Monthly Basis")
    fig.savefig(os.path.join(read_data.desktop, file_name), dpi=fig.dpi,
                bbox_inches='tight')
    plt.show()


def plot_line_graph(df, figure_title_name, local_figure_name):
    # plot the linegraph of sentiment of the selected transit neighborhooods on a monthly basis
    plt.subplots(figsize=(10, 10))
    months = list(range(1, 13))
    values = df.values
    rows = list(df.index)
    feature_stations = ['Wan Chai', 'Admiralty',  'Mong Kok East', 'Airport', 'Disneyland']
    plt.xlabel('Months')
    plt.ylabel('Sentiment(Percentage of Positive Tweets - Percentage of Negative Tweets) %')
    plt.title(figure_title_name)
    for index in range(values.shape[0]):
        if rows[index] in feature_stations:
            x_pos = np.arange(len(months))
            plt.plot(x_pos, values[index], label=rows[index], linewidth=5.0)
        else:
            x_pos = np.arange(len(months))
            plt.plot(x_pos, values[index], c='grey', alpha=0.6, linewidth=3.0)
    plt.legend()
    plt.savefig(os.path.join(plot_path, local_figure_name), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    # ========================================For the whole Tweets=====================================================
    # Overall Sentiment
    activity_dict_whole = compute_tweet_activity_for_all_stations(for_human_review=False)

    # This dictionary contains both positive percentage, negative percentage and pos min neg
    overall_sentiment_dict = {}

    for file in os.listdir(station_related_path_zh_en_cleaned):
        sentiment = compute_sentiment_for_one_station_ffnn(df_name=file, input_path=station_related_without_same_geo,
                                                           human_review=False,
                                                           for_review=False, output_dataframe=False,
                                                           sentiment_for_each_month=False,
                                                           compute_positive_percent=False)
        station_name = file[0:-4]
        overall_sentiment_dict[station_name] = sentiment
    print(overall_sentiment_dict)

    whole_df_pos_minus_neg = select_stations_for_overall_sentiment_plot(overall_sentiment_dict,
                                                          activity_dict_whole, human_review=False,
                                                                         positive_percent=False,
                                                                      negative_percent=False)
    print('------------------------------------------------')
    print('The average sentiment level is....(pos minus neg)')
    print(whole_df_pos_minus_neg['Sentiment'].mean())
    print('------------------------------------------------')
    whole_df_pos_percent = select_stations_for_overall_sentiment_plot(overall_sentiment_dict,
                                                                        activity_dict_whole, human_review=False,
                                                                        positive_percent=True,
                                                                        negative_percent=False)
    whole_df_neg_percent = select_stations_for_overall_sentiment_plot(overall_sentiment_dict,
                                                                        activity_dict_whole, human_review=False,
                                                                        positive_percent=False,
                                                                        negative_percent=True)
    print('Total number of transit neighborhoods shown on the overall sentiment plot is: ',
          whole_df_pos_minus_neg.shape[0])
    plot_overall_sentiment_for_whole_tweets(whole_df_pos_minus_neg,
                                            figure_title='Overall Sentiment Comparison(Positive Percent Minus Negative Percent)',
                                            saved_file_name='Overall_sentiment_whole_data_with_outlier(pos minus neg).png',
                                            y_label_name='Positive Percentage Minus Negative Percentage %', without_outlier=False)
    plot_overall_sentiment_for_whole_tweets(whole_df_pos_percent,
                                            figure_title='Overall Sentiment Comparison(Positive Percent)',
                                            saved_file_name='Overall_sentiment_whole_data_with_outlier(pos).png',
                                            without_outlier=False, y_label_name='Positive Percentage %')
    plot_overall_sentiment_for_whole_tweets(whole_df_neg_percent,
                                            figure_title='Overall Sentiment Comparison(Negative Percent)',
                                            saved_file_name='Overall_sentiment_whole_data_with_outlier(neg).png',
                                            without_outlier=False, y_label_name='Negative Percentage %')

    # One a monthly basis
    files = os.listdir(station_related_without_same_geo)

    activity_dict = compute_tweet_activity_for_all_stations(for_human_review=False, for_review=False,
                                                            on_monthly_basis=True)
    print(activity_dict)
    activity_dataframe = pd.DataFrame(activity_dict).T
    activity_dataframe = activity_dataframe[months]
    activity_dataframe.to_csv(os.path.join(read_data.desktop, 'by_month_activity.csv'))
    print('On a monthly basis, the number of transit neighborhoods we consider is.....')
    selected_stations = list(activity_dict.keys())
    print(len(selected_stations))
    print('======================')

    dictionaries_pos_minus_neg = []
    station_names_pos_minus_neg = []

    for file in files:
        month_sentiment_dict_pos_minus_neg = compute_sentiment_for_one_station_ffnn(df_name=file,
                                                                                    input_path=station_related_without_same_geo,
                                                                                    output_dataframe=False,
                                                                                    sentiment_for_each_month=True)
        station_names_pos_minus_neg.append(file[0:-4])
        dictionaries_pos_minus_neg.append(month_sentiment_dict_pos_minus_neg)

    print('=====================================================================================')

    sentiment_dict_by_month_pos_minus_neg = {}
    for index, station in enumerate(station_names_pos_minus_neg):
        sentiment_dict_by_month_pos_minus_neg[station] = dictionaries_pos_minus_neg[index]

    selected_sentiment_dict_by_month_pos_minus_neg = {station:sentiment_dict_by_month_pos_minus_neg[station] for
                                                       station in selected_stations}
    sentiment_for_selected_stations_by_month_pos_minus_neg = pd.DataFrame(
        selected_sentiment_dict_by_month_pos_minus_neg).T
    # Reorder the columns of a pandas dataframe
    sentiment_for_selected_stations_by_month_pos_minus_neg = \
        sentiment_for_selected_stations_by_month_pos_minus_neg[months]
    sentiment_for_selected_stations_by_month_pos_minus_neg.to_csv(os.path.join(read_data.desktop,
                                                                               'by_month_file.csv'))

    # plot the heatmap of the sentiment
    plot_heatmap(sentiment_for_selected_stations_by_month_pos_minus_neg,
                 y_label='Percentage of Positive Tweets Minus Percentage of Negative Tweets',
                 file_name='Station_sentiment_by_month_heatmap.png')
    # plot the heatmap of the activity
    plot_heatmap(activity_dataframe, y_label='Activity Level(Number of Tweets)',
                 file_name='Station_activity_by_month_heatmap.png')
    # line graph of the sentiment level
    plot_line_graph(sentiment_for_selected_stations_by_month_pos_minus_neg,
                  'Sentiment Level(Positive Percent-Negative Percent) On a Monthly Basis',
                  'pos_neg_scheme2.png')

    #
