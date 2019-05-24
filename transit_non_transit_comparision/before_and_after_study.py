import pandas as pd
import numpy as np
import os
import pytz
from datetime import datetime

from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import gensim

import read_data
import wordcloud_generate
import Topic_Modelling_for_tweets

from matplotlib import pyplot as plt
from matplotlib import rc
rc('mathtext', default='regular')
import seaborn as sns


months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
time_list = ['2016_7', '2016_8', '2016_9', '2016_10', '2016_11', '2016_12', '2017_1',
             '2017_2', '2017_3', '2017_4', '2017_5', '2017_6', '2017_7', '2017_8', '2017_9', '2017_10',
             '2017_11', '2017_12']

# Hong Kong and Shanghai share the same time zone.
# Hence, we transform the utc time in our dataset into Shanghai time
time_zone_hk = pytz.timezone('Asia/Shanghai')


class TransitNeighborhood_before_after(object):

    def __init__(self, tn_dataframe, non_tn_dataframe, oct_open, before_and_after, compute_positive,
                 compute_negative):
        # the dataframe which contains all the tweets for this TN
        self.tn_dataframe = tn_dataframe # the dataframe which records all the tweets posted in this TN
        self.non_tn_dataframe = non_tn_dataframe # the dataframe which records all the tweets posted in corresponding
        # non_tn
        self.oct_open = oct_open # boolean, check whether the station is opened on oct 23, 2016
        self.before_and_after = before_and_after # boolean, only newly built stations are considered in the before and
        # after study
        self.compute_positive = compute_positive # boolean, True if use positive percent as the sentiment metric
        self.compute_negative = compute_negative # boolean, True if use negative percent as the sentiment metric

        assert isinstance(self.oct_open, bool)
        assert isinstance(self.before_and_after, bool)

    def output_sent_act_dataframe(self):
        result_dict_tn = sentiment_by_month(self.tn_dataframe, compute_positive_percent=self.compute_positive,
                                             compute_negative_percent=self.compute_negative)
        result_dict_non_tn = sentiment_by_month(self.non_tn_dataframe, compute_positive_percent=self.compute_positive,
                                             compute_negative_percent=self.compute_negative)
        result_dataframe_tn = pd.DataFrame(list(result_dict_tn.items()), columns=['Date', 'Value'])
        result_dataframe_non_tn = pd.DataFrame(list(result_dict_non_tn.items()), columns=['Date', 'Value'])
        return result_dataframe_tn, result_dataframe_non_tn

    def line_map_comparison(self, line_labels:tuple, ylabel, plot_title_name,
                            saving_file_name, draw_sentiment=True):
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
        dataframe_with_sentiment_activity = tn_dataframe_sent_act.set_index('Date')
        # So that we could reorder it based on an ordered time list
        dataframe_for_plot = dataframe_with_sentiment_activity.loc[time_list]
        tpu_sent_act = non_tn_dataframe_sent_act.set_index('Date')
        tpu_dataframe_for_plot = tpu_sent_act.loc[time_list]
        x = np.arange(0, len(list(dataframe_for_plot.index)), 1)
        if draw_sentiment:  # draw the sentiment comparison plot: y1: TN; y2: tpu
            y1 = [value[0] for value in list(dataframe_for_plot['Value'])]
            y2 = [value[0] for value in list(tpu_dataframe_for_plot['Value'])]
        else:  # draw the activity comparison plot
            y1 = [np.log10(value[1]) for value in list(dataframe_for_plot['Value'])]
            y2 = [np.log10(value[1]) for value in list(tpu_dataframe_for_plot['Value'])]

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        lns1 = ax.plot(x, y1, 'g-', label=line_labels[0], linestyle='--', marker='o')
        lns2 = ax.plot(x, y2, 'y-', label=line_labels[1], linestyle='--', marker='^')
        # Whether to draw the vertical line which indicates the open date
        if self.before_and_after:
            if self.oct_open:
                plt.axvline(3.77, color='black')
                if draw_sentiment:
                    ax.text(2.8, 0, 'Opening Date: \nOct 23, 2016', horizontalalignment='center', color='black')
                else:
                    ax.text(2.8, 1.5, 'Opening Date: \nOct 23, 2016', horizontalalignment='center', color='black')
            else:
                plt.axvline(5.95, color='black')
                if draw_sentiment:
                    ax.text(5, 0, 'Opening Date: \nDec 28, 2016', horizontalalignment='center', color='black')
                else:
                    ax.text(5, 1.8, 'Opening Date: \nDec 28, 2016', horizontalalignment='center', color='black')
        else:
            pass

        # Add the legend
        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs)

        # Draw the average of the sentiment level
        if draw_sentiment:
            ax.axhline(y=0.40, color='r', linestyle='solid')
            ax.text(3, 0.43, 'Average Sentiment Level: 0.40', horizontalalignment='center', color='r')
            ax.set_ylim(-1, 1)
        else:  # here I don't want to draw the horizontal activity line as the activity level varies greatly between TNs
            pass

        ax.set_xlabel('Time')
        ax.set_ylabel(ylabel, color='k')
        ax.set_xticks(x)
        ax.set_xticklabels(time_list, rotation='vertical')
        plt.title(plot_title_name)
        plt.savefig(os.path.join(read_data.transit_non_transit_comparison_before_after, saving_file_name))
        plt.show()


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
def sentiment_by_month(df, compute_positive_percent=False, compute_negative_percent=False):
    try:
        df['month_plus_year'] = df.apply(lambda row: str(row['year'])+'_'+str(row['month']),
                                         axis=1)
    except:
        df['month_plus_year'] = df.apply(lambda row: str(row['hk_time'].year) + '_' + str(row['hk_time'].month),
                                         axis=1)
    dataframe_dict = {}
    for time, dataframe_by_time in df.groupby('month_plus_year'):
        dataframe_dict[time] = dataframe_by_time
    time_list = list(dataframe_dict.keys())
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


def get_tweets_based_on_date(station_name, start_date, end_date):
    dataframe_2016 = pd.read_pickle(os.path.join(read_data.station_related_path_2016, station_name+'.pkl'))
    dataframe_2017 = pd.read_pickle(os.path.join(read_data.station_related_2017_without_same_geo, station_name+'.pkl'))
    # select relevant columns
    dataframe_2016 = dataframe_2016[['user_id_str', 'hk_time', 'lang', 'cleaned_text', 'text',
                                     'sentiment', 'url', 'lat', 'lon']]
    dataframe_2017 = dataframe_2017[['user_id_str', 'hk_time', 'lang', 'cleaned_text', 'text',
                                     'sentiment', 'url', 'lat', 'lon']]
    dataframe_2016['month'] = dataframe_2016.apply(lambda row: row['hk_time'].month, axis=1)
    dataframe_2017['month'] = dataframe_2017.apply(lambda row: row['hk_time'].month, axis=1)
    # combined dataframe
    combined_dataframe = pd.concat([dataframe_2016, dataframe_2017])
    time_mask = (combined_dataframe['hk_time'] >= start_date) & (combined_dataframe['hk_time'] <= end_date)
    filtered_dataframe = combined_dataframe.loc[time_mask]
    return filtered_dataframe


def compare_sentiment(df, before_date, after_date):
    before_mask = (df['hk_time'] < before_date)
    before_dataframe = df.loc[before_mask]
    after_mask = (df['hk_time'] > after_date)
    after_dataframe = df.loc[after_mask]
    before_sentiment_dict = sentiment_by_month(before_dataframe, compute_positive_percent=False,
                                               compute_negative_percent=False)
    after_sentiment_dict = sentiment_by_month(after_dataframe, compute_positive_percent=False,
                                              compute_negative_percent=False)
    return before_sentiment_dict, after_sentiment_dict


def build_text_for_wordcloud_topic_model(df, oct_open=True, build_wordcloud=True):
    """
    :param df: the whole dataframe for before and after study
    :param oct_open: if the station is opened in October or not
    :param build_wordcloud: whether for drawing wordcloud or for topic modelling
    :return: text or dataframe which would be used to generate word cloud or build topic model
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
    :param file_name: the name of the saved picture
    :param color_func: color function
    :return: two wordclouds for before and after
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


def draw_word_count_histogram(df, file_name):
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
    plt.ylim((0, 450))
    # Check if it is appropriate to set the number of keywords as 7 in this dataframe
    plt.xticks(list(plt.xticks()[0]) + [7])
    plt.savefig(os.path.join(read_data.transit_non_transit_comparison_before_after, file_name))
    plt.show()


# Set the hyperparameter: the number of the topics
topic_modelling_search_params = {'n_components': [5, 6, 7]}


def build_topic_model(df, keyword_file_name, topic_predict_file_name, saving_path):
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
    tweet_word_count_list = [len(text) for text in data_ready]
    data_sentence_in_one_list = [' '.join(text) for text in data_ready]
    Topic_Modelling_for_tweets.get_lda_model(data_sentence_in_one_list,
                                             grid_search_params=topic_modelling_search_params,
                                             number_of_keywords=7,
                                             keywords_file=keyword_file_name,
                                             topic_predict_file=topic_predict_file_name,
                                             saving_path=saving_path)


# new_time_list_oct = ['2016_7', '2016_8', '2016_9', '2016_10_before', '2016_10_after',
#                      '2016_11', '2016_12', '2017_1', '2017_2', '2017_3', '2017_4', '2017_5', '2017_6',
#                      '2017_7', '2017_8', '2017_9', '2017_10', '2017_11', '2017_12']
#
# new_time_list_dec = ['2016_7', '2016_8', '2016_9', '2016_10', '2016_11',
#                      '2016_12_before', '2016_12_after', '2017_1', '2017_2', '2017_3',
#                      '2017_4', '2017_5', '2017_6', '2017_7', '2017_8', '2017_9', '2017_10',
#                      '2017_11', '2017_12']


if __name__ == '__main__':

    # path = r'C:\Users\Haoliang Chang\Desktop\transit_non_transit_compare'

    # For instance, if we want to compare the sentiment and activity level before and after the
    # openning date of the Whampoa MTR railway station in Hong Kong, since the station is opened on 23 Oct 2016,
    # we could specify the openning date using datatime package and output before and after dataframes
    october_23_start = datetime(2016, 10, 23, 0, 0, 0, tzinfo=time_zone_hk)
    october_23_end = datetime(2016, 10, 23, 23, 59, 59, tzinfo=time_zone_hk)
    december_28_start = datetime(2016, 12, 28, 0, 0, 0, tzinfo=time_zone_hk)
    december_28_end = datetime(2016, 12, 28, 23, 59, 59, tzinfo=time_zone_hk)
    start_date = datetime(2016, 5, 7, tzinfo=time_zone_hk)
    end_date = datetime(2017, 12, 31, tzinfo=time_zone_hk)

    whammpoa_dataframe = get_tweets_based_on_date('Whampoa', start_date, end_date)
    ho_man_tin_dataframe = get_tweets_based_on_date('Ho Man Tin', start_date, end_date)
    south_horizons_dataframe = get_tweets_based_on_date('South Horizons', start_date, end_date)
    lei_tung_dataframe = get_tweets_based_on_date('Lei Tung', start_date, end_date)
    wong_chuk_hang_dataframe = get_tweets_based_on_date('Wong Chuk Hang', start_date, end_date)
    ocean_park_dataframe = get_tweets_based_on_date('Ocean Park', start_date, end_date)

    whammpoa_dataframe.to_pickle(os.path.join(read_data.transit_non_transit_comparison_before_after,
                                              'whole_whampoa_dataframe.pkl'))
    ho_man_tin_dataframe.to_pickle(os.path.join(read_data.transit_non_transit_comparison_before_after,
                                                'whole_ho_man_tin_dataframe.pkl'))
    south_horizons_dataframe.to_pickle(os.path.join(read_data.transit_non_transit_comparison_before_after,
                                                    'whole_south_horizons_dataframe.pkl'))
    lei_tung_dataframe.to_pickle(os.path.join(read_data.transit_non_transit_comparison_before_after,
                                              'whole_lei_tung_dataframe.pkl'))
    wong_chuk_hang_dataframe.to_pickle(os.path.join(read_data.transit_non_transit_comparison_before_after,
                                                    'whole_wong_chuk_hang_dataframe.pkl'))
    ocean_park_dataframe.to_pickle(os.path.join(read_data.transit_non_transit_comparison_before_after,
                                                'whole_ocean_park_dataframe.pkl'))

    # draw_word_count_histogram(whammpoa_dataframe, file_name='whampoa_word_count_hist.png')
    # draw_word_count_histogram(ho_man_tin_dataframe, file_name='ho_man_tin_word_count_hist.png')
    # draw_word_count_histogram(south_horizons_dataframe, file_name='south_horizons_word_count_hist.png')
    # draw_word_count_histogram(lei_tung_dataframe, file_name='lei_tung_word_count_hist.png')
    # draw_word_count_histogram(wong_chuk_hang_dataframe, file_name='wung_chuk_hang_word_count_hist.png')
    # draw_word_count_histogram(ocean_park_dataframe, file_name='ocean_park_word_count_hist.png')


    # ================================='Activity and Sentiment Comparison'==========================================

    target_path = \
        r'F:\CityU\Datasets\Hong Kong Tweets 2017\transit_non_transit_comparision\before_and_after\compare_tn_and_nontn'
    oct_23 = pd.read_csv(os.path.join(target_path, 'oct_23_tpu_nontn_tweets.csv'), encoding='latin-1')
    dec_28 = pd.read_csv(os.path.join(target_path, 'dec_28_tpu_nontn_tweets.csv'), encoding='latin-1')

    whampoa_tn = TransitNeighborhood_before_after(tn_dataframe=whammpoa_dataframe, non_tn_dataframe=oct_23, before_and_after=True,
                                                  oct_open=True, compute_positive=False, compute_negative=False)
    ho_man_tin_tn = TransitNeighborhood_before_after(tn_dataframe=ho_man_tin_dataframe, non_tn_dataframe=oct_23,
                                                     before_and_after=True, oct_open=True, compute_positive=False,
                                                     compute_negative=False)
    south_horizons_tn = TransitNeighborhood_before_after(tn_dataframe=south_horizons_dataframe, non_tn_dataframe=dec_28,
                                                         before_and_after=True, oct_open=False, compute_positive=False,
                                                         compute_negative=False)
    wong_chuk_hang_tn = TransitNeighborhood_before_after(tn_dataframe=wong_chuk_hang_dataframe, non_tn_dataframe=dec_28,
                                                         before_and_after=True, oct_open=False, compute_positive=False,
                                                         compute_negative=False)
    ocean_park_tn = TransitNeighborhood_before_after(tn_dataframe=ocean_park_dataframe, non_tn_dataframe=dec_28,
                                                     before_and_after=True, oct_open=False, compute_positive=False,
                                                     compute_negative=False)
    lei_tung_tn = TransitNeighborhood_before_after(tn_dataframe=lei_tung_dataframe, non_tn_dataframe=dec_28,
                                                   before_and_after=True, oct_open=False, compute_positive=False,
                                                   compute_negative=False)

    whampoa_tn.line_map_comparison(line_labels=('Sentiment Level of Whampoa TN', 'Sentiment Level of Non-TN'),
                                   ylabel='Percentage of Positive Tweets Minus Percentage of Negative Tweets',
                                   plot_title_name='Sentiment Before and After Study: Whampoa',
                                   saving_file_name='Whampoa_sent_compare.png', draw_sentiment=True)
    whampoa_tn.line_map_comparison(line_labels=('Activity Level of Whampoa TN', 'Activity Level of Non-TN'),
                                   ylabel='Number of Tweets(log10)',
                                   plot_title_name='Activity Before and After Study: Whampoa',
                                   saving_file_name='Whampoa_act_compare.png', draw_sentiment=False)
    ho_man_tin_tn.line_map_comparison(line_labels=('Sentiment Level of Ho Man Tin TN', 'Sentiment Level of Non-TN'),
                                      ylabel='Percentage of Positive Tweets Minus Percentage of Negative Tweets',
                                      plot_title_name='Sentiment Before and After Study: Ho Man Tin',
                                      saving_file_name='Ho_Man_Tin_sent_compare.png', draw_sentiment=True)
    ho_man_tin_tn.line_map_comparison(line_labels=('Activity Level of Ho Man Tin TN', 'Activity Level of Non-TN'),
                                      ylabel='Number of Tweets(log10)',
                                      plot_title_name='Activity Before and After Study: Ho Man Tin',
                                      saving_file_name='Ho_Man_Tin_act_compare.png', draw_sentiment=False)
    wong_chuk_hang_tn.line_map_comparison(line_labels=('Sentiment Level of Wong Chuk Hang TN', 'Sentiment Level of Non-TN'),
                                      ylabel='Percentage of Positive Tweets Minus Percentage of Negative Tweets',
                                      plot_title_name='Sentiment Before and After Study: Wong Chuk Hang',
                                      saving_file_name='Wong_Chuk_Hang_sent_compare.png', draw_sentiment=True)
    wong_chuk_hang_tn.line_map_comparison(line_labels=('Activity Level of Wong Chuk Hang TN', 'Activity Level of Non-TN'),
                                      ylabel='Number of Tweets(log10)',
                                      plot_title_name='Activity Before and After Study: Wong Chuk Hang',
                                      saving_file_name='Wong_Chuk_Hang_act_compare.png', draw_sentiment=False)
    south_horizons_tn.line_map_comparison(
        line_labels=('Sentiment Level of South Horizons TN', 'Sentiment Level of Non-TN'),
        ylabel='Percentage of Positive Tweets Minus Percentage of Negative Tweets',
        plot_title_name='Sentiment Before and After Study: South Horizons',
        saving_file_name='South_Horizons_sent_compare.png', draw_sentiment=True)
    south_horizons_tn.line_map_comparison(
        line_labels=('Activity Level of South Horizons TN', 'Activity Level of Non-TN'),
        ylabel='Number of Tweets(log10)',
        plot_title_name='Activity Before and After Study: South Horizons',
        saving_file_name='South_Horizons_act_compare.png', draw_sentiment=False)
    ocean_park_tn.line_map_comparison(
        line_labels=('Sentiment Level of Ocean Park TN', 'Sentiment Level of Non-TN'),
        ylabel='Percentage of Positive Tweets Minus Percentage of Negative Tweets',
        plot_title_name='Sentiment Before and After Study: Ocean Park',
        saving_file_name='Ocean_Park_sent_compare.png', draw_sentiment=True)
    ocean_park_tn.line_map_comparison(
        line_labels=('Activity Level of Ocean Park TN', 'Activity Level of Non-TN'),
        ylabel='Number of Tweets(log10)',
        plot_title_name='Activity Before and After Study: Ocean Park',
        saving_file_name='Ocean_Park_act_compare.png', draw_sentiment=False)
    lei_tung_tn.line_map_comparison(line_labels=('Sentiment Level of Lei Tung TN', 'Sentiment Level of Non-TN'),
        ylabel='Percentage of Positive Tweets Minus Percentage of Negative Tweets',
        plot_title_name='Sentiment Before and After Study: Lei Tung',
        saving_file_name='Lei_Tung_sent_compare.png', draw_sentiment=True)
    lei_tung_tn.line_map_comparison(line_labels=('Activity Level of Lei Tung TN', 'Activity Level of Non-TN'),
                                    ylabel='Number of Tweets(log10)',
                                    plot_title_name='Activity Before and After Study: Lei Tung',
                                    saving_file_name='Lei_Tung_act_compare.png', draw_sentiment=False)

	# ===================================================================================================
	#
	# ======================================Wordcloud comparison=========================================
	before_text_whampoa, after_text_whampoa = build_text_for_wordcloud_topic_model(whammpoa_dataframe, oct_open=True)
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
	# ================================================================================================================
	#
	# =======================================Topic Modelling Comparison================================================
	before_dataframe_whampoa, after_dataframe_whampoa = \
		build_text_for_wordcloud_topic_model(whammpoa_dataframe, oct_open=True, build_wordcloud=False)
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

