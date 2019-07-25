import pandas as pd
import os
import numpy as np
import re
import pytz
from datetime import datetime
import csv

import before_and_after
import read_data

# statistics
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# visualization
from matplotlib import pyplot as plt
import seaborn as sns
from adjustText import adjust_text

# Load the month_list and the time_list for plot
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
time_list = ['2016_7', '2016_8', '2016_9', '2016_10', '2016_11', '2016_12', '2017_1',
             '2017_2', '2017_3', '2017_4', '2017_5', '2017_6', '2017_7', '2017_8', '2017_9', '2017_10',
             '2017_11', '2017_12']

# Hong Kong and Shanghai share the same time zone.
# Hence, we transform the utc time in our dataset into Shanghai time
time_zone_hk = pytz.timezone('Asia/Shanghai')
# Load the path for tn_dataframes
data_path = os.path.join(read_data.transit_non_transit_comparison_cross_sectional, 'tpu_data_more')
# Load a csv file which records the names of TPUs
tpu_dataframe = pd.read_csv(os.path.join(read_data.transit_non_transit_comparison_before_after, 'compare_tn_and_nontn',
                                         'tpu_data.csv'), encoding='latin-1')


class TransitNeighborhood_TPU(object):

    tpu_name_list = list(tpu_dataframe['SmallTPU'])
    tn_tpus = np.load(os.path.join(read_data.transit_non_transit_comparison, 'tn_tpus.npy'))
    non_tn_tpus = np.load(os.path.join(read_data.transit_non_transit_comparison, 'non_tn_tpus.npy'))

    def __init__(self, tpu_dataframe, oct_open: bool, before_and_after: bool, compute_positive: bool,
                 compute_negative: bool):
        """
        :param tpu_dataframe: the dataframe which saves all the tweets posted in this TPU
        :param oct_open: boolean, check whether the station is opened on oct 23, 2016
        :param before_and_after: boolean, only newly built stations are considered in the before and
        :param compute_positive: True if use positive percent as the sentiment metric
        :param compute_negative: True if use negative percent as the sentiment metric
        """
        self.tpu_dataframe = tpu_dataframe
        self.oct_open = oct_open
        self.before_and_after = before_and_after
        self.compute_positive = compute_positive
        self.compute_negative = compute_negative

    def output_sent_act_dataframe(self):
        """
        :return: a dataframe which saves the activity and sentiment in each month
        """
        result_dict_tn = before_and_after.sentiment_by_month(self.tpu_dataframe,
                                                             compute_positive_percent=self.compute_positive,
                                                             compute_negative_percent=self.compute_negative)
        result_dataframe_tn = pd.DataFrame(list(result_dict_tn.items()), columns=['Date', 'Value'])
        return result_dataframe_tn

    def line_map(self, ax, line_labels:tuple, ylabel:str, plot_title_name: str, saving_file_name: str,
                            draw_sentiment:bool=True):
        """
        :param ax: axis to make this plot
        :param line_labels: a tuple which records the line labels in the line graph
        :param ylabel: the ylabel of the final plot
        :param plot_title_name: the title of the final plot
        :param saving_file_name: the name of the saved file
        :param draw_sentiment: if True we draw sentiment comparison plot; Otherwise we draw activity comparison plot
        :return: the sentiment/activity comparison plot
        """
        tn_dataframe_sent_act = self.output_sent_act_dataframe()
        # Set one column as the index
        dataframe_with_sentiment_activity = tn_dataframe_sent_act.set_index('Date')
        # So that we could reorder it based on an ordered time list
        dataframe_for_plot = dataframe_with_sentiment_activity.loc[time_list]
        fig, ax = plt.subplots(1,1,figsize=(10,8))
        x = np.arange(0, len(list(dataframe_for_plot.index)), 1)
        if draw_sentiment:  # draw the sentiment comparison plot: y1: TN; y2: tpu
            y1 = [value[0] for value in list(dataframe_for_plot['Value'])]
        else:  # draw the activity comparison plot
            y1 = [np.log10(value[1]) for value in list(dataframe_for_plot['Value'])]

        lns1 = ax.plot(x, y1, 'g-', label=line_labels[0], linestyle='--', marker='o')
        # Whether to draw the vertical line which indicates the open date
        if self.before_and_after:
            if self.oct_open:
                plt.axvline(3.77, color='black')
            else:
                plt.axvline(5.95, color='black')
        else:
            pass

        # Add the legend
        lns = lns1
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs)

        # Draw the average of the sentiment level
        if draw_sentiment:
            # Specify the average sentiment across all TNs
            ax.axhline(y=0.40, color='r', linestyle='solid')
            ax.text(3, 0.43, 'Average Sentiment Level: 0.40', horizontalalignment='center', color='r')
            ax.set_ylim(-1, 1)
        else:  # here I don't want to draw the horizontal activity line as the activity level varies greatly between TNs
            pass

        ax.set_xlabel('Time')
        ax.set_ylabel(ylabel, color='k')
        ax.set_xticks(x)
        ax.set_xticklabels(time_list, rotation='vertical')
        ax.set_title(plot_title_name)
        plt.show()
        fig.savefig(os.path.join(read_data.transit_non_transit_comparison_cross_sectional, saving_file_name))

    @staticmethod
    def select_tpu_for_following_analysis(check_all_stations=False):
        tpu_activity_dict = {}
        for name in TransitNeighborhood_TPU.tpu_name_list:
            dataframe = pd.read_csv(os.path.join(data_path, name, name + '_data.csv'),
                                    encoding='utf-8', dtype='str', quoting=csv.QUOTE_NONNUMERIC)
            tpu_activity_dict[name] = dataframe.shape[0]
        selected_tpu_dict = {}
        # check_all_stations=False means that we only consider TPUs of which the number of posted tweets is bigger
        # than 100
        if not check_all_stations:
            for tpu_name in tpu_activity_dict.keys():
                if tpu_activity_dict[tpu_name] > 100:
                    selected_tpu_dict[tpu_name] = tpu_activity_dict[tpu_name]
                else:
                    pass
        else:
            selected_tpu_dict = tpu_activity_dict
        return selected_tpu_dict

    @staticmethod
    def construct_sent_act_dataframe(sent_dict, activity_dict):
        tpu_name_list = list(activity_dict.keys())
        sentiment_list =[]
        acitivity_list =[]
        for name in tpu_name_list:
            sentiment_list.append(sent_dict[name])
            acitivity_list.append(activity_dict[name])
        activity_log10_list = [np.log10(count) for count in acitivity_list]
        result_dataframe = pd.DataFrame({'tpu_name': tpu_name_list, 'Sentiment': sentiment_list,
                                         'activity':acitivity_list, 'Activity_log10': activity_log10_list})
        result_dataframe['tn_or_not'] = result_dataframe.apply(
            lambda row: TransitNeighborhood_TPU.check_tn_tpu_or_nontn_tpu(str(row['tpu_name']).encode('utf-8')),
            axis=1)
        return result_dataframe

    @staticmethod
    def check_tn_tpu_or_nontn_tpu(tpu_name):
        if tpu_name in TransitNeighborhood_TPU.tn_tpus:
            result = 'tn_tpu'
        elif tpu_name in TransitNeighborhood_TPU.non_tn_tpus:
            result = 'non_tn_tpu'
        else:
            result = 'not considered'
        return result

    @staticmethod
    def plot_overall_sentiment_for_whole_tweets(df, y_label_name:str, figure_title: str=None,
                                                saved_file_name: str=None, without_outlier=False):
        fig, ax = plt.subplots(figsize=(10, 10))
        if without_outlier:
            # outliers: these transit neighborhoods have very high pos/neg
            neglected_stations = ['WAC', 'STW', 'CKT', 'TWH']
            df = df.loc[~df['Station abbreviations'].isin(neglected_stations)]
        else:
            pass
        if 'Labels' in df.columns:
            cluster_zero_dataframe = df.loc[df['Labels'] == 0]
            cluster_one_dataframe = df.loc[df['Labels'] == 1]
            cluster_two_dataframe = df.loc[df['Labels'] == 2]
            other_stations_dataframe = df.loc[~df['Labels'].isin([0, 1, 2])]

            x_cluster_zero = list(cluster_zero_dataframe['Activity_log10'])
            y_cluster_zero = list(cluster_zero_dataframe['Sentiment'])
            x_cluster_one = list(cluster_one_dataframe['Activity_log10'])
            y_cluster_one = list(cluster_one_dataframe['Sentiment'])
            x_cluster_two = list(cluster_two_dataframe['Activity_log10'])
            y_cluster_two = list(cluster_two_dataframe['Sentiment'])
            x_cluster_non_label = list(other_stations_dataframe['Activity_log10'])
            y_cluster_non_label = list(other_stations_dataframe['Sentiment'])
            plt.xlabel('Tweet Activity(log10)')
            plt.ylabel(y_label_name)
            plt.title(figure_title)
            stations_abbreviations_for_annotations_cluster_zero = list(cluster_zero_dataframe['Station abbreviations'])
            stations_abbreviations_for_annotations_cluster_one = list(cluster_one_dataframe['Station abbreviations'])
            stations_abbreviations_for_annotations_cluster_two = list(cluster_two_dataframe['Station abbreviations'])
            stations_abbreviations_for_annotations_non_cluster_stations = \
                list(other_stations_dataframe['Station abbreviations'])

            p_cluster_zero = plt.scatter(x_cluster_zero, y_cluster_zero, color='r', marker=".", label='Cluster One')
            p_cluster_one = plt.scatter(x_cluster_one, y_cluster_one, color='m', marker=".", label='Cluster Two')
            p_cluster_two = plt.scatter(x_cluster_two, y_cluster_two, color='g', marker=".", label='Cluster Three')
            p_cluster_other = plt.scatter(x_cluster_non_label, y_cluster_non_label, color='k', marker=".",
                                          label='Stations Not Considered in Clustering Analysis')

            texts_cluster_zero = []
            for x, y, s in zip(x_cluster_zero, y_cluster_zero, stations_abbreviations_for_annotations_cluster_zero):
                texts_cluster_zero.append(plt.text(x, y, s, color='r'))
            texts_cluster_one = []
            for x, y, s in zip(x_cluster_one, y_cluster_one, stations_abbreviations_for_annotations_cluster_one):
                texts_cluster_one.append(plt.text(x, y, s, color='m'))
            texts_cluster_two = []
            for x, y, s in zip(x_cluster_two, y_cluster_two, stations_abbreviations_for_annotations_cluster_two):
                texts_cluster_two.append(plt.text(x, y, s, color='g'))
            other_stations_text = []
            for x, y, s in zip(x_cluster_non_label, y_cluster_non_label,
                               stations_abbreviations_for_annotations_non_cluster_stations):
                other_stations_text.append(plt.text(x, y, s, color='k'))

            # font = matplotlib.font_manager.FontProperties(family='Tahoma', weight='extra bold', size=8)

            adjust_text(texts_cluster_zero, only_move={'points': 'y', 'text': 'y'},
                        arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
            adjust_text(texts_cluster_one, only_move={'points': 'y', 'text': 'y'},
                        arrowprops=dict(arrowstyle="->", color='m', lw=0.5))
            adjust_text(texts_cluster_two, only_move={'points': 'y', 'text': 'y'},
                        arrowprops=dict(arrowstyle="->", color='g', lw=0.5))
            adjust_text(other_stations_text, only_move={'points': 'y', 'text': 'y'},
                        arrowprops=dict(arrowstyle="->", color='k', lw=0.5))
            plt.legend()
            fig.savefig(os.path.join(read_data.plot_path_2017, saved_file_name), dpi=fig.dpi, bbox_inches='tight')
            plt.show()
        elif 'tn_or_not' in df.columns:
            tn_tpu_dataframe = df.loc[df['tn_or_not'] == 'tn_tpu']
            non_tn_tpu_dataframe = df.loc[df['tn_or_not'] == 'non_tn_tpu']

            x_tpu_tn = list(tn_tpu_dataframe['Activity_log10'])
            y_tpu_tn = list(tn_tpu_dataframe['Sentiment'])
            x_tpu_nontn = list(non_tn_tpu_dataframe['Activity_log10'])
            y_tpu_nontn = list(non_tn_tpu_dataframe['Sentiment'])

            plt.xlabel('Tweet Activity(log10)')
            plt.ylabel(y_label_name)
            plt.title(figure_title)
            tpu_tn_annotations = list(tn_tpu_dataframe['tpu_name'])
            tpu_non_tn_annotations = list(non_tn_tpu_dataframe['tpu_name'])

            p_cluster_zero = plt.scatter(x_tpu_tn, y_tpu_tn, color='r', marker=".", label='TN TPUs')
            p_cluster_one = plt.scatter(x_tpu_nontn, y_tpu_nontn, color='g', marker="^", label='Non-TN TPUs')

            texts_tpu_tn = []
            for x, y, s in zip(x_tpu_tn, y_tpu_tn, tpu_tn_annotations):
                texts_tpu_tn.append(plt.text(x, y, s, color='r'))
            texts_tpu_non_tn = []
            for x, y, s in zip(x_tpu_nontn, y_tpu_nontn, tpu_non_tn_annotations):
                texts_tpu_non_tn.append(plt.text(x, y, s, color='g'))

            adjust_text(texts_tpu_tn, only_move={'points': 'y', 'text': 'y'},
                        arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
            adjust_text(texts_tpu_non_tn, only_move={'points': 'y', 'text': 'y'},
                        arrowprops=dict(arrowstyle="->", color='g', lw=0.5))

            plt.legend()
            fig.savefig(os.path.join(read_data.plot_path_2017, saved_file_name), dpi=fig.dpi, bbox_inches='tight')
            plt.show()
        else:
            x = list(df['Activity_log10'])
            y = list(df['Sentiment'])
            plt.xlabel('Tweet Activity(log10)')
            plt.ylabel(y_label_name)
            plt.title(figure_title)
            stations_abbreviations_for_annotations = list(df['tpu_name'])

            p1 = plt.scatter(x, y, color='red', marker=".")

            texts = []
            for x, y, s in zip(x, y, stations_abbreviations_for_annotations):
                texts.append(plt.text(x, y, s))

            # font = matplotlib.font_manager.FontProperties(family='Tahoma', weight='extra bold', size=8)

            adjust_text(texts, only_move={'points': 'y', 'text': 'y'},
                        arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
            fig.savefig(os.path.join(read_data.plot_path_2017, saved_file_name), dpi=fig.dpi, bbox_inches='tight')
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


def get_data_for_tpu(tpu_name, economic_dataframe, marry_dataframe, edu_dataframe):
    """
    :param tpu_name: the name of the tpu
    :param economic_dataframe: the dataframe contains median income and employment rate
    :param marry_dataframe: the dataframe which saves the marrital status
    :param edu_dataframe: the dataframe that saves the education information
    :return:
    """
    # The economic dataframe contains median income and employment rate
    # median income
    median_income = list(economic_dataframe.loc[
                             economic_dataframe['Small Tertiary Planning Unit Group'] == tpu_name
                             ]['Median Monthly Income from Main Employment(1)'])[0]
    # employment rate
    employment_rate = list(economic_dataframe.loc[
                               economic_dataframe['Small Tertiary Planning Unit Group'] == tpu_name
                               ]['Labour Force Participation Rate(2)'])[0]
    # marrital status
    start_row_index_marry = \
    marry_dataframe.loc[marry_dataframe['Small Tertiary Planning Unit Group'] == tpu_name].index.values[0]
    selected_tpu_dataframe = marry_dataframe.iloc[start_row_index_marry:start_row_index_marry + 6, :]
    marrital_rate = int(selected_tpu_dataframe.iloc[1, 4]) / int(selected_tpu_dataframe.iloc[5, 4])
    # education
    start_row_index_edu = \
    edu_dataframe.loc[edu_dataframe['Small Tertiary Planning Unit Group'] == tpu_name].index.values[0]
    selected_edu_dataframe = edu_dataframe.iloc[start_row_index_edu:start_row_index_edu + 8, :]
    diploma = selected_edu_dataframe.iloc[4, 4]
    sub_degree = selected_edu_dataframe.iloc[5, 4]
    degree = selected_edu_dataframe.iloc[6, 4]
    # if the value equals '-', it means zero
    if diploma == '-':
        diploma_num = 0
    else:
        diploma_num = int(diploma)
    if sub_degree == '-':
        sub_degree_num = 0
    else:
        sub_degree_num = int(sub_degree)
    if degree == '-':
        degree_num = 0
    else:
        degree_num = int(degree)
    numerator = diploma_num + sub_degree_num + degree_num
    denominator = int(selected_edu_dataframe.iloc[7, 4])
    edu_rate = numerator / denominator
    return median_income, employment_rate, marrital_rate, edu_rate


def build_social_demographic_dataframe(tpu_name_list, economic_dataframe, marry_dataframe, edu_dataframe):
    median_income_list = []
    employment_rate = []
    marry_list = []
    education_list = []
    for name in tpu_name_list:
        median_income, employ_rate, marry_status, edu = get_data_for_tpu(name, economic_dataframe,
                                                                         marry_dataframe, edu_dataframe)
        median_income_list.append(median_income)
        employment_rate.append(employ_rate)
        marry_list.append(marry_status)
        education_list.append(edu)

    tpu_2016_social_demographic_dataframe = pd.DataFrame(
        columns=['tpu_name', 'median_income', 'employment', 'marry', 'education'])
    tpu_2016_social_demographic_dataframe['tpu_name'] = tpu_name_list
    tpu_2016_social_demographic_dataframe['median_income'] = median_income_list
    tpu_2016_social_demographic_dataframe['employment'] = employment_rate
    tpu_2016_social_demographic_dataframe['education'] = education_list
    tpu_2016_social_demographic_dataframe['marry'] = marry_list
    return tpu_2016_social_demographic_dataframe


# def redefine_tpu_name(string):
#     if '-' in string:
#         result_string = re.sub('-', ' - ', string)
#     elif '&' in string:
#         result_string = re.sub('&', 'and', string)
#     elif ('-' in string) and ('&' in string):
#         result_string = re.sub('-', ' - ', re.sub('&', 'and', string))
#     else:
#         result_string = string
#     return result_string


def draw_boxplot(dataframe, column_name, title_name):
    fig, ax = plt.subplots(1,1)
    if column_name == 'Sentiment':
        y_label_name = '% of Positive Tweets - % of Negative Tweets'
    else:
        y_label_name = 'Number of Tweets'
    ax = sns.boxplot(y=column_name, x='tn_or_not',data=dataframe, width=0.5, palette='RdBu')
    ax.set(xlabel='TN or Not', ylabel=y_label_name)
    ax.set_xticks(np.arange(len([0,1])))
    x_tick_list = ['Non-TN TPUs', 'TN TPUs']
    ax.set_xticklabels(x_tick_list, fontsize=7)
    ax.set_title(title_name)
    plot_saving_path = os.path.join(read_data.transit_non_transit_comparison_cross_sectional, 'plots')
    fig.savefig(os.path.join(plot_saving_path, column_name+'.png'))
    plt.show()


def build_data_for_cross_sectional_study(tweet_data_path, saving_path):
    all_tweet_data = pd.read_csv(os.path.join(tweet_data_path, 'tweet_2016_2017_more_tweets.csv'),
                                 encoding='utf-8', dtype='str', quoting=csv.QUOTE_NONNUMERIC)
    tweet_2017 = all_tweet_data.loc[all_tweet_data['year'] == '2017']
    tpu_set = set(tpu_dataframe['SmallTPU'])
    for tpu in tpu_set:
        try:
            os.mkdir(os.path.join(saving_path, tpu))
            dataframe = tweet_2017.loc[tweet_2017['SmallTPU'] == tpu]
            dataframe.to_csv(os.path.join(saving_path, tpu, tpu+'_data.csv'), encoding='utf-8',
                         quoting=csv.QUOTE_NONNUMERIC)
        except WindowsError:
            pass


def draw_correlation_plot(dataframe):
    """
    :param dataframe: a dataframe which saves the data of independent variables in the regression analysis
    :return: the corrrelation table that would be saved in a local directory
    """
    fig, ax = plt.subplots(1,1,figsize=(10,8))
    # Use heatmap embeded in seaborn to draw the correlation matrix
    sns.heatmap(dataframe.corr(method='pearson'), annot=True, fmt='.4f',
                cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
    fig.savefig(os.path.join(
        read_data.transit_non_transit_comparison_cross_sectional, 'independent_correlation.png'))
    plt.show()


def compute_vif(dataframe):
    """
    :param dataframe: a dataframe which saves the data of independent variables in the regression analysis
    :return: a pandas series which records the VIF value for each predictor
    """
    X = add_constant(dataframe)
    result_series = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
    return result_series


if __name__ == '__main__':
    october_23_start = datetime(2016, 10, 23, 0, 0, 0, tzinfo=time_zone_hk)
    october_23_end = datetime(2016, 10, 23, 23, 59, 59, tzinfo=time_zone_hk)
    december_28_start = datetime(2016, 12, 28, 0, 0, 0, tzinfo=time_zone_hk)
    december_28_end = datetime(2016, 12, 28, 23, 59, 59, tzinfo=time_zone_hk)
    start_date = datetime(2016, 5, 7, tzinfo=time_zone_hk)
    end_date = datetime(2017, 12, 31, tzinfo=time_zone_hk)

    demographic_path = os.path.join(read_data.transit_non_transit_comparison_cross_sectional,
                                    'cross_sectional_independent_variables')
    income_employment_rate = pd.read_csv(os.path.join(demographic_path, 'Median Income and Employment Rate.csv'))
    marry_status_dataframe = pd.read_csv(os.path.join(demographic_path, 'Marital Status.csv'))
    education = pd.read_csv(os.path.join(demographic_path, 'Education.csv'))
    tpu_2016_name_list = list(income_employment_rate['Small Tertiary Planning Unit Group'])
    tpu_2016_social_demographic_dataframe = \
        build_social_demographic_dataframe(tpu_name_list=tpu_2016_name_list, economic_dataframe=income_employment_rate,
                                       marry_dataframe=marry_status_dataframe, edu_dataframe=education)
    # tpu_2016_social_demographic_dataframe.to_csv(os.path.join(demographic_path, 'social_demographic_combined.csv'))
    print('The combined social demographic data(median income, marry, employment, education) is......')
    print(tpu_2016_social_demographic_dataframe)
    # tpu_2016_social_demographic_dataframe.to_csv(os.path.join(read_data.desktop, 'tpu_data_social_demo.csv'),
    #                                              encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    print('--------------------------------------------------')

    # Find the tweets in each TPU
    build_data_for_cross_sectional_study(tweet_data_path=read_data.datasets,
                                         saving_path=os.path.join(read_data.transit_non_transit_comparison_cross_sectional, 'tpu_data_more'))

    activity_dict = TransitNeighborhood_TPU.select_tpu_for_following_analysis(check_all_stations=False)
    print('Total number of tpus we consider...')
    print(len(activity_dict.keys()))
    sentiment_dict = {}
    for tpu in activity_dict.keys():
        dataframe = pd.read_csv(os.path.join(read_data.transit_non_transit_comparison_cross_sectional, 'tpu_data_more',
                                             tpu, tpu+'_data.csv'), encoding='utf-8', dtype='str',
                                quoting=csv.QUOTE_NONNUMERIC)
        dataframe['sentiment'] = dataframe['sentiment'].astype(np.int)
        sentiment_dict[tpu] = pos_percent_minus_neg_percent(dataframe)
    #
    sentiment_activity_combined_dict = {}
    for tpu_name in sentiment_dict.keys():
        sentiment_activity_combined_dict[tpu_name] = (sentiment_dict[tpu_name], activity_dict[tpu_name])
    print('\nThe TPUs we consider and their sentiment and activity are...')
    print(sentiment_activity_combined_dict)
    print('-----------------------------------------------\n')

    # the tn_tpus
    # ****check this function****
    whole_tpu_sent_act_dataframe = TransitNeighborhood_TPU.construct_sent_act_dataframe(sent_dict=sentiment_dict,
                                                                                        activity_dict=activity_dict)
    print('--------------------------------------------------------------------')
    print('Total number of tweets we consider in cross sectional study...')
    print(sum(list(whole_tpu_sent_act_dataframe['activity'])))
    print('--------------------------------------------------------------------')
    # whole_tpu_sent_act_dataframe.to_csv(os.path.join(read_data.desktop, 'tpu_sent_act.csv'))
    # y_label = 'Percentage of Positive Tweets Minus Percentage of Negative Tweets'
    # Draw the tpu sentiment against activity
    # figure_title_name = 'Sentiment Against Activity Across TPUs'
    # TransitNeighborhood_TPU.plot_overall_sentiment_for_whole_tweets(df=whole_tpu_sent_act_dataframe,
    #                                                                 y_label_name=y_label,
    #                                                                 figure_title=figure_title_name,
    #                                                                 saved_file_name='tpu_sent_vs_act.png',
    #                                                                 without_outlier=False)
    print('--------------------------Activity---------------------------------')
    for index, dataframe in whole_tpu_sent_act_dataframe.groupby('tn_or_not'):
        print(index)
        print(dataframe['activity'].describe())
    print('-------------------------------------------------------------------')

    print('--------------------------Sentiment---------------------------------')
    for index, dataframe in whole_tpu_sent_act_dataframe.groupby('tn_or_not'):
        print(index)
        print(dataframe['Sentiment'].describe())
    print('-------------------------------------------------------------------')

    print('Building the regressiong model between sentiment/activity and social demographic variables...')
    # Build the regression model between the sentiment and social demographic variables
    # we did not consider the following tpu conflicts(tpu 2011 and tpu 2016 are different)
    not_considered_tpu = ['121 & 122', '123 & 124', '251, 252 & 256', '280 & 286',
                          '294 & 295', '624 & 629', '961 & 962', '963', '610 & 621', '631-633', '310',
                          '321', '515 & 517', '820 & 824', '829', '545 & 546', '543', '525 & 526',
                          '620 & 622', '641']
    # whole_tpu_sent_act_dataframe['tpu_name'] = \
    #     whole_tpu_sent_act_dataframe.apply(lambda row: redefine_tpu_name(row['tpu_name']), axis=1)
    tpu_sent_act_without_conflicts = \
        whole_tpu_sent_act_dataframe.loc[~whole_tpu_sent_act_dataframe['tpu_name'].isin(not_considered_tpu)]
    print("\nThe shape of the sentiment dataframe without conflicts is...")
    print(tpu_sent_act_without_conflicts.shape)
    print('-----------------------------------\n')
    # tpu_sent_act_without_conflicts.to_csv(os.path.join(read_data.desktop, 'tpu_sent_act_without_conflicts.csv'),
    #                                       encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    # fix the typo
    conflict_dict = {'731, 733 & 754': '731, 733 and 754', '288 & 289': '288 - 289',
                     '320, 324 & 329': '320, 324 and 329', '156 & 158': '156 and 158',
                     '832 & 834': '832 and 834', '146 & 147': '146 - 147', '293 & 296': '293 and 296',
                     '826 & 828': '826 and 828', '911-913': '911 - 913', '811-815': '811 - 815',
                     '931 & 933': '931 and 933', '711-712, 721 & 728': '711 - 712, 721 and 728',
                     '421 & 422': '421 - 422', '255 & 269': '255 and 269', '164 & 165': '164 - 165',
                     '971-974': '971 - 974', '175 & 176': '175 - 176', '190, 192 & 194': '190, 192 and 194',
                     '941-943': '941 - 943', '213 & 215-216': '213 and 215 - 216', '950 & 951': '950 - 951',
                     '423 & 428': '423 and 428', '756 & 761-762': '756 and 761 - 762',
                     '722 & 727': '722 and 727', '732, 751 & 753': '732, 751 and 753',
                     '181 & 182': '181 - 182', '741-744': '741 - 744', '193, 195 & 198': '193, 195 and 198'}
    tpu_sent_act_final = tpu_sent_act_without_conflicts.replace({'tpu_name': conflict_dict})
    combined_dataframe = tpu_sent_act_final.merge(tpu_2016_social_demographic_dataframe, on='tpu_name')
    print('The shape of the combined dataframe is : {}'.format(combined_dataframe.shape))
    combined_dataframe.to_csv(os.path.join(read_data.desktop, 'combined.csv'), encoding='utf-8',
                              quoting=csv.QUOTE_NONNUMERIC)
    print('----------------------------------------------------------')
    tn_or_not_dict = {'non_tn_tpu': 0, 'tn_tpu': 1}
    combined_dataframe = combined_dataframe.replace({'tn_or_not': tn_or_not_dict})
    tn_or_not_list = list(combined_dataframe['tn_or_not'])
    tpu_name_list_from_combined_data = list(combined_dataframe['tpu_name'])
    combined_dataframe['employment'] = combined_dataframe.apply(lambda row: row['employment'] / 100, axis=1)
    combined_dataframe.to_csv(os.path.join(demographic_path, 'combined_dataframe.csv'))
    # draw_boxplot(combined_dataframe, column_name='Sentiment', title_name='Sentiment Comparison')
    # draw_boxplot(combined_dataframe, column_name='activity', title_name='Activity Level Comparison')
    combined_dataframe = combined_dataframe[['Sentiment', 'activity', 'median_income', 'employment',
                                             'marry', 'education']]
    print('Social Demographic Data Description...')
    for column_name in ['median_income', 'employment', 'marry', 'education']:
        print('Coping with {}'.format(column_name))
        print(combined_dataframe[column_name].describe())
        print('-------------Done!----------------')

    print('Regression analysis starts..... ')
    normalized_combined_dataframe = (combined_dataframe - combined_dataframe.mean()) / combined_dataframe.std()
    normalized_combined_dataframe['tn_or_not'] = tn_or_not_list
    normalized_combined_dataframe['tpu_name'] = tpu_name_list_from_combined_data
    print(normalized_combined_dataframe.head(5))
    print(normalized_combined_dataframe.columns)
    # Check the correlation matrix of independent variables and compute VIF value for each independent variable
    dataframe_for_correlation_matrix = normalized_combined_dataframe[['median_income', 'employment',
                                                                      'marry', 'education', 'tn_or_not']]
    draw_correlation_plot(dataframe_for_correlation_matrix)
    result_vif_series = compute_vif(dataframe_for_correlation_matrix)
    print(result_vif_series)
    # Regression analysis
    reg_sent = smf.ols('Sentiment ~ median_income+employment+marry+education+tn_or_not',
                       normalized_combined_dataframe).fit()
    print(reg_sent.summary())
    reg_act = smf.ols('activity ~ median_income+employment+marry+education+tn_or_not',
                      normalized_combined_dataframe).fit()
    print(reg_act.summary())






