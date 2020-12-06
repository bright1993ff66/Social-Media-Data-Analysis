# necessary packages
import pandas as pd
import os
import numpy as np
import pytz
from datetime import datetime
import csv

# load my own modules
import before_and_after_final_tpu
import data_paths
import utils
import wordcloud_generate

# packages for regression
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# packages for visualization
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
# Data path which stores the tweet data for each TPU
data_path = os.path.join(data_paths.tweet_combined_path, 'cross_sectional_tpus')
# Load a csv file which saves the names of TPUs
tpu_dataframe = pd.read_csv(os.path.join(data_paths.transit_non_transit_comparison_cross_sectional,
                                         'cross_sectional_independent_variables',
                                         'tpu_names.csv'), encoding='utf-8')


class TransitNeighborhood_TPU(object):

    # Get the TPU name list
    tpu_name_list = list(tpu_dataframe['TPU Names'])
    # Get the TN tpus and non-TN TPUs
    tn_tpus = np.load(os.path.join(data_paths.transit_non_transit_comparison, 'tn_tpus.npy'))
    non_tn_tpus = np.load(os.path.join(data_paths.transit_non_transit_comparison, 'non_tn_tpus.npy'))

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
        :return: a dataframe which saves the activity and sentiment in each month for the tn dataframe
        """
        result_dict_tn = before_and_after_final_tpu.sentiment_by_month(self.tpu_dataframe,
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
        fig.savefig(os.path.join(data_paths.tweet_combined_path, 'cross_sectional_plots', saving_file_name))

    @staticmethod
    def select_tpu_for_following_analysis(check_all_stations=False):
        tpu_activity_dict = {}
        for name in TransitNeighborhood_TPU.tpu_name_list:
            dataframe = pd.read_csv(os.path.join(data_path, name, name + '_data.csv'),
                                    encoding='utf-8', dtype='str', quoting=csv.QUOTE_NONNUMERIC)
            tpu_activity_dict[name] = dataframe.shape[0]
        selected_tpu_dict = {}
        # check_all_stations=False means that we only consider TPUs of which the number of posted tweets is bigger
        # than or equal to 100
        if not check_all_stations:
            for tpu_name in tpu_activity_dict.keys():
                if tpu_activity_dict[tpu_name] >= 200:
                    selected_tpu_dict[tpu_name] = tpu_activity_dict[tpu_name]
                else:
                    pass
        else:
            selected_tpu_dict = tpu_activity_dict
        return selected_tpu_dict

    @staticmethod
    def build_dataframe_yearly(year_number: int):
        """
        create activity dictionary for each TPU for a specific year
        :param year_number: a integer which specifies the year we want to look at
        :return: an activity dict for one specified year
        """
        assert year_number in [2017, 2018]
        tpu_activity_dict_for_one_year = {}
        for name in TransitNeighborhood_TPU.tpu_name_list:
            dataframe = pd.read_csv(os.path.join(data_path, name, name + '_data.csv'),
                                    encoding='utf-8', dtype='str', quoting=csv.QUOTE_NONNUMERIC)
            dataframe_copy = dataframe.copy()
            if year_number == 2017:
                year_dataframe = dataframe_copy.loc[dataframe_copy['year'].isin(['2017.0'])]
            else:
                year_dataframe = dataframe_copy.loc[dataframe_copy['year'].isin(['2018.0'])]
            # Here, we don't consider tpus in which the number of posted tweets is less than 100 in one year
            if year_dataframe.shape[0] >= 100:
                tpu_activity_dict_for_one_year[name] = year_dataframe.shape[0]
            else:
                pass
        return tpu_activity_dict_for_one_year

    @staticmethod
    def build_dataframe_quarterly(quarter_number: int):
        """
        create a activity dictionary for each TPU unit given a specified quarter number
        :param quarter_number: a specified quarter number
        :return: a activity dict of each TPU unit for a specified quarter
        """
        assert quarter_number in [1, 2, 3, 4, 5, 6, 7, 8]
        tpu_activity_dict_for_one_quarter = {}
        for name in TransitNeighborhood_TPU.tpu_name_list:
            dataframe = pd.read_csv(os.path.join(data_path, name, name + '_data.csv'),
                                    encoding='utf-8', dtype='str', quoting=csv.QUOTE_NONNUMERIC)
            dataframe_copy = dataframe.copy()
            dataframe_copy['month_plus_year'] = dataframe_copy.apply(
                lambda row: str(int(float(row['year']))) + '_' + str(int(float(row['month'][:-2]))), axis=1)
            if quarter_number == 1:
                month_plus_year_list = ['2017_1', '2017_2', '2017_3']
            elif quarter_number == 2:
                month_plus_year_list = ['2017_4', '2017_5', '2017_6']
            elif quarter_number == 3:
                month_plus_year_list = ['2017_7', '2017_8', '2017_9']
            elif quarter_number == 4:
                month_plus_year_list = ['2017_10', '2017_11', '2017_12']
            elif quarter_number == 5:
                month_plus_year_list = ['2018_1', '2018_2', '2018_3']
            elif quarter_number == 6:
                month_plus_year_list = ['2018_4', '2018_5', '2018_6']
            elif quarter_number == 7:
                month_plus_year_list = ['2018_7', '2018_8', '2018_9']
            else:
                month_plus_year_list = ['2018_10', '2018_11', '2018_12']
            selected_quarter_dataframe = dataframe_copy.loc[dataframe_copy['month_plus_year'].isin(
                month_plus_year_list)]
            if selected_quarter_dataframe.shape[0] >= 30:
                tpu_activity_dict_for_one_quarter[name] = selected_quarter_dataframe.shape[0]
            else:
                pass
        return tpu_activity_dict_for_one_quarter

    @staticmethod
    def construct_sent_act_dataframe(sent_dict, activity_dict):
        tpu_name_list = list(activity_dict.keys())
        sentiment_list =[]
        acitivity_list =[]
        for name in tpu_name_list:
            sentiment_list.append(sent_dict[name])
            acitivity_list.append(activity_dict[name])
        activity_log10_list = [np.log10(count) if count != 0 else 0 for count in acitivity_list]
        activity_log2_list = [np.log2(count) if count != 0 else 0 for count in acitivity_list]
        result_dataframe = pd.DataFrame({'tpu_name': tpu_name_list, 'Sentiment': sentiment_list,
                                         'activity':acitivity_list, 'Activity_log10': activity_log10_list,
                                         'Activity_log2':activity_log2_list})
        # print(type(list(result_dataframe['tpu_name'])[0]))
        result_dataframe['tn_or_not'] = result_dataframe.apply(
            lambda row: TransitNeighborhood_TPU.check_tn_tpu_or_nontn_tpu(row['tpu_name']), axis=1)
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
    def check_not_considered(dataframe):
        for index, row in dataframe.iterrows():
            if row['activity'] >= 200:
                dataframe.loc[index, 'tn_or_not_considered'] = row['tn_or_not']
            else:
                dataframe.loc[index, 'tn_or_not_considered'] = 'not_considered'
        return dataframe

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
            fig.savefig(os.path.join(data_paths.plot_path_2017, saved_file_name), dpi=fig.dpi, bbox_inches='tight')
            # plt.show()
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
            plot_saving_path = os.path.join(data_paths.tweet_combined_path, 'cross_sectional_plots')
            fig.savefig(os.path.join(plot_saving_path, saved_file_name), dpi=fig.dpi, bbox_inches='tight')
            # plt.show()
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
            fig.savefig(os.path.join(data_paths.plot_path_2017, saved_file_name), dpi=fig.dpi, bbox_inches='tight')
            # plt.show()


# compute the percentage of positive Tweets: 2 is positive
def positive_percent(df):
    positive = 0
    for sentiment in list(df['sentiment']):
        if int(float(sentiment)) == 2:
            positive += 1
        else:
            pass
    try:
        result = positive/df.shape[0]
        return result
    except ZeroDivisionError:
        return 0


# compute the percentage of positive Tweets: 0 is negative
def negative_percent(df):
    negative = 0
    for sentiment in list(df['sentiment']):
        if int(float(sentiment)) == 0:
            negative += 1
        else:
            pass
    try:
        result = negative/df.shape[0]
        return result
    except ZeroDivisionError:
        return 0


# compute positive percentage minus negative percentage: metric used to evaluate the sentiment of an area
# https://www.sciencedirect.com/science/article/pii/S0040162515002024
def pos_percent_minus_neg_percent(df):
    pos_percent = positive_percent(df)
    neg_percent = negative_percent(df)
    return pos_percent - neg_percent


def get_data_for_tpu(tpu_name, economic_dataframe, marry_dataframe, edu_dataframe, population_dataframe):
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
    # average population in each tpu
    average_population = population_dataframe.loc[population_dataframe['TPU'] == tpu_name, 'avg_population'].tolist()[0]
    tpu_area = population_dataframe.loc[population_dataframe['TPU'] == tpu_name, 'ShapeArea'].tolist()[0]
    # print(average_population)
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
    return median_income, employment_rate, marrital_rate, edu_rate, average_population, tpu_area


def build_social_demographic_dataframe(tpu_name_list, economic_dataframe, marry_dataframe, edu_dataframe,
                                       population_dataframe):
    median_income_list = []
    employment_rate = []
    marry_list = []
    education_list = []
    average_population_list = []
    tpu_area_list = []
    for name in tpu_name_list:
        median_income, employ_rate, marry_status, edu, avg_population, tpu_area = get_data_for_tpu(tpu_name=name,
                                                                         economic_dataframe=economic_dataframe,
                                                                         marry_dataframe=marry_dataframe,
                                                                         edu_dataframe=edu_dataframe,
                                                                         population_dataframe=population_dataframe)
        median_income_list.append(median_income)
        employment_rate.append(employ_rate)
        marry_list.append(marry_status)
        education_list.append(edu)
        average_population_list.append(avg_population)
        tpu_area_list.append(tpu_area)

    tpu_2016_social_demographic_dataframe = pd.DataFrame(
        columns=['tpu_name', 'median_income', 'employment', 'marry', 'education', 'avg_population', 'ShapeArea'])
    tpu_2016_social_demographic_dataframe['tpu_name'] = tpu_name_list
    tpu_2016_social_demographic_dataframe['median_income'] = median_income_list
    tpu_2016_social_demographic_dataframe['employment'] = employment_rate
    tpu_2016_social_demographic_dataframe['education'] = education_list
    tpu_2016_social_demographic_dataframe['marry'] = marry_list
    tpu_2016_social_demographic_dataframe['avg_population'] = average_population_list
    tpu_2016_social_demographic_dataframe['ShapeArea'] = tpu_area_list
    return tpu_2016_social_demographic_dataframe


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
    plot_saving_path = os.path.join(data_paths.tweet_combined_path, 'cross_sectional_plots')
    fig.savefig(os.path.join(plot_saving_path, column_name+'.png'))
    plt.show()


def build_data_for_cross_sectional_study(tweet_data_path, saving_path, only_2017_2018=True):
    """
    construct tweet dataframe for each TPU unit
    :param tweet_data_path: path which is used to save all the filtered tweets
    :param saving_path: path which is used to save the tweets posted in each TPU
    :return:
    """
    all_tweet_data = pd.read_csv(os.path.join(tweet_data_path, 'tweet_combined_sentiment_without_bots.csv'),
                                 encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    if only_2017_2018: # Only consider the tweets posted in 2017 and 2018
        assert 2017.0 in list(all_tweet_data['year'])
        assert 2018.0 in list(all_tweet_data['year'])
        tweet_2017_2018 = all_tweet_data.loc[all_tweet_data['year'].isin([2017.0, 2018.0])]
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
        data_paths.tweet_combined_path, 'cross_sectional_plots', 'independent_correlation.png'))
    plt.show()


def compute_vif(dataframe):
    """
    :param dataframe: a dataframe which saves the data of independent variables in the regression analysis
    :return: a pandas series which records the VIF value for each predictor
    """
    X = add_constant(dataframe)
    result_series = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
    return result_series


def describle_dataframe(dataframe, message):
    """
    describle a summarized dataframe for each quarter
    :param dataframe: a summarized dataframe
    :param message: Any message you want to output
    :return: A descriptive statement about the activity and number of involved TPUs
    """
    number_of_tweets = sum(dataframe['activity'])
    number_of_tpus = dataframe.shape[0]
    print("{}, number of tweets: {}; number of involved TPUs: {}".format(message, number_of_tweets, number_of_tpus))


def regres_analysis(sent_act_dataframe, social_demographic_merged_filenanme):
    """
    Build a regression analysis model based on the tweet dataframe and social demographic data
    :param sent_act_dataframe: the tweet considered dataframe
    :param social_demographic_merged_filenanme: the filename that used to save the sentiment, activity and
    social demographic variables for each TPU
    :return: the regression analysis for sentiment and activity respectively
    """
    print('--------------------------Activity---------------------------------')
    for index, dataframe in sent_act_dataframe.groupby('tn_or_not'):
        print(index)
        print(dataframe['activity'].describe())
    print('-------------------------------------------------------------------')

    print('--------------------------Sentiment---------------------------------')
    for index, dataframe in sent_act_dataframe.groupby('tn_or_not'):
        print(index)
        print(dataframe['Sentiment'].describe())
    print('-------------------------------------------------------------------')
    #
    print('Building the regressiong model between sentiment/activity and social demographic variables...')
    combined_dataframe = sent_act_dataframe.merge(tpu_2016_social_demographic_dataframe, on='tpu_name')
    print('The shape of the combined dataframe is : {}'.format(combined_dataframe.shape))
    print('----------------------------------------------------------')
    tn_or_not_dict = {'non_tn_tpu': 0, 'tn_tpu': 1}
    combined_dataframe = combined_dataframe.replace({'tn_or_not': tn_or_not_dict})
    tn_or_not_list = list(combined_dataframe['tn_or_not'])
    tpu_name_list_from_combined_data = list(combined_dataframe['tpu_name'])
    combined_dataframe['employment'] = combined_dataframe.apply(lambda row: row['employment'] / 100, axis=1)
    draw_boxplot(combined_dataframe, column_name='Sentiment', title_name='Sentiment Comparison')
    draw_boxplot(combined_dataframe, column_name='Activity_log10', title_name='Activity Level Comparison')
    combined_dataframe = combined_dataframe[['Sentiment', 'activity', 'median_income', 'employment',
                                             'marry', 'education', 'avg_population', 'ShapeArea']]
    print('Social Demographic Data Description...')
    for column_name in ['median_income', 'employment', 'marry', 'education', 'avg_population', 'ShapeArea']:
        print('Coping with {}'.format(column_name))
        print(combined_dataframe[column_name].describe())
        print('-------------Done!----------------')

    print('Check the correlation between sentiment and activity...')
    correlation_value_sent_act = combined_dataframe['Sentiment'].corr(combined_dataframe['activity'])
    print('The correlation coefficient of sentiment and activity is :{}'.format(correlation_value_sent_act))

    print('Check the correlation between activity and avg_population per square meter...')
    # First, compute the activity level per square meter
    combined_dataframe['avg_activity'] = combined_dataframe.apply(lambda row: row['activity'] / row['ShapeArea'],
                                                                  axis=1)
    print(combined_dataframe.head())
    # Then compute the correlation coefficient...
    correlation_value_act_population = combined_dataframe['avg_activity'].corr(combined_dataframe['avg_population'])
    print('The correlation coefficient of activity and avg population is :{}'.format(correlation_value_act_population))

    print('Regression analysis starts..... ')
    normalized_combined_dataframe = (combined_dataframe - combined_dataframe.mean()) / combined_dataframe.std()
    normalized_combined_dataframe['tn_or_not'] = tn_or_not_list
    normalized_combined_dataframe['tpu_name'] = tpu_name_list_from_combined_data
    # Check the correlation matrix of independent variables and compute VIF value for each independent variable
    combined_dataframe['tn_or_not'] = tn_or_not_list
    dataframe_for_correlation_matrix = combined_dataframe[['median_income', 'employment', 'marry', 'education',
                                                           'tn_or_not', 'avg_population', 'avg_activity']]
    draw_correlation_plot(dataframe_for_correlation_matrix)
    result_vif_series = compute_vif(dataframe_for_correlation_matrix)
    print(result_vif_series)
    # Regression analysis
    reg_sent = smf.ols('Sentiment ~ median_income+employment+marry+education+avg_population+tn_or_not',
                       normalized_combined_dataframe).fit()
    print(reg_sent.summary())
    reg_act = smf.ols('avg_activity ~ median_income+employment+marry+education+avg_population+tn_or_not',
                      normalized_combined_dataframe).fit()
    print(reg_act.summary())


if __name__ == '__main__':
    # Specify some important dates
    october_23_start = datetime(2016, 10, 23, 0, 0, 0, tzinfo=time_zone_hk)
    october_23_end = datetime(2016, 10, 23, 23, 59, 59, tzinfo=time_zone_hk)
    december_28_start = datetime(2016, 12, 28, 0, 0, 0, tzinfo=time_zone_hk)
    december_28_end = datetime(2016, 12, 28, 23, 59, 59, tzinfo=time_zone_hk)

    # Build the dataframe for the social demographic variables for each TPU
    demographic_path = os.path.join(data_paths.transit_non_transit_comparison_cross_sectional,
                                    'cross_sectional_independent_variables')
    income_employment_rate = pd.read_csv(os.path.join(demographic_path, 'Median Income and Employment Rate.csv'))
    marry_status_dataframe = pd.read_csv(os.path.join(demographic_path, 'Marital Status.csv'))
    education = pd.read_csv(os.path.join(demographic_path, 'Education.csv'))
    avg_population_dataframe = pd.read_csv(os.path.join(demographic_path, 'avg_population_in_tpu.csv'))
    # print(avg_population_dataframe.head())
    tpu_2016_name_list = list(income_employment_rate['Small Tertiary Planning Unit Group'])
    tpu_2016_name_list.remove('Land')
    tpu_2016_social_demographic_dataframe = \
        build_social_demographic_dataframe(tpu_name_list=tpu_2016_name_list, economic_dataframe=income_employment_rate,
                                       marry_dataframe=marry_status_dataframe, edu_dataframe=education,
                                           population_dataframe=avg_population_dataframe)
    tpu_2016_social_demographic_dataframe.to_csv(os.path.join(demographic_path, 'social_demographic_combined.csv'))
    print('The combined social demographic data(median income, marry, employment, education) is......')
    print(tpu_2016_social_demographic_dataframe)

    # Create TPU-shape area dictionary. The area is saved in square meter
    tpu_area_dict = pd.Series(avg_population_dataframe.ShapeArea.values, index=avg_population_dataframe.TPU).to_dict()
    print(tpu_area_dict)
    print('--------------------------------------------------')

    print('-----------------------Deal with the tweet 2017 & tweet 2018 together--------------------------')
    # Find the tweets in each TPU and save them to a local directory
    build_data_for_cross_sectional_study(tweet_data_path=data_paths.tweet_combined_path,
                                         saving_path=os.path.join(data_paths.tweet_combined_path,
                                                                  'cross_sectional_tpus'))
    # We have built folder for each TPU to store tweets
    # Based on the created folders, select tpus which have at least 100 tweets in 2017
    activity_dict = TransitNeighborhood_TPU.select_tpu_for_following_analysis(check_all_stations=False)
    activity_dict_whole = TransitNeighborhood_TPU.select_tpu_for_following_analysis(check_all_stations=True)
    print('Total number of tpus we consider...')
    print(len(activity_dict.keys()))
    print('Total number of tweets we consider...')
    print(sum(activity_dict.values()))
    sentiment_dict = {}
    for tpu in activity_dict.keys():
        dataframe = pd.read_csv(os.path.join(data_path, tpu, tpu+'_data.csv'), encoding='utf-8', dtype='str',
                                quoting=csv.QUOTE_NONNUMERIC)
        # dataframe['sentiment'] = dataframe['sentiment'].astype(np.int)
        sentiment_dict[tpu] = pos_percent_minus_neg_percent(dataframe)

    sentiment_dict_whole = {}
    for tpu in activity_dict_whole.keys():
        dataframe = pd.read_csv(os.path.join(data_path, tpu, tpu + '_data.csv'), encoding='utf-8', dtype='str',
                                quoting=csv.QUOTE_NONNUMERIC)
        # dataframe['sentiment'] = dataframe['sentiment'].astype(np.int)
        sentiment_dict_whole[tpu] = pos_percent_minus_neg_percent(dataframe)

    # the tn_tpus
    whole_tpu_sent_act_dataframe = TransitNeighborhood_TPU.construct_sent_act_dataframe(sent_dict=sentiment_dict,
                                                                                        activity_dict=activity_dict)
    whole_tpu_sent_act_dataframe_all_considered = TransitNeighborhood_TPU.construct_sent_act_dataframe(
        sent_dict=sentiment_dict_whole, activity_dict=activity_dict_whole)

    whole_tpu_sent_act_dataframe.to_csv(os.path.join(data_paths.desktop, 'tpu_sent_act.csv'))
    # Check which tpu should not be considered
    whole_tpu_sent_act_dataframe_tn_or_not_considered = \
        TransitNeighborhood_TPU.check_not_considered(whole_tpu_sent_act_dataframe_all_considered)
    # Create the tpu name to match the shapefile
    whole_tpu_sent_act_dataframe_tn_or_not_considered['tpu_old_name'] = \
        whole_tpu_sent_act_dataframe_tn_or_not_considered.apply(
            lambda row: utils.tpu_name_match_reverse[row['tpu_name']]
            if row['tpu_name'] in  utils.tpu_name_match_reverse else row['tpu_name'], axis=1)
    whole_tpu_sent_act_dataframe_tn_or_not_considered.to_csv(os.path.join(data_paths.desktop,
                                                                    'tpu_sent_act_all_considered.csv'))
    y_label = 'Percentage of Positive Tweets Minus Percentage of Negative Tweets'
    # Draw the tpu sentiment against activity
    figure_title_name = 'Sentiment Against Activity Across TPUs'
    TransitNeighborhood_TPU.plot_overall_sentiment_for_whole_tweets(df=whole_tpu_sent_act_dataframe,
                                                                    y_label_name=y_label,
                                                                    figure_title=figure_title_name,
                                                                    saved_file_name='tpu_sent_vs_act.png',
                                                                    without_outlier=False)
    print('------------------------------------Done!---------------------------------------------')

    print('-----------------------Deal with the tweet 2017 & tweet 2018 respectively--------------------------')
    activity_dict_2017 = TransitNeighborhood_TPU.build_dataframe_yearly(year_number=2017)
    activity_dict_2018 = TransitNeighborhood_TPU.build_dataframe_yearly(year_number=2018)

    sentiment_dict_2017 = {}
    sentiment_dict_2018 = {}

    activity_list_yearly = [activity_dict_2017, activity_dict_2018]
    sentiment_list_yearly = [sentiment_dict_2017, sentiment_dict_2018]

    for activity_dict, sentiment_dict in zip(activity_list_yearly, sentiment_list_yearly):
        index_value = activity_list_yearly.index(activity_dict)
        for tpu in list(activity_dict.keys()):
            dataframe = pd.read_csv(
                os.path.join(data_path, tpu, tpu + '_data.csv'), encoding='utf-8', dtype='str',
                quoting=csv.QUOTE_NONNUMERIC)
            if index_value == 0:
                year_dataframe = dataframe.loc[dataframe['year'].isin(['2017.0'])]
            else:
                year_dataframe = dataframe.loc[dataframe['year'].isin(['2018.0'])]
            sentiment_dict[tpu] = pos_percent_minus_neg_percent(year_dataframe)

    whole_sent_act_year_2017 = TransitNeighborhood_TPU.construct_sent_act_dataframe(sent_dict=sentiment_dict_2017,
                                                                                    activity_dict=activity_dict_2017)
    whole_sent_act_year_2018 = TransitNeighborhood_TPU.construct_sent_act_dataframe(sent_dict=sentiment_dict_2018,
                                                                                    activity_dict=activity_dict_2018)

    print('The general information of the yearly based dataframe is...')
    describle_dataframe(whole_sent_act_year_2017, message='In 2017')
    describle_dataframe(whole_sent_act_year_2018, message='In 2018')

    print('--------------------------------------------------------------------')
    print('For instance, total number of tweets we consider in 2017 of the cross sectional study...')
    print(sum(list(whole_sent_act_year_2017['activity'])))
    print('--------------------------------------------------------------------')

    whole_sent_act_year_2017.to_csv(os.path.join(data_paths.desktop, 'tpu_sent_act_year_2017.csv'))
    whole_sent_act_year_2018.to_csv(os.path.join(data_paths.desktop, 'tpu_sent_act_year_2018.csv'))
    y_label = 'Percentage of Positive Tweets Minus Percentage of Negative Tweets'
    # Draw the tpu sentiment against activity
    figure_title_name_year_2017 = 'Sentiment Against Activity Across TPUs(Year 2017)'
    figure_title_name_year_2018 = 'Sentiment Against Activity Across TPUs(Year 2018)'

    TransitNeighborhood_TPU.plot_overall_sentiment_for_whole_tweets(df=whole_sent_act_year_2017,
                                                                    y_label_name=y_label,
                                                                    figure_title=figure_title_name_year_2017,
                                                                    saved_file_name='tpu_sent_vs_act_year_2017.png',
                                                                    without_outlier=False)
    TransitNeighborhood_TPU.plot_overall_sentiment_for_whole_tweets(df=whole_sent_act_year_2018,
                                                                    y_label_name=y_label,
                                                                    figure_title=figure_title_name_year_2018,
                                                                    saved_file_name='tpu_sent_vs_act_year_2018.png',
                                                                    without_outlier=False)
    print('------------------------------------Done!---------------------------------------------')

    # Analyze the result quarterly
    print('----------------Deal with each quarter of tweet 2017 & tweet 2018 respectively---------------------')
    print('-------------------------------------------------------------------')
    activity_dict_quarter_1 = TransitNeighborhood_TPU.build_dataframe_quarterly(quarter_number=1)
    activity_dict_quarter_2 = TransitNeighborhood_TPU.build_dataframe_quarterly(quarter_number=2)
    activity_dict_quarter_3 = TransitNeighborhood_TPU.build_dataframe_quarterly(quarter_number=3)
    activity_dict_quarter_4 = TransitNeighborhood_TPU.build_dataframe_quarterly(quarter_number=4)
    activity_dict_quarter_5 = TransitNeighborhood_TPU.build_dataframe_quarterly(quarter_number=5)
    activity_dict_quarter_6 = TransitNeighborhood_TPU.build_dataframe_quarterly(quarter_number=6)
    activity_dict_quarter_7 = TransitNeighborhood_TPU.build_dataframe_quarterly(quarter_number=7)
    activity_dict_quarter_8 = TransitNeighborhood_TPU.build_dataframe_quarterly(quarter_number=8)

    sentiment_dict_quarter_1 = {}
    sentiment_dict_quarter_2 = {}
    sentiment_dict_quarter_3 = {}
    sentiment_dict_quarter_4 = {}
    sentiment_dict_quarter_5 = {}
    sentiment_dict_quarter_6 = {}
    sentiment_dict_quarter_7 = {}
    sentiment_dict_quarter_8 = {}

    activity_dict_list = [activity_dict_quarter_1, activity_dict_quarter_2, activity_dict_quarter_3,
                          activity_dict_quarter_4, activity_dict_quarter_5, activity_dict_quarter_6,
                          activity_dict_quarter_7, activity_dict_quarter_8]
    sentiment_dict_list = [sentiment_dict_quarter_1, sentiment_dict_quarter_2, sentiment_dict_quarter_3,
                           sentiment_dict_quarter_4, sentiment_dict_quarter_5, sentiment_dict_quarter_6,
                           sentiment_dict_quarter_7, sentiment_dict_quarter_8]
    index_value_list = list(range(1, 9))

    for index_value, activity_dict, sentiment_dict in zip(index_value_list, activity_dict_list, sentiment_dict_list):
        print('Coping with the {} sent activity pair'.format(index_value))
        for tpu in list(activity_dict.keys()):
            dataframe = pd.read_csv(os.path.join(data_path, tpu, tpu + '_data.csv'), encoding='utf-8', dtype='str',
                                    quoting=csv.QUOTE_NONNUMERIC)
            if index_value == 0:
                month_plus_year_list = ['2017_1', '2017_2', '2017_3']
            elif index_value == 1:
                month_plus_year_list = ['2017_4', '2017_5', '2017_6']
            elif index_value == 2:
                month_plus_year_list = ['2017_7', '2017_8', '2017_9']
            elif index_value == 3:
                month_plus_year_list = ['2017_10', '2017_11', '2017_12']
            elif index_value == 4:
                month_plus_year_list = ['2018_1', '2018_2', '2018_3']
            elif index_value == 5:
                month_plus_year_list = ['2018_4', '2018_5', '2018_6']
            elif index_value == 6:
                month_plus_year_list = ['2018_7', '2018_8', '2018_9']
            else:
                month_plus_year_list = ['2018_10', '2018_11', '2018_12']
            # dataframe['sentiment'] = dataframe['sentiment'].astype(np.int)
            quarter_dataframe = dataframe.loc[dataframe['month_plus_year'].isin(month_plus_year_list)]
            sentiment_dict[tpu] = pos_percent_minus_neg_percent(quarter_dataframe)

    whole_sent_act_quarter_1 = TransitNeighborhood_TPU.construct_sent_act_dataframe(sent_dict=sentiment_dict_quarter_1,
                                                                                    activity_dict=activity_dict_quarter_1)
    whole_sent_act_quarter_2 = TransitNeighborhood_TPU.construct_sent_act_dataframe(sent_dict=sentiment_dict_quarter_2,
                                                                                    activity_dict=activity_dict_quarter_2)
    whole_sent_act_quarter_3 = TransitNeighborhood_TPU.construct_sent_act_dataframe(sent_dict=sentiment_dict_quarter_3,
                                                                                    activity_dict=activity_dict_quarter_3)
    whole_sent_act_quarter_4 = TransitNeighborhood_TPU.construct_sent_act_dataframe(sent_dict=sentiment_dict_quarter_4,
                                                                                    activity_dict=activity_dict_quarter_4)
    whole_sent_act_quarter_5 = TransitNeighborhood_TPU.construct_sent_act_dataframe(sent_dict=sentiment_dict_quarter_5,
                                                                                    activity_dict=activity_dict_quarter_5)
    whole_sent_act_quarter_6 = TransitNeighborhood_TPU.construct_sent_act_dataframe(sent_dict=sentiment_dict_quarter_6,
                                                                                    activity_dict=activity_dict_quarter_6)
    whole_sent_act_quarter_7 = TransitNeighborhood_TPU.construct_sent_act_dataframe(sent_dict=sentiment_dict_quarter_7,
                                                                                    activity_dict=activity_dict_quarter_7)
    whole_sent_act_quarter_8 = TransitNeighborhood_TPU.construct_sent_act_dataframe(sent_dict=sentiment_dict_quarter_8,
                                                                                    activity_dict=activity_dict_quarter_8)

    print('The general information of the quarterly based dataframe is...')
    describle_dataframe(whole_sent_act_quarter_1, message='In 2017, Quarter 1')
    describle_dataframe(whole_sent_act_quarter_2, message='In 2017, Quarter 2')
    describle_dataframe(whole_sent_act_quarter_3, message='In 2017, Quarter 3')
    describle_dataframe(whole_sent_act_quarter_4, message='In 2017, Quarter 4')
    describle_dataframe(whole_sent_act_quarter_5, message='In 2018, Quarter 1')
    describle_dataframe(whole_sent_act_quarter_6, message='In 2018, Quarter 2')
    describle_dataframe(whole_sent_act_quarter_7, message='In 2018, Quarter 3')
    describle_dataframe(whole_sent_act_quarter_8, message='In 2018, Quarter 4')

    print('--------------------------------------------------------------------')
    print('For instance, total number of tweets we consider in the second quarter of 2018 in the cross sectional study...')
    print(sum(list(whole_sent_act_quarter_6['activity'])))
    print('--------------------------------------------------------------------')
    whole_sent_act_quarter_1.to_csv(os.path.join(data_paths.desktop, '2017_tpu_sent_act_quarter_1.csv'))
    whole_sent_act_quarter_2.to_csv(os.path.join(data_paths.desktop, '2017_tpu_sent_act_quarter_2.csv'))
    whole_sent_act_quarter_3.to_csv(os.path.join(data_paths.desktop, '2017_tpu_sent_act_quarter_3.csv'))
    whole_sent_act_quarter_4.to_csv(os.path.join(data_paths.desktop, '2017_tpu_sent_act_quarter_4.csv'))
    whole_sent_act_quarter_5.to_csv(os.path.join(data_paths.desktop, '2018_tpu_sent_act_quarter_1.csv'))
    whole_sent_act_quarter_6.to_csv(os.path.join(data_paths.desktop, '2018_tpu_sent_act_quarter_2.csv'))
    whole_sent_act_quarter_7.to_csv(os.path.join(data_paths.desktop, '2018_tpu_sent_act_quarter_3.csv'))
    whole_sent_act_quarter_8.to_csv(os.path.join(data_paths.desktop, '2018_tpu_sent_act_quarter_4.csv'))
    y_label = 'Percentage of Positive Tweets Minus Percentage of Negative Tweets'
    # Draw the tpu sentiment against activity
    figure_title_name_quarter1 = 'Sentiment Against Activity Across TPUs(2017 Quarter 1)'
    figure_title_name_quarter2 = 'Sentiment Against Activity Across TPUs(2017 Quarter 2)'
    figure_title_name_quarter3 = 'Sentiment Against Activity Across TPUs(2017 Quarter 3)'
    figure_title_name_quarter4 = 'Sentiment Against Activity Across TPUs(2017 Quarter 4)'
    figure_title_name_quarter5 = 'Sentiment Against Activity Across TPUs(2018 Quarter 1)'
    figure_title_name_quarter6 = 'Sentiment Against Activity Across TPUs(2018 Quarter 2)'
    figure_title_name_quarter7 = 'Sentiment Against Activity Across TPUs(2018 Quarter 3)'
    figure_title_name_quarter8 = 'Sentiment Against Activity Across TPUs(2018 Quarter 4)'

    TransitNeighborhood_TPU.plot_overall_sentiment_for_whole_tweets(df=whole_sent_act_quarter_1,
                                                                    y_label_name=y_label,
                                                                    figure_title=figure_title_name_quarter1,
                                                                    saved_file_name='2017_tpu_sent_vs_act_quarter_1.png',
                                                                    without_outlier=False)
    TransitNeighborhood_TPU.plot_overall_sentiment_for_whole_tweets(df=whole_sent_act_quarter_2,
                                                                    y_label_name=y_label,
                                                                    figure_title=figure_title_name_quarter2,
                                                                    saved_file_name='2017_tpu_sent_vs_act_quarter_2.png',
                                                                    without_outlier=False)
    TransitNeighborhood_TPU.plot_overall_sentiment_for_whole_tweets(df=whole_sent_act_quarter_3,
                                                                    y_label_name=y_label,
                                                                    figure_title=figure_title_name_quarter3,
                                                                    saved_file_name='2017_tpu_sent_vs_act_quarter_3.png',
                                                                    without_outlier=False)
    TransitNeighborhood_TPU.plot_overall_sentiment_for_whole_tweets(df=whole_sent_act_quarter_4,
                                                                    y_label_name=y_label,
                                                                    figure_title=figure_title_name_quarter4,
                                                                    saved_file_name='2017_tpu_sent_vs_act_quarter_4.png',
                                                                    without_outlier=False)
    TransitNeighborhood_TPU.plot_overall_sentiment_for_whole_tweets(df=whole_sent_act_quarter_5,
                                                                    y_label_name=y_label,
                                                                    figure_title=figure_title_name_quarter5,
                                                                    saved_file_name='2018_tpu_sent_vs_act_quarter_1.png',
                                                                    without_outlier=False)
    TransitNeighborhood_TPU.plot_overall_sentiment_for_whole_tweets(df=whole_sent_act_quarter_6,
                                                                    y_label_name=y_label,
                                                                    figure_title=figure_title_name_quarter6,
                                                                    saved_file_name='2018_tpu_sent_vs_act_quarter_2.png',
                                                                    without_outlier=False)
    TransitNeighborhood_TPU.plot_overall_sentiment_for_whole_tweets(df=whole_sent_act_quarter_7,
                                                                    y_label_name=y_label,
                                                                    figure_title=figure_title_name_quarter7,
                                                                    saved_file_name='2018_tpu_sent_vs_act_quarter_3.png',
                                                                    without_outlier=False)
    TransitNeighborhood_TPU.plot_overall_sentiment_for_whole_tweets(df=whole_sent_act_quarter_8,
                                                                    y_label_name=y_label,
                                                                    figure_title=figure_title_name_quarter8,
                                                                    saved_file_name='2018_tpu_sent_vs_act_quarter_4.png',
                                                                    without_outlier=False)
    print('------------------------------------Done!---------------------------------------------')
    # regression analysis overall
    print('For 2017 and 2018...')
    regres_analysis(sent_act_dataframe=whole_tpu_sent_act_dataframe,
                    social_demographic_merged_filenanme='combined.csv')
    print('For 2017...')
    # regression analysis in 2017
    regres_analysis(sent_act_dataframe=whole_sent_act_year_2017,
                    social_demographic_merged_filenanme='combined_2017.csv')
    print('For 2018...')
    # regression analysis in 2018
    regres_analysis(sent_act_dataframe=whole_sent_act_year_2018,
                    social_demographic_merged_filenanme='combined_2018.csv')




