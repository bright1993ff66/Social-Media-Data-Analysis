# necessary packages
import pandas as pd
import os
import numpy as np
import pytz
from datetime import datetime
import csv

# load my own modules
import before_and_after_final_tpu
import read_data
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
data_path = os.path.join(read_data.transit_non_transit_comparison_cross_sectional, 'tpu_data_more')
# Load a csv file which saves the names of TPUs
tpu_dataframe = pd.read_csv(os.path.join(read_data.transit_non_transit_comparison_cross_sectional,
                                         'cross_sectional_independent_variables',
                                         'tpu_names.csv'), encoding='utf-8')


class TransitNeighborhood_TPU(object):

    # Get the TPU name list
    tpu_name_list = list(tpu_dataframe['TPU Names'])
    # Get the TN tpus and non-TN TPUs
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
        # than or equal to 100
        if not check_all_stations:
            for tpu_name in tpu_activity_dict.keys():
                if tpu_activity_dict[tpu_name] >= 100:
                    selected_tpu_dict[tpu_name] = tpu_activity_dict[tpu_name]
                else:
                    pass
        else:
            selected_tpu_dict = tpu_activity_dict
        return selected_tpu_dict

    @staticmethod
    def build_dataframe_quarterly(quarter_number:int):
        """
        create a activity dictionary for each TPU unit given a specified quarter number
        :param quarter_number: a specified quarter number
        :return: a activity dict of each TPU unit for a specified quarter
        """
        assert quarter_number in [1, 2, 3, 4]
        tpu_activity_dict_for_one_quarter = {}
        for name in TransitNeighborhood_TPU.tpu_name_list:
            dataframe = pd.read_csv(os.path.join(data_path, name, name + '_data.csv'),
                                    encoding='utf-8', dtype='str', quoting=csv.QUOTE_NONNUMERIC)
            dataframe_copy = dataframe.copy()
            if quarter_number == 1:
                quarter_dataframe = dataframe_copy.loc[dataframe_copy['month'].isin(['1.0', '2.0', '3.0'])]
            elif quarter_number == 2:
                quarter_dataframe = dataframe_copy.loc[dataframe_copy['month'].isin(['4.0', '5.0', '6.0'])]
            elif quarter_number == 3:
                quarter_dataframe = dataframe_copy.loc[dataframe_copy['month'].isin(['7.0', '8.0', '9.0'])]
            else:
                quarter_dataframe = dataframe_copy.loc[dataframe_copy['month'].isin(['10.0', '11.0', '12.0'])]
            # print('For TPU {}, the number of tweets posted in this quarter is: {}'.format(name,
            #                                                                               quarter_dataframe.shape[0]))
            if quarter_dataframe.shape[0] >= 30:
                tpu_activity_dict_for_one_quarter[name] = quarter_dataframe.shape[0]
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
        activity_log10_list = [np.log10(count) if count !=0 else 0 for count in acitivity_list]
        activity_log2_list = [np.log2(count) if count !=0 else 0 for count in acitivity_list]
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
    for sentiment in list(df['sentiment']):
        if int(float(sentiment))==2:
            positive+=1
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
        if int(float(sentiment))==0:
            negative+=1
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
    return median_income, employment_rate, marrital_rate, edu_rate, average_population


def build_social_demographic_dataframe(tpu_name_list, economic_dataframe, marry_dataframe, edu_dataframe,
                                       population_dataframe):
    median_income_list = []
    employment_rate = []
    marry_list = []
    education_list = []
    average_population_list = []
    for name in tpu_name_list:
        median_income, employ_rate, marry_status, edu, avg_population = get_data_for_tpu(tpu_name=name,
                                                                         economic_dataframe=economic_dataframe,
                                                                         marry_dataframe=marry_dataframe,
                                                                         edu_dataframe=edu_dataframe,
                                                                         population_dataframe=population_dataframe)
        median_income_list.append(median_income)
        employment_rate.append(employ_rate)
        marry_list.append(marry_status)
        education_list.append(edu)
        average_population_list.append(avg_population)

    tpu_2016_social_demographic_dataframe = pd.DataFrame(
        columns=['tpu_name', 'median_income', 'employment', 'marry', 'education', 'avg_population'])
    tpu_2016_social_demographic_dataframe['tpu_name'] = tpu_name_list
    tpu_2016_social_demographic_dataframe['median_income'] = median_income_list
    tpu_2016_social_demographic_dataframe['employment'] = employment_rate
    tpu_2016_social_demographic_dataframe['education'] = education_list
    tpu_2016_social_demographic_dataframe['marry'] = marry_list
    tpu_2016_social_demographic_dataframe['avg_population'] = average_population_list
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
    plot_saving_path = os.path.join(read_data.transit_non_transit_comparison_cross_sectional, 'plots')
    fig.savefig(os.path.join(plot_saving_path, column_name+'.png'))
    plt.show()


def build_data_for_cross_sectional_study(tweet_data_path, saving_path, only_2017=True):
    """
    :param tweet_data_path: path which is used to save all the filtered tweets
    :param saving_path: path which is used to save the tweets posted in each TPU
    :return:
    """
    all_tweet_data = pd.read_csv(os.path.join(tweet_data_path, 'tweet_2017_cross_sectional.csv'),
                                 encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    if only_2017:
        # Only consider tweets posted in 2017
        tweet_2017 = all_tweet_data.loc[all_tweet_data['year'] == 2017.0]
        # Change the name of the column
        tpu_set = set(tpu_dataframe['TPU Names'])
        for tpu in tpu_set:
            try:
                os.mkdir(os.path.join(saving_path, tpu))
                # Use the TPU_cross_sectional column
                dataframe = tweet_2017.loc[tweet_2017['TPU_cross_sectional'] == tpu]
                dataframe.to_csv(os.path.join(saving_path, tpu, tpu+'_data.csv'), encoding='utf-8',
                             quoting=csv.QUOTE_NONNUMERIC)
            except WindowsError:
                pass
    else:
        tpu_set = set(tpu_dataframe['TPU Names'])
        for tpu in tpu_set:
            try:
                os.mkdir(os.path.join(saving_path, tpu))
                dataframe = all_tweet_data.loc[all_tweet_data['TPU_cross_sectional'] == tpu]
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


if __name__ == '__main__':
    # Specify some important dates
    october_23_start = datetime(2016, 10, 23, 0, 0, 0, tzinfo=time_zone_hk)
    october_23_end = datetime(2016, 10, 23, 23, 59, 59, tzinfo=time_zone_hk)
    december_28_start = datetime(2016, 12, 28, 0, 0, 0, tzinfo=time_zone_hk)
    december_28_end = datetime(2016, 12, 28, 23, 59, 59, tzinfo=time_zone_hk)
    start_date = datetime(2016, 5, 7, tzinfo=time_zone_hk)
    end_date = datetime(2017, 12, 31, tzinfo=time_zone_hk)

    # Build the dataframe for the social demographic variables for each TPU
    demographic_path = os.path.join(read_data.transit_non_transit_comparison_cross_sectional,
                                    'cross_sectional_independent_variables')
    income_employment_rate = pd.read_csv(os.path.join(demographic_path, 'Median Income and Employment Rate.csv'))
    marry_status_dataframe = pd.read_csv(os.path.join(demographic_path, 'Marital Status.csv'))
    education = pd.read_csv(os.path.join(demographic_path, 'Education.csv'))
    avg_population_dataframe = pd.read_csv(os.path.join(demographic_path, 'avg_population_in_tpu.csv'))
    tpu_2016_name_list = list(income_employment_rate['Small Tertiary Planning Unit Group'])
    tpu_2016_name_list.remove('Land')
    tpu_2016_social_demographic_dataframe = \
        build_social_demographic_dataframe(tpu_name_list=tpu_2016_name_list, economic_dataframe=income_employment_rate,
                                       marry_dataframe=marry_status_dataframe, edu_dataframe=education,
                                           population_dataframe=avg_population_dataframe)
    tpu_2016_social_demographic_dataframe.to_csv(os.path.join(demographic_path, 'social_demographic_combined.csv'))
    print('The combined social demographic data(median income, marry, employment, education) is......')
    print(tpu_2016_social_demographic_dataframe)
    print('--------------------------------------------------')

    # Find the tweets in each TPU
    build_data_for_cross_sectional_study(tweet_data_path=read_data.transit_non_transit_comparison_cross_sectional,
                                         saving_path=os.path.join(
                                             read_data.transit_non_transit_comparison_cross_sectional, 'tpu_data_more'))
    # We have built folder for each TPU to store tweets
    # Based on the created folders, select tpus which have at least 100 tweets in 2017
    activity_dict = TransitNeighborhood_TPU.select_tpu_for_following_analysis(check_all_stations=False)
    print('Total number of tpus we consider...')
    print(len(activity_dict.keys()))
    print('Total number of tweets we consider...')
    print(sum(activity_dict.values()))
    # print(activity_dict)
    sentiment_dict = {}
    for tpu in activity_dict.keys():
        dataframe = pd.read_csv(os.path.join(read_data.transit_non_transit_comparison_cross_sectional, 'tpu_data_more',
                                             tpu, tpu+'_data.csv'), encoding='utf-8', dtype='str',
                                quoting=csv.QUOTE_NONNUMERIC)
        # dataframe['sentiment'] = dataframe['sentiment'].astype(np.int)
        sentiment_dict[tpu] = pos_percent_minus_neg_percent(dataframe)

    sentiment_activity_combined_dict = {}
    for tpu_name in sentiment_dict.keys():
        sentiment_activity_combined_dict[tpu_name] = (sentiment_dict[tpu_name], activity_dict[tpu_name])
    print('\nThe TPUs we consider and their sentiment and activity are...')
    print(sentiment_activity_combined_dict)
    print('-----------------------------------------------\n')

    # the tn_tpus
    whole_tpu_sent_act_dataframe = TransitNeighborhood_TPU.construct_sent_act_dataframe(sent_dict=sentiment_dict,
                                                                                        activity_dict=activity_dict)
    print('--------------------------------------------------------------------')
    print('Total number of tweets we consider in cross sectional study...')
    print(sum(list(whole_tpu_sent_act_dataframe['activity'])))
    print('--------------------------------------------------------------------')
    whole_tpu_sent_act_dataframe.to_csv(os.path.join(read_data.desktop, 'tpu_sent_act.csv'))
    y_label = 'Percentage of Positive Tweets Minus Percentage of Negative Tweets'
    # Draw the tpu sentiment against activity
    figure_title_name = 'Sentiment Against Activity Across TPUs'
    TransitNeighborhood_TPU.plot_overall_sentiment_for_whole_tweets(df=whole_tpu_sent_act_dataframe,
                                                                    y_label_name=y_label,
                                                                    figure_title=figure_title_name,
                                                                    saved_file_name='tpu_sent_vs_act.png',
                                                                    without_outlier=False)

    # Analyze the result quarterly
    print('-------------------------------------------------------------------')
    activity_dict_quarter_1 = TransitNeighborhood_TPU.build_dataframe_quarterly(quarter_number=1)
    activity_dict_quarter_2 = TransitNeighborhood_TPU.build_dataframe_quarterly(quarter_number=2)
    activity_dict_quarter_3 = TransitNeighborhood_TPU.build_dataframe_quarterly(quarter_number=3)
    activity_dict_quarter_4 = TransitNeighborhood_TPU.build_dataframe_quarterly(quarter_number=4)

    sentiment_dict_quarter_1 = {}
    sentiment_dict_quarter_2 = {}
    sentiment_dict_quarter_3 = {}
    sentiment_dict_quarter_4 = {}

    activity_dict_list = [activity_dict_quarter_1, activity_dict_quarter_2, activity_dict_quarter_3,
                          activity_dict_quarter_4]
    sentiment_dict_list = [sentiment_dict_quarter_1, sentiment_dict_quarter_2, sentiment_dict_quarter_3,
                           sentiment_dict_quarter_4]

    for activity_dict, sentiment_dict in zip(activity_dict_list, sentiment_dict_list):
        index_value = activity_dict_list.index(activity_dict)
        for tpu in list(activity_dict.keys()):
            dataframe = pd.read_csv(
                os.path.join(read_data.transit_non_transit_comparison_cross_sectional, 'tpu_data_more',
                             tpu, tpu + '_data.csv'), encoding='utf-8', dtype='str',
                quoting=csv.QUOTE_NONNUMERIC)
            if index_value == 0:
                quarter_dataframe = dataframe.loc[dataframe['month'].isin(['1.0', '2.0', '3.0'])]
            elif index_value == 1:
                quarter_dataframe = dataframe.loc[dataframe['month'].isin(['4.0', '5.0', '6.0'])]
            elif index_value == 2:
                quarter_dataframe = dataframe.loc[dataframe['month'].isin(['7.0', '8.0', '9.0'])]
            else:
                quarter_dataframe = dataframe.loc[dataframe['month'].isin(['10.0', '11.0', '12.0'])]
            # dataframe['sentiment'] = dataframe['sentiment'].astype(np.int)
            sentiment_dict[tpu] = pos_percent_minus_neg_percent(quarter_dataframe)

    print('The sentiment dict in quarter 1 is...')
    print(activity_dict_quarter_1['971 - 974'])
    print(sentiment_dict_list[0]['971 - 974'], sentiment_dict_list[1]['971 - 974'], sentiment_dict_list[2]['971 - 974'],
          sentiment_dict_list[3]['971 - 974'])

    # the tn_tpus
    whole_sent_act_quarter_1 = TransitNeighborhood_TPU.construct_sent_act_dataframe(sent_dict=sentiment_dict_list[0],
                                                                                    activity_dict=activity_dict_quarter_1)
    whole_sent_act_quarter_2 = TransitNeighborhood_TPU.construct_sent_act_dataframe(sent_dict=sentiment_dict_list[1],
                                                                                    activity_dict=activity_dict_quarter_2)
    whole_sent_act_quarter_3 = TransitNeighborhood_TPU.construct_sent_act_dataframe(sent_dict=sentiment_dict_list[2],
                                                                                    activity_dict=activity_dict_quarter_3)
    whole_sent_act_quarter_4 = TransitNeighborhood_TPU.construct_sent_act_dataframe(sent_dict=sentiment_dict_list[3],
                                                                                    activity_dict=activity_dict_quarter_4)
    print('The general information of the quarterly based dataframe is...')
    describle_dataframe(whole_sent_act_quarter_1, message='In Quarter 1')
    describle_dataframe(whole_sent_act_quarter_2, message='In Quarter 2')
    describle_dataframe(whole_sent_act_quarter_3, message='In Quarter 3')
    describle_dataframe(whole_sent_act_quarter_4, message='In Quarter 4')

    print('--------------------------------------------------------------------')
    print('For instance, total number of tweets we consider in the first quarter of the cross sectional study...')
    print(sum(list(whole_sent_act_quarter_1['activity'])))
    print('--------------------------------------------------------------------')
    whole_sent_act_quarter_1.to_csv(os.path.join(read_data.desktop, 'tpu_sent_act_quarter_1.csv'))
    whole_sent_act_quarter_2.to_csv(os.path.join(read_data.desktop, 'tpu_sent_act_quarter_2.csv'))
    whole_sent_act_quarter_3.to_csv(os.path.join(read_data.desktop, 'tpu_sent_act_quarter_3.csv'))
    whole_sent_act_quarter_4.to_csv(os.path.join(read_data.desktop, 'tpu_sent_act_quarter_4.csv'))
    y_label = 'Percentage of Positive Tweets Minus Percentage of Negative Tweets'
    # Draw the tpu sentiment against activity
    figure_title_name_quarter1 = 'Sentiment Against Activity Across TPUs(Quarter 1)'
    figure_title_name_quarter2 = 'Sentiment Against Activity Across TPUs(Quarter 2)'
    figure_title_name_quarter3 = 'Sentiment Against Activity Across TPUs(Quarter 3)'
    figure_title_name_quarter4 = 'Sentiment Against Activity Across TPUs(Quarter 4)'
    TransitNeighborhood_TPU.plot_overall_sentiment_for_whole_tweets(df=whole_sent_act_quarter_1,
                                                                    y_label_name=y_label,
                                                                    figure_title=figure_title_name_quarter1,
                                                                    saved_file_name='tpu_sent_vs_act_quarter_1.png',
                                                                    without_outlier=False)
    TransitNeighborhood_TPU.plot_overall_sentiment_for_whole_tweets(df=whole_sent_act_quarter_2,
                                                                    y_label_name=y_label,
                                                                    figure_title=figure_title_name_quarter2,
                                                                    saved_file_name='tpu_sent_vs_act_quarter_2.png',
                                                                    without_outlier=False)
    TransitNeighborhood_TPU.plot_overall_sentiment_for_whole_tweets(df=whole_sent_act_quarter_3,
                                                                    y_label_name=y_label,
                                                                    figure_title=figure_title_name_quarter3,
                                                                    saved_file_name='tpu_sent_vs_act_quarter_3.png',
                                                                    without_outlier=False)
    TransitNeighborhood_TPU.plot_overall_sentiment_for_whole_tweets(df=whole_sent_act_quarter_4,
                                                                    y_label_name=y_label,
                                                                    figure_title=figure_title_name_quarter4,
                                                                    saved_file_name='tpu_sent_vs_act_quarter_4.png',
                                                                    without_outlier=False)
    print('-------------------------------------------------------------------')

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
    #
    print('Building the regressiong model between sentiment/activity and social demographic variables...')
    combined_dataframe = whole_tpu_sent_act_dataframe.merge(tpu_2016_social_demographic_dataframe, on='tpu_name')
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
    draw_boxplot(combined_dataframe, column_name='Sentiment', title_name='Sentiment Comparison')
    draw_boxplot(combined_dataframe, column_name='Activity_log10', title_name='Activity Level Comparison')
    combined_dataframe = combined_dataframe[['Sentiment', 'activity', 'median_income', 'employment',
                                             'marry', 'education', 'avg_population']]
    print('Social Demographic Data Description...')
    for column_name in ['median_income', 'employment', 'marry', 'education', 'avg_population']:
        print('Coping with {}'.format(column_name))
        print(combined_dataframe[column_name].describe())
        print('-------------Done!----------------')

    print('Check the correlation between sentiment and activity...')
    correlation_value_sent_act = combined_dataframe['Sentiment'].corr(combined_dataframe['activity'])
    print('The correlation coefficient of sentiment and activity is :{}'.format(correlation_value_sent_act))

    print('Check the correlation between activity and avg_population per square meter...')
    correlation_value_act_population = combined_dataframe['activity'].corr(combined_dataframe['avg_population'])
    print('The correlation coefficient of activity and avg population is :{}'.format(correlation_value_act_population))

    print('Regression analysis starts..... ')
    normalized_combined_dataframe = (combined_dataframe - combined_dataframe.mean()) / combined_dataframe.std()
    normalized_combined_dataframe['tn_or_not'] = tn_or_not_list
    normalized_combined_dataframe['tpu_name'] = tpu_name_list_from_combined_data
    print(normalized_combined_dataframe.head(5))
    print(normalized_combined_dataframe.columns)
    # Check the correlation matrix of independent variables and compute VIF value for each independent variable
    combined_dataframe['tn_or_not'] = tn_or_not_list
    dataframe_for_correlation_matrix = combined_dataframe[['median_income', 'employment', 'marry', 'education',
                                                           'tn_or_not', 'avg_population']]
    draw_correlation_plot(dataframe_for_correlation_matrix)
    result_vif_series = compute_vif(dataframe_for_correlation_matrix)
    print(result_vif_series)
    # Regression analysis
    reg_sent = smf.ols('Sentiment ~ median_income+employment+marry+education+avg_population+tn_or_not',
                       normalized_combined_dataframe).fit()
    print(reg_sent.summary())
    reg_act = smf.ols('activity ~ median_income+employment+marry+education+avg_population+tn_or_not',
                      normalized_combined_dataframe).fit()
    print(reg_act.summary())

    ## Get wordcloud and topic modeling for specific TPUs
    # Specify the data path
    # tn_tpu_path = os.path.join(read_data.datasets, 'tpu_related_csvs', 'cross_sectional', 'tn_tpus')
    # nontn_tpu_path = os.path.join(read_data.datasets, 'tpu_related_csvs', 'cross_sectional', 'non_tn_tpus')
    # Load the corresponding dataframes
    # tpu_442 = utils.read_local_csv_file(path=nontn_tpu_path, filename='442_data.csv', dtype_str=False)
    # tpu_181_182 = utils.read_local_csv_file(path=nontn_tpu_path, filename='181 - 182_data.csv', dtype_str=False)
    # tpu_426 = utils.read_local_csv_file(path=nontn_tpu_path, filename='426_data.csv', dtype_str=False)
    # tpu_421_422 = utils.read_local_csv_file(path=nontn_tpu_path, filename='421 - 422_data.csv', dtype_str=False)

    # tpu_950_951 = utils.read_local_csv_file(path=tn_tpu_path, filename='950 - 951_data.csv', dtype_str=False)
    # tpu_153 = utils.read_local_csv_file(path=tn_tpu_path, filename='153_data.csv', dtype_str=False)
    # tpu_971_974 = utils.read_local_csv_file(path=tn_tpu_path, filename='971 - 974_data.csv', dtype_str=False)
    # tpu_135 = utils.read_local_csv_file(path=tn_tpu_path, filename='135_data.csv', dtype_str=False)

    # Create text for wordcloud
    # dataframe_list = [tpu_442, tpu_181_182, tpu_426, tpu_950_951, tpu_153, tpu_971_974, tpu_135]
    # tpu_442_text_for_wordcloud = wordcloud_generate.create_text_for_wordcloud(tpu_442)
    # tpu_181_182_text_for_wordcloud = wordcloud_generate.create_text_for_wordcloud(tpu_181_182)
    # tpu_426_text_for_wordcloud = wordcloud_generate.create_text_for_wordcloud(tpu_426)
    # tpu_421_422_text_for_wordcloud = wordcloud_generate.create_text_for_wordcloud(tpu_421_422)

    # tpu_950_951_text_for_wordcloud = wordcloud_generate.create_text_for_wordcloud(tpu_950_951)
    # tpu_153_text_for_wordcloud = wordcloud_generate.create_text_for_wordcloud(tpu_153)
    # tpu_971_974_text_for_wordcloud = wordcloud_generate.create_text_for_wordcloud(tpu_971_974)
    # tpu_135_text_for_wordcloud = wordcloud_generate.create_text_for_wordcloud(tpu_135)

    # Generate wordcloud
    # wordcloud_generate.generate_wordcloud(tpu_442_text_for_wordcloud, wordcloud_generate.circle_mask,
    #                                       file_name='tpu442_wordcloud.png',
    #                                       color_func=wordcloud_generate.green_func,
    #                                       saving_path=read_data.transit_non_transit_comparison_cross_sectional)
    # wordcloud_generate.generate_wordcloud(tpu_181_182_text_for_wordcloud, wordcloud_generate.circle_mask,
    #                                       file_name='tpu181_182_wordcloud.png',
    #                                       color_func=wordcloud_generate.green_func,
    #                                       saving_path=read_data.transit_non_transit_comparison_cross_sectional)
    # wordcloud_generate.generate_wordcloud(tpu_426_text_for_wordcloud, wordcloud_generate.circle_mask,
    #                                       file_name='tpu426_wordcloud.png',
    #                                       color_func=wordcloud_generate.green_func,
    #                                       saving_path=read_data.transit_non_transit_comparison_cross_sectional)
    # wordcloud_generate.generate_wordcloud(tpu_421_422_text_for_wordcloud, wordcloud_generate.circle_mask,
    #                                       file_name='tpu421_422_wordcloud.png',
    #                                       color_func=wordcloud_generate.green_func,
    #                                       saving_path=read_data.transit_non_transit_comparison_cross_sectional)

    # wordcloud_generate.generate_wordcloud(tpu_950_951_text_for_wordcloud, wordcloud_generate.circle_mask,
    #                                       file_name='tpu950_951_wordcloud.png',
    #                                       color_func=wordcloud_generate.red_func,
    #                                       saving_path=read_data.transit_non_transit_comparison_cross_sectional)
    # wordcloud_generate.generate_wordcloud(tpu_153_text_for_wordcloud, wordcloud_generate.circle_mask,
    #                                       file_name='tpu153_wordcloud.png',
    #                                       color_func=wordcloud_generate.red_func,
    #                                       saving_path=read_data.transit_non_transit_comparison_cross_sectional)
    # wordcloud_generate.generate_wordcloud(tpu_971_974_text_for_wordcloud, wordcloud_generate.circle_mask,
    #                                       file_name='tpu971_974_wordcloud.png',
    #                                       color_func=wordcloud_generate.red_func,
    #                                       saving_path=read_data.transit_non_transit_comparison_cross_sectional)
    # wordcloud_generate.generate_wordcloud(tpu_135_text_for_wordcloud, wordcloud_generate.circle_mask,
    #                                       file_name='tpu135_wordcloud.png',
    #                                       color_func=wordcloud_generate.red_func,
    #                                       saving_path=read_data.transit_non_transit_comparison_cross_sectional)




