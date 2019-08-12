import pandas as pd
import numpy as np
import os
import pytz
import csv
from scipy.stats import linregress
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import gensim

import read_data
import utils
import wordcloud_generate
import Topic_Modelling_for_tweets

from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
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


class TransitNeighborhood_Before_After(object):

    before_after_stations = ['Whampoa', 'Ho Man Tin', 'South Horizons', 'Wong Chuk Hang', 'Ocean Park',
                             'Lei Tung']

    def __init__(self, name, tn_dataframe, non_tn_dataframe, treatment_not_considered_dataframe, oct_open:bool,
                 before_and_after:bool, compute_positive:bool, compute_negative:bool):
        """
        :param name: the name of the studied area
        :param tn_dataframe: the dataframe which records all the tweets posted in the TN
        :param non_tn_dataframe: the dataframe which records all the tweets posted in corresponding non_tn
        :param treatment_not_considered_dataframe: the dataframe which records all the tweets posted in the not
               considered TN
        :param oct_open: check whether the station is opened on oct 23, 2016
        :param before_and_after: only True if the MTR station in this TN is built recently(in 2016)
        :param compute_positive: True if use positive percent as the sentiment metric
        :param compute_negative: True if use negative percent as the sentiment metric
        """
        self.name = name
        self.tn_dataframe = tn_dataframe
        self.non_tn_dataframe = non_tn_dataframe
        self.treatment_not_considered_dataframe = treatment_not_considered_dataframe
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
        result_dict_tn_not_considered = sentiment_by_month(self.treatment_not_considered_dataframe,
                                                           compute_positive_percent=self.compute_positive,
                                             compute_negative_percent=self.compute_negative)
        result_dataframe_tn = pd.DataFrame(list(result_dict_tn.items()), columns=['Date', 'Value'])
        result_dataframe_tn_not_considered = pd.DataFrame(list(result_dict_tn_not_considered.items()),
                                                          columns=['Date', 'Value'])
        result_dataframe_non_tn = pd.DataFrame(list(result_dict_non_tn.items()), columns=['Date', 'Value'])
        return result_dataframe_tn, result_dataframe_tn_not_considered, result_dataframe_non_tn

    def compute_abs_coeff_difference(self):
        # Compute the coefficient difference of sentiment against time before the treatment
        treatment_group_dataframe, _, control_group_dataframe = self.output_sent_act_dataframe()
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
        return abs(slope_tn_sentiment-slop_non_tn_sentiment), abs(slope_tn_activity-slope_non_tn_activity)

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
        tn_dataframe_sent_act, tn_not_considered_dataframe_sent_act, non_tn_dataframe_sent_act = self.output_sent_act_dataframe()
        # Set Date as the index and reorder rows based on time list
        tn_dataframe_with_sentiment_activity = tn_dataframe_sent_act.set_index('Date')
        tn_dataframe_for_plot = tn_dataframe_with_sentiment_activity.loc[time_list]
        non_tn_dataframe_with_sentiment_activity = non_tn_dataframe_sent_act.set_index('Date')
        non_tn_dataframe_for_plot = non_tn_dataframe_with_sentiment_activity.loc[time_list]
        tn_not_considered_dataframe_with_sentiment_activity = tn_not_considered_dataframe_sent_act.set_index('Date')
        tn_not_considered_dataframe_for_plot = tn_not_considered_dataframe_with_sentiment_activity .loc[time_list]
        # print(tn_dataframe_for_plot.head())
        # x is used in plot to record time in x axis
        x = np.arange(0, len(list(tn_dataframe_for_plot.index)), 1)
        if draw_sentiment:  # draw the sentiment comparison plot: y1: TN-TPUs; y2: non-TN-TPUs
            y1 = [value[0] for value in list(tn_dataframe_for_plot['Value'])]
            y2 = [value[0] for value in list(tn_not_considered_dataframe_for_plot['Value'])]
            y3 = [value[0] for value in list(non_tn_dataframe_for_plot['Value'])]
        else:  # draw the activity comparison plot. Use log10(num of tweets) instead
            if 'Kwun Tong' in plot_title_name:
                y1 = [value[1]/1397137.377 for value in list(tn_dataframe_for_plot['Value'])]
                y2 = [value[1]/2573440.774 for value in list(tn_not_considered_dataframe_for_plot['Value'])]
                y3 = [value[1]/1603042.196 for value in list(non_tn_dataframe_for_plot['Value'])]
            elif 'South Horizons' in plot_title_name:
                y1 = [value[1]/1401496.118 for value in list(tn_dataframe_for_plot['Value'])]
                y2 = [value[1]/1810822.755 for value in list(tn_not_considered_dataframe_for_plot['Value'])]
                y3 = [value[1]/5303151.669 for value in list(non_tn_dataframe_for_plot['Value'])]
            elif 'Ocean Park' in plot_title_name:
                y1 = [value[1]/2167755.637 for value in list(tn_dataframe_for_plot['Value'])]
                y2 = [value[1]/3807838.975 for value in list(tn_not_considered_dataframe_for_plot['Value'])]
                y3 = [value[1]/7143243.558 for value in list(non_tn_dataframe_for_plot['Value'])]
            else:
                wrong_message = 'You should set a proper title name!'
                return wrong_message

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        if draw_sentiment:
            lns1 = ax.plot(x, y1, 'r-', label=line_labels[0], linestyle='--', marker='o')
            lns2 = ax.plot(x, y2, 'y-', label=line_labels[1], linestyle='--', marker='o')
            lns3 = ax.plot(x, y3, 'g-', label=line_labels[2], linestyle='--', marker='^')
        else:
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
            lns1 = ax.plot(x, y1, 'r-', label=line_labels[0], linestyle='--', marker='o')
            lns2 = ax.plot(x, y2, 'y-', label=line_labels[1], linestyle='--', marker='o')
            lns3 = ax.plot(x, y3, 'g-', label=line_labels[2], linestyle='--', marker='^')

        # Whether to draw the vertical line that indicates the open date
        if self.before_and_after:
            if self.oct_open:
                # draw a verticle showing the openning date of the station
                plt.axvline(5.77, color='black')
                # if draw_sentiment:  # the ylim of sentiment and activity plots are different
                #     ax.text(4.3, 0, 'Opening Date: \nOct 23, 2016', horizontalalignment='center', color='black')
                # else:
                #     ax.text(4.3, 3.0, 'Opening Date: \nOct 23, 2016', horizontalalignment='center', color='black')
            else:
                # draw a verticle showing the openning date of the station
                plt.axvline(7.95, color='black')
                # if draw_sentiment:  # the ylim of sentiment and activity plots are different
                #     ax.text(6.5, 0, 'Opening Date: \nDec 28, 2016', horizontalalignment='center', color='black')
                # else:
                #     ax.text(6.5, 3.0, 'Opening Date: \nDec 28, 2016', horizontalalignment='center', color='black')
        else:
            pass

        # Add the legend
        lns = lns1 + lns2 + lns3
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs)

        # Draw the average of the sentiment level
        # This average sentiment level means the average sentiment of 93 500-meter TBs
        if draw_sentiment:
            ax.axhline(y=0.38, color='r', linestyle='solid')
            ax.text(3, 0.43, 'Average Sentiment Level: 0.38', horizontalalignment='center', color='r')
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

    def line_map_comparison_between_tpus_and_control(self, *tpu_names,
                                                     saving_path=read_data.transit_non_transit_comparison_before_after):
        # Get the sent act dataframe for each tpu
        dataframe_list = []
        tpu_name_list = list(tpu_names)
        for name in tpu_name_list:
            dataframe = utils.read_local_csv_file(path=os.path.join(
                read_data.transit_non_transit_comparison_before_after, 'tpu_data_with_visitors', name),
                filename=name+'_data.csv')
            dataframe_list.append(dataframe)
        dataframe_sent_act_dict = [sentiment_by_month(dataframe) for dataframe in dataframe_list]
        dataframe_sent_act_ordered = []
        for result_dict in dataframe_sent_act_dict:
            dataframe_sent_act = pd.DataFrame(list(result_dict.items()), columns=['Date', 'Value'])
            dataframe_sent_act_ordered.append(sort_data_based_on_date(dataframe_sent_act))
        non_tn_sent_act_dict = sentiment_by_month(self.non_tn_dataframe)
        non_tn_sent_act_result = pd.DataFrame(list(non_tn_sent_act_dict.items()), columns=['Date', 'Value'])
        non_tn_sent_act_ordered = sort_data_based_on_date(non_tn_sent_act_result)
        tn_not_considered_sent_act_dict = sentiment_by_month(self.treatment_not_considered_dataframe)
        tn_not_considered_result = pd.DataFrame(list(tn_not_considered_sent_act_dict.items()),
                                                columns=['Date', 'Value'])
        tn_not_considered_ordered = sort_data_based_on_date(tn_not_considered_result)

        # create the plots for sentiment and activity
        figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        time_list_numbers = list(range(len(time_list)))
        for tpu_name, sent_act_data in zip(tpu_name_list, dataframe_sent_act_ordered):
            sentiment_list = list(sent_act_data['sentiment'])
            activity_list = np.log10(list(sent_act_data['activity']))
            ax1.plot(time_list_numbers, sentiment_list, label=tpu_name)
            ax2.plot(time_list_numbers, activity_list, label=tpu_name)
        non_tn_sent_list = list(non_tn_sent_act_ordered['sentiment'])
        non_tn_act_list = np.log10(list(non_tn_sent_act_ordered['activity']))
        ax1.plot(time_list_numbers, non_tn_sent_list, label='Control Group')
        ax2.plot(time_list_numbers, non_tn_act_list, label='Control Group')
        tn_not_considered_sent_list = list(tn_not_considered_ordered['sentiment'])
        tn_not_considered_act_list = np.log10(list(tn_not_considered_ordered['activity']))
        ax1.plot(time_list_numbers, tn_not_considered_sent_list, label='Treatment Group Not Considered')
        ax2.plot(time_list_numbers, tn_not_considered_act_list, label='Treatment Group Not Considered')
        ax1.set_ylim(-1, 1)
        for ax in (ax1, ax2):
            ax.set_xticks(time_list_numbers)
            ax.set_xticklabels(time_list, rotation='vertical')
            ax.legend()
        ax1.set_ylabel('Percentage of Positive Tweets Minus Percentage of Negative Tweets')
        ax2.set_ylabel('Number of Posted Tweets(log10)')
        if '245' in tpu_name_list:
            ax1.axvline(5.7, color='black')
            ax2.axvline(5.7, color='black')
            ax1.text(3.3, -0.5, 'Opening Date: \nOct 23, 2016', horizontalalignment='center', color='black')
            ax2.text(3.3, 2.2, 'Opening Date: \nOct 23, 2016', horizontalalignment='center', color='black')
            ax1.set_title('Kwun Tong Line Treatment TPUs and Control Group Comparison', fontsize=10)
            ax2.set_title('Kwun Tong Line Treatment TPUs and Control Group Comparison', fontsize=10)
            # plt.suptitle('Treatment and Control Comparison: Kwun Tong Line')
            figure.savefig(os.path.join(saving_path, 'Kwun_Tong_Line_tpus_control.png'))
            plt.show()
        elif '174' in tpu_name_list:
            ax1.axvline(7.7, color='black')
            ax2.axvline(7.7, color='black')
            ax1.text(5, -0.5, 'Opening Date: \nDec 28, 2016', horizontalalignment='center', color='black')
            ax2.text(5, 2, 'Opening Date: \nDec 28, 2016', horizontalalignment='center', color='black')
            ax1.set_title('South Horizons & Lei Tung Treatment TPUs and Control Group Comparison', fontsize=8)
            ax2.set_title('South Horizons & Lei Tung Treatment TPUs and Control Group Comparison', fontsize=8)
            # plt.suptitle('Treatment and Control Comparison: South Horizons & Lei Tung')
            figure.savefig(os.path.join(saving_path, 'Souths_horizons_lei_tung_'+'_'.join(tpu_name_list)+'_control.png'))
            plt.show()
        else:
            ax1.axvline(7.7, color='black')
            ax2.axvline(7.7, color='black')
            ax1.text(5, -0.5, 'Opening Date: \nDec 28, 2016', horizontalalignment='center', color='black')
            ax2.text(5, 2, 'Opening Date: \nDec 28, 2016', horizontalalignment='center', color='black')
            ax1.set_title('Ocean Park & Wong Chuk Hang Treatment TPUs and Control Group Comparison', fontsize=8)
            ax2.set_title('Ocean Park & Wong Chuk Hang Treatment TPUs and Control Group Comparison', fontsize=8)
            # plt.suptitle('Treatment and Control Comparison: Ocean Park & Wong Chuk Hang')
            figure.savefig(
                os.path.join(saving_path, 'Ocean_park_wong_chuk_hang_' + '_'.join(tpu_name_list) + '_control.png'))
            plt.show()

    def draw_tweet_posting_time_comparison(self, saving_path):
        treatment_dataframe_copy = self.tn_dataframe.copy()
        treatment_not_considered_dataframe_copy = self.non_tn_dataframe.copy()
        control_dataframe_copy = self.non_tn_dataframe.copy()
        # Transform the string time to datetime object
        if isinstance(list(treatment_dataframe_copy['hk_time'])[0], str) or isinstance(
                list(treatment_not_considered_dataframe_copy['hk_time'])[0], str) or isinstance(
                list(control_dataframe_copy['hk_time'])[0], str):
            treatment_dataframe_copy['hk_time'] = treatment_dataframe_copy.apply(
                lambda row: TransitNeighborhood_Before_After.transform_string_time_to_datetime(row['hk_time']), axis=1)
            treatment_not_considered_dataframe_copy['hk_time'] = treatment_not_considered_dataframe_copy.apply(
                lambda row: TransitNeighborhood_Before_After.transform_string_time_to_datetime(row['hk_time']), axis=1)
            control_dataframe_copy['hk_time'] = control_dataframe_copy.apply(
                lambda row: TransitNeighborhood_Before_After.transform_string_time_to_datetime(row['hk_time']), axis=1)
        else:
            pass
        # Get the day column and hour column
        treatment_dataframe_copy['day'] = treatment_dataframe_copy.apply(lambda row: row['hk_time'].day, axis=1)
        treatment_dataframe_copy['hour'] = treatment_dataframe_copy.apply(lambda row: row['hk_time'].hour, axis=1)
        treatment_not_considered_dataframe_copy['day'] = treatment_not_considered_dataframe_copy.apply(
            lambda row: row['hk_time'].day, axis=1)
        treatment_not_considered_dataframe_copy['hour'] = treatment_not_considered_dataframe_copy.apply(
            lambda row: row['hk_time'].hour, axis=1)
        control_dataframe_copy['day'] = control_dataframe_copy.apply(
            lambda row: row['hk_time'].day, axis=1)
        control_dataframe_copy['hour'] = control_dataframe_copy.apply(
            lambda row: row['hk_time'].hour, axis=1)

        if self.oct_open:
            treatment_dataframe_before = treatment_dataframe_copy.loc[
                treatment_dataframe_copy['hk_time'] < october_23_start]
            treatment_dataframe_after = treatment_dataframe_copy.loc[
                treatment_dataframe_copy['hk_time'] > october_23_end]
            treatment_not_considered_dataframe_before = treatment_not_considered_dataframe_copy.loc[
                treatment_not_considered_dataframe_copy['hk_time'] < october_23_start]
            treatment_not_considered_dataframe_after = treatment_not_considered_dataframe_copy.loc[
                treatment_not_considered_dataframe_copy['hk_time'] > october_23_end]
            control_dataframe_before = control_dataframe_copy.loc[control_dataframe_copy['hk_time'] < october_23_start]
            control_dataframe_after = control_dataframe_copy.loc[control_dataframe_copy['hk_time'] > october_23_end]
        else:
            treatment_dataframe_before = treatment_dataframe_copy.loc[
                treatment_dataframe_copy['hk_time'] < december_28_start]
            treatment_dataframe_after = treatment_dataframe_copy.loc[
                treatment_dataframe_copy['hk_time'] > december_28_end]
            treatment_not_considered_dataframe_before = treatment_not_considered_dataframe_copy.loc[
                treatment_not_considered_dataframe_copy['hk_time'] < december_28_start]
            treatment_not_considered_dataframe_after = treatment_not_considered_dataframe_copy.loc[
                treatment_not_considered_dataframe_copy['hk_time'] > december_28_end]
            control_dataframe_before = control_dataframe_copy.loc[control_dataframe_copy['hk_time'] < december_28_start]
            control_dataframe_after = control_dataframe_copy.loc[control_dataframe_copy['hk_time'] > december_28_end]

        fig_treatment = plt.figure(figsize=(20, 8))
        fig_treatment_not_considered = plt.figure(figsize=(20, 8))
        fig_control = plt.figure(figsize=(20, 8))
        dataframe_dict = {'treatment': [treatment_dataframe_before, treatment_dataframe_after],
                          'treatment_not_considered': [treatment_not_considered_dataframe_before,
                                                       treatment_not_considered_dataframe_after],
                          'control': [control_dataframe_before, control_dataframe_after]}
        for setting in list(dataframe_dict.keys()):
            if setting == 'treatment':
                dataframe_list_treatment = dataframe_dict[setting]
                ax_treatment = [None] * 2
                for index, dataframe in enumerate(dataframe_list_treatment):
                    ax_treatment[index] = fig_treatment.add_subplot(1,2, index+1)
                    if index == 0:
                        sns.distplot(treatment_dataframe_before.hour, kde=False, label='Treatment Before',
                                     ax=ax_treatment[index], color='red', norm_hist=True)
                    else:
                        sns.distplot(treatment_dataframe_after.hour, kde=False, label='Treatment After',
                                     ax=ax_treatment[index], color='green', norm_hist=True)
                    ax_treatment[index].legend()
                    ax_treatment[index].set_ylim(0, 0.15)
                    ax_treatment[index].set_ylabel('Probability', color='k')
                fig_treatment.savefig(os.path.join(saving_path, self.name+'_treatment.png'))
            elif setting == 'treatment_not_considered':
                dataframe_list_treatment_not_considered = dataframe_dict[setting]
                ax_treatment_not_considered = [None] * 2
                for index, dataframe in enumerate(dataframe_list_treatment_not_considered):
                    ax_treatment_not_considered[index] = fig_treatment_not_considered.add_subplot(1,2, index+1)
                    if index == 0:
                        sns.distplot(treatment_not_considered_dataframe_before.hour, kde=False,
                                     label='Treatment Not Considered Before',
                                     ax=ax_treatment_not_considered[index], color='red', norm_hist=True)
                    else:
                        sns.distplot(treatment_not_considered_dataframe_after.hour, kde=False,
                                     label='Treatment Not Considered After',
                                     ax=ax_treatment_not_considered[index], color='green', norm_hist=True)
                    ax_treatment_not_considered[index].legend()
                    ax_treatment_not_considered[index].set_ylim(0, 0.15)
                    ax_treatment_not_considered[index].set_ylabel('Probability', color='k')
                    fig_treatment_not_considered.savefig(
                        os.path.join(saving_path, self.name + '_treatment_not_considered.png'))
            else:
                dataframe_list_control = dataframe_dict[setting]
                ax_control = [None] * 2
                for index, dataframe in enumerate(dataframe_list_control):
                    ax_control[index] = fig_control.add_subplot(1,2, index+1)
                    if index == 0:
                        sns.distplot(control_dataframe_before.hour, kde=False, label='Control Before',
                                     ax=ax_control[index], color='red', norm_hist=True)
                    else:
                        sns.distplot(control_dataframe_after.hour, kde=False, label='Control After',
                                     ax=ax_control[index], color='green', norm_hist=True)
                    ax_control[index].legend()
                    ax_control[index].set_ylim(0, 0.15)
                    ax_control[index].set_ylabel('Probability', color='k')
                fig_control.savefig(os.path.join(saving_path, self.name+'_control.png'))

    @staticmethod
    def build_data_for_longitudinal_study(tweet_data_path, saving_path):
        """
        :param tweet_data_path: path which is used to save all the filtered tweets
        :param saving_path: path which is used to save the tweets posted in each TPU
        :return:
        """
        all_tweet_data = pd.read_csv(os.path.join(tweet_data_path, 'tweet_2016_2017_with_visitors_tpu2016.csv'),
                                     encoding='utf-8', dtype='str', quoting=csv.QUOTE_NONNUMERIC)
        tpu_2016 = pd.read_csv(os.path.join(tweet_data_path, 'tpu_2016.csv'), encoding='utf-8', dtype='str',
                               quoting=csv.QUOTE_NONNUMERIC)
        print(tpu_2016.columns)
        tpu_set = set(tpu_2016['TPU'])
        for tpu in tpu_set:
            try:
                os.mkdir(os.path.join(saving_path, tpu))
                dataframe = all_tweet_data.loc[all_tweet_data['TPU_2016'] == tpu]
                dataframe.to_csv(os.path.join(saving_path, tpu, tpu + '_data.csv'), encoding='utf-8',
                                 quoting=csv.QUOTE_NONNUMERIC)
            except WindowsError:
                    pass

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
    for sentiment in list(df['sentiment']):
        if int(float(sentiment)) == 2:
            positive+=1
        else:
            pass
    return positive/df.shape[0]


# compute the percentage of positive Tweets: 0 is negative
def negative_percent(df):
    negative = 0
    for sentiment in list(df['sentiment']):
        if int(float(sentiment)) == 0:
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


# compute the sentiment level for each month
def sentiment_by_month(df, compute_positive_percent=False, compute_negative_percent=False):
    # check whether the value in the hk_time attribute is string or not
    if isinstance(list(df['hk_time'])[0], str):
        df['hk_time'] = df.apply(
            lambda row: TransitNeighborhood_Before_After.transform_string_time_to_datetime(row['hk_time']), axis=1)
    else:
        pass
    # Check whether one dataframe has the year and the month columns. If not, generate them!
    try:
        df['month_plus_year'] = df.apply(lambda row: str(int(float(row['year'])))+'_'+str(int(float(row['month']))),
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
        df_copy = df.copy()
        if isinstance(list(df_copy['hk_time'])[0], str):
            df_copy['hk_time'] = \
                df_copy.apply(
                    lambda row: TransitNeighborhood_Before_After.transform_string_time_to_datetime(row['hk_time']),
                    axis=1)
        else:
            pass
        df_before = df_copy.loc[df_copy['hk_time'] < open_date_start]
        df_after = df_copy.loc[df_copy['hk_time'] > open_date_end]
    else:
        open_date_start = december_28_start
        open_date_end = december_28_end
        df_copy = df.copy()
        if isinstance(list(df_copy['hk_time'])[0], str):
            df_copy['hk_time'] = \
                df_copy.apply(
                    lambda row: TransitNeighborhood_Before_After.transform_string_time_to_datetime(row['hk_time']),
                    axis=1)
        else:
            pass
        df_before = df_copy.loc[df_copy['hk_time'] < open_date_start]
        df_after = df_copy.loc[df_copy['hk_time'] > open_date_end]
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
    # save the processed text
    np.save(os.path.join(read_data.transit_non_transit_comparison_before_after, station_name+'_text.npy'), data_ready)
    text_count_list = [len(text) for text in data_ready]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.distplot(text_count_list, kde=False, hist=True)
    # check whether tweet count=7 is appropriate
    ax.axvline(np.median(text_count_list), color='#EB1B52', label='50% Percentile')
    ax.axvline(7, color='#FF9415', label='Number of Keywords in Topic Modelling')
    plt.xlim((0, 100))
    plt.ylim((0, 700))
    # Check if it is appropriate to set the number of keywords as 7 in this dataframe
    plt.xticks(list(plt.xticks()[0]) + [np.median(text_count_list)])
    plt.xticks(list(plt.xticks()[0]) + [7])
    plt.legend()
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


def build_treatment_control_tpu_compare_for_one_line(treatment_csv, control_1000_csv,
                                                     control_1500_csv):
    """
    :param treatment_csv: a csv file which records the tpus that intersect with the 500-meter buffers
    :param control_1000_csv: a csv file which records the tpus that intersect with 1000-meter buffers
    :param control_1500_csv: a csv file which records the tpus that intersect with 1500-meter buffers
    :return: tpu names for the treatment group and control groups
    """
    datapath = os.path.join(read_data.transit_non_transit_comparison_before_after,
                            'tpu_based_longitudinal_analysis')
    treatment_data = pd.read_csv(os.path.join(datapath, treatment_csv), encoding='utf-8')
    control_1000_data = pd.read_csv(os.path.join(datapath, control_1000_csv), encoding='utf-8')
    control_1500_data = pd.read_csv(os.path.join(datapath, control_1500_csv), encoding='utf-8')
    treatment_tpus_set = set(list(treatment_data['SmallTPU']))
    control_1000_set = set(list(control_1000_data['SmallTPU'])) - treatment_tpus_set
    control_1500_set = set(list(control_1500_data['SmallTPU'])) - treatment_tpus_set
    return treatment_tpus_set, control_1000_set, control_1500_set


def select_dataframe_for_treatment_control(treatment_set, control_set, treatment_not_considered_set,
                                           datapath, return_dataframe=False):
    treatment_dataframe_list = []
    control_dataframe_list = []
    treatment_dataframe_not_considered_list = []
    for tpu_name in treatment_set:
        dataframe_treatment = pd.read_csv(os.path.join(datapath, tpu_name, tpu_name+'_data.csv'), encoding='utf-8',
                                quoting=csv.QUOTE_NONNUMERIC, dtype='str', index_col=0)
        dataframe_treatment['month_plus_year'] = dataframe_treatment.apply(
            lambda row: row['year']+'_'+row['month'], axis=1)
        treatment_dataframe_list.append(dataframe_treatment)
    for tpu_name in control_set:
        dataframe_control = pd.read_csv(os.path.join(datapath, tpu_name, tpu_name+'_data.csv'), encoding='utf-8',
                                        quoting=csv.QUOTE_NONNUMERIC, dtype='str')
        dataframe_control['month_plus_year'] = dataframe_control.apply(
            lambda row: row['year'] + '_' + row['month'], axis=1)
        control_dataframe_list.append(dataframe_control)
    for tpu_name in treatment_not_considered_set:
        dataframe_control = pd.read_csv(os.path.join(datapath, tpu_name, tpu_name+'_data.csv'), encoding='utf-8',
                                        quoting=csv.QUOTE_NONNUMERIC, dtype='str')
        dataframe_control['month_plus_year'] = dataframe_control.apply(
            lambda row: row['year'] + '_' + row['month'], axis=1)
        treatment_dataframe_not_considered_list.append(dataframe_control)
    combined_treatment = pd.concat(treatment_dataframe_list, axis=0)
    combined_control = pd.concat(control_dataframe_list, axis=0)
    combined_not_considered_treatment = pd.concat(treatment_dataframe_not_considered_list, axis=0)
    print(
        'The size of the treatment group {}; The size of the not considered treatment group {}; The size of the control group {}'.format(
            combined_treatment.shape, combined_not_considered_treatment.shape, combined_control.shape))
    if return_dataframe:
        return combined_treatment, combined_not_considered_treatment, combined_control
    else:
        treatment_sent_act_dict = sentiment_by_month(combined_treatment)
        control_sent_act_dict = sentiment_by_month(combined_control)
        not_considered_treatment_sent_act_dict = sentiment_by_month(combined_not_considered_treatment)
        return treatment_sent_act_dict, not_considered_treatment_sent_act_dict, control_sent_act_dict


def sort_data_based_on_date(df):
    df_time_index = df.set_index('Date')
    df_for_plot = df_time_index.loc[time_list]
    df_for_plot['Date'] = time_list
    final_df = df_for_plot.reset_index(drop=True)
    final_df_copy = final_df.copy()
    final_df_copy['sentiment'] = final_df_copy.apply(lambda row: row['Value'][0], axis=1)
    final_df_copy['activity'] = final_df_copy.apply(lambda row: row['Value'][1], axis=1)
    return final_df_copy


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

    longitudinal_saving_path = os.path.join(read_data.transit_non_transit_comparison_before_after,
                                            'tpu_data_with_visitors')
    TransitNeighborhood_Before_After.build_data_for_longitudinal_study(tweet_data_path=read_data.datasets,
                                                                       saving_path=longitudinal_saving_path)

    # List the TPUs in the treatment group and control
    kwun_tong_line_treatment_selected = {'213', '236', '243', '245'}
    kwun_tong_line_treatment = {'213', '215', '217', '226', '236', '237', '241', '243', '244', '245'}
    kwun_tong_line_treatment_not_considered = kwun_tong_line_treatment - kwun_tong_line_treatment_selected
    print('For Kwun Tong line, '
          'the treatment not considered TPUs are: {}'.format(kwun_tong_line_treatment_not_considered))
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
    south_horizons_lei_tung_treatment_not_considered = south_horizons_lei_tung_treatment - south_horizons_lei_tung_treatment_selected
    south_horizons_lei_tung_control_1000 = \
        {'172', '173', '174', '175', '176'} - south_horizons_lei_tung_treatment - {'175', '176'}
    south_horizons_lei_tung_control_1500 = {'172', '173', '174', '175', '176', '182'} - \
                                           south_horizons_lei_tung_treatment - {'175', '176'}
    print('For South Horizons & Lei Tung, '
          'the treatment not considered TPUs are: {}'.format(south_horizons_lei_tung_treatment_not_considered))
    print('----------------------------------------------------------------------------------')
    print('For Souths Horizons&Lei Tung Line Extension: the treatment group is: {}; '
          'the control group 1000-meter is: {}; the control group 1500-meter is: {}'.format(
        south_horizons_lei_tung_treatment_selected, south_horizons_lei_tung_control_1000,
        south_horizons_lei_tung_control_1500))
    print('----------------------------------------------------------------------------------\n')
    ocean_park_wong_chuk_hang_treatment_selected = {'175'}
    ocean_park_wong_chuk_hang_treatment = {'175', '176', '191'}
    ocean_park_wong_chuk_hang_treatment_not_considered = ocean_park_wong_chuk_hang_treatment - ocean_park_wong_chuk_hang_treatment_selected
    ocean_park_wong_chuk_hang_control_1000 = {'173', '174', '175', '176', '183', '191'} - \
                                             ocean_park_wong_chuk_hang_treatment - south_horizons_lei_tung_treatment
    ocean_park_wong_chuk_hang_control_1500 = {'173', '174', '175', '176', '182', '183', '184', '191'} - \
                                             ocean_park_wong_chuk_hang_treatment - south_horizons_lei_tung_treatment
    print('Ocean Park & Wong Chuk Hang, '
          'the treatment not considered TPUs are: {}'.format(ocean_park_wong_chuk_hang_treatment_not_considered))
    print('----------------------------------------------------------------------------------')
    print('For Wong Chuk Hang&Ocean Park Line Extension: the treatment group is: {}; '
          'the control group 1000-meter is: {}; the control group 1500-meter is: {}'.format(
        ocean_park_wong_chuk_hang_treatment_selected, ocean_park_wong_chuk_hang_control_1000,
        ocean_park_wong_chuk_hang_control_1500))
    print('----------------------------------------------------------------------------------\n')

    # Get the dataframe for treatment group and control group
    print('For Kwun Tong Line extension...')
    print('treatment vs 1000-meter control group')
    kwun_tong_line_treatment_dataframe, kwun_tong_line_treatment_not_considered_dataframe, kwun_tong_line_control_1000_dataframe = \
        select_dataframe_for_treatment_control(treatment_set=kwun_tong_line_treatment_selected,
                                               control_set=kwun_tong_line_control_1000,
                                               treatment_not_considered_set=kwun_tong_line_treatment_not_considered,
                                               datapath=os.path.join(read_data.transit_non_transit_comparison_before_after,
                                                                     'tpu_data_with_visitors'),
                                               return_dataframe=True)
    print('treatment vs 1500-meter control group')
    _, _, kwun_tong_line_control_1500_dataframe = \
        select_dataframe_for_treatment_control(treatment_set=kwun_tong_line_treatment_selected,
                                               control_set=kwun_tong_line_control_1500,
                                               treatment_not_considered_set=kwun_tong_line_treatment_not_considered,
                                               datapath=os.path.join(
                                                   read_data.transit_non_transit_comparison_before_after,
                                                   'tpu_data_with_visitors'), return_dataframe=True)
    print('For South Horizons & Lei Tung...')
    print('treatment vs 1000-meter control group')
    south_horizons_lei_tung_treatment_dataframe, south_horizons_lei_tung_treatment_not_considered_dataframe, south_horizons_lei_tung_control_1000_dataframe = \
        select_dataframe_for_treatment_control(treatment_set=south_horizons_lei_tung_treatment_selected,
                                               control_set=south_horizons_lei_tung_control_1000,
                                               treatment_not_considered_set=south_horizons_lei_tung_treatment_not_considered,
                                               datapath=os.path.join(read_data.transit_non_transit_comparison_before_after,
                                                   'tpu_data_with_visitors'),
                                               return_dataframe=True)
    print('treatment vs 1500-meter control group')
    _, _, south_horizons_lei_tung_control_1500_dataframe = \
        select_dataframe_for_treatment_control(treatment_set=south_horizons_lei_tung_treatment_selected,
                                               control_set=south_horizons_lei_tung_control_1500,
                                               treatment_not_considered_set=south_horizons_lei_tung_treatment_not_considered,
                                               datapath=os.path.join(
                                                   read_data.transit_non_transit_comparison_before_after,
                                                   'tpu_data_with_visitors'),
                                               return_dataframe=True)
    print('For Ocean Park & Wong Chuk Hang...')
    print('treatment vs 1000-meter control group')
    ocean_park_wong_chuk_hang_treatment_dataframe, ocean_park_wong_chuk_hang_treatment_not_considered_dataframe, ocean_park_wong_chuk_hang_control_1000_dataframe = \
        select_dataframe_for_treatment_control(treatment_set=ocean_park_wong_chuk_hang_treatment_selected,
                                               control_set=ocean_park_wong_chuk_hang_control_1000,
                                               treatment_not_considered_set=ocean_park_wong_chuk_hang_treatment_not_considered,
                                               datapath=os.path.join(
                                                   read_data.transit_non_transit_comparison_before_after,
                                                   'tpu_data_with_visitors'),
                                               return_dataframe=True
                                               )
    print('treatment vs 1500-meter control group')
    _, _, ocean_park_wong_chuk_hang_control_1500_dataframe = \
        select_dataframe_for_treatment_control(treatment_set=ocean_park_wong_chuk_hang_treatment_selected,
                                               control_set=ocean_park_wong_chuk_hang_control_1500,
                                               treatment_not_considered_set=ocean_park_wong_chuk_hang_treatment_not_considered,
                                               datapath=os.path.join(
                                                   read_data.transit_non_transit_comparison_before_after,
                                                   'tpu_data_with_visitors'),
                                               return_dataframe=True
                                               )

    # Draw the word count
    draw_word_count_histogram(df=kwun_tong_line_treatment_dataframe, station_name='Kwun_Tong_Line',
                              saved_file_name='Kwun_Tong_Line_tweet_word_count.png')
    draw_word_count_histogram(df=south_horizons_lei_tung_treatment_dataframe,
                              station_name='south_horizons_lei_tung',
                              saved_file_name='South_horizons_lei_tung_line_tweet_word_count.png')
    draw_word_count_histogram(df=ocean_park_wong_chuk_hang_treatment_dataframe,
                              station_name='Ocean_park_wong_chuk_hang',
                              saved_file_name='Ocean_park_wong_chuk_hang_tweet_word_count.png')

    kwun_tong_line_extension_1000_control = TransitNeighborhood_Before_After(name = 'Kwun_Tong_Line',
        tn_dataframe=kwun_tong_line_treatment_dataframe,
        non_tn_dataframe=kwun_tong_line_control_1000_dataframe,
        treatment_not_considered_dataframe=kwun_tong_line_treatment_not_considered_dataframe,
        oct_open=True, before_and_after=True,
        compute_positive=False,
        compute_negative=False)
    kwun_tong_line_extension_1500_control = TransitNeighborhood_Before_After(name = 'Kwun_Tong_Line',
        tn_dataframe=kwun_tong_line_treatment_dataframe,
        non_tn_dataframe=kwun_tong_line_control_1500_dataframe,
        treatment_not_considered_dataframe=kwun_tong_line_treatment_not_considered_dataframe,
        oct_open=True, before_and_after=True,
        compute_positive=False,
        compute_negative=False)
    south_horizons_lei_tung_1000_control = TransitNeighborhood_Before_After(name='South_Horizons_Lei_Tung',
        tn_dataframe=south_horizons_lei_tung_treatment_dataframe,
        non_tn_dataframe=south_horizons_lei_tung_control_1000_dataframe,
        treatment_not_considered_dataframe=south_horizons_lei_tung_treatment_not_considered_dataframe,
        oct_open=False, before_and_after=True,
        compute_positive=False,
        compute_negative=False)
    south_horizons_lei_tung_1500_control = TransitNeighborhood_Before_After(name='South_Horizons_Lei_Tung',
        tn_dataframe=south_horizons_lei_tung_treatment_dataframe,
        non_tn_dataframe=south_horizons_lei_tung_control_1500_dataframe,
        treatment_not_considered_dataframe=south_horizons_lei_tung_treatment_not_considered_dataframe,
        oct_open=False, before_and_after=True,
        compute_positive=False,
        compute_negative=False)
    ocean_park_wong_chuk_hang_1000_control = TransitNeighborhood_Before_After(name='Ocean_Park_Wong_Chuk_Hang',
        tn_dataframe=ocean_park_wong_chuk_hang_treatment_dataframe,
        non_tn_dataframe=ocean_park_wong_chuk_hang_control_1000_dataframe,
        treatment_not_considered_dataframe=ocean_park_wong_chuk_hang_treatment_not_considered_dataframe,
        oct_open=False, before_and_after=True,
        compute_positive=False,
        compute_negative=False)
    ocean_park_wong_chuk_hang_1500_control = TransitNeighborhood_Before_After(name='Ocean_Park_Wong_Chuk_Hang',
        tn_dataframe=ocean_park_wong_chuk_hang_treatment_dataframe,
        non_tn_dataframe=ocean_park_wong_chuk_hang_control_1500_dataframe,
        treatment_not_considered_dataframe=ocean_park_wong_chuk_hang_treatment_not_considered_dataframe,
        oct_open=False, before_and_after=True,
        compute_positive=False,
        compute_negative=False)

    kwun_tong_line_extension_1000_control.draw_tweet_posting_time_comparison(
        saving_path=read_data.transit_non_transit_comparison_before_after)
    south_horizons_lei_tung_1500_control.draw_tweet_posting_time_comparison(
        saving_path=read_data.transit_non_transit_comparison_before_after)
    ocean_park_wong_chuk_hang_1500_control.draw_tweet_posting_time_comparison(
        saving_path=read_data.transit_non_transit_comparison_before_after)

    #Kwun Tong Line - Overall Comparison between treatment and control group
    kwun_tong_line_extension_1000_control.line_map_comparison(
        line_labels=('Treatment Group', 'Treatment Group Not Considered', 'Control Group(1000 meter)'),
        ylabel='Percentage of Positive Tweets Minus Percentage of Negative Tweets',
        draw_sentiment=True, plot_title_name='Kwun Tong Line Treatment Group and Control Group Comparison',
        saving_file_name='kwun_tong_line_sentiment_treatment_control_comparison_1000_meter.png')
    kwun_tong_line_extension_1500_control.line_map_comparison(
        line_labels=('Treatment Group', 'Treatment Group Not Considered', 'Control Group(1000 meter)'),
        ylabel='Percentage of Positive Tweets Minus Percentage of Negative Tweets',
        draw_sentiment=True, plot_title_name='Kwun Tong Line Treatment Group and Control Group Comparison',
        saving_file_name='kwun_tong_line_sentiment_treatment_control_comparison_1500_meter.png')
    kwun_tong_line_extension_1000_control.line_map_comparison(
        line_labels=('Treatment Group',  'Treatment Group Not Considered', 'Control Group(1000 meter)'),
        ylabel='Number of Posted Tweets Per Square Meter',
        draw_sentiment=False, plot_title_name='Kwun Tong Line Treatment Group and Control Group Comparison',
        saving_file_name='kwun_tong_line_activity_treatment_control_comparison_1000_meter.png')
    kwun_tong_line_extension_1500_control.line_map_comparison(
        line_labels=('Treatment Group', 'Treatment Group Not Considered', 'Control Group(1000 meter)'),
        ylabel='Number of Posted Tweets Per Square Meter',
        draw_sentiment=False, plot_title_name='Kwun Tong Line Treatment Group and Control Group Comparison',
        saving_file_name='kwun_tong_line_activity_treatment_control_comparison_1500_meter.png')

    # South Horizons and Lei Tung: Comparison between treatment group and control group
    # south_horizons_lei_tung_1000_control.line_map_comparison(
    #     line_labels=('Treatment Group', 'Control Group(1000 meter)'),
    #     ylabel='Percentage of Positive Tweets Minus Percentage of Negative Tweets',
    #     draw_sentiment=True, plot_title_name='South Horizons & Lei Tung Treatment Group and Control Group Comparison',
    #     saving_file_name='south_horizons_lei_tung_sentiment_treatment_control_comparison_1000_meter.png')
    south_horizons_lei_tung_1500_control.line_map_comparison(
        line_labels=('Treatment Group', 'Treatment Group Not Considered', 'Control Group(1500 meter)'),
        ylabel='Percentage of Positive Tweets Minus Percentage of Negative Tweets',
        draw_sentiment=True, plot_title_name='South Horizons & Lei Tung Treatment Group and Control Group Comparison',
        saving_file_name='south_horizons_lei_tung_sentiment_treatment_control_comparison_1500_meter.png')
    # south_horizons_lei_tung_1000_control.line_map_comparison(
    #     line_labels=('Treatment Group', 'Control Group(1000 meter)'),
    #     ylabel='Number of Posted Tweets(log10)',
    #     draw_sentiment=False, plot_title_name='South Horizons & Lei Tung Treatment Group and Control Group Comparison',
    #     saving_file_name='south_horizons_lei_tung_activity_treatment_control_comparison_1000_meter.png')
    south_horizons_lei_tung_1500_control.line_map_comparison(
        line_labels=('Treatment Group', 'Treatment Group Not Considered', 'Control Group(1500 meter)'),
        ylabel='Number of Posted Tweets Per Square Meter',
        draw_sentiment=False, plot_title_name='South Horizons & Lei Tung Treatment Group and Control Group Comparison',
        saving_file_name='south_horizons_lei_tung_activity_treatment_control_comparison_1500_meter.png')

    # Ocean Park and Wong Chuk Hang: Comparison between treatment group and control group
    # ocean_park_wong_chuk_hang_1000_control.line_map_comparison(
    #     line_labels=('Treatment Group', 'Control Group(1000 meter)'),
    #     ylabel='Percentage of Positive Tweets Minus Percentage of Negative Tweets',
    #     draw_sentiment=True, plot_title_name='Ocean Park & Wong Chuk Hang Treatment Group and Control Group Comparison',
    #     saving_file_name='ocean_park_wong_chuk_hang_sentiment_treatment_control_comparison_1000_meter.png'
    # )
    ocean_park_wong_chuk_hang_1500_control.line_map_comparison(
        line_labels=('Treatment Group', 'Treatment Group Not Considered', 'Control Group(1500 meter)'),
        ylabel='Percentage of Positive Tweets Minus Percentage of Negative Tweets',
        draw_sentiment=True, plot_title_name='Ocean Park & Wong Chuk Hang Treatment Group and Control Group Comparison',
        saving_file_name='ocean_park_wong_chuk_hang_sentiment_treatment_control_comparison_1500_meter.png'
    )
    # ocean_park_wong_chuk_hang_1000_control.line_map_comparison(
    #     line_labels=('Treatment Group', 'Control Group(1000 meter)'),
    #     ylabel='Number of Tweets(log10)',
    #     draw_sentiment=False, plot_title_name='Ocean Park & Wong Chuk Hang Treatment Group and Control Group Comparison',
    #     saving_file_name='ocean_park_wong_chuk_hang_activity_treatment_control_comparison_1000_meter.png'
    # )
    ocean_park_wong_chuk_hang_1500_control.line_map_comparison(
        line_labels=('Treatment Group', 'Treatment Group Not Considered', 'Control Group(1500 meter)'),
        ylabel='Number of Tweets Per Square Meter',
        draw_sentiment=False, plot_title_name='Ocean Park & Wong Chuk Hang Treatment Group and Control Group Comparison',
        saving_file_name='ocean_park_wong_chuk_hang_activity_treatment_control_comparison_1500_meter.png'
    )

    # Save the dataframes
    treatment_control_saving_path = os.path.join(read_data.transit_non_transit_comparison_before_after,
                                                 'three_areas_longitudinal_analysis')
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
    print('***treatment vs 1000 control****')
    kwun_tong_line_extension_1000_control.compute_abs_coeff_difference()
    print('***treatment vs 1500 control****')
    kwun_tong_line_extension_1500_control.compute_abs_coeff_difference()

    # Compare the tpus in treatment group with the control group
    kwun_tong_line_extension_1000_control.line_map_comparison_between_tpus_and_control('245', '213', '236', '243')
    kwun_tong_line_extension_1500_control.line_map_comparison_between_tpus_and_control('245', '213', '236', '243')
    south_horizons_lei_tung_1500_control.line_map_comparison_between_tpus_and_control('174')
    ocean_park_wong_chuk_hang_1500_control.line_map_comparison_between_tpus_and_control('175')

    # #=========================================Build the wordcloud============================================
    before_text_kwun_tong_line, after_text_kwun_tong_line = build_text_for_wordcloud_topic_model(
        kwun_tong_line_treatment_dataframe,
        oct_open=True, build_wordcloud=True)
    before_text_south_horizons_lei_tung, after_text_south_horizons_lei_tung = build_text_for_wordcloud_topic_model(
        south_horizons_lei_tung_treatment_dataframe,
        oct_open=False, build_wordcloud=True)
    before_text_ocean_park_wong_chuk_hang, after_text_ocean_park_wong_chuk_hang = build_text_for_wordcloud_topic_model(
        ocean_park_wong_chuk_hang_treatment_dataframe,
        oct_open=False, build_wordcloud=True)

    generate_wordcloud(before_text_kwun_tong_line, after_text_kwun_tong_line, mask=wordcloud_generate.circle_mask,
                       file_name_before='before_kwun_tong_line_wordcloud',
                       file_name_after="after_kwun_tong_line_wordcloud",
                       color_func=wordcloud_generate.green_func)
    generate_wordcloud(before_text_south_horizons_lei_tung, after_text_south_horizons_lei_tung,
                       mask=wordcloud_generate.circle_mask,
                       file_name_before='before_south_horizons_lei_tung_wordcloud',
                       file_name_after="after_south_horizons_lei_tung_wordcloud",
                       color_func=wordcloud_generate.green_func)
    generate_wordcloud(before_text_ocean_park_wong_chuk_hang, after_text_ocean_park_wong_chuk_hang,
                       mask=wordcloud_generate.circle_mask,
                       file_name_before='before_ocean_park_wong_chuk_hang_wordcloud',
                       file_name_after="after_ocean_park_wong_chuk_hang_wordcloud",
                       color_func=wordcloud_generate.green_func)
    # #=========================================================================================================
    #
    # #=======================================Topic Modelling===================================================
    before_dataframe_kwun_tong_line, after_dataframe_kwun_tong_line = \
        build_text_for_wordcloud_topic_model(kwun_tong_line_treatment_dataframe, oct_open=True, build_wordcloud=False)
    before_dataframe_south_horizons_lei_tung, after_dataframe_south_horizons_lei_tung = \
        build_text_for_wordcloud_topic_model(south_horizons_lei_tung_treatment_dataframe, oct_open=True,
                                             build_wordcloud=False)
    before_dataframe_ocean_park_wong_chuk_hang, after_dataframe_ocean_park_wong_chuk_hang = build_text_for_wordcloud_topic_model(
        ocean_park_wong_chuk_hang_treatment_dataframe, oct_open=True, build_wordcloud=False)


    before_and_after_dataframes_list = [before_dataframe_kwun_tong_line, after_dataframe_kwun_tong_line,
                                        before_dataframe_south_horizons_lei_tung, after_dataframe_south_horizons_lei_tung,
                                        before_dataframe_ocean_park_wong_chuk_hang,
                                        after_dataframe_ocean_park_wong_chuk_hang]

    name_list = ['before_kwun_tong_line', 'after_kwun_tong_line', 'before_south_horizons_lei_tung',
                 'after_south_horizons_lei_tung', 'before_ocean_park_wong_chuk_hang',
                 'after_ocean_park_wong_chuk_hang']

    for dataframe, file_name in zip(before_and_after_dataframes_list, name_list):
        print('-------------------' + file_name + ' starts--------------------------')
        build_topic_model(df=dataframe, keyword_file_name=file_name + '_keyword.pkl',
                          topic_predict_file_name=file_name + '_tweet_topic.pkl',
                          saving_path=read_data.before_and_after_topic_modelling_compare)
        print('------------------' + file_name + ' ends-----------------------------')
    # =================================================================================================================