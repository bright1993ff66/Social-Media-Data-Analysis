import pandas as pd
import os
import numpy as np
import pytz
from datetime import datetime
from matplotlib import pyplot as plt

import before_and_after
import read_data


months = before_and_after.months
time_list = before_and_after.time_list
# A list of the stations we consider in the cross sectional study
cross_sectional_study_station_names = ['Admiralty', 'Airport', 'Austin', 'Causeway Bay', 'Central',
                                       'Disneyland', 'East Tsim Sha Tsui', 'HKU', 'Hong Kong', 'Hung Hom',
                                       'Jordan', 'Kennedy Town', 'Kowloon', 'Kwun Tong', 'Mong Kok',
                                       'Mong Kok East', 'Prince Edward', 'Sai Ying Pun', 'Sheung Wan',
                                       'Tai Koo', 'Tin Hau', 'Tsim Sha Tsui', 'Wan Chai', 'Whampoa',
                                       'Yau Ma Tei']
region_one = ['Airport', 'Disneyland']
region_two = ['Kennedy Town', 'HKU',  'Sai Ying Pun', 'Sheung Wan', 'Hong Kong', 'Central', 'Admiralty',
              'Wan Chai', 'Causeway Bay', 'Tin Hau', 'Tai Koo']
region_three = ['Austin', 'East Tsim Sha Tsui', 'Hung Hom', 'Jordan', 'Kowloon', 'Mong Kok',
                'Mong Kok East', 'Prince Edward', 'Tsim Sha Tsui',  'Whampoa', 'Yau Ma Tei']
region_four = ['Kwun Tong']


# Hong Kong and Shanghai share the same time zone.
# Hence, we transform the utc time in our dataset into Shanghai time
time_zone_hk = pytz.timezone('Asia/Shanghai')
tns_path = os.path.join(read_data.transit_non_transit_comparison_cross_sectional, 'tns_dataframe')


class TransitNeighborhood_cross_sectional(object):

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
        result_dict_tn = before_and_after.sentiment_by_month(self.tn_dataframe,
                                                             compute_positive_percent=self.compute_positive,
                                                             compute_negative_percent=self.compute_negative)
        result_dict_non_tn = before_and_after.sentiment_by_month(self.non_tn_dataframe,
                                                                 compute_positive_percent=self.compute_positive,
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
            y1 = [value[1] for value in list(dataframe_for_plot['Value'])]
            y2 = [value[1] for value in list(tpu_dataframe_for_plot['Value'])]

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        lns1 = ax.plot(x, y1, 'g-', label=line_labels[0], linestyle='--', marker='o')
        lns2 = ax.plot(x, y2, 'y-', label=line_labels[1], linestyle='--', marker='^')
        # Whether to draw the vertical line which indicates the open date
        if self.before_and_after:
            if self.oct_open:
                plt.axvline(3.77, color='black')
            else:
                plt.axvline(5.95, color='black')
        else:
            pass

        # Add the legend
        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs)

        # Draw the average of the sentiment level
        if draw_sentiment:
            ax.axhline(y=0.43, color='r', linestyle='solid')
        else:  # here I don't want to draw the horizontal activity line as the activity level varies greatly between TNs
            pass

        ax.set_xlabel('Time')
        ax.set_ylabel(ylabel, color='g')
        ax.set_xticks(x)
        ax.set_xticklabels(time_list, rotation='vertical')
        plt.title(plot_title_name)
        plt.savefig(os.path.join(read_data.transit_non_transit_comparison_cross_sectional, saving_file_name))
        plt.show()


def create_dataframe_dicts_for_four_regions(region1_name_list, region2_name_list, region3_name_list, region4_name_list):
    region1_dict = {}
    region2_dict = {}
    region3_dict = {}
    region4_dict = {}
    regions_name_list = [region1_name_list, region2_name_list, region3_name_list, region4_name_list]
    for index, region in enumerate(regions_name_list):
        for station_name in region:
            dataframe =  before_and_after.get_tweets_based_on_date(station_name, start_date, end_date)
            dataframe.to_pickle(os.path.join(tns_path, station_name+'.pkl'))
            if index+1 == 1:
                region1_dict[station_name] = dataframe
            elif index+1 == 2:
                region2_dict[station_name] = dataframe
            elif index+1 == 3:
                region3_dict[station_name] = dataframe
            else:
                region4_dict[station_name] = dataframe
    return region1_dict, region2_dict, region3_dict, region4_dict


if __name__ == '__main__':
    october_23_start = datetime(2016, 10, 23, 0, 0, 0, tzinfo=time_zone_hk)
    october_23_end = datetime(2016, 10, 23, 23, 59, 59, tzinfo=time_zone_hk)
    december_28_start = datetime(2016, 12, 28, 0, 0, 0, tzinfo=time_zone_hk)
    december_28_end = datetime(2016, 12, 28, 23, 59, 59, tzinfo=time_zone_hk)
    start_date = datetime(2016, 5, 7, tzinfo=time_zone_hk)
    end_date = datetime(2017, 12, 31, tzinfo=time_zone_hk)

    region1_dataframe_dict, region2_dataframe_dict, region3_dataframe_dict, region4_dataframe_dict = \
        create_dataframe_dicts_for_four_regions(region_one, region_two, region_three, region_four)

    non_tn_path = os.path.join(read_data.transit_non_transit_comparison_cross_sectional, 'non_tns_dataframe')
    region1_nontn_dataframe = pd.read_csv(os.path.join(non_tn_path, 'region_one_tpu_nontn.csv'),
                                          encoding='latin-1')
    region2_nontn_dataframe = pd.read_csv(os.path.join(non_tn_path, 'region_two_tpu_nontn.csv'),
                                          encoding='latin-1')
    region3_nontn_dataframe = pd.read_csv(os.path.join(non_tn_path, 'region_three_tpu_nontn.csv'),
                                          encoding='latin-1')
    region4_nontn_dataframe = pd.read_csv(os.path.join(non_tn_path, 'region_four_tpu_nontn.csv'),
                                          encoding='latin-1')

    for name in list(region1_dataframe_dict.keys()):
        tn_object = TransitNeighborhood_cross_sectional(tn_dataframe=region1_dataframe_dict[name],
                                                        non_tn_dataframe=region1_nontn_dataframe,
                                                     before_and_after=False, oct_open=False, compute_positive=False,
                                                     compute_negative=False)
        tn_object.line_map_comparison(line_labels=('Sentiment Level of '+name+' TN', 'Sentiment Level of Non-TN'),
                                   ylabel='Percentage of Positive Tweets Minus Percentage of Negative Tweets',
                                   plot_title_name='Sentiment Cross Sectional Study: '+name,
                                   saving_file_name=name+'_sent_compare.png', draw_sentiment=True)
        tn_object.line_map_comparison(line_labels=('Activity Level of ' + name + ' TN', 'Activity Level of Non-TN'),
                                      ylabel='Number of Posted Tweets',
                                      plot_title_name='Activity Cross Sectional Study: ' + name,
                                      saving_file_name=name + '_act_compare.png', draw_sentiment=False)

    for name in list(region2_dataframe_dict.keys()):
        tn_object = TransitNeighborhood_cross_sectional(tn_dataframe=region2_dataframe_dict[name],
                                                        non_tn_dataframe=region2_nontn_dataframe,
                                                     before_and_after=False, oct_open=False, compute_positive=False,
                                                     compute_negative=False)
        tn_object.line_map_comparison(line_labels=('Sentiment Level of '+name+' TN', 'Sentiment Level of Non-TN'),
                                   ylabel='Percentage of Positive Tweets Minus Percentage of Negative Tweets',
                                   plot_title_name='Sentiment Cross Sectional Study: '+name,
                                   saving_file_name=name+'_sent_compare.png', draw_sentiment=True)
        tn_object.line_map_comparison(line_labels=('Activity Level of ' + name + ' TN', 'Activity Level of Non-TN'),
                                      ylabel='Number of Posted Tweets',
                                      plot_title_name='Activity Cross Sectional Study: ' + name,
                                      saving_file_name=name + '_act_compare.png', draw_sentiment=False)

    for name in list(region3_dataframe_dict.keys()):
        tn_object = TransitNeighborhood_cross_sectional(tn_dataframe=region3_dataframe_dict[name],
                                                        non_tn_dataframe=region3_nontn_dataframe,
                                                     before_and_after=False, oct_open=False, compute_positive=False,
                                                     compute_negative=False)
        tn_object.line_map_comparison(line_labels=('Sentiment Level of '+name+' TN', 'Sentiment Level of Non-TN'),
                                   ylabel='Percentage of Positive Tweets Minus Percentage of Negative Tweets',
                                   plot_title_name='Sentiment Cross Sectional Study: '+name,
                                   saving_file_name=name+'_sent_compare.png', draw_sentiment=True)
        tn_object.line_map_comparison(line_labels=('Activity Level of ' + name + ' TN', 'Activity Level of Non-TN'),
                                      ylabel='Number of Posted Tweets',
                                      plot_title_name='Activity Cross Sectional Study: ' + name,
                                      saving_file_name=name + '_act_compare.png', draw_sentiment=False)

    for name in list(region4_dataframe_dict.keys()):
        tn_object = TransitNeighborhood_cross_sectional(tn_dataframe=region4_dataframe_dict[name],
                                                        non_tn_dataframe=region4_nontn_dataframe,
                                                     before_and_after=False, oct_open=False, compute_positive=False,
                                                     compute_negative=False)
        tn_object.line_map_comparison(line_labels=('Sentiment Level of '+name+' TN', 'Sentiment Level of Non-TN'),
                                   ylabel='Percentage of Positive Tweets Minus Percentage of Negative Tweets',
                                   plot_title_name='Sentiment Cross Sectional Study: '+name,
                                   saving_file_name=name+'_sent_compare.png', draw_sentiment=True)
        tn_object.line_map_comparison(line_labels=('Activity Level of ' + name + ' TN', 'Activity Level of Non-TN'),
                                      ylabel='Number of Posted Tweets',
                                      plot_title_name='Activity Cross Sectional Study: ' + name,
                                      saving_file_name=name + '_act_compare.png', draw_sentiment=False)

