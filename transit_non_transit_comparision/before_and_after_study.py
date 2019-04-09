import pandas as pd
import numpy as np
import os
import pytz
from datetime import datetime

import read_data


months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Hong Kong and Shanghai share the same time zone.
# Hence, we transform the utc time in our dataset into Shanghai time
time_zone_hk = pytz.timezone('Asia/Shanghai')

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
    df['month_plus_year'] = df.apply(lambda row: str(row['hk_time'].year)+'_'+str(row['hk_time'].month), axis=1)
    dataframe_dict = {}
    for time, dataframe_by_time in df.groupby('month_plus_year'):
        dataframe_dict[time] = dataframe_by_time
    time_list = list(dataframe_dict.keys())
    tweet_month_sentiment = {}
    for time in time_list:
        if compute_positive_percent:
            tweet_month_sentiment[time] = (positive_percent(dataframe_dict[time]), dataframe_dict[time].shape[0])
        else:
            tweet_month_sentiment[time] = (pos_percent_minus_neg_percent(dataframe_dict[time]),
                                           dataframe_dict[time].shape[0])
    return tweet_month_sentiment


def get_tweets_based_on_date(station_name, start_date, end_date):
    dataframe_2016 = pd.read_pickle(os.path.join(read_data.station_related_path_2016, station_name+'.pkl'))
    dataframe_2017 = pd.read_pickle(os.path.join(read_data.station_related_path_zh_en, station_name+'.pkl'))
    # select relevant columns
    dataframe_2016 = dataframe_2016[['user_id_str', 'hk_time', 'month', 'lang', 'cleaned_text', 'text',
                                     'sentiment', 'url', 'lat', 'lon']]
    dataframe_2017 = dataframe_2017[['user_id_str', 'hk_time', 'month', 'lang', 'cleaned_text', 'text',
                                     'sentiment', 'url', 'lat', 'lon']]
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
    before_sentiment_dict = sentiment_by_month(before_dataframe, compute_positive_percent=False)
    after_sentiment_dict = sentiment_by_month(after_dataframe, compute_positive_percent=False)
    return before_sentiment_dict, after_sentiment_dict


def output_sentiment_comparision_dataframe(station_name, tweet_start_date, tweet_end_date, before_date, after_date,
                                           saving_path):
    whole_dataframe = get_tweets_based_on_date(station_name, tweet_start_date, tweet_end_date)
    before_dict, after_dict = compare_sentiment(whole_dataframe, before_date, after_date)
    print('For the ', station_name, ', before the openning date....')
    print(before_dict)
    print('After the openning date...')
    print(after_dict)
    dataframe_before = pd.DataFrame(list(before_dict.items()), columns=['Date', 'Value'])
    dataframe_before.to_csv(os.path.join(saving_path, station_name+'_before.pkl'))
    dataframe_after = pd.DataFrame(list(after_dict.items()), columns=['Date', 'Value'])
    dataframe_after.to_csv(os.path.join(saving_path, station_name+'_after.pkl'))


if __name__ == '__main__':

    # For instance, if we want to compare the sentiment and activity level before and after the
    # openning date of the Whampoa MTR railway station in Hong Kong, since the station is opened on 23 Oct 2016,
    # we could specify the openning date using datatime package and output before and after dataframes which record
	# the sentiment and activity level in each month
    october_23_start = datetime(2016, 10, 23, 0, 0, tzinfo=time_zone_hk)
    october_23_end = datetime(2016, 10, 23, 23, 59, tzinfo=time_zone_hk)
	# tweets we consider starting from 7th May 2016 to 31st Dec 2017
    start_date_whampoa = datetime(2016, 5, 7, tzinfo=time_zone_hk)
    end_date_whampoa = datetime(2017, 12, 31, tzinfo=time_zone_hk)

    output_sentiment_comparision_dataframe('Whampoa', tweet_start_date=start_date_whampoa,
                                           tweet_end_date=end_date_whampoa, before_date=october_23_start,
                                           after_date=october_23_end, saving_path=read_data.desktop)

