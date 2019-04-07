import pandas as pd
import os
import numpy as np
import read_data
import time
from datetime import datetime, timedelta
import pytz

import tweepy

# An API which is used to check whether an account is bot
import botometer

from collections import Counter

target_path = read_data.tweet_2016

# The key to run the botometer: https://github.com/IUNetSci/botometer-python
mashape_key = "XXXXX"

# The Twitter Developer Key - Got from: https://developer.twitter.com/en/apps
twitter_app_auth = {
    'consumer_key': 'XXXXX',
    'consumer_secret': 'XXXXX',
    'access_token': 'XXXXX',
    'access_token_secret': 'XXXXX',
  }

bom = botometer.Botometer(wait_on_ratelimit=True,
                          mashape_key=mashape_key,
                          **twitter_app_auth)

# Hong Kong and Shanghai share the same time zone.
# Hence, we transform the utc time in our dataset into Shanghai time
time_zone_hk = pytz.timezone('Asia/Shanghai')

# Function used to output a pandas dataframe for each user based on the user account number
def derive_dataframe_for_each_user(df, all_users):
    dataframes = []
    for user in all_users:
        dataframes.append(df[df['user_id_str']==user])
    return dataframes


# Based on the dataframe for each user, compute the time range between his or her first tweet and last tweet
def compute_time_range_for_one_user(df):
    user_id_str = list(df['user_id_str'])[0]
    first_row = list(df.head(1)['created_at'])[0]
    end_row = list(df.tail(1)['created_at'])[0]
    datetime_object_first_row = datetime.strptime(first_row, '%a %b %d %H:%M:%S %z %Y')
    datetime_object_last_row = datetime.strptime(end_row, '%a %b %d %H:%M:%S %z %Y')
    # add 8 hours
    first_time = datetime_object_first_row+timedelta(hours=8)
    last_time = datetime_object_last_row+timedelta(hours=8)
    time_range = last_time - first_time
    return (user_id_str, time_range.days)


# Check whether an account is bot or not based on the account number
# The id_str should be an integer
def check_bot(id_str):
    result = bom.check_account(int(id_str))
    return result['cap']['universal']


# The time of tweet we have collected is recorded as the UTC time
# Add 8 hours to get the Hong Kong time
def get_hk_time(df):
    changed_time_list = []
    for _, row in df.iterrows():
        time_to_change = datetime.strptime(row['created_at'], '%a %b %d %H:%M:%S %z %Y')
        # get the hk time
        changed_time = time_to_change.astimezone(time_zone_hk)
        changed_time_list.append(changed_time)
    df['hk_time'] = changed_time_list
    return df


def get_month_hk_time(timestamp):
    """
    :param timestamp: timestamp variable after passing the pandas dataframe to add_eight_hours function
    :return: when the tweet is posted
    """
    month_int = timestamp.month
    if month_int == 1:
        result = 'Jan'
    elif month_int == 2:
        result = 'Feb'
    elif month_int == 3:
        result = 'Mar'
    elif month_int == 4:
        result = 'Apr'
    elif month_int == 5:
        result = 'May'
    elif month_int == 6:
        result = 'Jun'
    elif month_int == 7:
        result = 'Jul'
    elif month_int == 8:
        result = 'Aug'
    elif month_int == 9:
        result = 'Sep'
    elif month_int == 10:
        result = 'Oct'
    elif month_int == 11:
        result = 'Nov'
    else:
        result = 'Dec'
    return result


if __name__ == '__main__':

    start_time = time.time()
    print('Data generation starts.....')
    whole_data = pd.read_pickle(os.path.join(target_path, 'raw_tweets_2016.pkl'))
    # 1. Only consider the English and Chinese tweets
    whole_data_zh_en = whole_data.loc[whole_data['lang'].isin(['zh', 'en'])]
    # 2. Delete the verified accounts
    whole_data_without_verified = whole_data_zh_en.loc[whole_data_zh_en['verified'].isin([False])]
    # 3. Only keep the tweets which have geoinformation
    whole_data_geocoded = whole_data_without_verified.dropna(axis=0, subset=['lat'])
    # 4. Remove the Travelers
    # Based on the regulations in Hong Kong, travelers could only stay at most 7 days
    all_users = set(list(whole_data_geocoded['user_id_str']))
    dataframes = derive_dataframe_for_each_user(whole_data_geocoded, all_users)
    time_range_list = []

    for df in dataframes:
        # Each value in time_range_list contains the user_id_str and the maximum time range betweet the first
        # tweet and the last tweet
        time_range_list.append(compute_time_range_for_one_user(df))

    locals = []

    for day in time_range_list:
        if day[1] > 7:
            locals.append(day[0])

    tweet_2016_zh_en_local = whole_data_geocoded[whole_data_geocoded['user_id_str'].isin(locals)]
    tweet_2016_zh_en_local.to_pickle(os.path.join(read_data.desktop, 'local_tweets_with_bots.pkl'))

    print()
    print('Data generation ends.....')
    print()

    # 5. Remove the bots
    print('Check bots starts....')
    bot_result_list = []
    accounts = list(tweet_2016_zh_en_local['user_id_str'])
    # The input of the check bot function should be integers
    account_integers = [int(number) for number in accounts]
    # Get a set of unique users and transform it to list
    account_integer_set_list = list(set(account_integers))

    print()
    print('===========================================================')
    print('The total number of users is: ', len(account_integer_set_list))
    print('===========================================================')
    print()

    for index in range(len(account_integer_set_list)):
        try:
            bot_result_list.append(check_bot(account_integer_set_list[index]))
            print('The ', index + 1, 'th account out of ', len(account_integer_set_list), ' is done...')
        except:
            bot_result_list.append(1)
            # In this case, the api shows that this account does not exit
            # We choose not to consider these accounts
            print('Something wrong with the ', index + 1, 'th account')

    check_bot_dataframe = pd.DataFrame({'account':account_integer_set_list, 'bot_score': bot_result_list})
    check_bot_dataframe.to_pickle(os.path.join(read_data.desktop, 'check_bot_dataframe.pkl'))

    dataframe_without_bot = check_bot_dataframe.loc[check_bot_dataframe['bot_score'] < 0.4]
    selected_accounts = list(dataframe_without_bot['account'])
    tweet_2016_zh_en_local_without_bot = \
        tweet_2016_zh_en_local.loc[tweet_2016_zh_en_local['user_id_str'].isin(selected_accounts)]
    # We forgot add eight hours when first generating the raw_tweets_final.pkl file
    # run the add_eight_hours function on raw_tweets_final.pkl file and delete these two lines of comments
    tweet_2016_zh_en_local_without_bot_hk_time = get_hk_time(tweet_2016_zh_en_local_without_bot)
    tweet_2016_zh_en_local_without_bot_hk_time['month'] = tweet_2016_zh_en_local_without_bot_hk_time.apply(
        lambda row: get_month_hk_time(row['hk_time']), axis=1)
    tweet_2016_zh_en_local_without_bot_hk_time.to_pickle(os.path.join(target_path, 'raw_tweets_final.pkl'))
    
    end_time = time.time()
    print('Check bots ends....')
    print("Total time is: ", end_time-start_time)

    # print(check_bot(19011829))
