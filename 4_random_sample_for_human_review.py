# Commonly used
import numpy as np
import pandas as pd
import os
import read_data
import string
from collections import Counter
import re
from datetime import datetime, timedelta
from codecs import encode

# A package for preprocessing(officially used in SemEval NLP competition)
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

from sklearn.utils import shuffle


def remove_u_plus(text):
    result = re.sub(pattern=r'U\+00', repl=r'', string=text)
    return result


def encode_decode(text):
    result = text.encode('unicode_escape').decode('utf-8')
    return result


# The time of tweet we have collected is recorded as the UTC time
# Add 8 hours to get the Hong Kong time
def add_eight_hours(df):
    changed_time_list = []
    for _, row in df.iterrows():
        time_to_change = datetime.strptime(row['created_at'], '%a %b %d %H:%M:%S %z %Y')
        # add 8 hours
        changed_time = time_to_change + timedelta(hours=8)
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


def show_emoji_in_tweet(text, emoji_dictionary):
    without_u = remove_u_plus(text)
    old_text = without_u
    old_text = old_text.encode('unicode_escape').decode('utf-8')
    result1 = re.sub(pattern='\\\\r', repl='', string=old_text)
    result2 = re.sub(pattern='\\\\n', repl='', string=result1)
    result3 = re.sub(pattern='\\\\x([a-z0-9]{2})', repl = '<\\1>', string=result2)
    old_text = result3
    for _, row in emoji_dictionary.iterrows():
        if row['R_Encoding'] in old_text:
            new_text = re.sub(pattern=row['R_Encoding'], repl=row['emoji'], string=old_text)
            old_text = new_text
        else:
            pass
        if row['R_Encoding_lower'] in old_text:
            new_text = re.sub(pattern=row['R_Encoding_lower'], repl=row['emoji'], string=old_text)
            old_text = new_text
        else:
            pass
    return old_text


def show_chinese_step1(text, emoji_dataset):
    result1 = re.sub('\<u\+', '\\'+'u', text.lower())
    result2 = re.sub('\>', '', result1)
    all_chars = result2.split()
    new_all_chars = []
    for char in all_chars:
        emoji_in_char = False
        for emoji in list(emoji_dataset['emoji']):
            if emoji in char:
                emoji_in_char = True
                new_char = char.encode('utf-8').decode('utf-8')
                new_all_chars.append(new_char)
            else:
                pass
        if not emoji_in_char:
            new_char = char.encode('utf-8').decode('unicode_escape')
            new_all_chars.append(new_char)
    return " ".join(new_all_chars)


def show_chinese_step2(text):
    result1 = re.sub('<', '\\x', text)
    result2 = encode(result1.encode().decode('unicode_escape', 'ignore'), 'raw_unicode_escape')
    result3 = result2.decode('utf-8', 'ignore')
    return result3


def show_chinese_step3(text):
    patterns = re.findall(pattern='\\\\u[a-z0-9]{4}', string=text)
    old_text = text
    for pattern in patterns:
        new_pattern = pattern.encode('utf-8').decode('unicode_escape', 'ignore')
        new_text = re.sub(pattern='\\'+pattern, repl=new_pattern, string=old_text)
        old_text = new_text
    return old_text


text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
              'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens

    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter="twitter",

    # corpus from which the word statistics are going to be used
    # for spell correction
    corrector="twitter",

    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)


def preprocessing_for_english(text_preprocessor, raw_text):
    preprocessed_text = ' '.join(text_preprocessor.pre_process_doc(str(raw_text)))
    # remove punctuations
    result = re.sub(u'[{}]'.format(string.punctuation), u'', preprocessed_text)
    return result


def clean_english_tweet_for_review(text, emoji_dictionary):
    text_with_emoji = show_emoji_in_tweet(text, emoji_dictionary)
    processed_text = preprocessing_for_english(text_processor, text_with_emoji)
    return processed_text


def clean_chinese_tweet_for_review(text, emoji_dictionary):
    tweet_with_emoji = show_emoji_in_tweet(text, emoji_dictionary)
    step1 = show_chinese_step1(tweet_with_emoji, emoji_dictionary)
    step2 = show_chinese_step2(step1)
    step3 = show_chinese_step3(step2)
    return step3


if __name__ == '__main__':
    # load the data
    # Use the tweets_filtering.py to get the final_uncleaned file
    final_uncleaned = pd.read_pickle(os.path.join(read_data.tweet_2017, 'final_uncleaned.pkl'))
    emoji_dict = pd.read_pickle(os.path.join(read_data.tweet_2017, 'emoji.pkl'))
    final_uncleaned_without_tl = final_uncleaned.loc[final_uncleaned['lang'] != 'tl']
    final_uncleaned_without_tl_hk_time = add_eight_hours(final_uncleaned_without_tl)
    final_uncleaned_without_tl_hk_time['month'] = final_uncleaned_without_tl_hk_time.apply(
        lambda row: get_month_hk_time(row['hk_time']), axis=1)
    all_zh = final_uncleaned_without_tl_hk_time.loc[final_uncleaned_without_tl_hk_time['lang'] == 'zh']
    all_en = final_uncleaned_without_tl_hk_time.loc[final_uncleaned_without_tl_hk_time['lang'] == 'en']
    # Then use the cross_sectional_study.py to get the tweets in each TN
    # Save the station related dataframes to read_data.prepare_for_the_review_path

    files = os.listdir(read_data.prepare_for_the_review_path)
    dataframes = []

    # The path should be the path which contains raw tweets waiting to be preprocessed
    # Here we only consider the TNs which have at least 100 tweets
    for file in files:
        dataframe = pd.read_pickle(os.path.join(read_data.prepare_for_the_review_path, file))
        if dataframe.shape[0] < 100:
            pass
        else:
            dataframe = dataframe[['text', 'url', 'user_id_str', 'lang', 'lat', 'lon', 'month']]
            dataframes.append(dataframe)

    whole_data_for_review = pd.concat(dataframes)
    review_data = whole_data_for_review.sample(n=5000)
    en_review = review_data.loc[review_data['lang'] == 'en']
    zh_review = review_data.loc[review_data['lang'] == 'zh']

    en_review['cleaned_text'] = en_review.apply(lambda row: clean_english_tweet_for_review(row['text'], emoji_dict),
                                        axis = 1)
    zh_review['cleaned_text'] = zh_review.apply(lambda row: clean_chinese_tweet_for_review(row['text'], emoji_dict),
                                        axis = 1)

    en_review.to_pickle(os.path.join(read_data.human_review_result_path, 'en_review.pkl'))
    zh_review.to_pickle(os.path.join(read_data.human_review_result_path, 'zh_review.pkl'))

    cleaned_review_data = pd.concat([zh_review, en_review])
    cleaned_review_data = shuffle(cleaned_review_data)
    cleaned_review_data.to_pickle(os.path.join(read_data.human_review_result_path, 'review_data.pkl'))
