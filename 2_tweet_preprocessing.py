# Commonly used
import re
import string
import os
import pandas as pd
from codecs import encode
import time

# Load some text resources in nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

# Shuffle the rows of the pandas dataframe
from sklearn.utils import shuffle

# pre-defined packages
import read_data
import utils

# An package which could cope with emoji
import emoji as ej

# A package for preprocessing(officially used in SemEval NLP competition)
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

# For Machine Translation
from google.cloud import translate

# Set the credential environment in jupyter notebook
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=r"XXXXX"
# Instantiates a client
translate_client = translate.Client()

# Get the stopwords and punctuations from the NLTK corpora and the string package respectively
stopwords = stopwords.words('english')
stopwords.extend(['atuser', 'atplace'])
stopwords = set(stopwords)

# Create the emoji word set
english_words = set(nltk.corpus.words.words())
english_words_lower = set(word.lower() for word in english_words)
emoji_dict = pd.read_pickle(os.path.join(read_data.tweet_2017, 'emoji.pkl'))
emoji_list = list(emoji_dict['emoji'])
english_words_lower.update(emoji_list)
# Add station names to the set
station_location = pd.read_csv(os.path.join(read_data.tweet_2017, 'station_location.csv'))
station_names_list = list(station_location['Name'])
names_lower = [word_tokenize(name.lower()) for name in station_names_list]
words = []
for word_list in names_lower:
    for word in word_list:
        words.append(word)
english_words_lower.update(words)

punctuations = string.punctuation

# Load the Wordnet Lemmatizer: change the word, for instance, dictionaries to dictionary
# The stemmer could output dict in the dictionaries case
lemmatizer = WordNetLemmatizer()

Object = utils.RegexpReplacer(patterns=utils.replacement_patterns)

# wid->with, u->you
prefixStr = '<div class="translation-text">'
postfixStr = '</div'

# Use regex to find all the emojis in a raw text
emoji_pattern = ej.get_emoji_regexp()


"""
def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet
"""


# Clean the txt data - my function
def clean_raw_text(raw_text, caller, remove_stopwords=True, lemmatize = True):

    # 1. Remove the meaningless links
    text_without_link = re.sub(r'http\S+', '', raw_text)

    # 2. Remove all punctuations
    tweet_without_punctuations = re.sub(u'[{}]'.format(punctuations), u'', text_without_link)

    # 3.Transform some patterns to more meaningful ones: you're -> you are. Then return a list of words.
    cleaned_tweet = Object.replace(tweet_without_punctuations)

    # 4. Word tokenize
    tokenized_tweet = word_tokenize(cleaned_tweet)

    # 5. Lowercase the words
    tweet_lower = [word.lower() for word in tokenized_tweet]

    # 6. Remove stopwords
    if remove_stopwords:
        without_stopwords = [word for word in tweet_lower if word not in stopwords]
        result = without_stopwords
    else:
        result = tweet_lower

    # 7. Lemmatization
    for index, word in enumerate(result):
        if lemmatize:
            result[index] = lemmatizer.lemmatize(word)

    # Return result based on different callers
    if caller == 'Topic Modelling':
        return result
    elif caller == 'bilstm':
        return ' '.join(result)
    else:
        raise RuntimeError


def show_translated_chinese(text):
    result1 = re.sub('\<u\+', '\\' + 'u', text.lower())
    result2 = re.sub('\>', '', result1)
    result3 = re.sub('\<([a-z0-9]{2})*', '', result2)
    all_chars = result3.split()
    for index, char in enumerate(all_chars):
        try:
            if translate_client.detect_language(char)['language'][:2] != 'en':
                all_chars[index] = translate_client.translate(char, target_language='en')['translatedText']
            else:
                all_chars[index] = char.encode('utf-8').decode('utf-8')
        except:
            pass
    return ' '.join(all_chars)


"""
def show_translated_chinese_for_review(text):
    all_chars = text.split()
    emojis = emoji_pattern.findall(text)
    for index, char in enumerate(all_chars):
        try:
            if translate_client.detect_language(char)['language'][:2] != 'en':
                if len(emojis) == 0:
                    all_chars[index] = translate_client.translate(char, target_language='en')['translatedText']
                else:
                    all_chars[index] = re.sub(emoji_pattern, '', all_chars[index])
                    all_chars[index] = translate_client.translate(char, target_language='en')['translatedText']
            else:
                all_chars[index] = char.encode('utf-8').decode('utf-8')
        except:
            print('The text we cannot process is: ', text)
            pass

        for emoji in emojis:
            all_chars[index] += emoji
    return ' '.join(all_chars)
"""


text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "elongated", "repeated"},
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


def nltk2wn_tag(nltk_tag):
  if nltk_tag.startswith('J'):
    return wordnet.ADJ
  elif nltk_tag.startswith('V'):
    return wordnet.VERB
  elif nltk_tag.startswith('N'):
    return wordnet.NOUN
  elif nltk_tag.startswith('R'):
    return wordnet.ADV
  else:
    return None


def lemmatize_sentence(sentence):
  nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
  wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
  res_words = []
  for word, tag in wn_tagged:
    if tag is None:
      res_words.append(word)
    else:
      res_words.append(lemmatizer.lemmatize(word, tag))
  return " ".join(res_words)


# delete unmeaningful terms
def delete_unmeaningful_terms(tweet):
    deleted_text = " ".join(w for w in nltk.wordpunct_tokenize(tweet) if w.lower() in english_words)
    return deleted_text


def preprocessing_for_english(text_preprocessor, raw_text):
    preprocessed_text = ' '.join(text_preprocessor.pre_process_doc(str(raw_text)))
    # remove punctuations
    result = re.sub(u'[{}]'.format(string.punctuation), u'', preprocessed_text)
    return result


def preprocessing_for_chinese(text_preprocessor, raw_text):
    preprocessed_text = ' '.join(text_preprocessor.pre_process_doc(str(raw_text)))
    # remove punctuations
    result1 = re.sub(u'[{}]'.format(string.punctuation), u'', preprocessed_text)
    # remove hashtag
    result2 = re.sub('hashtag', u'', result1)
    # remove url
    result3 = re.sub('url', '', result2)
    # replace the multiple blanks to one blank
    result4 = re.sub('\\s+', u' ', result3)
    # remove the digits
    result5 = re.sub('number', u'', result4)
    return result5


#=======================================For English Tweet=======================================================
def remove_u_plus(text):
    result = re.sub(pattern=r'U\+00', repl=r'', string=text)
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


def clean_english_tweet(text, emoji_dictionary):
    text_with_emoji = show_emoji_in_tweet(text, emoji_dictionary)
    processed_text = preprocessing_for_english(text_processor, text_with_emoji)
    return processed_text


# =====================================For Chinese Tweet================================================================
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


def clean_chinese_tweet(text, emoji_dictionary):
    tweet_with_emoji = show_emoji_in_tweet(text, emoji_dictionary)
    step1 = show_chinese_step1(tweet_with_emoji, emoji_dictionary)
    step2 = show_chinese_step2(step1)
    step3 = show_chinese_step3(step2)
    # For the preparation of review data, we don't need to show translated Chinese and preprocessing_for_chinese
    text_with_translated_chinese = show_translated_chinese(step3)
    processed_text = preprocessing_for_chinese(text_processor, text_with_translated_chinese)
    return processed_text

# ==================================================================================================


if __name__ == '__main__':

    """
    # Clean the Sentiment140 Dataset and Save
    sentiment140 = pd.read_pickle(os.path.join(read_data.path, 'sentiment140.pkl'))
    # If caller = 'bilstm', the cleaned text is string. Otherwise it's list
    sentiment140['text'] = sentiment140['text'].apply(lambda x: clean_raw_text(raw_text=x, caller='bilstm'))
    sentiment140_cleaned = sentiment140
    sentiment140_cleaned.to_pickle(os.path.join(read_data.path, 'sentiment140_cleaned.pkl'))

    
    # Clean the concatenated HK 2016 Tweet Data from May to October
    May_to_Oct_tweet = pd.read_pickle(os.path.join(read_data.path, 'concatenated_tweet_data.pkl'))
    # If caller = 'bilstm', the cleaned text is string. Otherwise it's list
    May_to_Oct_tweet['processed_text'] = May_to_Oct_tweet['processed_text'].apply(lambda x: clean_raw_text(raw_text=x, caller='bilstm'))
    May_to_Oct_tweet.to_pickle(os.path.join(read_data.path, 'concatenated_tweet_data_cleaned.pkl'))

    end = time.time()
    time_spent = end-start

    print('Total time spent is: ', time_spent)
    """
    # For 2016 tweets
    start_time_2016 = time.time()

    tweets_without_bot_2016 = pd.read_pickle(os.path.join(read_data.tweet_2016, 'raw_tweets_final.pkl'))
    en_tweets = tweets_without_bot_2016.loc[tweets_without_bot_2016['lang']=='en']
    zh_tweets = tweets_without_bot_2016.loc[tweets_without_bot_2016['lang']=='zh']

    print('Cleaning the English Tweets in 2016...')
    en_tweets['cleaned_text'] = en_tweets.apply(lambda row: clean_english_tweet(row['text'],
                                                                                emoji_dictionary=emoji_dict), axis=1)
    print('Done')
    print('Cleaning Chinese Tweets in 2016...')
    zh_tweets['cleaned_text'] = zh_tweets.apply(lambda row: clean_chinese_tweet(row['text'],
                                                                                emoji_dictionary=emoji_dict), axis=1)
    print('Done')
    final_tweets_2016 = pd.concat([en_tweets, zh_tweets])
    final_tweets_for_comparision_2016  = shuffle(final_tweets_2016)
    final_tweets_for_comparision_2016['cleaned_text'] = final_tweets_for_comparision_2016.apply(
        lambda row: lemmatize_sentence(row['cleaned_text']), axis=1)
    final_tweets_for_comparision_2016['cleaned_text'] = final_tweets_for_comparision_2016.apply(
        lambda row: delete_unmeaningful_terms(row['cleaned_text']), axis=1)
    final_tweets_for_comparision_2016_without_non_cleaned_text = \
        final_tweets_for_comparision_2016.loc[final_tweets_for_comparision_2016['cleaned_text'] != '']
    # Save the tweets in 2016 which could be used for transit non-transit comparision
    final_tweets_for_comparision_2016_without_non_cleaned_text.to_pickle(os.path.join(read_data.tweet_2016,
                                                                                      'tweet_2016_cleaned_text.pkl'))

    end_time = time.time()
    print("Total time for 2016: ", end_time-start_time_2016)

    # For 2017 tweets
    start_time_2017 = time.time()

    final_uncleaned = pd.read_pickle(os.path.join(read_data.tweet_2017, 'final_uncleaned.pkl'))
    final_uncleaned_without_tl = final_uncleaned.loc[final_uncleaned['lang'] != 'tl']
    all_zh = final_uncleaned_without_tl.loc[final_uncleaned_without_tl['lang'] == 'zh']
    all_en = final_uncleaned_without_tl.loc[final_uncleaned_without_tl['lang'] == 'en']

    print('Cleaning the English Tweets in 2017...')
    all_en['cleaned_text'] = all_en.apply(lambda row: clean_english_tweet(row['text'],
                                                                                emoji_dictionary=emoji_dict), axis=1)
    print('Done')
    print('Cleaning Chinese Tweets in 2017...')
    all_zh['cleaned_text'] = all_zh.apply(lambda row: clean_chinese_tweet(row['text'],
                                                                                emoji_dictionary=emoji_dict), axis=1)
    print('Done')
    final_tweets_2017 = pd.concat([all_en, all_zh])
    final_tweets_in_2017 = shuffle(final_tweets_2017)
    final_tweets_in_2017['cleaned_text'] = final_tweets_in_2017.apply(
        lambda row: lemmatize_sentence(row['cleaned_text']), axis=1)
    final_tweets_in_2017['cleaned_text'] = final_tweets_in_2017.apply(
        lambda row: delete_unmeaningful_terms(row['cleaned_text']), axis=1)
    final_tweets_in_2017_without_non_cleaned_text = \
        final_tweets_in_2017.loc[final_tweets_in_2017['cleaned_text'] != '']
    final_tweets_in_2017_without_non_cleaned_text.to_pickle(
        os.path.join(read_data.tweet_2017, 'final_zh_en_for_paper.pkl'))

    end_time_2017 = time.time()
    print("Total time: ", end_time_2017 - start_time_2017)