import re
import string
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import read_data
import utils
import pandas as pd

# Get the stopwords and punctuations from the NLTK corpora and the string package respectively
stopwords = stopwords.words('english')
stopwords.extend(['atuser', 'atplace'])
stopwords = set(stopwords)
punctuations = string.punctuation

# Load the Wordnet Lemmatizer: change the word, for instance, dictionaries to dictionary
# The stemmer could output dict in the dictionaries case
lemmer = WordNetLemmatizer()

Object = utils.RegexpReplacer(patterns=utils.replacement_patterns)


# Clean the txt data
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

    return ' '.join(result)


if __name__ == '__main__':
    # Clean the Sentiment140 Dataset and Save
    sentiment140 = pd.read_pickle(os.path.join(read_data.tweet_2016, 'sentiment140.pkl'))
    # If caller = 'bilstm', the cleaned text is string. Otherwise it's list
    sentiment140['text'] = sentiment140['text'].apply(lambda x: clean_raw_text(raw_text=x, caller='bilstm'),
                                                      axis=1)
    sentiment140_cleaned = sentiment140.copy()
    sentiment140_cleaned.to_pickle(os.path.join(read_data.tweet_2016, 'sentiment140_cleaned.pkl'))
