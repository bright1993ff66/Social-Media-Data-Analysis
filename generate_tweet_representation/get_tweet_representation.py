import os
working_directory = r'F:\CityU\Hong Kong Twitter 2016\emoji2vec'
os.chdir(working_directory)
print('The current working directory has changed to: ', os.getcwd())

#===================================================================================================================
# Impore Relevant Packages
# Commonly used
import math
import gensim.models as gs
import pickle as pk
import numpy as np
import pandas as pd
from collections import Counter
import time
import csv
import read_data
import utils

from sklearn.utils import shuffle

# This paper requires
import phrase2vec as p2v
from twitter_sentiment_dataset import TweetTrainingExample
from model import ModelParams

# tokenization
import nltk.tokenize as tk

from sklearn.model_selection import train_test_split

# Ignore the tedious warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Specify the random seed
random_seed = 777

# Some important paths
w2v_path = './data/word2vec/'


def list_of_array_to_array(list_array):
    shape = list(list_array[0].shape)
    shape[:0] = [len(list_array)]
    arr = np.concatenate(list_array).reshape(shape)
    return arr

def read_local_csv_file(path, filename):
    dataframe = pd.read_csv(os.path.join(path, filename), encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, dtype='str',
                            index_col=0)
    return dataframe


def prepare_tweet_vector_averages_for_prediction(tweets, p2v):
    """
    Take the vector sum of all tokens in each tweet

    Args:
        tweets: All tweets
        p2v: Phrase2Vec model

    Returns:
        Average vectors for each tweet
        Truth
    """
    tokenizer = tk.TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)

    avg_vecs = list()

    for tweet in tweets:
        tokens = tokenizer.tokenize(tweet)
        avg_vecs.append(np.sum([p2v[x] for x in tokens], axis=0) / len(tokens))

    return avg_vecs


# construct the whole review and sample datasets: sample for prediction; review for validation
def construct_whole_sample_datasets(en_sample, zh_sample, en_review, zh_review):
    final_review = pd.concat((en_review, zh_review), axis=0)
    final_sample = pd.concat((en_sample, zh_sample), axis=0)
    final_sample = final_sample.reset_index(drop=True)
    final_sample = shuffle(final_sample)
    sample_index = final_sample.index.tolist()
    final_review = final_review.reset_index(drop=True)
    final_review = final_review.reindex(sample_index)
    return final_sample, final_review


if __name__ == '__main__':

    #Set Global Variables for emoji2vec
    in_dim = 100  # Length of word2vec vectors
    out_dim = 100  # Desired dimension of output vectors
    pos_ex = 4
    neg_ratio = 1
    max_epochs = 40
    dropout = 0.1
    params = ModelParams(in_dim=in_dim, out_dim=out_dim, pos_ex=pos_ex, max_epochs=max_epochs,
                         neg_ratio=neg_ratio, learning_rate=0.001, dropout=dropout, class_threshold=0.5)

    e2v_ours_path = params.model_folder('unicode') + '/emoji2vec_100.bin'

    # Load the FastText word vectors and emoji vectors
    w2v = gs.FastText.load(os.path.join(w2v_path, 'fasttext_model'))
    e2v_ours = gs.KeyedVectors.load_word2vec_format(e2v_ours_path, binary=True)
    # Combine the word vectors and emoji vectors together
    p2v_our_emoji = p2v.Phrase2Vec(out_dim, w2v, e2v=e2v_ours)

    # #=========================For the unprocessed text========================================
    tweet_combined_dataframe = utils.read_local_csv_file(path=read_data.tweet_combined_path,
                                                 filename='tweet_combined_cleaned_translated.csv')
    text_unprocessed = list(tweet_combined_dataframe['cleaned_text'])
    #
    tweet_array = prepare_tweet_vector_averages_for_prediction(text_unprocessed, p2v_our_emoji)
    tweet_repre = list_of_array_to_array(tweet_array)

    np.save(os.path.join(read_data.tweet_combined_path, 'tweet_representations', 'tweet_combined_repre.npy'),
            tweet_repre)





