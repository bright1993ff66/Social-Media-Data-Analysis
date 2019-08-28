import os
# Change the current working directory to emoji2vec
working_directory = r'XXXX\emoji2vec'
os.chdir(working_directory)
print('The current working directory has changed to: ',os.getcwd())

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
import read_data

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

    # Set Global Variables for emoji2vec
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

    """
    # Create the training tweets and testing tweets
    train_tweets, test_tweets = tsd.load_training_test_sets()

    #  Prepare the Training, Testing Vectors and Corresponding Labels
    train_ours, trainy = tsd.prepare_tweet_vector_averages(train_tweets, p2v_our_emoji)
    trainy_nums = []
    for label in trainy:
        num_label = rename_labels(label)
        trainy_nums.append(num_label)

    test_ours, test_y = tsd.prepare_tweet_vector_averages(test_tweets, p2v_our_emoji)
    test_ours = np.array(test_ours)
    test_nums = []
    for label in test_y:
        num_label = rename_labels(label)
        test_nums.append(num_label)

    whole_tweets = np.concatenate((train_ours, test_ours), axis=0)
    whole_nums = np.concatenate((trainy_nums, test_nums), axis=0).tolist()
    """

    # Evaluate the model on the human review data
    # Load the files to generate tweet representations for our classifiers
    final_zh_sample_cleaned_and_translated = pd.read_pickle(
        os.path.join(read_data.tweet_2017, 'final_sample_cleaned_and_translated_2.pkl'))
    final_zh_sample_cleaned_and_translated.loc[
        final_zh_sample_cleaned_and_translated['cleaned_text'] == '', 'cleaned_text'] = \
        final_zh_sample_cleaned_and_translated['text']
    final_en_sample_cleaned = pd.read_pickle(
        os.path.join(read_data.tweet_2017, 'final_en_sample_cleaned_and_translated_2.pkl'))
    final_en_sample_cleaned.loc[final_en_sample_cleaned['cleaned_text'] == '', 'cleaned_text'] = \
        final_en_sample_cleaned['text']
    # Delete the Tweet which has conflicts among reviewers
    new_final_zh_sample = final_zh_sample_cleaned_and_translated.drop([381])
    # Load the human review result
    sample_path = r'F:\CityU\Datasets\Hong Kong Tweets 2017\human review\human review result'
    en_review = pd.read_excel(os.path.join(sample_path, 'en_sample.xlsx'))
    zh_review = pd.read_excel(os.path.join(sample_path, 'zh_sample.xlsx'))
    new_zh_review = zh_review.drop([381])
    # Use the 'final sentiment' column if you let the sentiment be neutral if one reviewer gives neutral
    # Use final_sentiment_2 if you want the sentiment of a tweet be neutral only if two revieweres give neutral
    final_sample, final_review = construct_whole_sample_datasets(en_sample=final_en_sample_cleaned,
                                                                 zh_sample=new_final_zh_sample,
                                                                 en_review=en_review, zh_review=new_zh_review)

    # final sample is used to compute the tweet representations
    # final review is used to save the human review result
    final_sample.to_pickle(os.path.join(read_data.desktop, 'final_sample.pkl'))
    final_review.to_pickle(os.path.join(read_data.desktop, 'final_review.pkl'))

    whole_sample_tweets = list(final_sample['cleaned_text'])
    tweets_representations_whole_sample = prepare_tweet_vector_averages_for_prediction(whole_sample_tweets,
                                                                                       p2v_our_emoji)
    tweets_representations_whole_sample_array = list_of_array_to_array(tweets_representations_whole_sample)
    # Scheme1: If one reviewer annotates neutral, the sentiment of a tweet would be neutral
    whole_review_result_scheme1 = list(final_review['final sentiment'])
    # Scheme2: The sentiment of a tweet would be neutral only if both two reviewers label it neutral
    whole_review_result_scheme2 = list(final_review['final sentiment_2'])
    # Save the tweet representation and the label
    np.save(os.path.join(read_data.tweet_representation_path, 'whole_sample_array'),
            tweets_representations_whole_sample_array)
    np.save(os.path.join(read_data.tweet_representation_path, 'whole_samply_label'), whole_review_result_scheme2)
    # Build the cross validation data and the test data. Then we store them
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(tweets_representations_whole_sample_array,
                                                                    whole_review_result_scheme2, test_size=0.2,
                                                                    random_state=random_seed)
    # Save the X_test and y_test for model selection
    np.save(os.path.join(read_data.tweet_representation_path, 'train_valid_cross_validation_data'),
            X_train_valid)
    np.save(os.path.join(read_data.tweet_representation_path, 'train_valid_cross_validation_label'),
            y_train_valid)
    np.save(os.path.join(read_data.tweet_representation_path, 'test_data_for_model_compare'), X_test)
    np.save(os.path.join(read_data.tweet_representation_path, 'test_label_for_model_compare'), y_test)

    # Load the tweets in 2016 - one for comparision with the previous papaer and another one for sentiment computation
    tweets_in_2016_dataframe_compare_with_yao = pd.read_pickle(os.path.join(read_data.tweet_2016,
                                                           'tweet_2016_compare_with_Yao.pkl'))
    tweets_in_2016_dataframe_compare_with_yao = \
        tweets_in_2016_dataframe_compare_with_yao.loc[tweets_in_2016_dataframe_compare_with_yao['cleaned_text'] != '']
    tweets_in_2016_compare_with_yao = list(tweets_in_2016_dataframe_compare_with_yao['cleaned_text'])
    # Get the representation of each tweet
    tweets_representations_2016_compare_with_yao = \
        prepare_tweet_vector_averages_for_prediction(tweets_in_2016_compare_with_yao, p2v_our_emoji)
    tweets_representations_2016_array_compare_with_yao = \
        list_of_array_to_array(tweets_representations_2016_compare_with_yao)
    np.save(os.path.join(read_data.tweet_representation_path, 'tweet_array_2016_compare_with_yao'),
            tweets_representations_2016_array_compare_with_yao)

    # Load the tweets in 2016
    tweets_in_2016_dataframe = pd.read_pickle(os.path.join(read_data.tweet_2016,
                                                           'final_zh_en_for_paper_hk_time_2016.pkl'))
    tweets_in_2016_dataframe = tweets_in_2016_dataframe.loc[tweets_in_2016_dataframe['cleaned_text'] != '']
    tweets_in_2016 = list(tweets_in_2016_dataframe['cleaned_text'])
    tweets_representations_2016 = prepare_tweet_vector_averages_for_prediction(tweets_in_2016, p2v_our_emoji)
    tweets_representations_2016_array = list_of_array_to_array(tweets_representations_2016)
    np.save(os.path.join(read_data.tweet_representation_path, 'tweet_array_2016'), tweets_representations_2016_array)

    # Load the whole Hong Kong 2017 Twitter dataset and make predictions
    final_whole_data = pd.read_pickle(os.path.join(read_data.tweet_2017,
                                                   'final_zh_en_for_paper_hk_time_2017.pkl'))
    final_whole_data = final_whole_data.loc[final_whole_data['cleaned_text'] != '']
    print(final_whole_data.shape)
    whole_tweets_in_2017 = list(final_whole_data['cleaned_text'])
    tweets_representation_whole_2017_tweets = \
        prepare_tweet_vector_averages_for_prediction(whole_tweets_in_2017, p2v_our_emoji)
    tweets_representation_whole_2017_tweets_array = list_of_array_to_array(tweets_representation_whole_2017_tweets)
    np.save(os.path.join(read_data.tweet_representation_path, 'tweet_array_2017'),
            tweets_representation_whole_2017_tweets_array)






