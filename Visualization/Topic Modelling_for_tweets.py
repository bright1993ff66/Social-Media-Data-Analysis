import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore",category=UserWarning)
from collections import Counter
import pandas as pd
import numpy as np
import os, re
import read_data

from wordcloud import STOPWORDS
import gensim
import spacy
from spacy.tokenizer import Tokenizer
from nltk.tokenize import word_tokenize

# sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from PIL import Image


# Load the tokenizer in SpaCy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
tokenizer = Tokenizer(nlp.vocab)

whole_tweets = pd.read_pickle(os.path.join(read_data.tweet_2017, 'final_zh_en_for_paper_hk_time_2017.pkl'))
stopwords = list(set(STOPWORDS))
strange_terms = ['allcaps', 'repeated', 'elongated', 'repeat', 'user', 'percent_c']
unuseful_terms = stopwords + strange_terms
unuseful_terms_set = set(unuseful_terms)


def process_words(texts, stop_words, bigram_mod, trigram_mod, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    texts = [[word for word in doc if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    # remove stopwords once more after lemmatization
    texts_out = [[word for word in doc if word not in stop_words] for doc in texts_out]
    return texts_out


def color_green(val):
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)

def make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)


# Show top n keywords for each topic
def show_topics(vectorizer, lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords


def get_lda_model(sentiment_text_in_one_list, grid_search_params, number_of_keywords, topic_predict_file, keywords_file):
    """
    :param sentiment_text_in_one_list: a text list. Each item of this list is a posted tweet
    :param grid_search_params: the dictionary which contains the values of hyperparameters for gridsearch
    :return: the result of lda model - the keywords of each topic
    """
    # 1. Vectorized the data
    vectorizer = CountVectorizer(analyzer='word',
                                 min_df=10,  # minimum reqd occurences of a word
                                 stop_words='english',  # remove stop words
                                 lowercase=True,  # convert all words to lowercase
                                 token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                                 # max_features=50000,             # max number of uniq words
                                 )
    text_vectorized = vectorizer.fit_transform(sentiment_text_in_one_list)

    # 2. Use the GridSearch to find the best hyperparameter
    # In this case, the number of topics is the hyperparameter we should tune
    lda = LatentDirichletAllocation(learning_method='batch')
    model = GridSearchCV(lda, param_grid=grid_search_params)
    model.fit(text_vectorized)
    # See the best model
    best_lda_model = model.best_estimator_
    # Model Parameters
    print("Best Model's Params: ", model.best_params_)
    # Log Likelihood Score
    print("Best Log Likelihood Score: ", model.best_score_)
    # Perplexity
    print("Model Perplexity: ", best_lda_model.perplexity(text_vectorized))

    # 3. Use the best model to fit the data
    # Create Document - Topic Matrix
    lda_output = best_lda_model.transform(text_vectorized)
    # column names
    topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]
    # index names
    docnames = ["Tweet" + str(i) for i in range(np.shape(text_vectorized)[0])]
    # Make the pandas dataframe
    # The df_document_topic dataframe just shows the dominant topic of each doc(tweet)
    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)
    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic
    df_document_topic.to_pickle(os.path.join(read_data.topic_modelling_path,
                                             topic_predict_file))
    # Apply Style
    # df_document_topics = df_document_topic.head(15).style.applymap(color_green).applymap(make_bold)
    # df_document_topics
    # Show the number of topics appeared among documents
    # df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
    # df_topic_distribution.columns = ['Topic Num', 'Num Documents']

    # 4. Get the keywords for each topic
    topic_keywords = show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=number_of_keywords)
    # Topic - Keywords Dataframe
    df_topic_keywords = pd.DataFrame(topic_keywords)
    df_topic_keywords.columns = ['Word ' + str(i) for i in range(df_topic_keywords.shape[1])]
    df_topic_keywords.index = ['Topic ' + str(i) for i in range(df_topic_keywords.shape[0])]
    df_topic_keywords.to_pickle(os.path.join(read_data.topic_modelling_path, keywords_file))
    return df_topic_keywords


if __name__ == '__main__':
    whole_tweets = whole_tweets[
        ['user_id_str', 'cleaned_text', 'sentiment', 'url', 'lat', 'lon', 'lang', 'retweet_count',
         'retweeted']]
    negative_tweets = whole_tweets.loc[whole_tweets['sentiment'] == 0]
    positive_tweets = whole_tweets.loc[whole_tweets['sentiment'] == 2]

    # Gridsearch params
    search_params = {'n_components': [10, 15, 20, 25, 30]}

    # Get the topic model for negative tweets
    whole_text_negative = list(negative_tweets['cleaned_text'])
    tokenized_whole_text_negative = [word_tokenize(text) for text in whole_text_negative]
    bigram_negative = gensim.models.Phrases(tokenized_whole_text_negative, min_count=5,
                                   threshold=100)  # higher threshold fewer phrases.
    trigram_negative = gensim.models.Phrases(bigram_negative[tokenized_whole_text_negative], threshold=100)
    bigram_mod_negative = gensim.models.phrases.Phraser(bigram_negative)
    trigram_mod_negative = gensim.models.phrases.Phraser(trigram_negative)
    negative_data_ready = process_words(tokenized_whole_text_negative, stop_words=unuseful_terms_set,
                                        bigram_mod=bigram_mod_negative, trigram_mod=trigram_mod_negative)
    negative_data_sentence_in_one_list = [' '.join(text) for text in negative_data_ready]
    get_lda_model(negative_data_sentence_in_one_list, grid_search_params=search_params, number_of_keywords=10,
                  keywords_file='negative_tweets_topic_modelling_result.pkl',
                  topic_predict_file='negative_tweet_topic.pkl')

    # Get the topic model for positive tweets
    whole_text_positive = list(positive_tweets['cleaned_text'])
    tokenized_whole_text_positive = [word_tokenize(text) for text in whole_text_positive]
    bigram_positive = gensim.models.Phrases(tokenized_whole_text_positive, min_count=5,
                                   threshold=100)  # higher threshold fewer phrases.
    trigram_positive = gensim.models.Phrases(bigram_positive[tokenized_whole_text_positive], threshold=100)
    bigram_mod_positive = gensim.models.phrases.Phraser(bigram_positive)
    trigram_mod_positive = gensim.models.phrases.Phraser(trigram_positive)
    positive_data_ready = process_words(tokenized_whole_text_positive, stop_words=unuseful_terms_set,
                                        bigram_mod=bigram_mod_positive, trigram_mod=trigram_mod_positive)
    positive_data_sentence_in_one_list = [' '.join(text) for text in positive_data_ready]
    get_lda_model(negative_data_sentence_in_one_list, grid_search_params=search_params, number_of_keywords=10,
                  keywords_file='positive_tweets_topic_modelling_result.pkl',
                  topic_predict_file='positive_tweet_topic.pkl')