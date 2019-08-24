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

# Specify the random seed
random_seed = 777

whole_tweets = pd.read_pickle(os.path.join(read_data.tweet_2017, 'final_2017_with_sentiment_smote.pkl'))

# gensim.corpora.MmCorpus.serialize('MmCorpusTest.mm', corpus)
# gensim.corpora.MmCorpus.serialize('MmCorpusTest.mm', corpus)
stopwords = list(set(STOPWORDS))
strange_terms = ['allcaps', 'repeated', 'elongated', 'repeat', 'user', 'percent_c', 'hong kong', 'hong',
                 'kong', 'u_u', 'u_u_number', 'u_u_u_u', 'u_number', 'elongate', 'u_number_u',
                 'u', 'number', 'm', 'will', 'hp', 'grad', 'ed', 'boo']
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


# Delete users which have same geoinformation and the total number of posted tweets is bigger than 10
def delete_bots_have_same_geoinformation(df):
    users = set(list(df['user_id_str']))
    bot_account = []
    for user in users:
        dataframe = df.loc[df['user_id_str']==user]
        # If only one unqiue geoinformation is found and more than 10 tweets are posted, we regard this account as bot
        if (len(set(dataframe['lat'])) == 1 and dataframe.shape[0]>10) or (len(set(dataframe['lon'])) == 1
                                                                           and dataframe.shape[0]>10):
            bot_account.append(user)
        else:
            pass
    cleaned_df = df.loc[~df['user_id_str'].isin(bot_account)]
    return cleaned_df


def get_lda_model(sentiment_text_in_one_list, grid_search_params, number_of_keywords, topic_predict_file,
                  keywords_file, topic_number, grid_search_or_not = True, saving_path = read_data.topic_modelling_path):
    """
    :param sentiment_text_in_one_list: a text list. Each item of this list is a posted tweet
    :param grid_search_params: the dictionary which contains the values of hyperparameters for gridsearch
    :param number_of_keywords: number of keywords to represent a topic
    :param topic_predict_file: one file which contains the predicted topic for each tweet
    :param keywords_file: one file which saves all the topics and keywords
    :param topic_number: The number of topics we use(this argument only works if grid_search_or_not = False)
    :param grid_search_or_not: Whether grid search to get 'best' number of topics
    :param saving_path: path used to save the results 
    """
    # 1. Vectorized the data
    vectorizer = CountVectorizer(analyzer='word',
                                 min_df=1,  # minimum occurences of a word
                                 stop_words='english',  # remove stop words
                                 lowercase=True,  # convert all words to lowercase
                                 token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                                 # max_features=50000,             # max number of uniq words
                                 )
    text_vectorized = vectorizer.fit_transform(sentiment_text_in_one_list)

    # 2. Use the GridSearch to find the best hyperparameter
    # In this case, the number of topics is the hyperparameter we should tune
    lda = LatentDirichletAllocation(learning_method='batch', random_state=random_seed)
    if grid_search_or_not:
        model = GridSearchCV(lda, param_grid=grid_search_params)
    else:
        model = GridSearchCV(lda, param_grid={'n_components': [topic_number]})
    model.fit(text_vectorized)
    # See the best model
    best_lda_model = model.best_estimator_
    if grid_search_or_not:
        # Model Parameters
        print("Best Model's Params: ", model.best_params_)
        # Log Likelihood Score
        print("Best Log Likelihood Score: ", model.best_score_)
        # Perplexity
        print("Model Perplexity: ", best_lda_model.perplexity(text_vectorized))
    else:
        # Show the number of topics we use
        print('The number of topics we use: {}'.format(topic_number))
        # Log likelihood score
        print('The loglikelihood score is {}'.format(model.best_score_))
        # Perplexity
        print("Model Perplexity: {}".format(best_lda_model.perplexity(text_vectorized)))

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
    df_document_topic.to_pickle(os.path.join(saving_path,
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
    df_topic_keywords.to_pickle(os.path.join(saving_path, keywords_file))


def plot_topic_num(topic_df, filename):
    # Use Counter to get the topic distribution
    topic_count_dict = Counter(topic_df['dominant_topic'])
    topic_list = []
    count_list = []
    # Order a key based on the corresponding value
    for key, value in sorted(topic_count_dict.items(), key=lambda item: item[1], reverse=False):
        topic_list.append(key)
        count_list.append(value)

    fig, ax = plt.subplots(1, 1, figsize=(6, 10))

    y_pos = np.arange(len(np.arange(len(topic_list))))
    plt.barh(y_pos, count_list, align='center', alpha=0.5, color='black')
    plt.yticks(y_pos, topic_list)
    plt.ylabel('Topic Number')
    plt.xlabel('Count')
    plt.savefig(os.path.join(read_data.plot_path_2017, filename))
    plt.show()
