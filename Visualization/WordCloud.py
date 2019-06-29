import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)
from collections import Counter
import pandas as pd
import numpy as np
import os, re
import string

import read_data
import Topic_Modelling_for_tweets

import gensim
from gensim import corpora, models
import spacy

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt
from PIL import Image

plot_path = read_data.plot_path

# gensim.corpora.MmCorpus.serialize('MmCorpusTest.mm', corpus)
unuseful_terms_set = Topic_Modelling_for_tweets.unuseful_terms_set

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# the regex used to detect words is a combination of normal words, ascii art, and emojis
# 2+ consecutive letters (also include apostrophes), e.x It's
normal_word = r"(?:\w[\w']+)"
# 2+ consecutive punctuations, e.x. :)
ascii_art = r"(?:[{punctuation}][{punctuation}]+)".format(punctuation=string.punctuation)
# a single character that is not alpha_numeric or other ascii printable
emoji = r"(?:[^\s])(?<![\w{ascii_printable}])".format(ascii_printable=string.printable)
regexp = r"{normal_word}|{ascii_art}|{emoji}".format(normal_word=normal_word, ascii_art=ascii_art,
                                                     emoji=emoji)
symbola_font_path = os.path.join(read_data.plot_path, 'Symbola_Hinted.ttf')

circle_mask = np.array(Image.open(r"F:\CityU\Datasets\Hong Kong Tweets 2017\circle.png"))

# Change the color of the wordcloud
def green_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(127, 0%%, %d%%)" % np.random.randint(49, 100)


def red_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(19, 0%%, %d%%)" % np.random.randint(49, 100)


def generate_wordcloud(words, mask, file_name, color_func):
    """
    :param words: the text we use to draw the wordcloud. The datatype should be string
    :param mask: a mask we load to draw the plot
    :param file_name: the name of the saved pic
    :param color_func: the function which defines the color of words
    :return: the wordcloud of the input text
    """
    # stopwords argument in word_cloud: specify the words we neglect when outputing the wordcloud
    word_cloud = WordCloud(width = 512, height = 512, background_color='white', stopwords=unuseful_terms_set,
                           mask=mask, max_words=800).generate(words)
    plt.figure(figsize=(15,13), facecolor = 'white', edgecolor='black')
    plt.imshow(word_cloud.recolor(color_func=color_func, random_state=3), interpolation="bilinear")
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(plot_path, file_name))
    plt.show()


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


def create_text_for_wordcloud(df):
    """
    :param df: the pandas dataframe which contains the text of tweets
    :return: the process text for word cloud generation
    """
    tweet_text = list(df['cleaned_text'])
    tokenized_text_list = [word_tokenize(text) for text in tweet_text]
    bigram = models.phrases.Phrases(tokenized_text_list, min_count=5,
                                   threshold=100)  # higher threshold fewer phrases.
    bigram_mod = models.phrases.Phraser(bigram)
    trigram = models.phrases.Phrases(bigram_mod[tokenized_text_list])
    trigram_mod = models.phrases.Phraser(trigram)
    processed_text = process_words(tokenized_text_list, bigram_mod=bigram_mod, trigram_mod=trigram_mod,
                               stop_words=unuseful_terms_set)
    text_in_list = [' '.join(text) for text in processed_text]
    text_ready = ' '.join(text_in_list)
    return text_ready


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


if __name__ == '__main__':
    # Load a mask
    whole_tweets = pd.read_pickle(os.path.join(read_data.tweet_2017, 'final_2017_with_sentiment_smote.pkl'))
    # Just in case: filter out the users whose tweets have exactly the same geoinformation
    whole_tweets_filterd = delete_bots_have_same_geoinformation(whole_tweets)

    negative_tweets = whole_tweets_filterd.loc[whole_tweets_filterd['sentiment'] == 0]
    positive_tweets = whole_tweets_filterd.loc[whole_tweets_filterd['sentiment'] == 2]

    negative_text = create_text_for_wordcloud(negative_tweets)
    positive_text = create_text_for_wordcloud(positive_tweets)

    # Generate word cloud for positive text
    generate_wordcloud(positive_text, circle_mask, file_name='positive_wordcloud.png',
                       color_func=green_func)
    # Generate word cloud for negative text
    generate_wordcloud(negative_text, circle_mask, file_name='negative_wordcloud.png',
                       color_func=red_func)