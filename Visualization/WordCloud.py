import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)
from collections import Counter
import pandas as pd
import numpy as np
import os, re
import read_data

import gensim
from gensim import corpora, models
import spacy

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt
from PIL import Image

plot_path = r'F:\CityU\Datasets\Hong Kong Tweets 2017\plots'

# gensim.corpora.MmCorpus.serialize('MmCorpusTest.mm', corpus)
stopwords = list(set(STOPWORDS))
strange_terms = ['allcaps', 'repeated', 'elongated', 'repeat', 'user', 'percent_c']
unuseful_terms = stopwords + strange_terms
unuseful_terms_set = set(unuseful_terms)

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


# Change the color of the wordcloud
def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0, 0%%, %d%%)" % np.random.randint(60, 100)


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
                           mask=mask, max_words=500).generate(words)
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
    tweet_text = list(df['cleaned_text'])
    tokenized_text_list = [word_tokenize(text) for text in tweet_text]
    bigram = models.Phrases(tokenized_text_list, min_count=5,
                                   threshold=100)  # higher threshold fewer phrases.
    trigram = models.Phrases(bigram[tokenized_text_list])
    bigram_mod = models.phrases.Phraser(bigram)
    trigram_mod = models.phrases.Phraser(trigram)
    processed_text = process_words(tokenized_text_list, bigram_mod=bigram_mod, trigram_mod=trigram_mod,
                               stop_words=unuseful_terms_set)
    text_in_list = [' '.join(text) for text in processed_text]
    text_ready = ' '.join(text_in_list)
    return text_ready



if __name__ == '__main__':
    # Load a mask
    circle_mask = np.array(Image.open(r"F:\CityU\Datasets\Hong Kong Tweets 2017\circle.png"))
    whole_tweets = pd.read_pickle(os.path.join(read_data.tweet_2017, 'final_zh_en_for_paper_hk_time_2017.pkl'))
    negative_tweets = whole_tweets.loc[whole_tweets['sentiment'] == 0]
    positive_tweets = whole_tweets.loc[whole_tweets['sentiment'] == 2]

    negative_text = create_text_for_wordcloud(negative_tweets)
    positive_text = create_text_for_wordcloud(positive_tweets)

    # Generate word cloud for positive text
    generate_wordcloud(positive_text, circle_mask, file_name='negative_wordcloud.png',
                       color_func=grey_color_func)
    # Generate word cloud for negative text
    generate_wordcloud(negative_text, circle_mask, file_name='negative_wordcloud.png',
                       color_func=grey_color_func)