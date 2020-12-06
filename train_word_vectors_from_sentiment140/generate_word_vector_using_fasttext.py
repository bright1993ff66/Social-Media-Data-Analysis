import pandas as pd
from sklearn.model_selection import train_test_split
import time
import os
import data_paths

from gensim.models import FastText
from gensim.models.word2vec import LineSentence

# Load the dataset
sentiment140 = pd.read_pickle(os.path.join(data_paths.tweet_2016, 'sentiment140_cleaned.pkl'))

if __name__ == '__main__':
    all_text_list = list(sentiment140['text'])

    with open(os.path.join(path, 'sentiment_cleaned_text.txt'), 'w') as f:
        for text in all_text_list:
            f.write("%s\n" % text)

    all_text = LineSentence(os.path.join(data_paths.tweet_2016, 'sentiment_cleaned_text.txt'))

    print('Generating FastText Vectors ..')
    # embedding size
    start = time.time()
    fasttext_model = FastText(size=100, window=3, min_count=1, iter=10)
    fasttext_model.build_vocab(all_text)
    fasttext_model.train(all_text, total_examples=fasttext_model.corpus_count, epochs=fasttext_model.iter)

    print('FastText Created in {} seconds.'.format(time.time() - start))

    fasttext_model.save(os.path.join(data_paths.word_vector_path, 'fasttext_model'))
