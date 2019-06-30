# Description :call_me_hand:

This folder contains the codes about how we generate pre-trained word embedding from a tweet dataset called [Sentiment 140](http://help.sentiment140.com/for-students)

## 1. What is Sentiment 140?

[Sentiment 140](http://help.sentiment140.com/for-students) is dataset which contains 1.6 million tweets collected by graduate students in Stanford.

For more information, please go to this paper: [Twitter Sentiment Classification using Distant Supervision](https://cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf)

## 2. How to get the pre-trained word representation from this dataset?

Since this project includes the analysis of social media data in Hong Kong, I consider using [Sentiment 140](http://help.sentiment140.com/for-students) to generate pre-trained word vectors, which could be used in the following sentiment analysis.

The [clean_sentiment140.py](https://github.com/bright1993ff66/Social-Media-Data-Analysis/blob/master/train_word_vectors_from_sentiment140/clean_sentiment140.py) contains the codes about how to clean the text of this dataset.

Then based on the cleaned file, the [generate_word_vector_using_fasttext.py](https://github.com/bright1993ff66/Social-Media-Data-Analysis/blob/master/train_word_vectors_from_sentiment140/generate_word_vector_using_fasttext.py) uses [FastText model from Gensim](https://radimrehurek.com/gensim/models/fasttext.html) to generate word vectors.



