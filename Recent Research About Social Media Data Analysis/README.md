# Recent Research About the Social Media Data Analysis

This folder will record some recent works about the analysis of social media data, which will cover various topics such as [Machine Learning](<https://en.wikipedia.org/wiki/Machine_learning>), [Natural Language Processing(NLP)](<https://en.wikipedia.org/wiki/Natural_language_processing>), [Computational Linguistics](<https://en.wikipedia.org/wiki/Computational_linguistics>),  [Computer Vision](https://en.wikipedia.org/wiki/Computer_vision) and [Intelligent Transportation Systems(ITS)](<https://en.wikipedia.org/wiki/Intelligent_transportation_system>). 

Social media data contains various datatypes about a specific user, ranging from raw text, time to location. Merging multi-type data structures derived from social media data could be very useful when coping with some real-world problems such as identifying risky comments, traffic event detection, etc.

The following sections are arranged as follows. In **Section 1**, a list of some research subfields would be given. Each subfield is attached with some relevant  papers. In **Section 2**, Some popular tools and useful datasets would be listed. Finally in **Section 3**, some interesting but unanswered research problems are shown and possible solutions are also given.

The contents of this repo would be updated on a regular basis :happy:

## 1. Recent Research Progress in Social Media Data Analysis

Social Media Data Analysis has been seen in various research fields. Some hot research topics are given below:

- Social media text analysis
- Social media user profiling, including link prediction, user modeling, etc.
- Combine the text in social media with other resources such as time, location, author description, demographic information, etc.to solve some real-world problems

Some interesting research topics are given below:

### 1.1 Social Media Text Normalization

This subarea tries to transform the unstructured social media data to the structured raw text, which could be used for the NLP downstream tasks. Normalize the social media data has always been challenging, mainly due to:

- spelling inconsistence
- the free-form adoption of new terms
- regular violations of grammars
- strange tokens(emoji, emoticons, etc.)
- multilingual issue

Some influential works are listed below:

- [A Broad-Coverage Normalization System for Social Media Language](https://www.aclweb.org/anthology/P12-1109)
- [ekphrasis](https://github.com/cbaziotis/ekphrasis): A Tweet normalization tool developed in this paper [DataStories at SemEval-2017 Task 4: Deep LSTM with Attention for Message-level and Topic-based Sentiment Analysis](https://www.aclweb.org/anthology/S17-2126)
- ...

### 1.2 Sentiment Analysis of Social Media Data

[Sentiment Analysis](<https://en.wikipedia.org/wiki/Sentiment_analysis>) is a traditional NLP problem which wants to study the subjective sentiment information from raw text. The following lists some recent works about the sentiment analysis of social media data:

- [Sentiment Analysis of Twitter Data](https://www.aclweb.org/anthology/W11-0705)
- [emoji2vec: Learning Emoji Representations from their Description](https://arxiv.org/abs/1609.08359)
- [DataStories at SemEval-2017 Task 4: Deep LSTM with Attention for
  Message-level and Topic-based Sentiment Analysis](https://www.aclweb.org/anthology/S17-2126)
- ...

Moreover, since many social media posts contain videos, some researchers try to directory analyze the emotion of videos. Some representative works are given here:

- [Context-Dependent Sentiment Analysis in User-Generated Videos](https://github.com/SenticNet/contextual-utterance-level-multimodal-sentiment-analysis)
- ...

### 1.3 Event Detection From Social Media

Based on the text from social media, researchers have done a lot of work such as abusive comments detection, traffic event detection. Some works are listed here:

- [Detecting Comments Showing Risk for Suicide in YouTube](https://link.springer.com/chapter/10.1007/978-3-030-02686-8_30) :kissing_heart:
- [A deep learning approach for detecting traffic accidents from social media data](https://www.sciencedirect.com/science/article/pii/S0968090X1730356X)
- ...

### 1.4 Response Generation

Automatically generate response based on some comments posted in social media platform. Some works are given here:

- [Data-Driven Response Generation in Social Media](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/mt_chat.pdf)
- ...

### 1.5 Text Style Transfer

Social Media text has a totally different text style comparing to formal text such as news and Wiki articles. The text style transfer could be used to forbid the offensive language in the social media:

- [Style Transfer from Non-Parallel Text by Cross Alignment](https://papers.nips.cc/paper/7259-style-transfer-from-non-parallel-text-by-cross-alignment.pdf)

- [Fighting Offensive Language on Social Media with Unsupervised Text Style Transfer](https://www.aclweb.org/anthology/P18-2031.pdf)
- ...

### 1.6 Social Media User Profiling

This task wants to describe each user and the user interaction using mathematical representations(waiting to be updated...)

## 2. Some Datasets and Python Tools

The following Python tools could be very useful for analyzing social media data, which covers text analysis, geographical data analysis(since social media data contains location information), temporal analysis. 

- [SpaCy](<https://spacy.io/>): Build sophisticated NLP models
- [NLTK](https://www.nltk.org/): Classic NLP tool
- [Gensim](https://radimrehurek.com/gensim/): Very useful tools to build topic models
- [arcpy](<http://desktop.arcgis.com/en/arcmap/10.3/analyze/arcpy/what-is-arcpy-.htm>): Spatial analysis
- [datetime](<https://docs.python.org/3/library/datetime.html>): Cope with the time objects
- [tweepy](<http://www.tweepy.org/>): Fetch tweets in real-time manner
- [Google Places](<https://developers.google.com/places/web-service/intro>): Find latitude and longitude for any places
- [Google Translate](https://cloud.google.com/translate/): Translate multiple sources languages to the target language you want

Of course, packages like [Pandas](<https://pandas.pydata.org/>) and [Tensorflow](<https://www.tensorflow.org/>) are also highly recommended.

Moreover, the following lists some datasets for social media data analysis:

- [Datasets for Social Network Analysis](https://aminer.org/data-sna): A great collection of social media datasets. 
- [Social Computting Datasets](http://socialcomputing.asu.edu/pages/datasets): This collection of dataset, maintained by Arizona State University, contains network data from multiple types of social network
- [Higgs Twitter Dataset](https://snap.stanford.edu/data/higgs-twitter.html): This dataset has been built after monitoring the spreading processes on Twitter before, during and after **the announcement of the discovery of a new particle with the features of the elusive Higgs boson on 4th July 2012**.
- [BlogCatalog](http://socialcomputing.asu.edu/datasets/BlogCatalog): This dataset is a social blog directory which manages the bloggers and their blogs. Meanwhile, the bloggers also provide their interested topics, which are regarded as labels
- [Sentiment 140](<http://help.sentiment140.com/for-students>): 1.6 million tweets
- [Tweets with traffic-related labels](<https://data.mendeley.com/datasets/c3xvj5snvv/1>): useful for traffic-related tweet analysis
- [TwitterCrawl](https://wiki.illinois.edu/wiki/display/forward/Dataset-UDI-TwitterCrawl-Aug2012): This dataset is a subset of Twitter. It contains 284 million following relationships, 3 million user profiles and 50 million tweets. The dataset was collected at May 2011.
- [Chinese NLP Corpus](https://github.com/SophonPlus/ChineseNlpCorpus): A large collection of Chinese 'Tweet' dataset, mostly from Chinese social media Platform like [sina weibo](https://www.weibo.com/login.php)

## 3. Future Research Directions

Waiting to be updated soon...:blue_heart:

