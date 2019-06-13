# Recent Research About the Social Media Data Analysis

This folder will records some recent works about the analysis of social media data, which will cover various topics such as [Machine Learning](<https://en.wikipedia.org/wiki/Machine_learning>), [Natural Language Processing(NLP)](<https://en.wikipedia.org/wiki/Natural_language_processing>), [Computational Linguistics](<https://en.wikipedia.org/wiki/Computational_linguistics>),  and [Intelligent Transportation Systems(ITS)](<https://en.wikipedia.org/wiki/Intelligent_transportation_system>). Social media data contains various datatypes about a specific user, ranging from raw text, time to location. Merging multi-type data structures derived from social media data could be very useful when coping with some real-world problems such as identifying risky comments, traffic event detection, etc.

The following sections are arranged as follows. In Section 1, a list of some research subfields would be given. Each subfield is attached with some relevant  papers. In Section 2, Some popular NLP tools and useful datasets would be shown. Finally in Section 3, some interesting but unanswered research problems are listed and possible solutions are also given.

The contents of this repo would be updated on a regular basis.

## 1. Recent Research Progress in Social Media Data Analysis

Generally speaking, the research based on social media data could be classified into two groups:

- Social media text analysis
- Combine the text in social media with other resources such as time, location, author description, demographic information, etc. to solve some real-world problems

Some hot research topics are given below:

### 1.1 Social Media Text Normalization

This subarea tries to transform the unstructured social media data to the structured raw text, which could be used for the NLP downstream tasks. Normalize the social media data has always been challenging, mainly due to:

- spelling inconsistence
- the free-form adoption of new terms
- regular violations of grammars
- strange tokens(emoji, emoticons, etc.)
- multilingual issue

Some influential works are listed below:

- [A Broad-Coverage Normalization System for Social Media Language][Liu2012]
- ...

### 1.2 Sentiment Analysis of Social Media Data

[Sentiment Analysis](<https://en.wikipedia.org/wiki/Sentiment_analysis>) is a traditional NLP problem which wants to study the subjective sentiment information from raw text. The following lists some recent works about the sentiment analysis of social media data:

- [Sentiment Analysis of Twitter Data][Passonneau]

### 1.3 Event Detection From Social Media

Traffic event detection, abusive comments detection, text style transfer

### 1.4 Response Generation

Automatically generate response based on some comments posted in social media platform

## 2. Some Datasets and Tools

The following Python tools could be very useful for analyzing social media data, which covers text analysis, geographical data analysis(since social media data contains location information), temporal analysis. 

- [SpaCy](<https://spacy.io/>): Build sophisticated NLP models
- [arcpy](<http://desktop.arcgis.com/en/arcmap/10.3/analyze/arcpy/what-is-arcpy-.htm>): Spatial analysis
- [datetime](<https://docs.python.org/3/library/datetime.html>): Cope with the time objects
- [tweepy](<http://www.tweepy.org/>): Fetch tweets in real-time manner
- [Google Places](<https://developers.google.com/places/web-service/intro>): Find latitude and longitude for any places

Of course, packages like [Pandas](<https://pandas.pydata.org/>) and [Tensorflow](<https://www.tensorflow.org/>) are also highly recommended.

Moreover, the following lists some datasets for social media data analysis:

- [Sentiment 140](<http://help.sentiment140.com/for-students>): 1.6 million tweets
- [Tweets with traffic-related labels](<https://data.mendeley.com/datasets/c3xvj5snvv/1>): useful for traffic-related tweet analysis

## 3. Future Research Directions

Waiting to be updated....

## References

[Liu2012]: <https://www.aclweb.org/anthology/P12-1109>	"A Broad-Coverage Normalization System for Social Media Language"
[Passonneau]:  <https://www.aclweb.org/anthology/W11-0705>	"Sentiment Analysis of Twitter Data"



