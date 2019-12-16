# Analysis of Geo-coded Social Media Data in HK

## 1. Introduction

In this repository, I will show how to analyze the geo-coded social media data posted in Hong Kong. The general procedure is the following:

1. Tweet filtering. For more information, please check the following  Jupyter notebooks:

   - [tweet_filtering_process]( https://github.com/bright1993ff66/Social-Media-Data-Analysis/blob/master/tweet_filtering_final_github.ipynb )
2. Tweet text preprocessing

   -  Please check the [clean the text sample notebook]( https://github.com/bright1993ff66/Social-Media-Data-Analysis/blob/master/clean_the_text_sample.ipynb) for how to get the raw Chinese tweet text
   -  Please check the [tweet cleaning notebook](https://github.com/bright1993ff66/Social-Media-Data-Analysis/blob/master/tweet_cleaning_final_github.ipynb) to know how we clean, translate and preprocess the tweet for this work
3. Generate tweet representation using [FastText](https://fasttext.cc/) word embedding based on [sentiment140](http://help.sentiment140.com/for-students)
4. Manually label the sentiment of 5000 tweets randomly sampled from our tweet dataset
5. Build Sentiment analysis classifiers and conduct cross validation. To check how to train the word embedding model based on sentiment140, please check the [train_word_vectors_from_sentiment140](https://github.com/bright1993ff66/Social-Media-Data-Analysis/tree/master/train_word_vectors_from_sentiment140) folder. To generate the tweet representation for each tweet of our own dataset, please visit the [emoji2vec](https://github.com/bright1993ff66/Social-Media-Data-Analysis/blob/master/generate_tweet_representation/emoji2vec.ipynb) notebook or the code [get_tweet_representation.py](https://github.com/bright1993ff66/Social-Media-Data-Analysis/blob/master/generate_tweet_representation/get_tweet_representation.py)
6. Cross sectional analysis and longitudinal analysis
7. Difference-in-difference analysis
8. Result visualization(word cloud, topic modelling, etc)

## 2. Prerequisite Python Packages

In this project, I am using Python 3.5 to analyze the tweets. You could install all relevant packages by running the following code in the command line:

```shell
pip install -r requirements.txt
```

However, in the [transit_non_transit_comparison](https://github.com/bright1993ff66/Social-Media-Data-Analysis/tree/master/transit_non_transit_comparision) folder, you need the [ArcPy](https://pro.arcgis.com/en/pro-app/arcpy/get-started/what-is-arcpy-.htm) package to do the geographical analysis. This package is only supported in **Python 2+** and could only be imported after downloading the [ArcGIS](https://www.esri.com/en-us/arcgis/about-arcgis/overview).

## 3. Some Results

To be continued.....

## License

[MIT](https://github.com/bright1993ff66/Social-Media-Data-Analysis/blob/master/LICENSE)