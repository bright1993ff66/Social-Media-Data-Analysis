# Analysis of Geo-coded Social Media Data in HK

## 1. Introduction

In this repository, I will show how to analyze the geo-coded social media data posted in Hong Kong. The general procedure is the following:

1. Tweet filtering. For more information, please check the following two Jupyter notebooks:

   - [Tweet filtering without visitors](https://github.com/bright1993ff66/Social-Media-Data-Analysis/blob/master/tweet_filtering_process_without_visitors.ipynb)
   - [Tweet filtering process for tweets posted by visitors](https://github.com/bright1993ff66/Social-Media-Data-Analysis/blob/master/tweet_filtering_process_visitors.ipynb)

   The final dataset we use is the combined datafile derived from the above two Jupyter notebooks

2. Tweet text preprocessing

   -  Please check the [clean the text sample notebook]( https://github.com/bright1993ff66/Social-Media-Data-Analysis/blob/master/clean_the_text_sample.ipynb )

3. Generate tweet representation using [FastText](https://fasttext.cc/) word embedding based on [sentiment140](http://help.sentiment140.com/for-students)

4. Manually label 5000 tweets randomly sampled from our tweet dataset

5. Sentiment analysis classifiers cross validation and sentiment prediction

6. Cross sectional analysis and longitudinal analysis

7. Result visualization(word cloud, topic modelling, etc)

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