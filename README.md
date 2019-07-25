# Analysis of Geo-coded Social Media Data in HK

## 1. Introduction

In this repository, I will show how to analyze the geo-coded social media data posted in Hong Kong. The general procedure is the following:

1. Tweet filtering 
2. Tweet text preprocessing 
3. Generate tweet representation using [FastText](https://fasttext.cc/) word embedding
4. Locate tweets based on their geoinformation(latitude and longitude)
5. Sentiment analysis model selection and classification
6. Result visualization(word cloud, topic modelling, put the tweets on the map, etc)

## 2. Prerequisite Python Packages

In this project, I am using Python 3.5 to analyze the tweets. You could install all relevant packages by running the following code in the command line:

```shell
pip install -r requirements.txt
```

However, in the [transit_non_transit_comparison](https://github.com/bright1993ff66/Social-Media-Data-Analysis/tree/master/transit_non_transit_comparision) folder, you need the [ArcPy](https://pro.arcgis.com/en/pro-app/arcpy/get-started/what-is-arcpy-.htm) package to do the geographical analysis. This package is only supported in **Python 2+** and could only be imported after downing the [ArcGIS](https://www.esri.com/en-us/arcgis/about-arcgis/overview).

## 3. Some Results

To be continued.....

## License

[MIT](https://github.com/bright1993ff66/Social-Media-Data-Analysis/blob/master/LICENSE)