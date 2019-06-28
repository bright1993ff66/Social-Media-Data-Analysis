# Datasets Description:star:

This project mainly uses the following datasets：

## 1. Geographic Shapefiles of Hong Kong

The [shapefiles](https://github.com/bright1993ff66/Social-Media-Data-Analysis/tree/master/Datasets/shapefiles) folder contains the geometric shapefiles for Hong Kong, which show the borderlines for the [Small Tertiary Planning Units(TPU)](https://www.bycensus2016.gov.hk/en/bc-dp-tpu.html). The following figure shows the TPU units in HK: ![TPU Units](https://github.com/bright1993ff66/Social-Media-Data-Analysis/blob/master/Figures/HK_TPU.png).

Then, you could use [ArcMap](http://desktop.arcgis.com/en/arcmap/) or [ArcPy](https://pro.arcgis.com/en/pro-app/arcpy/get-started/what-is-arcpy-.htm) to map the tweets on the map.

## 2. Location of Mass Transit Railway(MTR) Stations in Hong Kong

The [station_location](https://github.com/bright1993ff66/Social-Media-Data-Analysis/blob/master/Datasets/station_location.csv) file contains the latitude and longitude information of MTR stations in Hong Kong. The geoinformation of these stations could be computed using the [Google Places API](https://developers.google.com/places/web-service/intro). For more information about how to use this API to find these locations, please go to [this Python3 notebook](https://github.com/bright1993ff66/Social-Media-Data-Analysis/blob/master/Use Google Places API to Get Location of MTR Stations.ipynb).

## 3. The Emoji Dataset

Since this project considers emojis in tweet sentiment classification, we first build the emoji dataset which records the Emoji names, Unicode representation of emojis, R language representation of emojis based on [Full Emoji List, v12.0](https://unicode.org/emoji/charts-12.0/full-emoji-list.html) .

The code which is used to build this dataset is given [here](https://github.com/bright1993ff66/Social-Media-Data-Analysis/blob/master/Datasets/build_emoji_dataframe.py). The emoji dataset we constructed is given  [here](https://github.com/bright1993ff66/Social-Media-Data-Analysis/blob/master/Datasets/emoji.pkl)







