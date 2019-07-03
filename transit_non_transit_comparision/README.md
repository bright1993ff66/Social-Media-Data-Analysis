# Description :world_map:

This folder contains codes needed to cope with geographic shapefiles of Hong Kong and the collected tweets. All the codes could be split into the following categories:

## 1. Prepare the data and paths

Firstly, we list all the directories needed in this folder

- The paths being used in dealing with shapefiles are given here: [read_data.py](https://github.com/bright1993ff66/Social-Media-Data-Analysis/blob/master/transit_non_transit_comparision/read_data.py)
- Other paths could be found in [here](https://github.com/bright1993ff66/Social-Media-Data-Analysis/blob/master/read_data.py)

Then, in the codes, the following three files could be prepared previously using [ArcMap](http://desktop.arcgis.com/en/arcmap/):

- ```tpu_4326.shp``` is the shapefile of Tertiary Planning Units(TPUs) in Hong Kong, which is equivalent to the shapefiles saved in [here](https://github.com/bright1993ff66/Social-Media-Data-Analysis/tree/master/Datasets/shapefiles).
- ```tn_in_tpu.shp``` is the shapefile which contains the location of each MTR station and the TPU unit that each MTR station belongs to
- ```tweets_in_tpu.shp``` is the shapefile which contains the geoinformation of the collected tweets and the TPU unit that each tweet belongs to

Based on the shapefiles above, we start preparing the datasets for the cross sectional study and longitudinal study.

### 1.1 Prepare the dataset for cross sectional study

<span style='color:red'>In the cross sectional study</span>, the TPU units which intersect with 500-meter geographic buffer are classified as TN-TPUs while the other TPUs are classified as the Non-TN TPUs:

- The code [find_intersected_tpus.py](https://github.com/bright1993ff66/Social-Media-Data-Analysis/blob/master/transit_non_transit_comparision/find_tweets_in_each_tn.py) helps us to find all the TPUs which intersect with the 500-meter station buffers. 
- The following figure shows the TN TPUs and Non-TN TPUs in the cross sectional study: ![TN TPUs and Non-TN TPUs](https://github.com/bright1993ff66/Social-Media-Data-Analysis/blob/master/Figures/tn_tpus_nontn_tpus.png)

Then, to get all the collected tweets in each TPU,  we use the code [prepare_datasets_cross_sectional.py](https://github.com/bright1993ff66/Social-Media-Data-Analysis/blob/master/transit_non_transit_comparision/prepare_datasets_cross_sectional.py) to add the TPU information to each tweet.

### 1.2 Prepare the dataset for longitudinal study

<span style='color:red'>In the longitudinal study</span>, we need to get tweets for both the treatment group and the control group:

- For instance, for Ho Man Tin MTR station, the following plot shows the buffer area, annulus area and the tweets posted in these areas: ![Longitudinal Plot](https://github.com/bright1993ff66/Social-Media-Data-Analysis/blob/master/Figures/longitudinal_study_plot.png)
- The codes needed to locate the tweets in these areas could be found in [prepare_datasets_longitudinal.py](https://github.com/bright1993ff66/Social-Media-Data-Analysis/blob/master/transit_non_transit_comparision/prepare_datasets_longitudinal.py)

## 2. Cross Sectional and Longitudinal Studies

<span style='color:red'>For the cross sectional study</span>, the code [cross_sectional_study.py](https://github.com/bright1993ff66/Social-Media-Data-Analysis/blob/master/transit_non_transit_comparision/cross_sectional_study.py) load the social demographic variables, normalize these data, and then build the regression model between the sentiment and the social demographic variables. All the social demographic variables we considered could be found in this [site](https://www.bycensus2016.gov.hk/en/bc-dp-tpu.html)

<span style='color:red'>For the longitudinal study</span>, the code [before_and_after_study.py](https://github.com/bright1993ff66/Social-Media-Data-Analysis/blob/master/transit_non_transit_comparision/before_and_after_study.py) draw the dashed line plot of sentiment for newly built stations on a monthly basis from July 7, 2016 to December 31, 2017. Moreover, it also shows the difference in tweet content before and after the opening of these stations, by drawing the wordcloud and topic modelling result

## 3. Difference in Difference Analysis

At last, the code [difference_in_difference_analysis.py](https://github.com/bright1993ff66/Social-Media-Data-Analysis/blob/master/transit_non_transit_comparision/difference_in_difference_analysis.py) illustrates how to build the difference in difference model to check the effectiveness of the transit neighborhood investment in the longitudinal study. 

The idea of selecting treatment and control groups and how to build the DID model are given in the following papers:

- [Transit-oriented economic development: The impact of light rail on new business starts in the Phoenix, AZ Region, USA](https://journals.sagepub.com/doi/full/10.1177/0042098017724119)
- [Do light rail transit investments increase employment opportunities? The case of Charlotte, North Carolina](https://rsaiconnect.onlinelibrary.wiley.com/doi/full/10.1111/rsp3.12184)

## More results would be updated in the following month...:blush:

