import numpy as np
import os
import sys
import arcpy
import re
import arcgisscripting
import pandas as pd
import read_data
import csv

# Give the workspace first
workspace_path = \
    r'F:\CityU\Datasets\Hong Kong Tweets 2017\transit_non_transit_comparision\before_and_after\compare_tn_and_nontn'
arcpy.env.workspace = workspace_path
arcpy.env.overwriteOutput = True
# List all the shapefiles in the work space
feature_classes = arcpy.ListFeatureClasses()
print "The feature classes in this directory are: "
print feature_classes
print "Then, set the coordinate system..."
WGS84 = arcpy.SpatialReference('WGS 1984')
# Set all the shapefile in the WGS 1984 system
for file in feature_classes:
    arcpy.DefineProjection_management(file, WGS84)
print 'Done!'

print type(feature_classes)

# Create the personal GDB management: could only be run for once
# arcpy.CreatePersonalGDB_management(
#     out_folder_path=
#     "F:/CityU/Datasets/Hong Kong Tweets 2017/transit_non_transit_comparision/before_and_after/compare_tn_and_nontn",
#     out_name="myDatabase", out_version="CURRENT")


class TPU(object):

    # Class variables, which include
    # tpu area shapefile
    tpu_4326 = 'tpu_4326.shp'
    # the center of each transit neighborhood
    tn_points = 'tn_in_tpu.shp'
    # the tweets shapefile
    tweets = 'tweets_in_tpu.shp'
    # tn intersected tpus
    tn_tpus = np.load(os.path.join(read_data.transit_non_transit_comparison, 'tn_tpus.npy'))
    # tpus not intersected with tn
    non_tn_tpus = np.load(os.path.join(read_data.transit_non_transit_comparison, 'non_tn_tpus.npy'))
    # whole tpu set
    whole_tpus = np.concatenate([tn_tpus, non_tn_tpus])

    def __init__(self, tpu_name):
        self.tpu_name = tpu_name
        self.tn_tpu_or_not = tpu_name in TPU.tn_tpus

    @staticmethod
    def get_tweets_for_one_tpu(tpu_name):
        # Load the tweets feature layer
        arcpy.MakeFeatureLayer_management(in_features=TPU.tweets, out_layer='tweets_lyr')
        selected_tweets_layer = arcpy.SelectLayerByAttribute_management(in_layer_or_view='tweets_lyr',
                                                                     selection_type='NEW_SELECTION',
                                                                     where_clause=""" "SmallTPU" = '{}' """.format(
                                                                         tpu_name))
        if ('&' in tpu_name) or ('-' in tpu_name):
            final_name = re.sub('[\&\-\,]', '_', tpu_name)
            arcpy.CopyFeatures_management(in_features=selected_tweets_layer,
                                              out_feature_class=os.path.join(
                                                  read_data.cross_sectional_data_path, final_name,
                                                  final_name+'_tweets'))
        else:
            final_name = tpu_name
            arcpy.CopyFeatures_management(in_features=selected_tweets_layer,
                                          out_feature_class=os.path.join(
                                              read_data.cross_sectional_data_path, final_name,
                                              final_name + '_tweets'))
        TPU.transform_local_shapefile_to_csv(file_path=os.path.join(read_data.cross_sectional_data_path, final_name),
                                             shapefile=final_name+'_tweets.shp', csvfile=final_name+'_tweets.csv')


    @staticmethod
    def get_field_names(file_path, file):
        directory = os.path.join(file_path, file)
        columns = [field.name for field in arcpy.ListFields(directory)]
        return columns

    @staticmethod
    def read_local_shapefile(file_path, file, layer_name):
        directory = os.path.join(file_path, file)
        arcpy.MakeFeatureLayer_management(in_features=directory, out_layer=layer_name)

    @staticmethod
    def transform_local_shapefile_to_csv(file_path, shapefile, csvfile):
        input_directory = os.path.join(file_path, shapefile)
        target_directory = os.path.join(file_path, csvfile)
        fields = arcpy.ListFields(input_directory)
        field_names = [field.name for field in fields]
        with open(target_directory, 'wb') as f:
            w = csv.writer(f)
            w.writerow(field_names)
            for row in arcpy.SearchCursor(input_directory):
                field_vals = [row.getValue(field.name) for field in fields]
                w.writerow([unicode(value).encode('utf-8') for value in field_vals])
                del row


if __name__ == '__main__':

    # Read the file which saves the location of MTR stations
    tpu_dataframe = pd.read_csv(os.path.join(read_data.tpu_4326_data_path, 'tpu_data.csv'))
    tpus_list = list(tpu_dataframe['SmallTPU'])
    for tpu in tpus_list:
        if ('&' in tpu) or ('-' in tpu):
            final_name_to_use = re.sub('[\&\-\,]', '_', tpu)
            print 'Creating folder for TPU: {}'.format(final_name_to_use)
            try:
                os.mkdir(os.path.join(read_data.cross_sectional_data_path, final_name_to_use))
            # The os.mkdir command could not create new folder with the same name
            except WindowsError:
                pass
        else:
            final_name_to_use = tpu
            print 'Creating folder for TPU: {}'.format(final_name_to_use)
            try:
                os.mkdir(os.path.join(read_data.cross_sectional_data_path, final_name_to_use))
            # The os.mkdir command could not create new folder with the same name
            except WindowsError:
                pass

    print 'Generating tweets for each TPU....'

    tpu_list = TPU.whole_tpus
    for tpu in tpus_list:
        if ('&' in tpu) or ('-' in tpu):
            final_name_to_use = re.sub('[\&\-\,]', '_', tpu)
            print 'generating tweets for TPU: {}'.format(final_name_to_use)
            TPU.get_tweets_for_one_tpu(tpu_name=final_name_to_use)
        else:
            final_name_to_use = tpu
            print 'generating tweets for TPU: {}'.format(final_name_to_use)
            TPU.get_tweets_for_one_tpu(tpu_name=final_name_to_use)