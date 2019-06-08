import numpy as np
import os
import sys
import arcpy
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


class TransitNeighborhood(object):

    # Class variables, which include
    # tpu area shapefile
    tpu_4326 = 'tpu_4326.shp'
    # the center of each transit neighborhood
    tn_points = 'tn_in_tpu.shp'
    # the tweets shapefile
    tweets = 'tweets_in_tpu.shp'

    def __init__(self, station_name):
        self.station_name = station_name

    def create_buffer(self, buffer_radius, saving_path, saved_file_name):
        # The 'tn_points_lyr' could only be created for once. So add a try except pair
        arcpy.MakeFeatureLayer_management(in_features=TransitNeighborhood.tn_points, out_layer='tn_points_lyr')
        buffer_argument = str(buffer_radius) + ' Meters'
        if self.station_name == 'AsiaWorld':
            selected_layer = arcpy.SelectLayerByAttribute_management(in_layer_or_view='tn_points_lyr',
                                                                     selection_type='NEW_SELECTION',
                                                                     where_clause=""" "Name" = '{}' """.format(
                                                                         'AsiaWorld-Expo'))
        else:
            selected_layer = arcpy.SelectLayerByAttribute_management(in_layer_or_view='tn_points_lyr',
                                                                     selection_type='NEW_SELECTION',
                                                                     where_clause=""" "Name" = '{}' """.format(
                                                                         self.station_name))
        arcpy.Buffer_analysis(in_features=selected_layer,
                                  out_feature_class=os.path.join(saving_path, saved_file_name),
                                  buffer_distance_or_field=buffer_argument,
                                  line_side="FULL",
                                  line_end_type="ROUND",
                                  dissolve_option="NONE",
                                  dissolve_field="",
                                  method="PLANAR")

    def intersect_analysis(self, TN_buffer_lyr, saving_path, saved_file_name):
        # The 'tn_points_lyr' could only be created for once. So add a try except pair
        arcpy.MakeFeatureLayer_management(in_features=TransitNeighborhood.tpu_4326, out_layer='tpu_lyr')
        shapefile_list = ['tpu_lyr', TN_buffer_lyr]
        arcpy.Intersect_analysis(in_features=shapefile_list,
                                     out_feature_class=os.path.join(saving_path, saved_file_name),
                                     join_attributes="ALL",
                                     cluster_tolerance="-1 Unknown",
                                     output_type="INPUT")

    def get_intersected_tpus(self, file_path):
        # Load the tpu feature layer
        arcpy.MakeFeatureLayer_management(in_features=TransitNeighborhood.tpu_4326, out_layer='tpu_lyr')
        # Load the tweets feature layer
        arcpy.MakeFeatureLayer_management(in_features=TransitNeighborhood.tweets, out_layer='tweets_lyr')
        # Load the intersected layer
        arcpy.MakeFeatureLayer_management(
            in_features=os.path.join(file_path, self.station_name+'_intersected_tpus_lyr.shp'),
            out_layer=self.station_name+'_tpu_lyr_tn')
        # Iterate over a cursor and get the selected TPUs which intersect with the corresponding TNs
        cursor = arcpy.SearchCursor(self.station_name+'_tpu_lyr_tn')
        intersected_TPUs_list = []
        for row in cursor:
            intersected_TPUs_list.append(row.getValue(u'SmallTPU'))
        print '----------------------------------------'
        print "Coping with the {} buffer, the interesected TPUs are...".format(self.station_name)
        print intersected_TPUs_list
        # Output a feature layer based on the selected TPUs
        selectedStr = "', '".join(intersected_TPUs_list)
        where_statement = """{} IN ('{}')""".format(arcpy.AddFieldDelimiters('tpu_lyr', "SmallTPU"),
                                                    selectedStr)
        print 'The where statement is: '
        print where_statement
        # arcpy.MakeFeatureLayer_management(in_features=TransitNeighborhood.tpu_4326, out_layer='tpu_lyr')
        selected_tpus_layer = arcpy.SelectLayerByAttribute_management(in_layer_or_view='tpu_lyr',
                                                                      selection_type='NEW_SELECTION',
                                                                      where_clause=where_statement)
        selected_tweets_layer = arcpy.SelectLayerByAttribute_management(in_layer_or_view='tweets_lyr',
                                                                        selection_type='NEW_SELECTION',
                                                                        where_clause=where_statement)
        arcpy.CopyFeatures_management(in_features=selected_tpus_layer, out_feature_class=os.path.join(file_path,
                                                                                                      self.station_name+'_intersected_tpus'))
        # arcpy.CopyFeatures_management(in_features=selected_tweets_layer, out_feature_class=os.path.join(file_path,
        #                                                                                               self.station_name + '_intersected_tweets'))
        # TransitNeighborhood.transform_local_shapefile_to_csv(file_path=file_path,
        #                                                      shapefile=self.station_name + '_intersected_tweets.shp',
        #                                                      csvfile=self.station_name + '_tn_tweets.csv')

    def spatial_join_analysis(self, file_path, file):
        arcpy.MakeFeatureLayer_management(in_features=TransitNeighborhood.tweets, out_layer='tweets_lyr')
        arcpy.MakeFeatureLayer_management(
            in_features=os.path.join(file_path, self.station_name + '_intersected_tpus.shp'),
            out_layer=self.station_name + '_tpus_considered')
        arcpy.SpatialJoin_analysis(target_features=self.station_name + '_tpus_considered',
                                   join_features='tweets_lyr',
                                   out_feature_class=os.path.join(file_path, file),
                                   match_option='CONTAINS')

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
    # #
    # # Get the column names of a local shapefile
    # # path = r'C:\Users\Haoliang Chang\Desktop\check\Tseung Kwan O'
    # # field_names = TransitNeighborhood.get_field_names(file_path=path, file='Tseung Kwan O_lyr.shp')
    # # print(field_names)

    # Read the file which saves the location of MTR stations
    station_location = pd.read_csv(os.path.join(read_data.tweet_2017_path, 'station_location.csv'))
    station_names_list = list(station_location['Name'])
    for index, name in enumerate(station_names_list):
        if name == 'AsiaWorld-Expo':
            # prevent the invalid character error
            new_name = 'AsiaWorld'
            station_names_list[index] = new_name
            try:
                os.mkdir(os.path.join(read_data.intersected_tpu_path, new_name))
            # The os.mkdir command could not create new folder with the same name
            except WindowsError:
                pass
        else:
            try:
                os.mkdir(os.path.join(read_data.intersected_tpu_path, name))
            except WindowsError:
                pass

    for name in station_names_list:
        TNs = TransitNeighborhood(name)
        # First Step: Save the Transit neighborhood area for each station
        TNs.create_buffer(buffer_radius=500, saving_path=os.path.join(read_data.intersected_tpu_path, name),
                          saved_file_name=name+'_buffer_lyr')
        TransitNeighborhood.read_local_shapefile(file_path=os.path.join(read_data.intersected_tpu_path, name),
                                                 file=name+'_buffer_lyr.shp', layer_name=name+'_buffer_lyr')
        # Second Step: Do the intersect analysis, see which TPUs intersect with one specific TN
        TNs.intersect_analysis(TN_buffer_lyr=name+'_buffer_lyr',
                               saving_path=os.path.join(read_data.intersected_tpu_path, name),
                               saved_file_name=name+'_intersected_tpus_lyr')
        TransitNeighborhood.read_local_shapefile(file_path=os.path.join(read_data.intersected_tpu_path, name),
                                                 file=name+'_intersected_tpus_lyr.shp',
                                                 layer_name=name+'_intersected_tpus_lyr')
        # Third Step: Based on the second step, output these TPUs
        TNs.get_intersected_tpus(file_path=os.path.join(read_data.intersected_tpu_path, name))

     # transform the tpu_4326 shapefile to the corresponding csv file
    TransitNeighborhood.transform_local_shapefile_to_csv(file_path=workspace_path,
                                                         shapefile='tpu_4326.shp',
                                                         csvfile='tpu_data.csv')
    # tranform the tweets data to the csv file
    TransitNeighborhood.transform_local_shapefile_to_csv(file_path=workspace_path, shapefile='tweets_in_tpu.shp',
                                                 csvfile='all_tweets_with_tpu.csv')

    # Find the intersected tn tpus and non-tn tpus
    TPU_list = []
    for file in os.listdir(read_data.intersected_tpu_path):
        new_path = os.path.join(read_data.intersected_tpu_path, file)
        dataframe = pd.read_csv(os.path.join(new_path, file + '_tn_tweets.csv'), encoding='latin-1')
        if float('nan') in list(dataframe['sentiment']):
            print('There is something wrong with the dataframe: ', file)
        else:
            pass
        TPU_set = set(list(dataframe['SmallTPU']))
        for tpu in TPU_set:
            TPU_list.append(tpu)
    TPU_list_str = [str(tpu) for tpu in TPU_list]
    tn_tpu_set = set(TPU_list_str)
    whole_tpu_shapefile_dataframe = pd.read_csv(os.path.join(workspace_path, 'tpu_data.csv'))
    whole_tpu_list = [str(tpu) for tpu in list(whole_tpu_shapefile_dataframe['SmallTPU'])]
    non_tpu_set = set(whole_tpu_list) - tn_tpu_set
    tn_tpu_array, non_tn_tpu_array = np.array(list(tn_tpu_set)), np.array(list(non_tpu_set))
    np.save(os.path.join(read_data.transit_non_transit_comparison, 'tn_tpus.npy'), tn_tpu_array)
    np.save(os.path.join(read_data.transit_non_transit_comparison, 'non_tn_tpus.npy'), non_tn_tpu_array)
