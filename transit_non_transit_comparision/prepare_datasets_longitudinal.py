import numpy as np
import os
import sys
import time
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


class MTR_Station(object):

    # Class variables, which include
    # tpu area shapefile
    tpu_4326 = 'tpu_4326.shp'
    # the center of each transit neighborhood
    tn_points = 'tn_in_tpu.shp'
    # the tweets shapefile
    tweets = 'tweets_in_tpu.shp'
    # stations involved in the before and after study
    before_after_stations = ['Whampoa', 'Ho Man Tin', 'South Horizons', 'Wong Chuk Hang', 'Ocean Park',
                             'Lei Tung']

    def __init__(self, station_name):
        self.station_name = station_name

    def create_buffer(self, buffer_radius, saving_path, saved_file_name):
        # The 'tn_points_lyr' could only be created for once. So add a try except pair
        arcpy.MakeFeatureLayer_management(in_features=MTR_Station.tn_points, out_layer='tn_points_lyr')
        buffer_argument = str(buffer_radius) + ' Meters'
        selected_layer = arcpy.SelectLayerByAttribute_management(in_layer_or_view='tn_points_lyr',
                                                                     selection_type='NEW_SELECTION',
                                                                     where_clause=""" "Name" = '{}' """.format(
                                                                         self.station_name))
        # create buffer
        arcpy.Buffer_analysis(in_features=selected_layer,
                                  out_feature_class=os.path.join(saving_path, saved_file_name),
                                  buffer_distance_or_field=buffer_argument,
                                  line_side="FULL",
                                  line_end_type="ROUND",
                                  dissolve_option="NONE",
                                  dissolve_field="",
                                  method="PLANAR")

    def erase_center_circle(self, bigger_radius, smaller_radius):
        """
        :param bigger_radius: the radius of the bigger circle
        :param smaller_radius:  the radius of the smaller circle
        """
        MTR_Station.read_local_shapefile(file_path=os.path.join(read_data.longitudinal_data_path, self.station_name),
                                         file=self.station_name+'_{}m_buffer_lyr.shp'.format(bigger_radius),
                                         layer_name=self.station_name+'_{}m_bigger_buffer_lyr'.format(bigger_radius))
        MTR_Station.read_local_shapefile(file_path=os.path.join(read_data.longitudinal_data_path, self.station_name),
                                         file=self.station_name + '_{}m_buffer_lyr.shp'.format(smaller_radius),
                                         layer_name=self.station_name + '_{}m_smaller_buffer_lyr'.format(smaller_radius))
        saved_file = os.path.join(read_data.longitudinal_data_path, self.station_name,
                                  self.station_name+'{}_erase_{}_layer'.format(bigger_radius, smaller_radius))
        arcpy.Erase_analysis(in_features=self.station_name+'_{}m_bigger_buffer_lyr'.format(bigger_radius),
                             erase_features=self.station_name + '_{}m_smaller_buffer_lyr'.format(smaller_radius),
                             out_feature_class=saved_file)

    def get_tweets_for_annulus(self, bigger_radius, smaller_radius):
        # Load the select_features
        MTR_Station.read_local_shapefile(file_path=os.path.join(read_data.longitudinal_data_path, self.station_name),
                                         file=self.station_name+'{}_erase_{}_layer.shp'.format(bigger_radius, smaller_radius),
                                         layer_name=self.station_name+'_selected_annulus_lyr')
        # Load the tweets layer
        arcpy.MakeFeatureLayer_management(in_features=MTR_Station.tweets, out_layer='tweets_lyr')
        # Select tweets using SelectLayerByLocation_management
        selected_tweets = arcpy.SelectLayerByLocation_management(in_layer='tweets_lyr',
                                                                 select_features=self.station_name+'_selected_annulus_lyr',
                                                                 overlap_type='COMPLETELY_WITHIN')
        selected_tweets_in_tpu = arcpy.SelectLayerByAttribute_management(in_layer_or_view=selected_tweets,
                                                                         selection_type='SUBSET_SELECTION',
                                                                         where_clause=""" "SmallTPU" <> '' """)
        output_directory = os.path.join(read_data.longitudinal_data_path, self.station_name,
                                        self.station_name+'_tweets_annulus',
                                        self.station_name+'_{}_erase_{}.shp'.format(bigger_radius, smaller_radius))
        arcpy.CopyFeatures_management(in_features=selected_tweets_in_tpu, out_feature_class=output_directory)
        # Transform the local shapefile to csv
        MTR_Station.transform_local_shapefile_to_csv(file_path=os.path.join(read_data.longitudinal_data_path, self.station_name,
                                        self.station_name+'_tweets_annulus'),
                                                     shapefile=self.station_name+'_{}_erase_{}.shp'.format(bigger_radius, smaller_radius),
                                                     csvfile=self.station_name+'_{}_erase_{}.csv'.format(bigger_radius, smaller_radius))

    def get_tweets_for_buffer(self, buffer_radius):
        # Load the select_features
        MTR_Station.read_local_shapefile(file_path=os.path.join(read_data.longitudinal_data_path, self.station_name),
                                         file=self.station_name + '_{}m_buffer_lyr.shp'.format(buffer_radius),
                                         layer_name=self.station_name + '_{}m_buffer_lyr'.format(buffer_radius))
        # Load the tweets layer
        arcpy.MakeFeatureLayer_management(in_features=MTR_Station.tweets, out_layer='tweets_lyr')
        # Select tweets using SelectLayerByLocation_management
        selected_tweets = arcpy.SelectLayerByLocation_management(in_layer='tweets_lyr',
                                                                 select_features=self.station_name + '_{}m_buffer_lyr'.format(buffer_radius),
                                                                 overlap_type='COMPLETELY_WITHIN')
        selected_tweets_in_buffer = arcpy.SelectLayerByAttribute_management(in_layer_or_view=selected_tweets,
                                                                         selection_type='SUBSET_SELECTION',
                                                                         where_clause=""" "SmallTPU" <> '' """)
        output_directory = os.path.join(read_data.longitudinal_data_path, self.station_name,
                                        self.station_name + '_{}m_tweets.shp'.format(buffer_radius))
        arcpy.CopyFeatures_management(in_features=selected_tweets_in_buffer, out_feature_class=output_directory)
        # Transform the local shapefile to csv
        MTR_Station.transform_local_shapefile_to_csv(
            file_path=os.path.join(read_data.longitudinal_data_path, self.station_name),
            shapefile=self.station_name + '_{}m_tweets.shp'.format(buffer_radius),
            csvfile=self.station_name + '_{}m_tn_tweets.csv'.format(buffer_radius))


    @staticmethod
    def intersect_analysis(TN_buffer_lyr, saving_path, saved_file_name):
        # The 'tn_points_lyr' could only be created for once. So add a try except pair
        arcpy.MakeFeatureLayer_management(in_features=MTR_Station.tpu_4326, out_layer='tpu_lyr')
        shapefile_list = ['tpu_lyr', TN_buffer_lyr]
        arcpy.Intersect_analysis(in_features=shapefile_list,
                                     out_feature_class=os.path.join(saving_path, saved_file_name),
                                     join_attributes="ALL",
                                     cluster_tolerance="-1 Unknown",
                                     output_type="INPUT")

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
    starting_time = time.clock()

    # Read the file which saves the location of MTR stations
    station_location = pd.read_csv(os.path.join(read_data.tweet_2017_path, 'station_location.csv'))
    station_names_list = list(station_location['Name'])
    for index, name in enumerate(station_names_list):
        try:
            if name in MTR_Station.before_after_stations:
                os.mkdir(os.path.join(read_data.longitudinal_data_path, name))
            else:
                pass
        except WindowsError:
            pass

    # Create 500m, 1000m and 1500m buffers based on stations
    print 'Creating the 500m, 1000m, and 1500m buffers for each involved station....'
    for name in MTR_Station.before_after_stations:
        station_object = MTR_Station(station_name=name)
        try:
            station_object.create_buffer(buffer_radius=500,
                                         saving_path=os.path.join(read_data.longitudinal_data_path, name),
                                         saved_file_name=name+'_500m_buffer_lyr')
            station_object.create_buffer(buffer_radius=1000,
                                         saving_path=os.path.join(read_data.longitudinal_data_path, name),
                                         saved_file_name=name + '_1000m_buffer_lyr')
            station_object.create_buffer(buffer_radius=1500,
                                         saving_path=os.path.join(read_data.longitudinal_data_path, name),
                                         saved_file_name=name + '_1500m_buffer_lyr')
        except arcgisscripting.ExecuteError:
            pass
    print 'Done'

    # Use Erase_analysis to create the shapefile for the annulus
    print 'Getting the shapefile for annulus....'
    big_small_circle_pair = [('500', '1000'), ('500', '1500'), ('1000', '1500')]
    for name in MTR_Station.before_after_stations:
        station_object = MTR_Station(station_name=name)
        for pair in big_small_circle_pair:
            try:
                station_object.erase_center_circle(bigger_radius=pair[1], smaller_radius=pair[0])
            except arcgisscripting.ExecuteError:
                pass
    print 'Done!'

    # Use arcpy.SelectLayerByLocation_management to get tweets in each annulus
    print '-------------------------------------------------------'
    print 'Getting tweets for the buffers...'
    for name in MTR_Station.before_after_stations:
        print 'Copeing with {}'.format(name)
        station_object = MTR_Station(station_name=name)
        # choose the radius:[('500', '1000'), ('500', '1500'), ('1000', '1500')]
        radius_list = [500, 1000, 1500]
        for radius in radius_list:
            station_object.get_tweets_for_buffer(buffer_radius=radius)
        print name + ' is done!'
    print 'All of them Done!'
    print '-------------------------------------------------------'

    # Use arcpy.SelectLayerByLocation_management to get tweets in each annulus
    print '-------------------------------------------------------'
    print 'Getting tweets for the annulus...'
    for name in MTR_Station.before_after_stations:
        print 'Copeing with {}'.format(name)
        try:
            os.mkdir(os.path.join(read_data.longitudinal_data_path, name, name+'_tweets_annulus'))
        except WindowsError:
            pass
        station_object = MTR_Station(station_name=name)
        radius_pair_list = [(500, 1000), (500, 1500), (1000, 1500)]
        for radius_pair in radius_pair_list:
            station_object.get_tweets_for_annulus(bigger_radius=radius_pair[1], smaller_radius=radius_pair[0])
        print name+'is done!'
    print 'All of them Done!'

    end_time = time.clock()
    print 'Total time for constructing the datasets for longitudinal study is {}'.format(end_time-starting_time)
    print '-------------------------------------------------------'




