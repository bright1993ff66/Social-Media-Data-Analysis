import numpy as np
import os
import sys
import time
import arcpy
import arcgisscripting
import pandas as pd
import data_paths
import csv

# Give the workspace first
tweet_filtering_path = r'F:\CityU\Datasets\tweet_filtering'
before_and_after = r'F:\CityU\Datasets\Hong Kong Tweets 2017\transit_non_transit_comparision\before_and_after'
arcpy.env.workspace = os.path.join(before_and_after, 'circle_annulus', 'shapefiles')
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

def read_local_shapefile(file_path, file, layer_name):
    directory = os.path.join(file_path, file)
    arcpy.MakeFeatureLayer_management(in_features=directory, out_layer=layer_name)


if __name__ == '__main__':

    # load the tweet shapefile
    print 'Loading the tweet shapefile...'
    read_local_shapefile(file_path=tweet_filtering_path, file='tweet_2016_2017_final.shp', layer_name='tweet_lyr')
    print 'Done'

    # Create a list for each area which saves the treatment area and annulus areas
    whampoa_ho_man_tin_shapefile_list = [u'500_whampoa_ho_man_tin_dissolve.shp',
                                         u'1000_minus_500_whampoa_dissolve.shp',
                                         u'1500_minus_500_whampoa_dissolve.shp']
    south_horizons_lei_tung_shapefile_list = [u'500_south_horizons_lei_tung_dissolve.shp',
                                              u'1000_minus_500_south_horizons_dissolve.shp',
                                              u'1500_minus_500_south_horizons_dissolve.shp']
    ocean_park_wong_chuk_hang_shapefile_list = [u'500_ocean_park_wong_chuk_hang_dissolve.shp',
                                                u'1000_minus_500_ocean_park_dissolve.shp',
                                                u'1500_minus_500_ocean_park_dissolve.shp']

    print '---------------------------------------------------------'
    print 'Conduct intersect analysis and save index_num list...'
    print '---------------------------------------------------------'
    print 'For Whampoa & Ho Man Tin...'
    for shapefile in whampoa_ho_man_tin_shapefile_list:
        print 'Coping with the shapfile {}'.format(shapefile)
        index_num_list = []
        read_local_shapefile(file_path=os.path.join(before_and_after, 'circle_annulus', 'shapefiles'),
                             file=shapefile, layer_name='{}_lyr'.format(shapefile[:3]))
        output_path = os.path.join(before_and_after, 'circle_annulus', 'intersect_shapefiles')
        arcpy.Intersect_analysis(in_features=['tweet_lyr', '{}_lyr'.format(shapefile[:3])],
                                 out_feature_class=os.path.join(output_path, '{}_intersect.shp'.format(shapefile[:-4])))
        read_local_shapefile(file_path=output_path, file='{}_intersect.shp'.format(shapefile[:-4]),
                             layer_name='{}_intersect_lyr'.format(shapefile[:-4]))
        rows = arcpy.SearchCursor('{}_intersect_lyr'.format(shapefile[:-4]))
        for row in rows:
            index_num_value = row.getValue('index_num')
            index_num_list.append(index_num_value)
        # print "The index_num list is {}".format(index_num_list)
        list_saving_path = os.path.join(before_and_after, 'circle_annulus', 'tweet_for_three_areas', 'index_num_list')
        np.save(os.path.join(list_saving_path, '{}_index_list.npy'.format(shapefile[:-4])), index_num_list)

    print 'For South Horizons & Lei Tung...'
    for shapefile in south_horizons_lei_tung_shapefile_list:
        print 'Coping with the shapfile {}'.format(shapefile)
        index_num_list = []
        read_local_shapefile(file_path=os.path.join(before_and_after, 'circle_annulus', 'shapefiles'),
                             file=shapefile, layer_name='{}_lyr'.format(shapefile[:3]))
        output_path = os.path.join(before_and_after, 'circle_annulus', 'intersect_shapefiles')
        arcpy.Intersect_analysis(in_features=['tweet_lyr', '{}_lyr'.format(shapefile[:3])],
                                 out_feature_class=os.path.join(output_path, '{}_intersect.shp'.format(shapefile[:-4])))
        read_local_shapefile(file_path=output_path, file='{}_intersect.shp'.format(shapefile[:-4]),
                             layer_name='{}_intersect_lyr'.format(shapefile[:-4]))
        rows = arcpy.SearchCursor('{}_intersect_lyr'.format(shapefile[:-4]))
        for row in rows:
            index_num_value = row.getValue('index_num')
            index_num_list.append(index_num_value)
        # print "The index_num list is {}".format(index_num_list)
        list_saving_path = os.path.join(before_and_after, 'circle_annulus', 'tweet_for_three_areas', 'index_num_list')
        np.save(os.path.join(list_saving_path, '{}_index_list.npy'.format(shapefile[:-4])), index_num_list)

    print 'For Ocean Park & Wong Chuk Hang...'
    for shapefile in ocean_park_wong_chuk_hang_shapefile_list:
        print 'Coping with the shapfile {}'.format(shapefile)
        index_num_list = []
        read_local_shapefile(file_path=os.path.join(before_and_after, 'circle_annulus', 'shapefiles'),
                             file=shapefile, layer_name='{}_lyr'.format(shapefile[:3]))
        output_path = os.path.join(before_and_after, 'circle_annulus', 'intersect_shapefiles')
        arcpy.Intersect_analysis(in_features=['tweet_lyr', '{}_lyr'.format(shapefile[:3])],
                                 out_feature_class=os.path.join(output_path, '{}_intersect.shp'.format(shapefile[:-4])))
        read_local_shapefile(file_path=output_path, file='{}_intersect.shp'.format(shapefile[:-4]),
                             layer_name='{}_intersect_lyr'.format(shapefile[:-4]))
        rows = arcpy.SearchCursor('{}_intersect_lyr'.format(shapefile[:-4]))
        for row in rows:
            index_num_value = row.getValue('index_num')
            index_num_list.append(index_num_value)
        # print "The index_num list is {}".format(index_num_list)
        list_saving_path = os.path.join(before_and_after, 'circle_annulus', 'tweet_for_three_areas', 'index_num_list')
        np.save(os.path.join(list_saving_path, '{}_index_list.npy'.format(shapefile[:-4])), index_num_list)