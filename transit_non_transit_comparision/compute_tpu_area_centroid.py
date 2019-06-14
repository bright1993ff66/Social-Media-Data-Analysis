import os
import sys
import arcpy
import time
import csv

import read_data

# Give the workspace first
workspace_path = \
    r'XXXXX'
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


def compute_polygon_attribute(layer_name, new_field_name, expression):
    arcpy.AddField_management(layer_name, new_field_name, "FLOAT")
    arcpy.CalculateField_management(in_table=layer_name, field=new_field_name, expression=expression,
                                    expression_type='PYTHON_9.3')


def read_local_shapefile(file_path, file, layer_name):
    directory = os.path.join(file_path, file)
    arcpy.MakeFeatureLayer_management(in_features=directory, out_layer=layer_name)


def get_field_names(file_path, file):
    directory = os.path.join(file_path, file)
    columns = [field.name for field in arcpy.ListFields(directory)]
    return columns


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

    starting_time = time.clock()

    tpu_4326_field_names = get_field_names(file_path=workspace_path, file='tpu_4326.shp')

    print tpu_4326_field_names

    # Compute the area and the centroid of each tpu
    if 'Shape_area' not in tpu_4326_field_names:
        try:
            area_expression = "!SHAPE.AREA@SQUAREKILOMETERS!"  # use square kilometers to measure area
            centroid_expression_lon = "!SHAPE.CENTROID.X!"
            centroid_expression_lat = "!SHAPE.CENTROID.Y!"
            # Load the local shapefile which contains the polygon shapes
            read_local_shapefile(file_path=workspace_path, file='tpu_4326.shp',
                                                     layer_name='tpu_4326_lyr')
            compute_polygon_attribute(layer_name='tpu_4326_lyr',
                                      new_field_name='Shape_area', expression=area_expression)
            compute_polygon_attribute(layer_name='tpu_4326_lyr',
                                      new_field_name='cent_lon', expression=centroid_expression_lon)
            compute_polygon_attribute(layer_name='tpu_4326_lyr',
                                      new_field_name='cent_lat', expression=centroid_expression_lat)
        except Exception as e:
            # If an error occurred, print line number and error message
            tb = sys.exc_info()[2]
            print "Line {0}".format(tb.tb_lineno)
            print e.message
    else:
        print '*****We have got the area and the centroids of TPUs*****'

    # record the tpus and their centroids
    print '\nSaving the tpus with area and centroid geoinformation.....'
    transform_local_shapefile_to_csv(file_path=workspace_path, shapefile='tpu_4326.shp',
                                     csvfile='tpu_4326_with_area_centroids.csv')

    end_time = time.clock()

    print "Total time for computing the area and the centroids for tpus is {}.".format(
        end_time-starting_time)



