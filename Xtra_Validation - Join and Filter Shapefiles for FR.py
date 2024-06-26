# -*- coding: utf-8 -*-
"""
To use this script, you will need to have been using FugroViewer to produce POI files
Each POI file will be derived from a LAS and the initial shapes files from a project's VA shapefiles
The initial shapes file will have a nearly identical name to the POI file

@author: mburnett
"""

import os
import geopandas as gpd

# Directories containing point and polygon shapefiles
pts_directory = r'C:\Users\--directory--'
poly_directory = r'C:\Users\--directory--'
output_fileName = "Michael_FR.shp"

# Function to read all shapefiles from a directory
def join_and_filter_shp(pts_directory, poly_directory):
    counter = 1
    pts_files = os.listdir(pts_directory)
#    poly_files = os.listdir(poly_directory)
    for file in pts_files:
        if file.endswith(".shp"):
            temp_poly_file = file.replace("_poi","InitialShapes")
            pts_shp = gpd.read_file(os.path.join(pts_directory, file))
            poly_shp = gpd.read_file(os.path.join(poly_directory, temp_poly_file))
            spatial_join_polys = gpd.sjoin(poly_shp,pts_shp,how='left',op='contains')
            # Filter out rows where 'index_right' is NaN
            filtered_sj = spatial_join_polys[~spatial_join_polys['index_right'].isna()]
            # use the first SHP as the SHP to append everything to
            if counter == 1: 
                shapefile_out = filtered_sj
            else:
                shapefile_out = shapefile_out.append(filtered_sj)
            counter += 1
            
    return shapefile_out

out_shp = join_and_filter_shp(pts_directory, poly_directory)

# Output the result to a new shapefile
output_path = r"C:\Users\--directory--\Validation\TestPlatforms\FR"
output = os.path.join(output_path,output_fileName)
out_shp.to_file(output)

print(f"Output written to {output}")