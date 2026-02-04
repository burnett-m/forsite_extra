import rasterio
from rasterio.features import geometry_mask
import matplotlib.pyplot as plt
import os
import geopandas as gpd
gpd.options.io_engine = "pyogrio"
import numpy as np
import pandas as pd
import argparse
import sys
import boto3
from tempfile import TemporaryDirectory


def s1_comparisonWith_CHM(poly, bucket, SHPDir, CHMDir, OutLocal, OutDir, productionPolys, minHeight, s3Client):
    # Load production polygons and access relevant polygon
    prodPolys_gdf = gpd.read_file(productionPolys)
    temp_poly_gdf = prodPolys_gdf[prodPolys_gdf['PRODGRIDID'] == poly]

    # Initialize stats DataFrame
    stats_df = pd.DataFrame(columns=["PRODGRIDID","Px_UncoveredByStage1","Total_Px","UncoveredArea_Percent","AverageHeight_Uncovered"])

    print(f"Processing {poly}")

    # Load the stage1 shapefile
    temp_shp = gpd.read_file(os.path.join(SHPDir,f"{poly}InitialShapes_s1.zip"))
    # Load the CHM raster
    with rasterio.open(os.path.join(CHMDir,f"CHM_HR_{poly}.tif")) as src:
        temp_chm = src.read(1)
        crs = src.crs
        transform = src.transform
        # pixel_area = abs(src.transform[0] * src.transform[4])  # Calculate pixel area from affine transform
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    # Make sure CRS matches
    temp_poly_gdf = temp_poly_gdf.to_crs(crs)

    # Mask out the buffered area outside the production polygon
    mask = geometry_mask(
        geometries=temp_poly_gdf.geometry,
        invert=True,                  # We want to KEEP pixels inside the polygon
        transform=transform,
        out_shape=temp_chm.shape
    )
    # Create a geometry mask representative of the stage1 coverage
    shp_mask = geometry_mask(temp_shp.geometry, out_shape=temp_chm.shape,
                            transform=transform, invert=True)
    # Create a boolean array where pixels are valid (not NaN and above minHeight)
    raster_valid_withBufferedCHM = (temp_chm != np.nan) & (temp_chm >= minHeight)
    raster_valid = raster_valid_withBufferedCHM & mask
    # Identify uncovered areas: valid CHM pixels not covered by stage1 shapefile within the production polygon
    covered_area_mask = raster_valid & shp_mask #& mask


    # --- Step 5: Count and Area ---
    uncovered_pixel_count = np.sum(raster_valid) - np.sum(covered_area_mask)
    valid_pixels_list = temp_chm[covered_area_mask].tolist() # List all valid pixel heights in uncovered area

    # Add stats to DataFrame
    stats_df.loc[len(stats_df)] = [poly, uncovered_pixel_count, np.sum(raster_valid), ((uncovered_pixel_count/np.sum(raster_valid))*100),np.mean(valid_pixels_list)]

    print(f"Pixels not covered by stage1 shapefile: {uncovered_pixel_count}")
    print(f"Total pixel >{minHeight:.0f}m: {np.sum(raster_valid)}")
    print(f"Total uncovered area: {((uncovered_pixel_count/np.sum(raster_valid))*100):.2f}%")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 8))
    # Plot the valid raster data as a background
    show_raster = np.where(raster_valid_withBufferedCHM, temp_chm, np.nan)
    img = ax.imshow(show_raster, cmap='viridis', extent=extent, origin='upper')
    plt.colorbar(img, ax=ax, label=f'CHM (m) >= {minHeight}m')
    # Overlay uncovered areas (red mask)
    masked_overlay = np.where(covered_area_mask, 1, np.nan)
    ax.imshow(masked_overlay, cmap='Reds', alpha=0.8, extent=extent, origin='upper')
    # Plot the polygon outline on top (in red)
    temp_poly_gdf.boundary.plot(ax=ax, color='red', linewidth=2)
    ax.set_title(f"{poly}_CHM Coverage with Uncovered Areas Highlighted")
    
    # Save and close plots
    localCHMcoverage = os.path.join(OutLocal, f"{poly}_CHMcoverage.jpeg")
    plt.savefig(localCHMcoverage, dpi=300, bbox_inches='tight')
    plt.close()
    s3Client.upload_file(localCHMcoverage, bucket, os.path.join(OutDir,f"{poly}_CHMcoverage.jpeg"))
    
    localStatsCsv = os.path.join(OutLocal,f"{poly}_S1chmStats.csv")
    stats_df.to_csv(localStatsCsv, index=False)
    s3Client.upload_file(localStatsCsv, bucket, os.path.join(OutDir,f"{poly}_S1chmStats.csv"))

        
def main():
    ############################

    # Checks for sufficient number of arguments and exits if untrue
    if len(sys.argv) < 8:
        print('Usage:  stage1_QC_batch.py s3Bucket initialShapes_dir chm_dir output_dir productionPolygonsSHP minHeight filesToProcessList NumFilesToProcessPerContainer')
        exit(-1)

    # Adds and sets up arguments with help information to the stage1_QC_batch.py script
    ap = argparse.ArgumentParser()
    ap.add_argument('s3_bucket', help='Name of bucket that contains source DEM/DSM and config')
    ap.add_argument("SHPDir", type=str, help="Main directory containing shapefiles.")
    ap.add_argument("CHMDir", type=str, help="Main directory containing CHM rasters.")
    ap.add_argument("OutDir", type=str, help="Output directory.")
    ap.add_argument("productionPolys", type=str, help="Path to the production polygons shapefile.")
    ap.add_argument("minHeight", type=float, help="Minimum height threshold for CHM (default: 5m).")
    ap.add_argument('filesToProcessList', help='filename of filesToProcessList on the S3')
    ap.add_argument('num_files_to_Process', help='number of files to process with this container instance',type=int)

    # parses the arguments and makes them available for use
    args = ap.parse_args()

    d = vars(args)
    for key in d:
        print(key, ":", d[key])

    # Set up the file index based on the array index  and the number of files to process argument
    if "AWS_BATCH_JOB_ARRAY_INDEX" in os.environ:
        sys.stdout.write("AWS_BATCH_JOB_ARRAY_INDEX : " + os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX') + "\n")
        sys.stdout.flush()
        fileIndexS = os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX')
        fileIndex = int(fileIndexS) * args.num_files_to_Process
    else:
        fileIndex = 0

    # Store s3 bucket name in the input_bucket variable    
    input_bucket=args.s3_bucket

    s3 = boto3.client('s3')

    # List files in directory
    response = s3.list_objects_v2(Bucket=input_bucket, Prefix=args.filesToProcessList.rsplit('/', 1)[0]+'/')
    if 'Contents' in response:
        print(f"Files in {args.SHPDir}:")
        for obj in response['Contents']:
            print(obj['Key'])

    # Download the filestoprocess list
    if len(args.filesToProcessList) > 0: # Exclude base folder name from model folder download
        try:
            s3.download_file(input_bucket, args.filesToProcessList, os.path.basename(args.filesToProcessList), ExtraArgs={'RequestPayer':'requester'})
        except:
            print("Could not download filesToProcessList " + input_bucket + "/" + args.filesToProcessList) 
            exit(-1)
        else:
            print("Downloaded filesToProcessList " + input_bucket + "/" + args.filesToProcessList + "\n")

    if os.path.isfile(os.path.basename(args.filesToProcessList)) > 0:
        list_file = open(os.path.basename(args.filesToProcessList), 'r')
        standID_list = list_file.read().splitlines()

    # Compute the paths
    shpRemote = f"s3://{args.s3_bucket}/{args.SHPDir}"
    chmRemote = f"s3://{args.s3_bucket}/{args.CHMDir}"
    outLocal = TemporaryDirectory() # add delete=False argument for debug
    print(f"Using temp directory {outLocal} for output")
    prodPolysRemote = f"s3://{args.s3_bucket}/{args.productionPolys}"
    
    for x in range(args.num_files_to_Process):
        if fileIndex + x < len(standID_list):
            poly_to_process = standID_list[fileIndex + x]
            poly_to_process = poly_to_process.replace(".laz","").replace(".las","")
            print(f"Processing {poly_to_process} from index {fileIndex + x}")
            s1_comparisonWith_CHM(poly_to_process, args.s3_bucket, shpRemote, chmRemote, outLocal.name, args.OutDir, prodPolysRemote, args.minHeight, s3)


if __name__ == "__main__":
    main()
