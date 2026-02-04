import os
import shutil
from pathlib import Path
import tqdm
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import subprocess
import laspy
from shapely.geometry import MultiPoint, Polygon
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd
import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as cm

# Function to copy files
def copy_files(chunk):
    for src, dst in chunk:
        shutil.copy(src, dst)

# Function to move files
def gtqc_finishedStemCollect(gtqc_FinalDir, vaPrefixPattern, completedGT):
    os.makedirs(os.path.join(gtqc_FinalDir,"voxel"), exist_ok=True) # Create the voxel directory and the 'accepted' and 'rejected' folders within it
    folders = [os.path.join(gtqc_FinalDir,fold) for fold in os.listdir(gtqc_FinalDir) if fold.startswith(vaPrefixPattern)] # Collect only folder that start with the VA naming scheme
    src_dst_pairs = [] # Create full Source and Destination file paths
    for fold in folders:
        trees_dir = os.path.join(gtqc_FinalDir, fold, 'trees')
        unprocessed = [os.path.join(trees_dir, f) for f in os.listdir(os.path.join(trees_dir)) if f.endswith('.las') or f.endswith('.laz')]
        if completedGT == True: # Collected approved and rejected stems
            if not os.path.exists(os.path.join(trees_dir, 'approved')) :
                print(f"Approved folder does not exist in {trees_dir}. Skipping this folder.")
                continue
            if not os.path.exists(os.path.join(trees_dir, 'rejected')):
                print(f"Rejected folder does not exist in {trees_dir}. Skipping this folder.")
                continue
            approved = [os.path.join(trees_dir, 'approved', f) for f in os.listdir(os.path.join(trees_dir, 'approved')) if f.endswith('.las') or f.endswith('.laz')]
            rejected = [os.path.join(trees_dir, 'rejected', f) for f in os.listdir(os.path.join(trees_dir, 'rejected')) if f.endswith('.las') or f.endswith('.laz')]
            os.makedirs(os.path.join(gtqc_FinalDir,"voxel", 'approved'), exist_ok=True)
            os.makedirs(os.path.join(gtqc_FinalDir,"voxel", 'rejected'), exist_ok=True)
        else:
            approved = []
            rejected = []

        master_stem_list = {
            'unprocessed': [],
            'approved': [],
            'rejected': []
        }
        # Add all source files to dictionary
        master_stem_list['unprocessed'].extend(unprocessed)
        master_stem_list['approved'].extend(approved)
        master_stem_list['rejected'].extend(rejected)
        

        # build src, dst pairs
        for src in master_stem_list['unprocessed'] + master_stem_list['approved'] + master_stem_list['rejected']:
            status = 'unprocessed'
            if 'approved' in os.path.dirname(src):
                status = 'approved'
            elif 'rejected' in os.path.dirname(src):
                status = 'rejected'
            
            if status == 'unprocessed':
                dst = os.path.join(gtqc_FinalDir,"voxel", os.path.basename(src))
            else:
                dst = os.path.join(gtqc_FinalDir,"voxel", status, os.path.basename(src))
            src_dst_pairs.append((src, dst))

    # Prepare multi-thread processing
    print(f"There are {len(src_dst_pairs)} files to copy")
    chunk_size = len(src_dst_pairs) // os.cpu_count()
    if chunk_size == 0:
        chunk_size = 1
    chunks = [src_dst_pairs[i:i + chunk_size] for i in range(0, len(src_dst_pairs), chunk_size)]
    print(f"The chunk size is {chunk_size} and there are {len(chunks)} chunks")

    print("--- Starting the multithread processing ---")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = executor.map(copy_files, chunks)
        for result in results:
            pass

    print("--- Finished copying files ---")
    if completedGT == False: # If running the model over GT that hasn't been QC'ed
        # Run the laszip tool
        print("--- Converting LAZ to LAS ---")
        command = 'laszip -i *.laz' # Define the command to be run
        subprocess.run(command, shell=True, cwd=os.path.join(gtqc_FinalDir,"voxel")) # Execute the command in the specified directory

    if completedGT == True:
        # Add labels CSVs
        folderPaths = [os.path.join(gtqc_FinalDir,"voxel", 'rejected'),os.path.join(gtqc_FinalDir,"voxel", 'approved')]
        for folderPath in folderPaths:
            filenames = [f for f in os.listdir(folderPath) if f.endswith(".las")] # List all filenames in the directory (without full path)
            filenames = [f.replace(".las", '') for f in filenames] # Remove the ".laz" extension from each filename
            if len(filenames) == 0:
                filenames = [f for f in os.listdir(folderPath) if f.endswith(".laz")] # List all filenames in the directory (without full path)
                filenames = [f.replace(".laz", '') for f in filenames] # Remove the ".laz" extension from each filename
            label = folderPaths.index(folderPath) # Get label - 0 = Rejected | 1 = approved
            temp_labels_df = pd.DataFrame({'BOX': filenames,'Label': [label] * len(filenames)}) # Create a DataFrame with "BOX" column containing the filenames and "Labels" column
            temp_labels_df.to_csv(os.path.join(folderPath,"Labels.csv"))

            # Run the laszip tool
            print("--- Converting LAZ to LAS ---")
            command = 'laszip -i *.laz' # Define the command to be run
            subprocess.run(command, shell=True, cwd=folderPath) # Execute the command in the specified directory


# Function to read LAZ file and extract points
def read_laz(file_path): # Function to read LAZ file and extract points
    las = laspy.read(file_path) # Read the LAZ file using laspy
    # Extract X, Y, Z coordinates as numpy arrays
    x = las.x
    y = las.y
    z = las.z
    xy_stack = np.column_stack((x, y)) # stack items into an array
    max_z = np.argmax(z) # Get index of max
    # Get the x and y of the max Z location
    max_z_XY = xy_stack[max_z]

    return xy_stack, max_z_XY, np.max(z), np.min(z)

# Function to create convex hull
def create_convex_hull(points): # Function to create convex hull
    multipoint = MultiPoint([tuple(point) for point in points]) # Create a Shapely MultiPoint object
    convex_hull = multipoint.convex_hull # Get the convex hull of the multipoint
    print(type(convex_hull))
    return convex_hull

# Function to process LAS and SHP files
def LASandSHP_prep(gtqc_FinalDir,completedGT):
    print("--- Starting to process convex hull shapefiles ---")
    # Access specific folder paths
    if completedGT == True:
        folderPaths = [os.path.join(gtqc_FinalDir,"voxel", 'rejected'),os.path.join(gtqc_FinalDir,"voxel", 'approved')]
    if completedGT == False:
        folderPaths = [os.path.join(gtqc_FinalDir,"voxel")]

    for folder in folderPaths:
        lazFiles = [os.path.join(folder,file) for file in os.listdir(folder) if file.endswith(".las")] # List LAS files
        if len(lazFiles) == 0:
            print("Did not convert all LAZ to LAS")
        outputSHP = os.path.join(folder,"laz_convex_hull.shp")
        counter = 0
        ch_gdf = gpd.GeoDataFrame()

        def process_laz_file(lazF, counter):
            try:
                points, maxz_XY, maxZ, minZ = read_laz(lazF)  # Read LAS as ind. points
                convex_hull = create_convex_hull(points)  # Convex Hull
                temp_gdf = gpd.GeoDataFrame({
                    "ID": counter,  # Create convex hull SHP attributes
                    "UNIQUEID": os.path.basename(lazF).replace(".las", ""),
                    "ELEVATION": float(minZ),
                    "HEIGHT": float(maxZ) - float(minZ),
                    "PEAKX": float(maxz_XY[0]),
                    "PEAKY": float(maxz_XY[1]),
                    "geometry": [convex_hull]
                })
                temp_gdf.to_file(lazF.replace(".las", ".shp"))  # Write individual convex hull SHP
                return temp_gdf
            except Exception as e:
                print(f"Error processing {lazF}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {executor.submit(process_laz_file, lazF, idx): lazF for idx, lazF in enumerate(lazFiles)}
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                try:
                    result = future.result()
                    if result is not None:
                        ch_gdf = ch_gdf._append(result, ignore_index=True)  # Add to overall SHP
                except Exception as e:
                    print(f"Error in future: {e}")

        try:
            ch_gdf.to_file(outputSHP)  # Write overall SHP
        except Exception as e:
            print(f"Error writing overall SHP: {e}")
        except Exception as e:
            print(e)

# Function to run LasFromCoords
def LasFromCoords(gtqc_FinalDir,completedGT,LasToLocalCoords):
    print("--- Processing LasFromCoords ---")
    # Access specific folder paths
    if completedGT == True:
        folderPaths = [os.path.join(gtqc_FinalDir,"voxel", 'rejected'),os.path.join(gtqc_FinalDir,"voxel", 'approved')]
    if completedGT == False:
        folderPaths = [os.path.join(gtqc_FinalDir,"voxel")]

    for folder in folderPaths:
        lasFiles = [os.path.join(folder,file) for file in os.listdir(folder) if file.endswith(".laz")]
        heightThresh = "0.0" # Predefined threshold
        
        # Produce dictionary with all LAS and SHP files
        file_dict = dict()
        for file in lasFiles:
            if os.path.exists(file) and os.path.exists(file.replace(".laz",".shp")):
                file_dict[file] = file.replace(".laz",".shp")
            else:
                print(f"File: {file} has no shapefile of a convex hull.")

        # Run the LasFromLocalCoords tool
        # print(file_dict)
        os.makedirs(os.path.join(folder,"_"), exist_ok=True)
        for lasF, shpF in tqdm.tqdm(file_dict.items()):
            # try:
            arg = [
                LasToLocalCoords,
                lasF,
                shpF,
                os.path.join(folder,"_"),
                heightThresh]
            # print(arg)
            command = subprocess.Popen(arg) # Run command
            return_code = command.wait()
            # except Exception as e:     
            #     print(f"Failed Seg the Las {lasF} due to {e}.")
            if return_code != 0:  
                print(f"{shpF}: {str(return_code)}.") # if return_code is 0, then correctly output the txt

# Function to use the LasFromPoly
def LASFromPoly(lazDir, polyShp, outputDir, nThreads=4):
    # Locate LASFromPoly tool
    if os.path.exists(os.environ['LOCALAPPDATA'] + '\\Dropbox\\info.json'):
        with open(os.environ['LOCALAPPDATA'] + '\\Dropbox\\info.json') as f:
            LASFromPoly_Path = json.load(f)['business']['root_path'] + r"\Project Data\OR\_TSI30Release\LASFromPoly_5.exe"
    elif os.path.exists(os.environ['APPDATA'] + '\\Dropbox\\info.json'):
        with open(os.environ['APPDATA'] + '\\Dropbox\\info.json') as f:
            LASFromPoly_Path = json.load(f)['business']['root_path'] + r"\Project Data\OR\_TSI30Release\LASFromPoly_5.exe"
    else:
        print("Dropbox\\info.json location not not found")
        exit()

    os.makedirs(outputDir,exist_ok=True)
    # Prepare LASFromPoly command
    args1 = [ 
        LASFromPoly_Path,
        lazDir + "\\",
        polyShp,
        outputDir + "\\",
        "nThreads=" + str(nThreads)
    ]
    # args1 = f"\"{os.path.basename(LASFromPoly_Path)}\" \"{lazDir}\" \"{polyShp}\" \"{outputDir}\" nThreads={nThreads}"

    result = subprocess.run(args1, shell=False, cwd = outputDir) # Run command
    [os.rename(os.path.join(outputDir, file), os.path.join(outputDir, file.replace(".txt",".laz"))) for file in os.listdir(outputDir) if file.endswith(".txt")]

# Function for downloading LAZ files from S3 when there are no GT boxes
def va_laz_to_local(va_merged_stems, productionPolys, s3_lazDir, gtqc_FinalDir):
    va_merged_stems_df = gpd.read_file(va_merged_stems) # Read the shapefile containing the merged stems
    if "BASE_OBJEC" not in va_merged_stems_df.columns:
        print("BASE_OBJECT column not found in the shapefile.")
        va_merged_stems_df["BASE_OBJEC"] = va_merged_stems_df["UNIQUE_ID"] # Create the BASE_OBJECT column from the UNIQUE_ID column
        va_merged_stems_df.to_file(va_merged_stems) # Save the updated shapefile
    if "PRODGRIDID" not in va_merged_stems_df.columns:
        print("PRODGRIDID column not found in the shapefile.")
        productionPolys_df = gpd.read_file(productionPolys) # Read the production polygons shapefile
        if va_merged_stems_df.crs != productionPolys_df.crs:
            print("CRS mismatch, reprojecting...")
            va_merged_stems_df = va_merged_stems_df.to_crs(productionPolys_df.crs)

        va_merged_stems_df = gpd.sjoin(va_merged_stems_df, productionPolys_df, how="left")

    # Prepare the AWS command line
    os.makedirs(os.path.join(gtqc_FinalDir,"voxel"), exist_ok=True) # Create the directory for the laz files if it doesn't exist
    prodGridID = va_merged_stems_df["PRODGRIDID"].unique()
    print(f"There are {len(prodGridID)} unique PRODGRIDID in the shapefile")
    if len(prodGridID) < 2:
        print("No PRODGRIDID found in the shapefile. Using STANDID instead.")
        prodGridID = va_merged_stems_df["STANDID"].unique()
        print(prodGridID)

    commandList = [f"aws s3 cp s3://{s3_lazDir}{prodID}.laz {gtqc_FinalDir}/{prodID}.laz" for prodID in prodGridID]
    print(commandList)

    # Execute the commands in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(subprocess.run, command, shell=True) for command in commandList]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                print(f"Error: {e}")

    # Convert any existing LAZ files in the voxel directory to LAS
    print("Converting LAZ to LAS")
    command = 'laszip -i *.laz' # Define the command to be run
    subprocess.run(command, shell=True, cwd=gtqc_FinalDir) # Execute the command in the specified directory
    os.makedirs(os.path.join(gtqc_FinalDir,"LAS"), exist_ok=True) # Create the LAS directory to isloate the LAS files
    [shutil.move(os.path.join(gtqc_FinalDir,f), os.path.join(gtqc_FinalDir,"LAS",f)) for f in os.listdir(os.path.join(gtqc_FinalDir)) if f.endswith('.laz')] # Move the LAS files to the LAS directory

    print("Clipping the LAZ files to the polygons")
    LASFromPoly(os.path.join(gtqc_FinalDir,"LAS")+"\\", va_merged_stems, os.path.join(gtqc_FinalDir,"voxel")+"\\") # Clip the laz files to the polygons

# Function for downloading LAZ files from S3 for VAs
def va_laz_from_IS(testPlatform_Dir, s3_lazDir, vaLAZ_Dir):#, vaPrefixPattern): #,aws_profile):
    first_is_file = [f for f in os.listdir(testPlatform_Dir) if f.endswith("001InitialShapes.shp")][0]
    vaPrefixPattern = first_is_file.split("001InitialShapes.shp")[0] # Extract the VA prefix pattern from the first IS shapefile found in the directory
    shapefile_outputname = os.path.join(vaLAZ_Dir,"temp_shp", "all_"+vaPrefixPattern+"stems.shp")
    if os.path.exists(shapefile_outputname):
        va_merged_stems_df = gpd.read_file(shapefile_outputname)
        prodGridID = va_merged_stems_df["STANDID"].unique()
    else:
        # Merge all the shapefiles in the testPlatform_Dir that start with the vaPrefixPattern
        va_merged_stems_df = pd.concat([gpd.read_file(os.path.join(testPlatform_Dir, f)) for f in os.listdir(testPlatform_Dir) if f.startswith(vaPrefixPattern) and f.endswith('.shp')], ignore_index=True)
        # Access all prodgridIDs
        prodGridID = va_merged_stems_df["STANDID"].unique()

        if "BASE_OBJEC" not in va_merged_stems_df.columns:
            print("BASE_OBJECT column not found in the shapefile.")
            va_merged_stems_df.loc[:,"BASE_OBJEC"] = va_merged_stems_df.loc[:,"UNIQUE_ID"] # Create the BASE_OBJECT column from the UNIQUE_ID column
        
        # Create NAME column for the LAZ file names
        va_merged_stems_df.loc[:,"NAME"] = va_merged_stems_df.loc[:,"UNIQUE_ID"]+".laz"
        
        # va_merged_stems_df.to_file(os.path.join(vaLAZ_Dir,"temp_shp", shapefile_outputname)) # Save the updated shapefile
        va_merged_stems_df.to_file(shapefile_outputname) # Save the updated shapefile

    # commandList = [f"aws s3 cp s3://{s3_lazDir}{prodID}.laz {os.path.join(vaLAZ_Dir,"temp_laz")}\\\\{prodID}.laz --profile {aws_profile}" for prodID in prodGridID]
    commandList = [f"aws s3 cp s3://{s3_lazDir}{prodID}.laz {os.path.join(vaLAZ_Dir,"temp_laz")}\\{prodID}.laz" for prodID in prodGridID]
    print(commandList)

    # Execute the commands in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(subprocess.run, command, shell=True) for command in commandList]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                print(f"Error: {e}")

    # Prepare the AWS command line
    # os.makedirs(os.path.join(vaLAZ_Dir,"temp","voxel"), exist_ok=True) # Create the directory for the laz files if it doesn't exist
    print(f"There are {len(prodGridID)} unique STANDID in the shapefile")

    print("Clipping the LAZ files to the polygons")
    # LASFromPoly(os.path.join(vaLAZ_Dir,"temp_laz")+"\\", os.path.join(vaLAZ_Dir,"temp_shp", shapefile_outputname), os.path.join(vaLAZ_Dir,"voxel")+"\\") # Clip the laz files to the polygons
    LASFromPoly(os.path.join(vaLAZ_Dir,"temp_laz")+"\\", shapefile_outputname, os.path.join(vaLAZ_Dir,"voxel")+"\\") # Clip the laz files to the polygons  
    
    # Create buffered LAZ files 
    os.makedirs(os.path.join(vaLAZ_Dir,"voxel","buffered"), exist_ok=True) # Create the directory for the buffered laz files if it doesn't exist
    buffered_shapefile_outputname = os.path.join(vaLAZ_Dir,"temp_shp", "all_"+vaPrefixPattern+"stems_buffered.shp")
    buffered_va_merged_stems_df = gpd.GeoDataFrame(va_merged_stems_df.drop(columns="geometry"),geometry=va_merged_stems_df.geometry.buffer(3), crs=va_merged_stems_df.crs)
    buffered_va_merged_stems_df.to_file(buffered_shapefile_outputname) # Save the updated shapefile
    LASFromPoly(os.path.join(vaLAZ_Dir,"temp_laz")+"\\", buffered_shapefile_outputname, os.path.join(vaLAZ_Dir,"voxel","buffered")+"\\") # Clip the laz files to the polygons  

    # Delete temp directories if the number of LAZ files in voxel matches the number of stems in the shapefile
    voxel_laz_files = [file for file in os.listdir(os.path.join(vaLAZ_Dir,"voxel")) if file.endswith('.laz') or file.endswith('.las')]
    subprocess.run(['laszip', '-i', '*.las'], shell=True, cwd=os.path.join(vaLAZ_Dir,"voxel")) # Convert any LAZ files to LAS in the voxel directory
    subprocess.run(['laszip', '-i', '*.las'], shell=True, cwd=os.path.join(vaLAZ_Dir,"voxel","buffered")) # Convert any LAZ files to LAS in the buffered voxel directory
    if len(voxel_laz_files) == len(va_merged_stems_df):
        # Delete source directories
        shutil.rmtree(os.path.join(vaLAZ_Dir,"temp_laz")) # Remove temp_laz directory
        print(f"temp_laz and temp_shp directories deleted")
    else:
        print(f"temp_laz and temp_shp directories NOT deleted as the number of LAZ files in voxel directory {len(os.listdir(os.path.join(vaLAZ_Dir,'voxel')))} does not match the number of stems in the shapefile {len(va_merged_stems_df)}")

    return vaPrefixPattern

# Prepare the Frag stems
def prepareFrag(fileList, frag_idList, output_dir): # Prepare frag
    for file in fileList:
        try:
            temp_shp = gpd.read_file(file)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

        filtered_shp = temp_shp[temp_shp["UNIQUE_ID"].isin(frag_idList)]
        filtered_shp["NAME"] = filtered_shp["UNIQUE_ID"]

        if file == fileList[0]:
            fullSHP = filtered_shp
        else:
            fullSHP = pd.concat([fullSHP,filtered_shp])
        # Access the LAZ/LAS files
        #newFileName = os.path.basename(file)#.replace("InitialShapes.shp",lasORlaz)
    if len(fullSHP) > 0:
        fullSHP.to_file(os.path.join(output_dir,"_FRAG.shp"))

# Run and display the model results
def evaluate_model(y_test,y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    conMat = cm(y_test, y_pred)
    r_precision = conMat[0][0]/(conMat[0][0]+conMat[0][1])
    r_recall = conMat[0][0]/(conMat[0][0]+conMat[1][0])
    a_precision = conMat[1][1]/(conMat[1][1]+conMat[1][0])
    a_recall = conMat[1][1]/(conMat[1][1]+conMat[0][1])
    r_f1 = (2 * r_precision * r_recall) / (r_precision + r_recall)
    a_f1 = (2 * a_precision * a_recall) / (a_precision + a_recall)

    print(f"The model's accuracy is {accuracy*100:.2f}%")
    print("-------------------")
    print("--------  Rej. App.  Precision")
    print(f"Rejected {conMat[0]}    {r_precision:.2f}")
    print(f"Approved {conMat[1]}    {a_precision:.2f}")
    print(f"Recall   {r_recall:.2f}  {a_recall:.2f}")
    print("-------------------")
    print(f"Rejected F1 Score: {r_f1:.3f}")
    print(f"Approved F1 Score: {a_f1:.3f}")

# Define the function that plots the histograms
def plot_dict_of_dicts_histograms(data_dict, results_shp, savedFile, bins=20, figsize=(12, 80), subplot_cols=3,all_usos_values=['0.', '1.', '2.'], title_prefix="Distribution of"):
    """
    Plots histograms of the values for each item in a dictionary of dictionaries
    all within a single figure.

    Args:
        data_dict (dict): A dictionary where keys are identifiers and values are
                           dictionaries containing numerical data.
        bins (int or sequence): The number of bins or the bin edges for the histograms.
        figsize (tuple): The total size of the figure (width, height) in inches.
        subplot_cols (int): The number of columns for the subplots in the figure.
        title_prefix (str): A prefix to add to the title of each histogram.
    """
    num_plots = len(data_dict) # Get the number of plots
    if num_plots == 0: # Check if there are any plots to show
        print("No data to plot.")
        return

    subplot_rows = (num_plots + subplot_cols - 1) // subplot_cols  # Calculate required rows

    fig, axes = plt.subplots(subplot_rows, subplot_cols, figsize=figsize)
    axes = axes.flatten()  # Flatten the 2D array of axes for easy indexing

    for i, (key, inner_dict) in enumerate(data_dict.items()):
        keys = list(inner_dict.keys()) # Access plot names
        values = list(inner_dict.values()) # Access probabilities

        # Create a dictionary to hold counts for each category
        category_counts = defaultdict(lambda: [0] * len(all_usos_values))
        
        # Match the keys with the results_shp NAME column
        matched_data = results_shp[results_shp['NAME'].isin(keys)]
        
        if not matched_data.empty:
            usos_groups = matched_data.groupby('USOS') # Divide data based on USOS groups
            ax = axes[i]
            plot_dict = {}
            # Create a dictionary to hold the probabilities for each USOS group
            for usos_value, group in usos_groups:
                group_keys = group['NAME'].tolist() # Access the stem IDs
                group_values = [inner_dict[k] for k in group_keys if k in inner_dict] # Connect the stem IDs to the probabilities
                plot_dict[usos_value] = group_values # Add all probabilities for each USOS group to dictionary
            # Print probability dictionaries
            print(f"Probabilities for {key}")
            print(plot_dict)
            filtered_plot_dict = {k: v for k, v in plot_dict.items() if v}  # Filter out empty lists
            
            bins = np.linspace(0, 1, 11)  # Define probability bins (e.g., 10 bins between 0 and 1)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])  # for x-axis positioning

            # Collect histogram counts for each group
            hist_data = []
            labels = []
            for group_name, values in filtered_plot_dict.items():
                counts, _ = np.histogram(values, bins=bins)
                hist_data.append(counts)
                labels.append(group_name)

            hist_data = np.array(hist_data) # Convert list to array to help display data

            bottom = np.zeros_like(hist_data[0]) # Tell matplotlib where the bottom parameter is of each bar on the y-axis
            colors = plt.cm.tab10.colors # Define a color map for the bars

            # Create a consistent color mapping for labels
            label_colors = {label: colors[i % len(colors)] for i, label in enumerate(all_usos_values)}

            # Plot the stacked histogram
            for i, counts in enumerate(hist_data):
                ax.bar(bin_centers, counts, width=0.09, bottom=bottom,
                    label=labels[i], color=label_colors[labels[i]])
                bottom += counts

            # Format plot
            ax.set_xticks(bin_centers)
            ax.set_xticklabels([f"{x:.1f}" for x in bin_centers])
            
            # Compute statistics for the histogram
            temp_sum = sum(values)
            temp_mean = temp_sum / len(values)
            temp_median = sorted(values)[len(values) // 2] if len(values) % 2 == 1 else (sorted(values)[len(values) // 2 - 1] + sorted(values)[len(values) // 2]) / 2
            
            ax.axvline(temp_mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {temp_mean:.3f}')
            ax.axvline(temp_median, color='purple', linestyle='dashed', linewidth=1, label=f'Median: {temp_median:.3f}')
            ax.set_title(f"{title_prefix} {key}", fontsize='small')
            ax.set_xlabel("Probability", fontsize='smaller')
            ax.set_ylabel("Frequency", fontsize='smaller')
            ax.legend()


    plt.tight_layout()  # Adjust layout to prevent overlapping titles/labels
    plt.savefig(savedFile, bbox_inches='tight')
    plt.close()

# Rename SHP column for the LASFromPoly tool
def rename_SHPcolumn(file,new_lasPlots_dir, voxel_dir):
    temp_shp = gpd.read_file(file.replace(".shp", ".dbf"))
    temp_shp.rename(columns={'UNIQUE_ID': 'NAME'}, inplace=True)
    temp_shp.to_file(file)
    # Clip the plot laz
    LASFromPoly(new_lasPlots_dir, file, voxel_dir, nThreads=4)