import numpy as np
import pandas as pd
import os
import tqdm
import stopit
import utils
from scipy.stats import kurtosis
from scipy.spatial import ConvexHull
import math
import glob
import laspy
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from utils.GapFraction_utils import *

CONCURRENT_PROCESSES = 6

def get_voxel_descriptors(txt_file):
    # start_time = time.time()
    # print("processing: " + txt_file)
    pc_data = pcDataset(txt_file)
    pc_data.setParams(
        voxelSize = None,
        voxelNum = [7,7,7],
        rotAxis = "Z",
        rotAngle = 0.0,
        spacing=None,
        voxelNormalization=False,
        pcNormalization=True
    )
    # step0_time = time.time() - start_time
    voxel = pc_data.pcVoxelization()
        
    # step1_time = time.time() - start_time - step0_time
    
    max_x, max_y, max_z = max(voxel.keys())
    ptDensity = np.zeros((max_x + 1, max_y + 1, max_z + 1))
    ptExtCoeff = np.zeros((max_x + 1, max_y + 1, max_z + 1))
    ptGapFrac = np.zeros((max_x + 1, max_y + 1, max_z + 1))
    # step2_time = time.time() - start_time - step1_time
    
    for voxel_index, voxel_info in voxel.items():
        x, y, z = voxel_index
        ptDensity[x, y, z] = len(voxel_info['points'])
        ptExtCoeff[x, y, z] = np.round(voxel_info['projCoeff'], 10)
        ptGapFrac[x, y, z] = np.round(voxel_info['gapFrac'], 10)
    # step3_time = time.time() - start_time - step1_time - step2_time
    
    flat_density = ptDensity.flatten().tolist()
    flat_extinction = ptExtCoeff.flatten().tolist()
    flat_gap_fraction = ptGapFrac.flatten().tolist()
    # step4_time = time.time() - start_time - step1_time - step2_time - step3_time
    
    box_name = os.path.basename(os.path.dirname(txt_file))
    #box_name = "_".join(tree_name.split("_")[:-2])
    
    row = flat_density + flat_extinction + flat_gap_fraction + [box_name]#, tree_name]
    # step5_time = time.time() - start_time - step1_time - step2_time - step3_time - step4_time
    
    # print(step0_time, step1_time, step2_time, step3_time, step4_time, step5_time)
    
    return row

def process_tree_list(tree_list, concurrent_processes=CONCURRENT_PROCESSES, run_in_parallel=True):
    f"""This function processes a list of tree files and returns a DataFrame of voxel descriptors.

    Args:
        tree_list (list): A list of paths to tree txt files.
        concurrent_processes (int, optional): The number of concurrent processes to use. Defaults to {CONCURRENT_PROCESSES}.

    Returns:
        pandas.DataFrame: A DataFrame of voxel descriptors. The columns are as follows:
            - {", ".join([f"{metric}{{x}}_{{y}}_{{z}}" for metric in ["ptDensity", "ptExtCoeff", "ptGapFrac"] for x in range(7) for y in range(7) for z in range(7)])}
            - BOX
            - TREE
    """
    base_column_names = [
        f"{metric}{x}_{y}_{z}"
        for metric in ["ptDensity", "ptExtCoeff", "ptGapFrac"]
        for x in range(7) for y in range(7) for z in range(7)
    ] + ["BOX"]#, "TREE"]
    results = []

    if run_in_parallel == True:
        with ProcessPoolExecutor(concurrent_processes) as executor:
            #results = list(tqdm(executor.map(get_voxel_descriptors, tree_list), total=len(tree_list)))
            # Submit each task and collect the futures
            futures = {executor.submit(get_voxel_descriptors, tree): tree for tree in tree_list}

            for future in tqdm(futures, total=len(futures)):
                tree = futures[future]
                try:
                    # Try to get the result within the timeout period
                    result = future.result(timeout=10)
                    results.append(result)
                    #print(f"Finished processing tree: {tree}")
                except TimeoutError as t:
                    # If the function takes too long, skip and move on
                    print(t)
                    print(f"Timeout occurred for tree: {tree}")
                    future.cancel()  # Cancel the not done task
                    continue
                except Exception as e:
                    # Handle any other exceptions that might occur during processing
                    print(f"Error occurred for tree: {tree}, error: {e}")
    if run_in_parallel == False:
        results = []

        for tree in tqdm(tree_list, total=len(tree_list)):
            try:
                result = get_voxel_descriptors(tree)
                results.append(result)
                # print(f"Finished processing tree: {tree}")
            except Exception as e:
                print(f"Error occurred for tree: {tree}, error: {e}")
                continue
        
    return pd.DataFrame(results, columns=base_column_names)

# Function to extract geometric features from LAZ files
def extract_geometric_features(file_path):
    try:
        # Open LAZ file using laspy
        las = laspy.read(file_path)
        print(f"Processing {file_path} with {len(las.points)} points.")
        
        # Extract point cloud data (x, y, z coordinates)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        points -= points.min(axis=0)

        # Feature 1: Bounding box dimensions (max - min for x, y, z)
        bbox_dimensions = np.max(points, axis=0) - np.min(points, axis=0)

        # Feature 2: Variance in x, y, z directions (spread of points)
        variance = np.var(points, axis=0)

        # Feature 3: Point density (number of points / volume of bounding box)
        bbox_volume = np.prod(bbox_dimensions)
        point_density = len(points) / bbox_volume if bbox_volume > 0 else 0

        # Feature 4: Centroid (mean position of the points)
        centroid = np.mean(points, axis=0)

        # Feature 5: Eigenvalues of the covariance matrix (describing point cloud shape)
        cov_matrix = np.cov(points.T)
        eigenvalues, _ = np.linalg.eig(cov_matrix)

        # Features 6: Mean, StD, skewness, and kurtosis intensities
        intensity_std = np.std(las.intensity)
        intensity_mean = np.mean(las.intensity)
        intensity_skew = 3 * ((intensity_mean - np.median(las.intensity)/ intensity_std))
        intensity_kurt = kurtosis(las.intensity)
        intensity = [intensity_std,intensity_mean,intensity_skew,intensity_kurt]

        # Feature 7: Convex Hull  &  Feature 8: Nearest Neighbour Distances
        height_bins = np.linspace(np.min(las.z),np.max(las.z),num=10) # Produce a normalized set of height bins
        bin_diff = (height_bins[1]-height_bins[0])/2 # Get half the difference of the height bins
        convex_hull_list = []
        nearest_neighbour = []
        for height in height_bins:
            if height == height_bins[0]: # Min bin mask
                mask = (las.z <= height+bin_diff)
            if height == height_bins[len(height_bins)-1]: # Max bin mask
                mask = (las.z >= height-bin_diff)
            else: # All other bins mask
                mask = (las.z >= height-bin_diff) & (las.z <= height+bin_diff)
            temp_las = las[mask] # Mask the LAS
            points2D = np.column_stack((temp_las.x,temp_las.y)) 
            try: # Calculate convex hull at each bin
                hull = ConvexHull(points2D) 
                convex_hull_area = hull.volume
            except Exception as e:
                print(f"Error convex hull calculation at: {e}")
                convex_hull_area = 0
            convex_hull_list.append(convex_hull_area)
            
            # Nearest Neighbour Mean Distances
            try:
                # Compute NN distances
                distances = np.sqrt((temp_las.x[:, None] - temp_las.x) ** 2 + (temp_las.y[:, None] - temp_las.y) ** 2 + (temp_las.z[:, None] - temp_las.z) ** 2)
                np.fill_diagonal(distances, np.nan)  # Ignore self-distance
                mean_distance_to_neighbor = np.mean(np.nanmin(distances, axis=1)) # Get the min non NaN value
                if math.isnan(mean_distance_to_neighbor):
                    mean_distance_to_neighbor = 0
            except Exception as e:
                print(f"Nearest Neighbour error at: {e}")
                mean_distance_to_neighbor = 0
            nearest_neighbour.append(mean_distance_to_neighbor)


        # Return a concatenated feature vector
        results = np.concatenate([bbox_dimensions, variance, [point_density], centroid, eigenvalues, intensity, convex_hull_list, nearest_neighbour])
        results = [[0] if x is None else x for x in results]
        results = [x.real for x in results]

        return results
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Function to process files in parallel
def extract_geometric_features_in_parallel(file_paths, concurrent_processes=CONCURRENT_PROCESSES):
    with ProcessPoolExecutor(concurrent_processes) as executor:
        results = list(executor.map(extract_geometric_features, file_paths))
    return results


def extract_geometric_features_agt(file_path):
    try:
        # Open LAZ file using laspy
        with laspy.read(file_path) as las:
        # las = laspy.read(file_path)
            print(f"Processing {file_path} with {len(las.points)} points.")
            
            # Extract point cloud data (x, y, z coordinates)
            points = np.vstack((las.x, las.y, las.z)).transpose()
            points -= points.min(axis=0)

            # Feature 1: Bounding box dimensions (max - min for x, y, z)
            bbox_dimensions = np.max(points, axis=0) - np.min(points, axis=0)

            # Feature 2: Variance in x, y, z directions (spread of points)
            variance = np.var(points, axis=0)

            # Feature 3: Point density (number of points / volume of bounding box)
            bbox_volume = np.prod(bbox_dimensions)
            point_density = len(points) / bbox_volume if bbox_volume > 0 else 0

            # Feature 4: Centroid (mean position of the points)
            centroid = np.mean(points, axis=0)

            # Feature 5: Eigenvalues of the covariance matrix (describing point cloud shape)
            cov_matrix = np.cov(points.T)
            eigenvalues, _ = np.linalg.eig(cov_matrix)

            # Features 6: Mean, StD, skewness, and kurtosis intensities
            intensity_std = np.std(las.intensity)
            intensity_mean = np.mean(las.intensity)
            intensity_skew = 3 * ((intensity_mean - np.median(las.intensity)/ intensity_std))
            intensity_kurt = kurtosis(las.intensity)
            intensity = [intensity_std,intensity_mean,intensity_skew,intensity_kurt]

            # Feature 7: Convex Hull  &  Feature 8: Nearest Neighbour Distances
            height_bins = np.linspace(np.min(las.z),np.max(las.z),num=10) # Produce a normalized set of height bins
            bin_diff = (height_bins[1]-height_bins[0])/2 # Get half the difference of the height bins
            convex_hull_list = []
            nearest_neighbour = []
            for height in height_bins:
                if height == height_bins[0]: # Min bin mask
                    mask = (las.z <= height+bin_diff)
                if height == height_bins[len(height_bins)-1]: # Max bin mask
                    mask = (las.z >= height-bin_diff)
                else: # All other bins mask
                    mask = (las.z >= height-bin_diff) & (las.z <= height+bin_diff)
                temp_las = las[mask] # Mask the LAS
                points2D = np.column_stack((temp_las.x,temp_las.y)) 
                try: # Calculate convex hull at each bin
                    hull = ConvexHull(points2D) 
                    convex_hull_area = hull.volume
                except Exception as e:
                    print(f"Error convex hull calculation at: {e}")
                    convex_hull_area = 0
                convex_hull_list.append(convex_hull_area)
                
                # Nearest Neighbour Mean Distances
                try:
                    # Compute NN distances
                    distances = np.sqrt((temp_las.x[:, None] - temp_las.x) ** 2 + (temp_las.y[:, None] - temp_las.y) ** 2 + (temp_las.z[:, None] - temp_las.z) ** 2)
                    np.fill_diagonal(distances, np.nan)  # Ignore self-distance
                    mean_distance_to_neighbor = np.mean(np.nanmin(distances, axis=1)) # Get the min non NaN value
                    if math.isnan(mean_distance_to_neighbor):
                        mean_distance_to_neighbor = 0
                except Exception as e:
                    print(f"Nearest Neighbour error at: {e}")
                    mean_distance_to_neighbor = 0
                nearest_neighbour.append(mean_distance_to_neighbor)


            # Return a concatenated feature vector
            results = np.concatenate([bbox_dimensions, variance, [point_density], centroid, eigenvalues, intensity, convex_hull_list, nearest_neighbour])
            results = [[0] if x is None else x for x in results]
            results = [x.real for x in results]

            return results
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None