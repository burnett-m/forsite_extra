import numpy as np
import os
from tqdm.notebook import tqdm
# import cv2
# from skimage.morphology import skeletonize
# from scipy.ndimage import distance_transform_edt
import math
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import random
from scipy.spatial import ConvexHull

def pcNorm(cloud, rotationAxis):

    # normalize the pc to range [-0.5,0.5] and keep the 3-D ratio based on the height Z

    maxPC = np.max(cloud,axis=0)
    minPC = np.min(cloud,axis=0)

    if rotationAxis.upper() == "Z":

        ratio = 1/(maxPC[2]-minPC[2])

    if rotationAxis.upper() == "X":

        ratio = 1/(maxPC[1]-minPC[1])
    
    if rotationAxis.upper() == "Y":

        ratio = 1/(maxPC[0]-minPC[0])

    return cloud*ratio

def GeneratingRotMatrix(
    rotAxes:str="Z",
    rotAngle:float=10):
    """
    Generating a dictionary of rotation matrices with a given rotation angle in degrees
    the default rotInterval is 10
    """    
    #
    if rotAxes.upper() == 'X':
        # rotate 'rotationAngle' around X
        # right dot: dot(point_cloud,rotMatrix)
        rotationMatrix = np.array([
            [1,0,0],
            [0, math.cos(rotAngle*math.pi/180), math.sin(rotAngle*math.pi/180)],
            [0, -math.sin(rotAngle*math.pi/180), math.cos(rotAngle*math.pi/180)],
            ])
            
        
    elif rotAxes.upper() == "Y":
        # rotate 'rotationAngle' around Y
        # right dot: dot(point_cloud,rotMatrix)
        rotationMatrix = np.array([
            [math.cos(rotAngle*math.pi/180), 0, -math.sin(rotAngle*math.pi/180)],
            [0,1,0],
            [math.sin(rotAngle*math.pi/180), 0, math.cos(rotAngle*math.pi/180)],
            ])
            
        
    elif rotAxes.upper() == "Z":
        # rotate 'rotationAngle' around Z
        # right dot: dot(point_cloud,rotMatrix)
        rotationMatrix = np.array([
            [math.cos(rotAngle*math.pi/180), math.sin(rotAngle*math.pi/180), 0],
            [-math.sin(rotAngle*math.pi/180), math.cos(rotAngle*math.pi/180), 0],
            [0, 0, 1]
            ])
            
    else:
        print("Please input the correct name of the rotation axis.")

    return rotationMatrix

def pcRotation(cloud, rotationMatrix):

    return np.dot(cloud,rotationMatrix)

def calArea(pt1,pt2,pt3):

    alphaX,alphaY,alphaZ = pt2[0]-pt1[0],pt2[1]-pt1[1],pt2[2]-pt1[2]
    betaX,betaY,betaZ = pt3[0]-pt1[0],pt3[1]-pt1[1],pt3[2]-pt1[2]
    
    crossProd = (alphaY*betaZ-alphaZ*betaY)**2+\
        (alphaZ*betaX-alphaX*betaZ)**2+\
            (alphaX*betaY-alphaY*betaX)**2

    areaS=math.sqrt(crossProd)/2

    return areaS

def isCollinear(points):

    pts = np.array(points)

    # Select the first point as the reference point
    refPt = pts[0]
    
    # Create vectors from the reference point to all other points
    vectors = [p - refPt for p in pts[1:]]
    
    # Check if all vectors are collinear
    for i in range(1, len(vectors)):
        crossProduct = np.cross(vectors[0], vectors[i])
        if not np.allclose(crossProduct, 0):
            return False
    return True
    
def calDistPoints(points):

    points = np.array(points)

    maxDist = np.linalg.norm(points[0]-points[1])

    for ptNum in range(2,points.shape[0]):

        dist = np.linalg.norm(points[0]-points[ptNum])

        if dist > maxDist:

            maxDist = dist
        
    return maxDist

def isCoplanar(points):
    # Select the first point as the reference point
    refPoint = points[0]
    
    # Create vectors from the reference point to all other points
    vectors = points[1:] - refPoint
    
    # Form a matrix from these vectors
    matrix = vectors.T
    
    # Compute the rank of the matrix
    rank = np.linalg.matrix_rank(matrix)
    
    # If the rank is 2 or less, the points are coplanar
    return rank <= 2

def projCoordOnCoplane(points):
    # Select three points to define the plane
    crossProd = 0

    while np.sum(crossProd)==0:

        sampleList = [i for i in range(points.shape[0])]
        sampleList = random.sample(sampleList,3)
        
        p1, p2, p3 = points[sampleList,:]
        
        # Calculate the normal vector of the plane
        v1 = p2 - p1
        v2 = p3 - p1
        #
        crossProd = np.cross(v1, v2)

    normal = crossProd / np.linalg.norm(crossProd)# Normalize the normal vector    
    
    # Project all points onto the plane
    projPoints = []
    for p in points:
        # Calculate the vector from p1 to the current point
        vec = p - p1
        # Calculate the distance from the point to the plane
        dist2Plane = np.dot(vec, normal)
        # Calculate the projected point
        projPoint = p - dist2Plane * normal
        projPoints.append(projPoint)
    
    # Define a 2D coordinate system on the plane
    u = v1 / np.linalg.norm(v1)  # First axis (normalized)
    w = np.cross(normal, u)  # Second axis (automatically perpendicular to u and normal)
    w = w / np.linalg.norm(w)  # Normalize
    
    # Convert projected points to 2D coordinates
    projPoints = np.array(projPoints)
    projPt2D = []
    for p in projPoints:
        vec = p - p1
        x = np.dot(vec, u)
        y = np.dot(vec, w)
        projPt2D.append([x, y])
    
    return np.array(projPt2D)


def CreatePixelCoords(
    points, 
    corners,
    spacingPara=0.1):

    """
    input: 
        points: xCol, yCol: points[:,0], points[:,1], the two axes of the projected plane
        spacingPara: the spacing distance in meters;
    output:
        img: np.array 
    """
    #
    xCol = points[:,0]
    yCol = points[:,1]
    pxValCol = (((points[:,2]-corners[:,2].min())/(corners[:,2].max()-corners[:,2].min())) * 255).astype('uint8')
    #
    if corners.any():
        xCoordi = np.floor((np.asarray(xCol) - corners[:,0].min())/spacingPara).astype(int)
        yCoordi = np.floor((np.asarray(yCol) - corners[:,1].min())/spacingPara).astype(int)
    else:
        xCoordi = np.floor((np.asarray(xCol) - np.asarray(xCol).min())/spacingPara).astype(int)
        yCoordi = np.floor((np.asarray(yCol) - np.asarray(yCol).min())/spacingPara).astype(int)
    #
    new_points = np.column_stack([xCoordi,yCoordi,pxValCol])
    #
    return new_points


def projImg(
    points, 
    corners,
    imgSize,
    spacingPara=0.1):
    #
    new_points = CreatePixelCoords(points, corners=corners, spacingPara=spacingPara)
    # 
    dens_new_points = np.copy(new_points)

    # image bands:
    # img = np.multiply(np.ones(([X.shape[0],X.shape[1],pxValCol.shape[1]]),dtype="uint8"),255)
    img = np.zeros(([dens_new_points[:,1].max()+1,dens_new_points[:,0].max()+1]),dtype="uint8")
    #
    img[dens_new_points[:,1], dens_new_points[:,0]] = dens_new_points[:,2]
    #
    img = np.flipud(img)
    #
    # imgF = np.multiply(np.ones(([imgSize,imgSize,pxValCol.shape[1]]),dtype="uint8"),255)
    imgF = np.zeros(([imgSize[0],imgSize[1]]),dtype="uint8")
    #
    if img.shape[0] >= imgF.shape[0] and img.shape[1] < imgF.shape[1]:
        #
        imgF[:, int(np.ceil((imgSize[0]-img.shape[1])/2)):int(np.ceil((imgSize[0]-img.shape[1])/2))+img.shape[1]] = img[0:imgF.shape[0],:]
        #
    elif img.shape[0] < imgF.shape[0] and img.shape[1] >= imgF.shape[1]:
        #
        imgF[int(np.ceil((imgSize[1]-img.shape[0])/2)):int(np.ceil((imgSize[1]-img.shape[0])/2))+img.shape[0], :] = img[:,0:imgF.shape[1]:]
        #
    elif img.shape[0] >= imgF.shape[0] and img.shape[1] >= imgF.shape[1]:
        #
        imgF = img[0:imgF.shape[0]:,0:imgF.shape[1]:]
        #
    else:
        #
        imgF[int(np.ceil((imgSize[0]-img.shape[0])/2)):int(np.ceil((imgSize[0]-img.shape[0])/2))+img.shape[0],
             int(np.ceil((imgSize[1]-img.shape[1])/2)):int(np.ceil((imgSize[1]-img.shape[1])/2))+img.shape[1]] = img
        #
    # print("%.2f seconds spent in generating one image."%(time.time()-start_time))
    #
    return imgF.astype(float)

# Create a custom Dataset class

class pcDataset():
    #
    def __init__(self, pcTxTPath:str):
        """
        Args:
            pcTxTPath: path of a txt file containing centralized points with [x, y, z] for each line in the txt
        """
        pcData = np.unique(np.loadtxt(pcTxTPath),axis=0) # remove duplicated points
        if pcData.shape[1]>3:
            self.pcData=pcData[:,0:3]
        else:
            self.pcData=pcData
        # Define the voxel size
        self.voxelSize = np.array([0.5, 0.5, 2.0])
        # Define the voxel number
        self.voxelNum = None
        # define the rotation axis
        self.rotAxis = None
        # define the rotation angle
        self.rotAngle = None
        # define the projected image resolution
        self.spacing = None
        # define the voxel Normalization
        self.voxelNormalization = None
        # define the point cloud Normalization
        self.pcNormalization = None
        #        
        return
           
    def setParams(
        self,   
        voxelSize = None,
        voxelNum = None,
        outputPath = None,
        rotAxis = None,
        rotAngle = None,
        spacing = None,
        voxelNormalization = None,
        pcNormalization = None):
        
        if voxelNum:
            self.voxelNum = voxelNum
        if outputPath:
            self.outputPath = outputPath
        if rotAxis:
            self.rotAxis = rotAxis
        if rotAngle:
            self.rotAngle = rotAngle        
        if voxelNormalization:
            self.voxelNormalization = voxelNormalization   
        if pcNormalization:
            self.pcNormalization = pcNormalization      
        if voxelSize:
            self.voxelSize = np.array(voxelSize)
        if spacing:
            self.spacing = spacing
            
        return

    def pcVoxelization(self):

        if self.rotAxis and (self.rotAngle or self.rotAngle==0):

            self.pcData = pcRotation(
                cloud=self.pcData, 
                rotationMatrix=GeneratingRotMatrix(
                    rotAxes=self.rotAxis,
                    rotAngle=self.rotAngle
                ))    

        if self.voxelNormalization and self.pcNormalization:
            print(f"voxel and pc data cannot be normalized simultaneously.")
            return
        elif self.voxelNormalization:
            tempSize = (np.max(self.pcData, axis=0)-np.min(self.pcData, axis=0))/np.array(self.voxelNum)
            if self.rotAxis.upper() == "X":
                self.voxelSize = np.array([max(tempSize[0],tempSize[2]),tempSize[1],max(tempSize[0],tempSize[2])])
                self.spacing = max(tempSize[0],tempSize[2])/5   
            elif self.rotAxis.upper() == "Y":
                self.voxelSize = np.array([tempSize[0],max(tempSize[1],tempSize[2]),max(tempSize[1],tempSize[2])])
                self.spacing = max(tempSize[1],tempSize[2])/5 
            elif self.rotAxis.upper() == "Z":
                self.voxelSize = np.array([max(tempSize[:2]),max(tempSize[:2]),tempSize[2]])
                self.spacing = max(tempSize[:2])/5 
        elif self.pcNormalization:
            self.pcData = pcNorm(self.pcData,self.rotAxis)
            tempSize = (np.max(self.pcData, axis=0)-np.min(self.pcData, axis=0))/np.array(self.voxelNum)
            if self.rotAxis.upper() == "X":
                self.voxelSize = np.array([max(tempSize[0],tempSize[2]),tempSize[1],max(tempSize[0],tempSize[2])])
                self.spacing = max(tempSize[0],tempSize[2])/5   
            elif self.rotAxis.upper() == "Y":
                self.voxelSize = np.array([tempSize[0],max(tempSize[1],tempSize[2]),max(tempSize[1],tempSize[2])])
                self.spacing = max(tempSize[1],tempSize[2])/5 
            elif self.rotAxis.upper() == "Z":
                self.voxelSize = np.array([max(tempSize[:2]),max(tempSize[:2]),tempSize[2]])
                self.spacing = max(tempSize[:2])/5 

        if self.voxelNum:
            wholeGridSize = self.voxelSize*np.array(self.voxelNum)
            # Calculate the centroid of the point cloud
            centroidCoords = np.array([0,0,0]) # as the original pc was centralized
            # Compute the minimum coordinates (grid origin) based on the centroid
            minCoords = centroidCoords - (wholeGridSize / 2)
            numVoxels = self.voxelNum
        else:
            # Compute the minimum and maximum coordinates (bounding box)
            minCoords = np.floor(np.min(self.pcData, axis=0) / self.voxelSize) * self.voxelSize
            maxCoords = np.ceil(np.max(self.pcData, axis=0) / self.voxelSize) * self.voxelSize
            # Calculate the number of voxels in each dimension
            numVoxels = np.ceil((maxCoords - minCoords) / self.voxelSize).astype(int)        

        # Initialize a dictionary to store voxel corners and points
        voxelDict = {}

        # Function to calculate and store the voxel corners
        def calVoxelCorners(origin, size):
            x, y, z = origin
            dx, dy, dz = size
            
            corners = np.array([
                [x, y, z],
                [x + dx, y, z],
                [x, y + dy, z],
                [x, y, z + dz],
                [x + dx, y + dy, z],
                [x + dx, y, z + dz],
                [x, y + dy, z + dz],
                [x + dx, y + dy, z + dz]
            ])
            return corners

        # Iterate over the entire grid and store every voxel's corners and points
        for i in range(numVoxels[0]):
            for j in range(numVoxels[1]):
                for k in range(numVoxels[2]):
                    voxelOrigin = minCoords + np.array([i, j, k]) * self.voxelSize
                    voxelIndex = (i, j, k)
                    voxelCorners = calVoxelCorners(voxelOrigin, self.voxelSize)
                    
                    # Identify points within this voxel
                    withinVoxel = np.all((self.pcData >= voxelOrigin) & (self.pcData < voxelOrigin + self.voxelSize), axis=1)
                    pointsInVoxel = self.pcData[withinVoxel]
                    
                    # Store the corners and points in the dictionary
                    voxelDict[voxelIndex] = {
                        'corners': voxelCorners,
                        'points': pointsInVoxel.tolist()
                    }
        # calculate the projection coefficient of each voxel
        for voxelInd, voxelInfoDict in voxelDict.items():
            if len(voxelInfoDict['points'])==0:
                voxelInfoDict['projCoeff']=0.0
                voxelInfoDict['gapFrac']=1.0
            else:
                if len(voxelInfoDict['points'])==1:
                    voxelInfoDict['projCoeff']=\
                        (self.spacing**2)/(self.voxelSize[0]*self.voxelSize[1])
                if len(voxelInfoDict['points'])==2:
                    pt1=voxelInfoDict['points'][0]
                    pt2=voxelInfoDict['points'][1]
                    projPt1 = [pt1[0],pt1[1],0]
                    projPt2 = [pt2[0],pt2[1],0]
                    voxelInfoDict['projCoeff']=\
                        (np.linalg.norm(np.array(projPt1)-np.array(projPt2)))/\
                            (np.linalg.norm(np.array(pt1)-np.array(pt2)))
                if len(voxelInfoDict['points'])==3:
                    pt1=voxelInfoDict['points'][0]
                    pt2=voxelInfoDict['points'][1]
                    pt3=voxelInfoDict['points'][2]
                    projPt1 = [pt1[0],pt1[1],0]
                    projPt2 = [pt2[0],pt2[1],0]
                    projPt3 = [pt3[0],pt3[1],0]                    
                    try:
                        if isCollinear(voxelInfoDict['points']):#collinear
                            projPtsDist = calDistPoints([projPt1,projPt2,projPt3])
                            ptsDist = calDistPoints(voxelInfoDict['points'])
                            # print(projPtsDist/ptsDist)
                            voxelInfoDict['projCoeff']=projPtsDist/ptsDist
                        else:
                            voxelInfoDict['projCoeff']=\
                                calArea(projPt1,projPt2,projPt3)/calArea(pt1,pt2,pt3)  
                    except Exception as e:
                        print(f"Failed in calculating 3 points: {voxelInfoDict['points']} in {voxelInd} by rotating {self.rotAxis} due to {e}.")
                if len(voxelInfoDict['points'])>3:
                    projPts=list()
                    for pt in voxelInfoDict['points']:
                        projPts.append([pt[0],pt[1]])
                    try:
                        if len(set(np.array(projPts)[:,0]))==1: # collinear along x
                            projPtsDist = abs(np.array(projPts)[:,1].max()-np.array(projPts)[:,1].min())
                            ptsDist = math.sqrt((np.array(voxelInfoDict['points'])[:,1].max()-np.array(voxelInfoDict['points'])[:,1].min())**2+\
                                                (np.array(voxelInfoDict['points'])[:,2].max()-np.array(voxelInfoDict['points'])[:,2].min())**2)
                            # print(projPtsDist/ptsDist)
                            voxelInfoDict['projCoeff']=projPtsDist/ptsDist
                        elif len(set(np.array(projPts)[:,1]))==1: # collinear along y
                            projPtsDist = abs(np.array(projPts)[:,0].max()-np.array(projPts)[:,0].min())
                            ptsDist = math.sqrt((np.array(voxelInfoDict['points'])[:,0].max()-np.array(voxelInfoDict['points'])[:,0].min())**2+\
                                                (np.array(voxelInfoDict['points'])[:,2].max()-np.array(voxelInfoDict['points'])[:,2].min())**2)
                            # print(projPtsDist/ptsDist)
                            voxelInfoDict['projCoeff']=projPtsDist/ptsDist                            
                        elif isCoplanar(np.array(voxelInfoDict['points'])):# coplanar
                            if isCollinear(voxelInfoDict['points']): # collinear in the 3-D space
                                voxelInfoDict['projCoeff']=\
                                    calDistPoints(projPts)/\
                                        calDistPoints(voxelInfoDict['points'])
                            elif isCollinear(projPts): # collinear in the projected plane
                                distZ = np.array(voxelInfoDict['points'])[:,2].max()-np.array(voxelInfoDict['points'])[:,2].min()
                                distX = np.array(voxelInfoDict['points'])[:,0].max()-np.array(voxelInfoDict['points'])[:,0].min()
                                distY = np.array(voxelInfoDict['points'])[:,1].max()-np.array(voxelInfoDict['points'])[:,1].min()
                                projPtsDist = np.sqrt(distX**2+distY**2)
                                ptsDist = np.sqrt(projPtsDist**2+distZ**2)
                                voxelInfoDict['projCoeff']=projPtsDist/ptsDist
                            else: # not collinear
                                voxelInfoDict['projCoeff']=\
                                    ConvexHull(np.array(projPts)).volume/\
                                        (ConvexHull(projCoordOnCoplane(np.array(voxelInfoDict['points']))).volume)
                        else:
                            try:
                                voxelInfoDict['projCoeff']=\
                                    ConvexHull(np.array(projPts)).volume/\
                                        (ConvexHull(np.array(voxelInfoDict['points'])).area/2)
                            except: #using qhull options to joggle the input points   
                                try:                             
                                    voxelInfoDict['projCoeff']=\
                                        ConvexHull(np.array(projPts),qhull_options="QJ").volume/\
                                            (ConvexHull(np.array(voxelInfoDict['points']),qhull_options="QJ").area/2)
                                except Exception as e:
                                    print(f"Failed in calculating more than 3 points {projPts} in {voxelInd} by rotating {self.rotAxis} due to {e}.")
                    except Exception as e:
                        print(f"Failed in calculating {projPts} in {voxelInd} by rotating {self.rotAxis} due to {e}.")
                #
                img=projImg(
                    points=np.array(voxelInfoDict['points']),
                    corners=voxelInfoDict['corners'],
                    imgSize = [int(self.voxelSize[0]/self.spacing),int(self.voxelSize[1]/self.spacing)],
                    spacingPara=self.spacing)
                voxelInfoDict['gapFrac']=\
                    np.sum(img==0)/\
                        ((self.voxelSize[0]/self.spacing)*(self.voxelSize[1]/self.spacing))

        return voxelDict
    
    def __voxelDimension__(self):

        # Compute the minimum and maximum coordinates (bounding box)
        minCoords = np.floor(np.min(self.pcData, axis=0) / self.voxelSize) * self.voxelSize
        maxCoords = np.ceil(np.max(self.pcData, axis=0) / self.voxelSize) * self.voxelSize

        numVoxels = np.ceil((maxCoords - minCoords) / self.voxelSize).astype(int)

        if self.voxelNum:            
            numVoxels = self.voxelNum

        return numVoxels

    def __getVoxel__(self, idx:list):

        voxelDim = self.__voxelDimension__()        

        if len(idx)==3 and idx[0]<voxelDim[0] and idx[1]<voxelDim[1] and idx[2]<voxelDim[2]:
            
            voxel = self.pcVoxelization()[tuple(idx)]

            return voxel 

        else:

            return f"{idx} is out of the voxel indices."
        
def visualizeVoxPC(data, voxel_dict, figsize=(24, 16),transparentRatio=0.1,viewAngle=[0,90]):
    # Plot the point cloud
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', marker='o', s=1, label='Points')
    ax.view_init(elev=viewAngle[0],azim=viewAngle[1])

    # Function to plot a voxel given its corners
    def plot_voxel(ax, corners, color='r', alpha=transparentRatio):
        edges = [
            [corners[0], corners[1]],
            [corners[0], corners[2]],
            [corners[0], corners[3]],
            [corners[1], corners[4]],
            [corners[1], corners[5]],
            [corners[2], corners[4]],
            [corners[2], corners[6]],
            [corners[3], corners[5]],
            [corners[3], corners[6]],
            [corners[4], corners[7]],
            [corners[5], corners[7]],
            [corners[6], corners[7]],
        ]
        
        for edge in edges:
            ax.plot3D(*zip(*edge), color=color, alpha=alpha)

    # Visualize each voxel from the voxel_dict
    for voxel_index, voxel_data in voxel_dict.items():
        corners = voxel_data['corners']
        points_in_voxel = voxel_data['points']
        
        # Optional: Change voxel color based on the number of points
        if len(points_in_voxel) > 0:
            color = 'r'  # Color for non-empty voxels
        else:
            color = 'r'  # Color for empty voxels
        
        plot_voxel(ax, corners, color)

    # Set the aspect ratio to be equal
    ax.set_box_aspect([1, 1, 1])

    # Set axis limits to be the same for proper scaling
    max_range = np.array(
        [data[:, 0].max()-data[:, 0].min(), 
         data[:, 1].max()-data[:, 1].min(), 
         data[:, 2].max()-data[:, 2].min()]).max() / 2.0
    mid_x = (data[:, 0].max() + data[:, 0].min()) * 0.5
    mid_y = (data[:, 1].max() + data[:, 1].min()) * 0.5
    mid_z = (data[:, 2].max() + data[:, 2].min()) * 0.5
    # ax.set_xlim(data[:, 0].min(), data[:, 0].max())
    # ax.set_ylim(data[:, 1].min(), data[:, 1].max())
    # ax.set_zlim(data[:, 2].min(), data[:, 2].max())
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=viewAngle[0],azim=viewAngle[1])
    plt.title("3D Point Cloud with Voxel Grid")

    # Add a legend to differentiate voxel colors
    ax.legend(loc='best')

    plt.show()

def calDescfromGapFraction(
    pcTxtPath:str,
    procParaDict:dict,
):

    outputData = pd.DataFrame()

    outputData["UNIQUE_ID"] = [os.path.basename(pcTxtPath).split('.txt')[0]]

    for rotAxis, procInfoDict in procParaDict.items():
        
        rotAngle = procInfoDict["rotAngle"]
        spacing = procInfoDict["spacing"]
        voxelSize = procInfoDict["voxelSize"]
        voxelNum = procInfoDict["voxelNum"]
        voxelNormalization = procInfoDict["voxelNormalization"]
        pcNormalization = procInfoDict["pcNormalization"]
        #
        #
        # meanProjCoeff = list()
        # gapFraction=list()

        #
        # pixNumVoxel = (voxelSize[0]/spacing)*(voxelSize[1]/spacing)
        # fullPixVoxel = np.ones((voxelNum[0],voxelNum[1]))*pixNumVoxel
        #
        # create a pcDataset class
        tempVoxel = pcDataset(pcTxtPath)
        tempVoxel.setParams(
            voxelSize = voxelSize,
            voxelNum = voxelNum,
            rotAxis = rotAxis,
            rotAngle = rotAngle,
            spacing = spacing,
            voxelNormalization = voxelNormalization,
            pcNormalization = pcNormalization
        )
        voxel = tempVoxel.pcVoxelization()
        maxX,maxY,maxZ = max(voxel.keys())
        # generate density and extinction coefficients
        ptDensity = np.zeros((maxX+1,maxY+1,maxZ+1))
        ptExtCoeff = np.zeros((maxX+1,maxY+1,maxZ+1))
        # empty pixels
        ptGapFrac = np.zeros((maxX+1,maxY+1,maxZ+1))

        for ptInx,ptInfo in voxel.items():
            ptDensity[ptInx[0],ptInx[1],ptInx[2]] = len(ptInfo['points'])
            ptExtCoeff[ptInx[0],ptInx[1],ptInx[2]] = ptInfo['projCoeff']        #
            ptGapFrac[ptInx[0],ptInx[1],ptInx[2]] = ptInfo['gapFrac']

        # # generate mean projection coefficient and gap fraction of all Z layers
        # sliceProjCoeff=list()
        # sliceGapFrac=list()
        # for zLayer in range(maxZ+1):
        #     layerDensity=ptDensity[:,:,zLayer]
        #     # projection coefficients for a single slice plane: sliceProjCoeff
        #     # gap fraction: sliceGapFrac
        #     if np.sum(layerDensity==0) !=0:
        #         sliceProjCoeff.append(
        #             ptExtCoeff[:,:,zLayer].sum()/np.sum(layerDensity==0))
        #         #
        #         emptyPixRatio=ptGapFrac[:,:,zLayer]
        #         # ## following the functions in the article
        #         # emptyPixRatio[emptyPixRatio==1]=0
        #         # # print(emptyPixRatio)
        #         # sliceGapFrac.append(
        #         #     ((voxelNum[0]*voxelNum[1]-np.sum(layerDensity==0))*np.sum(emptyPixRatio)+
        #         #     np.sum(layerDensity==0))/(voxelNum[0]*voxelNum[1]))
        #         #
        #         ## slice based integration             
        #         sliceGapFrac.append(np.prod(emptyPixRatio))
        #     else:
        #         sliceProjCoeff.append(0.0)
        #         sliceGapFrac.append(1.0)

        # meanProjCoeff.append(sliceProjCoeff)
        # gapFraction.append(sliceGapFrac)
            

        # # final slice-based mean projection coefficients
        # slicedMeanProjCoeff = np.array(meanProjCoeff).sum(axis=0).T
        # slicedGapFraction = np.array(gapFraction).sum(axis=0).T

        ##

        for zNum in range(ptExtCoeff.shape[2]):

            tempDF = pd.DataFrame()

            ptExtCoeffSlice = ptExtCoeff[:,:,zNum]

            # print(np.sum(ptExtCoeffSlice)/(ptExtCoeffSlice.shape[0]*ptExtCoeffSlice.shape[1]))

            tempDF[f"{rotAxis}{str(int(rotAngle))}S{str(zNum)}EC"] = \
                [np.sum(ptExtCoeffSlice)/(ptExtCoeffSlice.shape[0]*ptExtCoeffSlice.shape[1])]    

            ptGapFracSlice = ptGapFrac[:,:,zNum]

            # print(np.sum(ptGapFracSlice)/(ptGapFracSlice.shape[0]*ptGapFracSlice.shape[1]))

            tempDF[f"{rotAxis}{str(int(rotAngle))}S{str(zNum)}GFS"] = \
                [np.sum(ptGapFracSlice)/(ptGapFracSlice.shape[0]*ptGapFracSlice.shape[1])]
            
            tempDF[f"{rotAxis}{str(int(rotAngle))}S{str(zNum)}LAI"] = \
                [-np.log(np.sum(ptGapFracSlice)/(ptGapFracSlice.shape[0]*ptGapFracSlice.shape[1]))/\
                (np.sum(ptExtCoeffSlice)/(ptExtCoeffSlice.shape[0]*ptExtCoeffSlice.shape[1]))
                if np.sum(ptExtCoeffSlice)!=0 else 0.0]
            
            outputData=pd.concat([outputData, tempDF],axis=1)
            
            # outputData[f"{rotAxis}{str(int(rotAngle))}S{str(zNum)}GFP"] = \
            #     [np.prod(ptGapFracSlice)]

            # print(np.prod(ptExtCoeffSlice))

            # plt.imshow(ptExtCoeffSlice,cmap='gray')
            # plt.colorbar()
            # plt.show()
        ptCanopyGapFrac = np.zeros((ptGapFrac.shape[0],ptGapFrac.shape[1]))

        for x in range(ptGapFrac.shape[0]):
            for y in range(ptGapFrac.shape[1]):
                ptCanopyGapFrac[x,y]= 1.0 if np.prod(ptGapFrac[x,y,:]) == 1.0 else 0.0

        # print(np.sum(ptCanopyGapFrac==0)/(ptCanopyGapFrac.shape[0]*ptCanopyGapFrac.shape[1]))

        outputData[f"{rotAxis}{str(int(rotAngle))}CGF"] = \
            [np.sum(ptCanopyGapFrac==0)/(ptCanopyGapFrac.shape[0]*ptCanopyGapFrac.shape[1])]
        
    return outputData

