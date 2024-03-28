import pandas as pd
import os
import numpy as np
import subprocess
import json
import shutil
import subprocess

parentBaseDir = r"D:\_Models"
Tset_folderName = r"Training_Set"
classifierDir = r"D:\***\Classifier.exe"
modelName = r"M45-1345_ITI_RFE"
tuneType = "/eb" # "/ea" - addOnly / "/eb" - AddRemove / "/er" - RemoveOnly
CCweight = 5
CVweight = 1
TEweight = 4
hbtFolder = "hbt"

def tuner_newModel(modelName,tuneType,CCweight,TEweight,CVweight,parentBaseDir,Tset_folderName,classifierDir):

    id_or_1 = "ID" #"1"
    largeHisto_colName = "ID" #"TREEID"
    jsonPatternToCopy = "TEMPLATEpattern"
    
    parentDir = os.path.join(parentBaseDir,modelName)

    os.chdir(parentDir)

    if not os.path.exists(hbtFolder):
        os.makedirs(hbtFolder)

    trainingSet = os.path.join(parentDir,Tset_folderName,os.listdir(Tset_folderName)[0])    

##    # Read the larger CSV file as a pandas dataframe
    for file in os.listdir(parentBaseDir):
        if file.endswith("Histogram.csv"):
            larger_df = pd.read_csv(os.path.join(parentBaseDir,file),dtype={largeHisto_colName: str})
        if file.endswith(".json"):
            if tuneType == "/er":
                jsonModel = modelName.split("-")[0] + "_remove"
            if tuneType == "/ea":
                jsonModel = modelName.split("-")[0] + "_add"
            if tuneType == "/eb":
                jsonModel = modelName.split("-")[0] + "_addRemove"
            awkC = """awk -v pattern={0} -v replacement={1} "{2} 1" {3} > {4}"""
            awkCommand = awkC.format(jsonPatternToCopy,jsonModel,"gsub(pattern, replacement)",os.path.join(parentBaseDir,file),os.path.join(parentDir,"ProjectParameters",file))
            #print(awkCommand)
            
            subprocess.call(awkCommand,shell=True)

    if tuneType == "/er":
        # Access add list and test examples
        for file in os.listdir():
            if file.endswith('TestExamples.csv'):
                teFile = file
        # Make add list and test examples histogram subsets
        TE = pd.read_csv(teFile,dtype={id_or_1:str})
        larger_df[larger_df[largeHisto_colName].isin(TE[id_or_1])].to_csv(r"ProjectParameters\TestExamples.csv",index=False)
        # Fix combined percentages file
        os.chdir(r'ProjectParameters')
        for file in os.listdir():
            if file.endswith('CombinedPercentages.csv'):
                combP = file
                combinedP = pd.read_csv(file,header=None)
            if file.endswith('.prj'):
                prjFile = file

        combinedP.iloc[0,2:] = np.nan
        ccIdx = combinedP.where(combinedP=='Canopy Cover').dropna(how='all').dropna(axis=1)
        cvIdx = combinedP.where(combinedP=='CV').dropna(how='all').dropna(axis=1) 
        teIdx = combinedP.where(combinedP=='Test eXamples').dropna(how='all').dropna(axis=1)
        combinedP.at[1,ccIdx.columns[0]+2] = str(CCweight)
        combinedP.at[1,cvIdx.columns[0]+2] = str(CVweight)
        combinedP.at[1,teIdx.columns[0]+2] = str(TEweight)

        combinedP.to_csv(combP,index=False,header=None)

        # Run the classifier
        with open(os.path.join(parentDir,hbtFolder, r"TextFile.txt"), "w") as file:
            file.write("Nothing")

        histoDir = trainingSet
        prjDir = os.path.join(parentDir,r"ProjectParameters",prjFile)
        #prjDir = parentDir + r"/ProjectParameters/" + prjFile
        outputDir = os.path.join(parentDir,hbtFolder,"TextFile.txt")
        #outputDir = parentDir + r"/hbt/TextFile.txt"

        cmd = [classifierDir, histoDir, prjDir, outputDir, "/a", "6", "/p", "0.1", "/b", tuneType, "/dn"]
        #cmd = "Classifier "+histoDir+" "+prjDir+" "+outputDir+" /e "+addDir+" /n 98 /p 0.1 "+examples+" /dn"
        try:
           os.chdir(os.path.dirname(classifierDir))
           subprocess.Popen(cmd)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")

    else:  
        # Access add list and test examples
        for file in os.listdir():
            if file.endswith('addList.csv'):
                addListFile = file
            if file.endswith('TestExamples.csv'):
                teFile = file
        # Make add list and test examples histogram subsets
        addList = pd.read_csv(addListFile,dtype={id_or_1:str})
        larger_df[larger_df[largeHisto_colName].isin(addList[id_or_1])].to_csv(r"ProjectParameters\ExtraExamples.csv",index=False)
        TE = pd.read_csv(teFile,dtype={id_or_1:str})
        larger_df[larger_df[largeHisto_colName].isin(TE[id_or_1])].to_csv(r"ProjectParameters\TestExamples.csv",index=False)

        # Fix combined percentages file
        os.chdir(r'ProjectParameters')
        for file in os.listdir():
            if file.endswith('CombinedPercentages.csv'):
                combP = file
                combinedP = pd.read_csv(file,header=None)
            if file.endswith('.prj'):
                prjFile = file

        combinedP.iloc[0,2:] = np.nan
        ccIdx = combinedP.where(combinedP=='Canopy Cover').dropna(how='all').dropna(axis=1)
        cvIdx = combinedP.where(combinedP=='CV').dropna(how='all').dropna(axis=1) 
        teIdx = combinedP.where(combinedP=='Test eXamples').dropna(how='all').dropna(axis=1)
        combinedP.at[1,ccIdx.columns[0]+2] = str(CCweight)
        combinedP.at[1,cvIdx.columns[0]+2] = str(CVweight)
        combinedP.at[1,teIdx.columns[0]+2] = str(TEweight)

        combinedP.to_csv(combP,index=False,header=None)

        # Run the classifier
        with open(os.path.join(parentDir,hbtFolder, r"TextFile.txt"), "w") as file:
            file.write("Nothing")
        
        histoDir = trainingSet
        prjDir = os.path.join(parentDir,r"ProjectParameters",prjFile)
        #prjDir = parentDir + r"/ProjectParameters/" + prjFile
        outputDir = os.path.join(parentDir,hbtFolder,"TextFile.txt")
        #outputDir = parentDir + r"/hbt/TextFile.txt"
        addDir = parentDir + r"/" + addListFile.replace(".csv","_Histogram.csv")

        cmd = [classifierDir, histoDir, prjDir, outputDir, "/a", "6", "/p", "0.1", "/b", tuneType, "/dn"]
        #cmd = "Classifier "+histoDir+" "+prjDir+" "+outputDir+" /e "+addDir+" /n 98 /p 0.1 "+examples+" /dn"
        try:
           os.chdir(os.path.dirname(classifierDir))
           subprocess.Popen(cmd)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")

tuner_newModel(modelName,tuneType,CCweight,TEweight,CVweight,parentBaseDir,Tset_folderName,classifierDir)
