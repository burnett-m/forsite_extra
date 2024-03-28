import pandas as pd
import os
import shutil
import subprocess

parentBaseDir = r"E:\_Models"
Tset_folderName = r"Training_Set"
histoFile = r"E:\_Models\M361-2029_RFE\M361-2044.csv"
descriptorType = "_RFE"
classifierDir = r"E:/***/Classifier.exe"

def buildNewHist(histoFile,parentBaseDir,Tset_folderName,descriptorType,classifierDir):

    histo = pd.read_csv(histoFile,dtype={"Unnamed: 0": str})

    # Read the larger CSV file as a pandas dataframe
    for file in os.listdir(parentBaseDir):
        if file.endswith("Histogram.csv"):
            larger_df = pd.read_csv(os.path.join(parentBaseDir,file),dtype={'TREEID': str})

    histoFile_split = histoFile.split("\\")
    histoFile_split[0] += "/"

    # Change old Classifier folder
    if os.path.exists(os.path.join(*histoFile_split[0:-1],"Classifier")):
        os.rename(os.path.join(*histoFile_split[0:-1],"Classifier"),os.path.join(*histoFile_split[0:-1],histoFile_split[-2]))
        if os.path.exists(os.path.join(*histoFile_split[0:-1],"Training_Set")):
            shutil.move(os.path.join(*histoFile_split[0:-1],"Training_Set"),os.path.join(*histoFile_split[0:-1],histoFile_split[-2],"Training_Set"))
        
        
    # Make new dir
    if not os.path.exists(os.path.join(*histoFile_split[0:-1],"Training_Set")):
        os.makedirs(os.path.join(*histoFile_split[0:-1],"Training_Set"))
        os.makedirs(os.path.join(*histoFile_split[0:-1],"Classifier"))
        with open(os.path.join(*histoFile_split[0:-1],"Classifier") +r"/TextFile.txt", "w") as file:
                file.write("Nothing")
        
    outFile = os.path.join(*histoFile_split[0:-1], "Training_Set",histoFile_split[-1])
    larger_df[larger_df['TREEID'].isin(histo["Unnamed: 0"])].to_csv(outFile,index=False)

    os.rename(outFile,outFile.replace(".csv",f"{descriptorType}.csv"))
    # Rename folder
    newPath = os.path.join(*histoFile_split[0:-2],histoFile_split[-1].replace(".csv",f"{descriptorType}"))
    os.rename(os.path.join(*histoFile_split[0:-1]),newPath)

    # Set up harvest block tune
    histoDir = os.path.join(newPath,"Training_Set",outFile.split("\\")[-1].replace(".csv",f"{descriptorType}.csv"))
    prjDir = os.path.join(newPath,r"ProjectParameters")
    for file in os.listdir(prjDir):
        if file.endswith(".prj"):
            prjFile = os.path.join(prjDir,file)
    classOutput = os.path.join(newPath,"Classifier", r"TextFile.txt")

    cmd = [classifierDir, histoDir, prjFile, classOutput, "/p", "0.1", "/en", "/dn"]

    try:
       os.chdir(os.path.dirname(classifierDir))
       subprocess.Popen(cmd)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")


buildNewHist(histoFile,parentBaseDir,Tset_folderName,descriptorType,classifierDir)
