# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 05:20:00 2023

@author: mburnett
"""
import os 
import pandas as pd
import xlwings as xl

dir_path = os.chdir(r"C:\--scriptDirectory--")
from helper.scorecard import *
from helper.addRemoveList import addRemove

pd.set_option('display.expand_frame_repr', False) # Allow more columns when printing to console

# Get main directories
parentDir = r'C:\--directory--\Validation'
pDir = parentDir + r'\00_AllSpecies'
mFolder = parentDir + r'\Results\Michael'

# Update scorecard and stem test
def models(model):
    scCard(model,mFolder,pDir)
    indEx(model,mFolder,pDir)
    
# Update working histogram
def workingHisto(model):
    initial(model, mFolder, pDir)
    
# Update a model with an add/remove list
def addRem(modelFolderName,newModel,listName): # Use the model number, just new model number, and name of person who made the list
    addRemove(modelFolderName, newModel, listName, mFolder)

# Get important stem test stem list
def stemT_results(modelName):
    global resultsDF
    # Access stem test
    StemTestFiles = []
    for file in os.listdir(pDir):
        if file.endswith(".xlsm"):
            if "Michael" in file and "StemTest" in file and file[0] != "~":
                StemTestFiles.append(file)
    
    # Access latest stem test
    for file in StemTestFiles:
        #if file[-6].isalpha():
            stemTest = xl.Book(os.path.join(pDir,file),read_only=True)
            
    temp_sheet = stemTest.sheets[modelName]
    # Find row with ID in cell
    for row in range(1,1000):
        if temp_sheet.range(("A"+str(row))).value == "ID":
            resultsStartCell = "A"+str(row)
      
            # Read results as pd.DF
    resultsDF = temp_sheet.range(resultsStartCell+":K105817").options(pd.DataFrame,header=1,index=False).value
    #resultsDF['ID'] = resultsDF.index

def copyStems(vaNum,species):
    global va_stemsOut
    global va_stems
    species = species.upper()
    vaID = 'AreaC_'+str(vaNum)
    #va_stems = resultsDF.loc[(resultsDF['Type']=='VA') & (resultsDF['Correct?']==0)]
    va_stems = resultsDF.loc[(resultsDF['Correct?']==0)]
    va_stemsOut = va_stems.loc[(va_stems['Location']==vaID) & (va_stems['Label']==species)]
    va_stemsOut[['ID','Label']].to_clipboard(header=None,index=None)
#va_stems[['ID','Label']].to_clipboard(header=None,index=None)
    
# Access Excel Spreadsheet to specific sheet
def va_score(vaSheet): # Returns score of Excel Worksheet and grab all data
    # Access scorecard
    ScorecardFiles = []
    for file in os.listdir(pDir):
        if file.endswith(".xlsm"):
            if "Michael" in file and "Scorecard" in file and file[0] != "~":
                ScorecardFiles.append(file)
    # Access latest scorecard
    for file in ScorecardFiles:
        overallScoreCard = xl.Book(os.path.join(pDir,file),read_only=True)
    
    global ScoreCard, ScoreCard_df, ScoreCard_main, ScoreCard_bleeds, ScoreCard_stats # Make all variables global
    
    for sheet in overallScoreCard.sheets: # Loop through all the sheets in the workbook
        if sheet.name.startswith(vaSheet): # Check if the sheet name starts with vaSheet
            ScoreCard = overallScoreCard.sheets[sheet]
            # Get the range of the sheet data
            rng = sheet.used_range
            # Convert the range to a pandas data frame
            ScoreCard_df = rng.options(pd.DataFrame, index=False, header=True).value
            break # Break the loop
            
    # Access the Main Tables
    ScoreCard_main = ScoreCard.range('A1:I21').options(pd.DataFrame,header=1).value
    ScoreCard_bleeds = ScoreCard.range('F35:I46').options(pd.DataFrame,header=1).value
    ScoreCard_stats = ScoreCard.range('R44:X1000').options(pd.DataFrame,header=1).value
    ScoreCard_statsExtra = ScoreCard.range('K44:R1000').options(pd.DataFrame,header=1,index=False).value
    ScoreCard_statsExtra = ScoreCard_statsExtra.set_index(ScoreCard_statsExtra.iloc[:,7]) # Set index to be same as ScoreCard_stats
    ScoreCard_statsExtra = ScoreCard_statsExtra.drop(ScoreCard_statsExtra.columns[[6,7]],axis=1) # Remove extra columns
    ScoreCard_stats = pd.concat([ScoreCard_stats,ScoreCard_statsExtra],sort=False,axis=1) # Merge both dataframes to have more info for stats
    print(ScoreCard_main)
    
# Investigate VA table
def va(num): # You need to run va_scorecard first!!!
    """
    Parameters
    ----------
    num : Integer
        The VA number you want to view
    ScoreCard : va_excel variable
        Use the output from va_excel.

    Returns
    -------
    qc_va : VA Description
        DESCRIPTION.

    """
    start = 8 + ((num - 1) * 13)
    qc_va = ScoreCard.range('BH'+str(start+1)+":BS"+str(start+13)).options(pd.DataFrame,header=1).value
    qc_va = qc_va.loc[:,(qc_va.iloc[5,:] != 0)] # Remove irrelevant columns
    qc_va = qc_va.drop(index=[None,'VA OS + CC%','OS + CR SSQ'],axis=0)
    
    overall = ScoreCard.range("BT"+str(start+3)+":BX"+str(start+7)).options(pd.DataFrame,header=1).value
    overall = overall.drop(index=[None],axis=0)

    print('Overall Results :')
    print(overall)
    print('')
    print(qc_va) 
  