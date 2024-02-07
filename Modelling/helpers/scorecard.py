# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 05:13:13 2023

@author: mburnett
"""

# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np
import xlwings as xl
#import re

# Access Excel Spreadsheet to specific sheet
def va_scorecard(vaSheet,pDir): # Returns score of Excel Worksheet and grab all data
    # Access scorecard
    ScorecardFiles = []
    for file in os.listdir(pDir):
        if file.endswith(".xlsm"):
            if "Michael" in file and "Scorecard" in file and file[0] != "~":
                ScorecardFiles.append(file)
    # Access latest scorecard
    for file in ScorecardFiles:
        overallScoreCard = xl.Book(os.path.join(pDir,file),read_only=True)
    
    for sheet in overallScoreCard.sheets: # Loop through all the sheets in the workbook
        if sheet.name.startswith(vaSheet): # Check if the sheet name starts with vaSheet
            ScoreCard = overallScoreCard.sheets[sheet]
            # Get the range of the sheet data
            rng = sheet.used_range
            # Convert the range to a pandas data frame
            ScoreCard_df = rng.options(pd.DataFrame, index=False, header=True).value
            break # Break the loop
            
    # Access the Main Tables
    ScoreCard_main = ScoreCard.range('A1:I21').options(pd.DataFrame,header=1,index=False).value
    ScoreCard_stats = ScoreCard.range('R44:X329').options(pd.DataFrame,header=1).value
    ScoreCard_statsExtra = ScoreCard.range('K44:R329').options(pd.DataFrame,header=1,index=False).value
    ScoreCard_statsExtra = ScoreCard_statsExtra.set_index(ScoreCard_statsExtra.iloc[:,7]) # Set index to be same as ScoreCard_stats
    ScoreCard_statsExtra = ScoreCard_statsExtra.drop(ScoreCard_statsExtra.columns[[6,7]],axis=1) # Remove extra columns
    ScoreCard_stats = pd.concat([ScoreCard_stats,ScoreCard_statsExtra],sort=False,axis=1) # Merge both dataframes to have more info for stats
    
    
    return ScoreCard, ScoreCard_df, ScoreCard_main



# Investigate VA table
def va(ScoreCard,num):
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

    print('Overall Results :')
    print(overall)
    print('')
    print(qc_va)
    return qc_va
    


# UPDATED VERSIONS
def scCard(newFold,mFolder,pDir):
    os.chdir(mFolder)
    for fold in os.listdir():
        if fold[0:3] == newFold:
            newFolder = fold
        if fold[0:4] == newFold:
            newFolder = fold
    Folder = mFolder + '/' + newFolder
    os.chdir(Folder)
    for file in os.listdir():
        if file.endswith('Scorecard.csv'):
            csvFile = file
    # Read the scorecard CSV file
    score = pd.read_csv(csvFile,delimiter=",",names=list(range(29))).dropna(axis='columns', how='all')
    score = score.iloc[:,1:]
    # Read the score to the clipboard and manually paste in Scorecard XLSM
    #score.to_clipboard(excel=True,index=False,header=False)
    
    # Access scorecard
    ScorecardFiles = []
    for file in os.listdir(pDir):
        if file.endswith(".xlsm"):
            if "Michael" in file and "Scorecard" in file and file[0] != "~":
                ScorecardFiles.append(file)
    
    # Access latest scorecard
    for file in ScorecardFiles:
        #if file[-6].isalpha():
            scorecard = xl.Book(os.path.join(pDir,file),read_only=True)
        # if file[-6].isdigit():
        #     maxFile = file[-6]
    
    end_sheet_index = None
    for i, sheet in enumerate(scorecard.sheets):
        if sheet.name == 'END':
            end_sheet_index = i
            break
    
    if end_sheet_index is not None:
        # Copy the sheet before the 'END' sheet
        new_sheet = scorecard.sheets[end_sheet_index-1].copy(before=scorecard.sheets[end_sheet_index])
        
        # Rename the copied sheet
        new_sheet.name = newFolder
        new_sheet.range('AR9').options(index=False,header=False).value = score
        scorecard.save()
        
def indEx(newFold,mFolder,pDir):
    os.chdir(mFolder)
    for fold in os.listdir():
        if fold[0:3] == newFold:
            newFolder = fold
        if fold[0:4] == newFold:
            newFolder = fold
    Folder = mFolder + '/' + newFolder
    os.chdir(Folder)
    for file in os.listdir():
        if file.endswith('ExampleResults.csv'):
            csvFile = file
    # Read the scorecard CSV file
    indEx = pd.read_csv(csvFile, header=None,skiprows=18) # Change skiprows to match which row to start from. 28=30
    ind = indEx.iloc[1:len(indEx),0:4]
    # Read the results to the clipboard and manually paste in StemTest XLSM
    #ind.to_clipboard(excel=True,index=False,header=None)
    
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
        # if file[-6].isdigit():
        #     maxFile = file[-6]
        
    end_sheet_index = None
    for i, sheet in enumerate(stemTest.sheets):
        if sheet.name == 'Summary':
            end_sheet_index = i
            break
    
    if end_sheet_index is not None:
        # Copy the sheet before the 'END' sheet
        new_sheet = stemTest.sheets[end_sheet_index-1].copy(before=stemTest.sheets[end_sheet_index])
        
        # Rename the copied sheet
        new_sheet.name = newFolder
        # Find row with ID in cell
        for row in range(1,1000):
            if new_sheet.range(("A"+str(row))).value == "ID":
                resultsStartCell = "A"+str(row+1)
                deleStartCell = "A"+str(row+2)
        new_sheet.range(deleStartCell+':D105817').clear_contents()
        new_sheet.range(resultsStartCell).options(index=False,header=False).value = ind
        stemTest.save()
        
def initial(newFold,mFolder,pDir):
    os.chdir(mFolder)
    for fold in os.listdir():
        if fold[0:3] == newFold:
            newFolder = fold
        if fold[0:4] == newFold:
            newFolder = fold
    Folder = mFolder + '/' + newFolder
    os.chdir(Folder)
    for file in os.listdir():
        if file.endswith('initial.csv'):
            csvFile = file
    initi = pd.read_csv(csvFile,header=None) # Change skiprows to match which row to start from. 28=30
    init = initi.iloc[1:len(initi),0:2]
    #init.to_clipboard(excel=True,index=False,header=None)
    
    # Access working histogram
    WorkingHistoFiles = []
    for file in os.listdir(pDir):
        if file.endswith(".xlsx"):
            if "Michael" in file and "WorkingHistogram" in file and file[0] != "~":
                WorkingHistoFiles.append(file)
           
    # Access latest working histo
    for file in WorkingHistoFiles:
            workingHisto = xl.Book(os.path.join(pDir,file),read_only=True)

    end_sheet_index = None
    end_sheet_index = len(workingHisto.sheets)

    if end_sheet_index is not None:
        # Copy the sheet before the 'END' sheet
        new_sheet = workingHisto.sheets[end_sheet_index-1].copy()
        
        # Rename the copied sheet
        new_sheet.name = newFolder
        new_sheet.range('B22:C105817').clear_contents()
        new_sheet.range('B21').options(index=False,header=False).value = init
        workingHisto.save()
   
def stemT_results(modelName,pDir):
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
    resultsDF['ID'] = resultsDF.index
    