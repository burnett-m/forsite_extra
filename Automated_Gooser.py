# -*- coding: utf-8 -*-
"""

Automatic Gooser

@author: mburnett
"""
import os
import pandas as pd
import xlwings as xl
import numpy as np
import re

# Inputs
user = "Michael"
vaSheet = 'M126'
pDir = r'C:\--directory--\Validation\00_AllSpecies'
resultsDir = os.path.join(r"C:\--directory--\Validation\Results",user)


# Prepare file directories
resultsFolder = [r for r in os.listdir(resultsDir) if r.split("-")[0] == vaSheet][0]
# Get the combined Percentages file from relevant folder
for file in os.listdir(os.path.join(resultsDir,resultsFolder)):
    if file.endswith("CombinedPercentages.csv"):
        combinedP_file = os.path.join(resultsDir,resultsFolder,file) # get full path and file name

# Prepare combined percentages file and its components
combinedP = pd.read_csv(combinedP_file,header=None)
combinedP.iloc[0,2:] = np.nan
ccIdx = combinedP.where(combinedP=='Canopy Cover').dropna(how='all').dropna(axis=1).columns[0] # Get column where "Canopy Cover" appears
cvIdx = combinedP.where(combinedP=='CV').dropna(how='all').dropna(axis=1).columns[0] # Get column where "CV" appears
teIdx = combinedP.where(combinedP=='Test eXamples').dropna(how='all').dropna(axis=1).columns[0]
speciesNames = combinedP.loc[2,ccIdx:cvIdx-1].tolist() # Get list of species
VAs = combinedP.loc[3:len(combinedP),0].tolist() # Get list of VAs
for x in range(3,len(combinedP)): # Divide every balue by 0.95 and change FR to 0
    combinedP.loc[x,cvIdx - 1] = 0
    for y in range(ccIdx,cvIdx-2):
        combinedP.iloc[x,y] = float(combinedP.iloc[x,y])/0.95

# Access scorecard and stem test for specific VA
ScorecardFiles = []
StemTestFiles = []
for file in os.listdir(pDir):
    if file.endswith(".xlsm"):
        if user in file and "Scorecard" in file and file[0] != "~":
            ScorecardFiles.append(file)
        if user in file and "StemTest" in file and file[0] != "~":
            StemTestFiles.append(file)
# Access latest scorecard
for file in ScorecardFiles:
    overallScoreCard = xl.Book(os.path.join(pDir,file),read_only=True)
# Access latest stem test
for file in StemTestFiles:
    overallStemTest = xl.Book(os.path.join(pDir,file),read_only=True)
# Access correct scorecard
for sheet in overallScoreCard.sheets: # Loop through all the sheets in the workbook
    if sheet.name.startswith(vaSheet): # Check if the sheet name starts with vaSheet
        ScoreCard = overallScoreCard.sheets[sheet]
        # Get the range of the sheet data
        rng = sheet.used_range
        # Convert the range to a pandas data frame
        ScoreCard_df = rng.options(pd.DataFrame, index=False, header=False).value
        break # Break the loop
# Access correct stem test
for sheet in overallStemTest.sheets:
    if sheet.name.startswith(vaSheet):
        stemTest = overallStemTest.sheets[sheet]
        rng = sheet.used_range
        stemTest_df = rng.options(pd.DataFrame,index=False,header=False).value
        break

# Get index of where the VA SSQ tables sit
scTotals_Idx = ScoreCard_df.where(ScoreCard_df=="Totals").dropna(how='all').dropna(axis=1).columns[0]
scVA_Idx = scTotals_Idx + 3
scVAend_Idx = scVA_Idx + 1 + len(speciesNames)
ScoreCard_VASSQ = ScoreCard_df.loc[8:15000,scVA_Idx:scVAend_Idx]#.index(start=0)#.options(pd.DataFrame,header=1,index=False).value
ScoreCard_VASSQ = ScoreCard_VASSQ.reset_index(drop=True)

# Make subset DF from stem test
st_Idx_x = stemTest_df.where(stemTest_df=="ID").dropna(how='all').dropna(axis=1).index[0]
#st_Idx_y = stemTest_df.where(stemTest_df=="ID").dropna(how='all').dropna(axis=1).columns[0] # Implied it's column 0
stemTest_vals = stemTest_df.loc[st_Idx_x+1:150000,0:18] # Make DF of important section of stem test
stemTest_vals.columns = stemTest_df.loc[st_Idx_x,0:18]

# Get index of VA SSQs
bfCount_Idx_x = ScoreCard_df.where(ScoreCard_df=="BF Count").dropna(how='all').dropna(axis=1).index[0]
bfCount_Idx_y = ScoreCard_df.where(ScoreCard_df=="BF Count").dropna(how='all').dropna(axis=1).columns[0] + 2 # Get real start location
VA_SSQs_df = ScoreCard_df.loc[bfCount_Idx_x + 1:1001,bfCount_Idx_y:bfCount_Idx_y+4] # Make DF of the VA SSQs
VA_SSQs_df.columns = ScoreCard_df.iloc[bfCount_Idx_x,bfCount_Idx_y:bfCount_Idx_y+5] # Add column headers
VA_SSQs_df = VA_SSQs_df.sort_values('Block',ascending=False)
VA_SSQs_df = VA_SSQs_df.reset_index(drop=True)

copyStems_DF = pd.DataFrame()
for va in range(0,len(VA_SSQs_df)):
    va_num = int(re.findall(r'\d+',VA_SSQs_df.iloc[va,0])[0]) # Get the VA number
    
    # Access SSQs data for that VA
    start = ((va_num - 1) * 13)
    qc_va = ScoreCard_VASSQ.loc[start:start+12,:]
    qc_va = qc_va.loc[:,(qc_va.iloc[5,:] != 0)] # Remove irrelevant columns
    qc_va = qc_va.reset_index(drop=True)
    
    # Look only at species with SSQ > 0.75
    ssq_filter = qc_va.iloc[6,2:].tolist()
    ssq_filter_list = [item for item in ssq_filter if item > 0.75]
    qc_va_filtered = qc_va.loc[:,(qc_va.loc[6,2:].isin(ssq_filter_list))]
    
    # Make dummy combinedPercentages row to fill and populate
    dummy_combinedP = combinedP.loc[[va_num+2],ccIdx:cvIdx-1]
    dummy_combinedP.columns = combinedP.iloc[2,ccIdx:cvIdx] 
    dummy_combinedP = dummy_combinedP.reset_index(drop=True)
    
    # Find relevant combinedPercentages file conversions and fill dummy version

    printCommands = []
    for col in range(0,len(qc_va_filtered.columns)):
        if qc_va_filtered.iloc[3,col] == 0:
            dummy_combinedP[qc_va_filtered.iloc[0,col]] = -0.2
            printCommands.append(f"{qc_va_filtered.iloc[0,col]} :\n This species should not be in this VA. It has SSQ of {qc_va_filtered.iloc[6,col]} \n -0.2 in combinedPercentages file")
        elif qc_va_filtered.iloc[5,col] > qc_va_filtered.iloc[3,col]:
            printCommands.append(f"{qc_va_filtered.iloc[0,col]} :\n SSQ of {qc_va_filtered.iloc[6,col]} \n  TSI is overcalling this species, but it exists in VA, so no change to combinedPercentages file")
        elif qc_va_filtered.iloc[3,col] == 1:
            temp_stems = stemTest_vals.loc[(stemTest_vals['Correct?']==0)]
            temp_stemsOut = temp_stems.loc[(temp_stems['Location']==VA_SSQs_df.iloc[va,0]) & (temp_stems['Label']==qc_va_filtered.iloc[0,col])]
            #copyStems_DF = copyStems_DF.append(temp_stemsOut[['ID','Label']])
            printCommands.append(f"{qc_va_filtered.iloc[0,col]} :\n SSQ of {qc_va_filtered.iloc[6,col]} \n This species is 100% in this VA, so no change to combinedPercentages file")
        elif qc_va_filtered.iloc[3,col] > 0.8:
            temp_stems = stemTest_vals.loc[(stemTest_vals['Correct?']==0)]
            temp_stemsOut = temp_stems.loc[(temp_stems['Location']==VA_SSQs_df.iloc[va,0]) & (temp_stems['Label']==qc_va_filtered.iloc[0,col])]
            #copyStems_DF = copyStems_DF.append(temp_stemsOut[['ID','Label']])
            dummy_combinedP[qc_va_filtered.iloc[0,col]] = 1
            printCommands.append(f"{qc_va_filtered.iloc[0,col]} :\n SSQ of {qc_va_filtered.iloc[6,col]} \n Canopy cover raised to 1 because it's greater than 0.8")
        else:
            temp_stems = stemTest_vals.loc[(stemTest_vals['Correct?']==0)]
            temp_stemsOut = temp_stems.loc[(temp_stems['Location']==VA_SSQs_df.iloc[va,0]) & (temp_stems['Label']==qc_va_filtered.iloc[0,col])]
            #copyStems_DF = copyStems_DF.append(temp_stemsOut[['ID','Label']])
            dummy_combinedP[qc_va_filtered.iloc[0,col]] = dummy_combinedP[qc_va_filtered.iloc[0,col]] + 0.2
            printCommands.append(f"{qc_va_filtered.iloc[0,col]} :\n SSQ of {qc_va_filtered.iloc[6,col]} \n Canopy cover raised by 0.2")
    
    # Interact with user
    print("--------------------------------------------")
    print("--------------------------------------------")
    ## List number of stems in add list and how many would be added
    ## Print the SSQs of this VA
    
    print(VA_SSQs_df.iloc[va,0])
    print(*printCommands,sep="\n")
    print("--------------------------------------------")
    print(f"There are {len(copyStems_DF)} stems in the add list already. Accepting this VA would add another {len(temp_stemsOut)} stems to it.")   
    print("--------------------------------------------")
    print("Answers : 1 = Accept, 0 = Only add stems to list, End = Stop goosing")
    question = input("Do you accept?")
    if question == "1":
        combinedP.loc[[va_num+2],ccIdx:cvIdx-1] = dummy_combinedP.values
        copyStems_DF = pd.concat([copyStems_DF,temp_stemsOut[['ID','Label']]],axis=0)
        ## Insert part to append to add list here
    elif question == "0":
        copyStems_DF = pd.concat([copyStems_DF,temp_stemsOut[['ID','Label']]],axis=0)
    elif question == "End":
        break
    # else:
    #     "Get yourself organized! You had a typo! One more try, otherwise it's End"
    #     if question == 1:
    #         combinedP.loc[[va_num+2],ccIdx:cvIdx-1] = dummy_combinedP
    #         continue
    #     elif question == 0:
    #         continue
    #     else:
    #         break

        

# ask canopy cover and test examples questions
cc_question = input("What is the canopy cover weight?")
combinedP.at[1,ccIdx+2] = str(cc_question)
te_question = input("What is the test examples weight?")
combinedP.at[1,teIdx+2] = str(te_question)

combinedP.to_csv(combinedP_file,index=False,header=None)
copyStems_DF.to_csv(os.path.join(resultsDir,resultsFolder,vaSheet + "-addList.csv"),index=False)

