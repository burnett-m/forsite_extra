# -*- coding: utf-8 -*-
"""

Automatic Gooser

@author: mburnett
"""
import os
import pandas as pd
import xlwings as xl
import numpy as np
#import re

# Inputs
user = "Michael"
vaSheet = 'M126'
pDir = r'C:\--directory--\Validation\00_AllSpecies'
resultsDir = os.path.join(r"C:\--directory--\Validation\Results",user)

# Additional Parameters
ssqThreshold_Upper = 0.4 # SSQ threshold to consider when evaluating
add_Subtract_decimal = 0.2 # This is the number to add or subtract the canopy cover to in the combinedPercentages file
SC_startSSQrow = 8 # starting from 0 = 1, the row where the first VA appears in the SSQ section of the scorecard
# Leave this next parameter as an empty list if you don't want to use this option
specifiedSpecies_List = [] 
stemTest_fileName = "StemTest"
scorecard_fileName = "Scorecard"


# Prepare file directories
resultsFolder = [r for r in os.listdir(resultsDir) if r.split("-")[0] == vaSheet][0]
# Get the combined Percentages file from relevant folder
for file in os.listdir(os.path.join(resultsDir,resultsFolder)):
    if file.endswith("CombinedPercentages.csv"):
        combinedP_file = os.path.join(resultsDir,resultsFolder,file) # get full path and file name

if os.path.exists(combinedP_file) is False:
   print("You forgot to add the combinedPercentages.csv file!") 
# Prepare combined percentages file and its components
combinedP = pd.read_csv(combinedP_file,header=None)
combinedP.iloc[0,2:] = np.nan
ccIdx = combinedP.where(combinedP=='Canopy Cover').dropna(how='all').dropna(axis=1).columns[0] # Get column where "Canopy Cover" appears
cvIdx = combinedP.where(combinedP=='CV').dropna(how='all').dropna(axis=1).columns[0] # Get column where "CV" appears
teIdx = combinedP.where(combinedP=='Test eXamples').dropna(how='all').dropna(axis=1).columns[0]
speciesNames = combinedP.loc[2,2:].unique().tolist() # Get list of species
speciesNames = [ssp for ssp in speciesNames if str(ssp) != "nan"] # remove nan from list
VAs = combinedP.loc[3:len(combinedP),0].tolist() # Get list of VAs
for x in range(3,len(combinedP)): # Divide every value by 0.95 and change FR to 0
    combinedP.loc[x,cvIdx - 1] = 0
    for y in range(ccIdx,cvIdx-2):
        combinedP.iloc[x,y] = float(combinedP.iloc[x,y])/0.95

# Access scorecard and stem test for specific VA
ScorecardFiles = []
StemTestFiles = []
for file in os.listdir(pDir):
    if file.endswith(".xlsm") or file.endswith("xlsx"):
        if user in file and scorecard_fileName in file and file[0] != "~":
            ScorecardFiles.append(file)
        if user in file and stemTest_fileName in file and file[0] != "~":
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

# Make subset DF from stem test
correct_colName = "Correct?"
st_Idx_x = stemTest_df.where(stemTest_df=="ID").dropna(how='all').dropna(axis=1).index[0]
#st_Idx_y = stemTest_df.where(stemTest_df=="ID").dropna(how='all').dropna(axis=1).columns[0] # Implied it's column 0
stemTest_vals = stemTest_df.loc[st_Idx_x+1:150000,0:18] # Make DF of important section of stem test
stemTest_vals.columns = stemTest_df.loc[st_Idx_x,0:18] # Rename columns

# Get index of VA SSQs
bfCount_Idx_x = ScoreCard_df.where(ScoreCard_df=="BF Count").dropna(how='all').dropna(axis=1).index[0]
bfCount_Idx_y = ScoreCard_df.where(ScoreCard_df=="BF Count").dropna(how='all').dropna(axis=1).columns[0] + 2 # Get real start location
VA_SSQs_df = ScoreCard_df.loc[bfCount_Idx_x + 1:1001,bfCount_Idx_y:bfCount_Idx_y+4] # Make DF of the VA SSQs
VA_SSQs_df.columns = ScoreCard_df.iloc[bfCount_Idx_x,bfCount_Idx_y:bfCount_Idx_y+5] # Add column headers
VA_SSQs_df_0_0 = VA_SSQs_df.iloc[0,0]
VA_SSQs_df = VA_SSQs_df.sort_values('Block',ascending=False)
VA_SSQs_df = VA_SSQs_df.reset_index(drop=True)

# Get index of where the VA SSQ tables sit
scTotals_Idx = ScoreCard_df.where(ScoreCard_df=="Totals").dropna(how='all').dropna(axis=1).columns[0]
SC_temp_range = ScoreCard_df.loc[8,scTotals_Idx:].reset_index(drop=True)
for cell in range(1,50):
    if SC_temp_range[cell] == VA_SSQs_df_0_0:
        scVA_Idx = cell + scTotals_Idx - 1
        break
scVAend_Idx = scVA_Idx + 1 + len(speciesNames)
ScoreCard_VASSQ = ScoreCard_df.loc[SC_startSSQrow:15000,scVA_Idx:scVAend_Idx] # Access scorecard VA-specific SSQs
ScoreCard_VASSQ = ScoreCard_VASSQ.reset_index(drop=True) # Reindex starting from 0

# Set up dummy combined percentages and add list function
def gossing_values(qc_va_filtered,stemTest_vals,va_os_cc_idx,ssq_idx,dummy_combinedP_idx,va_num,manualTweak):
    global dummy_combinedP, temp_stemsOut, printCommands
    temp_stemsOut = pd.DataFrame() # Make sure it's empty if there's nothing to add to add list
    printCommands = []
    dummy_combinedP = combinedP.loc[dummy_combinedP_idx,ccIdx:ccIdx+len(speciesNames)-1]
    dummy_combinedP.index = combinedP.iloc[2,ccIdx:ccIdx+len(speciesNames)] 
    for col in range(0,len(qc_va_filtered.columns)):
        if qc_va_filtered.iloc[va_os_cc_idx,col] == 0: # Change all species that have relevant SSQs and CC% is 0 to -0.2
            if manualTweak == True:
                temp_value = input(f"{qc_va_filtered.iloc[0,col]} = 0 in canopy cover. What negative value should it be set to?")
                if "-" in temp_value:
                    dummy_combinedP[qc_va_filtered.iloc[0,col]] = float(temp_value)
                else:
                    dummy_combinedP[qc_va_filtered.iloc[0,col]] = float(temp_value) * -1
            else:
                dummy_combinedP[qc_va_filtered.iloc[0,col]] = add_Subtract_decimal * -1
                printCommands.append(f"{qc_va_filtered.iloc[0,col]} :\n This species should not be in this VA. It has CC% of - {add_Subtract_decimal} in combinedPercentages file")
        elif qc_va_filtered.iloc[ssq_idx,col] > qc_va_filtered.iloc[va_os_cc_idx,col]: # Ignore all species that have relevant SSQs and CC% exists
            if manualTweak == True:
                temp_value = input(f"{qc_va_filtered.iloc[0,col]} is in this VA, but it's overcalled. How much would you like to subtract its canopy cover by?")
                dummy_combinedP[qc_va_filtered.iloc[0,col]] = float(dummy_combinedP[qc_va_filtered.iloc[0,col]]) - float(temp_value)
            else:
                printCommands.append(f"{qc_va_filtered.iloc[0,col]} :\n SSQ of {qc_va_filtered.iloc[ssq_idx,col]} \n  TSI is overcalling this species, but it exists in VA, so no change to combinedPercentages file")
        ######## Add option to reduce overcalled species by 0.2
        elif qc_va_filtered.iloc[va_os_cc_idx,col] == 1: # Ignore all species that have relevant SSQs and CC% is 1, but collect stems for add list
            temp_stems = stemTest_vals.loc[(stemTest_vals[correct_colName]==0)]
            temp_stemsOut = pd.concat([temp_stemsOut,temp_stems.loc[(temp_stems['Location']==va_num) & (temp_stems['Label']==qc_va_filtered.iloc[0,col])]])
            #copyStems_DF = copyStems_DF.append(temp_stemsOut[['ID','Label']])
            printCommands.append(f"{qc_va_filtered.iloc[0,col]} :\n SSQ of {qc_va_filtered.iloc[ssq_idx,col]} \n This species is 100% in this VA, so no change to combinedPercentages file")
        elif qc_va_filtered.iloc[va_os_cc_idx,col] > 0.8 and qc_va_filtered.iloc[va_os_cc_idx,col] < 1: # Change all species that have relevant SSQs and CC% is close to 1 to 1 and collect stems for add list
            temp_stems = stemTest_vals.loc[(stemTest_vals[correct_colName]==0)]
            temp_stemsOut = pd.concat([temp_stemsOut,temp_stems.loc[(temp_stems['Location']==va_num) & (temp_stems['Label']==qc_va_filtered.iloc[0,col])]])
            #copyStems_DF = copyStems_DF.append(temp_stemsOut[['ID','Label']])
            dummy_combinedP[qc_va_filtered.iloc[0,col]] = 1
            printCommands.append(f"{qc_va_filtered.iloc[0,col]} :\n SSQ of {qc_va_filtered.iloc[ssq_idx,col]} \n Canopy cover raised to 1 because it's greater than 0.8")
        elif qc_va_filtered.iloc[va_os_cc_idx,col] <= 0.8: # Change all species that have relevant SSQs and CC% is up by 0.2 and collect stems for add list
            temp_stems = stemTest_vals.loc[(stemTest_vals[correct_colName]==0)]
            temp_stemsOut = pd.concat([temp_stemsOut,temp_stems.loc[(temp_stems['Location']==va_num) & (temp_stems['Label']==qc_va_filtered.iloc[0,col])]])
            if manualTweak == True:
                temp_value = input("Canopy cover is <= 0.8. How much do you want to add to it?")
                #copyStems_DF = copyStems_DF.append(temp_stemsOut[['ID','Label']])
                dummy_combinedP[qc_va_filtered.iloc[0,col]] = float(dummy_combinedP[qc_va_filtered.iloc[0,col]]) + float(temp_value)
            else:
                #copyStems_DF = copyStems_DF.append(temp_stemsOut[['ID','Label']])
                dummy_combinedP[qc_va_filtered.iloc[0,col]] = float(dummy_combinedP[qc_va_filtered.iloc[0,col]]) + add_Subtract_decimal 
                printCommands.append(f"{qc_va_filtered.iloc[0,col]} :\n SSQ of {qc_va_filtered.iloc[ssq_idx,col]} \n Canopy cover raised by {add_Subtract_decimal}")
    return dummy_combinedP, temp_stemsOut, printCommands

# Set up selecting subsets for add list
def addList_subset(temp_stemsOut,copyStems_DF):
    #global copyStems_DF
    print("--------------------------------------------\n--------------------------------------------")
    print("Options : 1 = select the tallest half of the stems, 0 = select the shortest half of the stems, ANY OTHER NUMBER = every nth stem is selected (if 3, then every 3rd stem is selected in descending order based on height starting with the tallest incorrect stem)")
    fractionQuestion = input("Which option do you choose?")
    if fractionQuestion == "1": # Tallest half
        temp_stemsOut = temp_stemsOut.sort_values("Height",ascending=False)
        temp_num = (len(temp_stemsOut)+2//2)//2
        temp_stemsOut = temp_stemsOut.iloc[0:temp_num,:] # Select the top half (rounded up)
        copyStems_DF = pd.concat([copyStems_DF,temp_stemsOut[['ID','Label']]],axis=0)
    elif fractionQuestion == "0": # Shortest half
        temp_stemsOut = temp_stemsOut.sort_values("Height",ascending=True)
        temp_num = (len(temp_stemsOut)+2//2)//2
        temp_stemsOut = temp_stemsOut.iloc[0:temp_num,:] # Select the bottom half (rounded up)
        copyStems_DF = pd.concat([copyStems_DF,temp_stemsOut[['ID','Label']]],axis=0)
    else:
        temp_stemsOut = temp_stemsOut.sort_values("Height",ascending=False)
        temp_stemsOut = temp_stemsOut.iloc[::int(fractionQuestion),:] # select every nth stem
        copyStems_DF = pd.concat([copyStems_DF,temp_stemsOut[['ID','Label']]],axis=0)
    return copyStems_DF
    
    

copyStems_DF = pd.DataFrame()
for va in range(0,len(VA_SSQs_df)):
    #va_num = int(re.findall(r'\d+',VA_SSQs_df.iloc[va,0])[0]) # Get the VA number
    va_num = VA_SSQs_df.iloc[va,0]
    
    
    # Access SSQs data for that VA
    #start = ((va_num - 1) * 13)
    start = ScoreCard_VASSQ.iloc[:,1].where(ScoreCard_VASSQ.iloc[:,1]==va_num).dropna().index[0] 
    qc_va = ScoreCard_VASSQ.loc[start:start+12,:]
    ssq_idx = qc_va.iloc[:,1].where(qc_va.iloc[:,1]=="OS CR SSQ").dropna().index[0] - qc_va.index[0] # Get index relative to the rest of df
    va_os_cc_idx = qc_va.iloc[:,1].where(qc_va.iloc[:,1]=="VA OS CC%").dropna().index[0] - qc_va.index[0] # Get index relative to the rest of df
    qc_va = qc_va.loc[:,(qc_va.iloc[ssq_idx,:] != 0)] # Remove irrelevant columns
    qc_va = qc_va.reset_index(drop=True)
    
    # Look only at species with SSQ > ssqThreshold_Upper
    ssq_filter = qc_va.iloc[ssq_idx,2:].tolist()
    ssq_filter_list = [item for item in ssq_filter if item > ssqThreshold_Upper]
    qc_va_filtered = qc_va.loc[:,(qc_va.loc[ssq_idx,2:].isin(ssq_filter_list))]
    
    # If an option was made to goose based on only one species
    if len(specifiedSpecies_List) > 0 and len(set(qc_va_filtered.iloc[0,:].tolist()).intersection(specifiedSpecies_List)) == 0:
        continue
    
    # Make dummy combinedPercentages row to fill and populate
    #dummy_combinedP = combinedP.loc[[va_num+2],ccIdx:cvIdx-1]
    dummy_combinedP_idx = combinedP.iloc[:,0].where(combinedP.iloc[:,0]==va_num).dropna().index[0]
    # dummy_combinedP = combinedP.loc[dummy_combinedP_idx,ccIdx:cvIdx-1]
    # dummy_combinedP.index = combinedP.iloc[2,ccIdx:cvIdx] 
#    dummy_combinedP = dummy_combinedP.reset_index(drop=True)
    
    # Find relevant combinedPercentages file conversions and fill dummy version
    # temp_stemsOut = pd.DataFrame() # Make sure it's empty if there's nothing to add to add list
    # printCommands = []
    gossing_values(qc_va_filtered,stemTest_vals,va_os_cc_idx,ssq_idx,dummy_combinedP_idx,va_num,False)
    
    # Interact with user
    print("--------------------------------------------\n--------------------------------------------\n--------------------------------------------\n--------------------------------------------")
    ## List number of stems in add list and how many would be added
    ## Print the SSQs of this VA
    
    print(VA_SSQs_df.iloc[va,0])
    print(*printCommands,sep="\n")
    print("--------------------------------------------")
    print(f"There are {len(copyStems_DF)} stems in the add list already. Accepting this VA would add another {len(temp_stemsOut)} stems to it.")   
    print("--------------------------------------------")
    print("Answers : 1 = Accept, 2 = Accept but do not add to add list, 3 = Accept but select a portion of stems to add to add list, 0 = Only add stems to list, manual = Manually input values to add or subtract to in the combinedPercentages file, End = Stop goosing")
    question = input("Do you accept?")
    if question == "1": # add stems to add list and configure combinedPercentages file
        combinedP.loc[dummy_combinedP_idx,ccIdx:ccIdx+len(speciesNames)-1] = dummy_combinedP.values
        if len(temp_stemsOut.index) != 0:
            copyStems_DF = pd.concat([copyStems_DF,temp_stemsOut[['ID','Label']]],axis=0)
    elif question == "2": # Only configure combined percentages file without adding to add list
        combinedP.loc[dummy_combinedP_idx,ccIdx:ccIdx+len(speciesNames)-1] = dummy_combinedP.values
    elif question == "0": # add stems to add list, but don't touch combinedPercentages file
        copyStems_DF = pd.concat([copyStems_DF,temp_stemsOut[['ID','Label']]],axis=0)
    elif question == "3": # select a fraction of the stems to add to the addList
        combinedP.loc[dummy_combinedP_idx,ccIdx:ccIdx+len(speciesNames)-1] = dummy_combinedP.values
        if len(temp_stemsOut) > 0:
            addList_subset(temp_stemsOut,copyStems_DF)
        else:
            print("Wise up! There's nothing in the add list for this VA!!!")
    elif question == "manual":
        print("--------------------------------------------")
        gossing_values(qc_va_filtered,stemTest_vals,va_os_cc_idx,ssq_idx,dummy_combinedP_idx,va_num,True)
        combinedP.loc[dummy_combinedP_idx,ccIdx:ccIdx+len(speciesNames)-1] = dummy_combinedP.values
        print("--------------------------------------------")
        if len(temp_stemsOut.index) != 0:
            addList_question = input("y = Add stems to add list, n = Don't add stems to add list, 3 = Select a portion of stems to add to add list \n Which do you select?")
            if addList_question == "y":
                copyStems_DF = pd.concat([copyStems_DF,temp_stemsOut[['ID','Label']]],axis=0)
            elif addList_question == "n":
                continue
            elif addList_question == "3":
                if len(temp_stemsOut) > 0:
                    addList_subset(temp_stemsOut,copyStems_DF)
                else:
                    print("Wise up! There's nothing in the add list for this VA!!!")
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

copyStems_DF = copyStems_DF.drop_duplicates(subset=["ID"],keep="first")       

# ask canopy cover and test examples questions
cc_question = input("What is the canopy cover weight?")
combinedP.at[1,ccIdx+2] = str(cc_question)

### Test Examples
# Prepare test examples list
teString = " 0 : All incorrect stems \n "
for species in range(1,len(speciesNames)+1):
    teString += str(species) + " : All correct and incorrect " + speciesNames[species - 1] + " stems \n "
teString += str(len(speciesNames)+1) + " : Don't make TestExamples.csv"

# Create test examples list
print("--------------------------------------------")
print("--------------------------------------------")
print("Select one of the following options for test examples")
print(teString)
print("If you'd like to select more than one species, separate each number by a comma without any spaces")
# User interaction
te_typeQ = input("Which option is your preference for test examples?")
if te_typeQ == "0":
    te_list_temp = stemTest_vals.loc[(stemTest_vals[correct_colName]==0)] # Collect all stems that are incorrect
    te_list = te_list_temp[["ID","Label"]] # only keep first two columns
    # Determine Test Examples weight
    te_question = input("What is the test examples weight?")
    combinedP.at[1,teIdx+2] = str(te_question)
te_speciesNums = [str(x) for x in list(range(1,len(speciesNames)+1))] # Make list of numbers as strings
if te_typeQ in te_speciesNums:
    spNum = int(te_typeQ) - 1
    te_list_temp = stemTest_vals.loc[(stemTest_vals['Label']==speciesNames[spNum])] # Collect all stems from that species
    te_list = te_list_temp[["ID","Label"]]
    # Determine Test Examples weight
    te_question = input("What is the test examples weight?")
    combinedP.at[1,teIdx+2] = str(te_question)
if "," in te_typeQ: # Give option to select more than one species for the test examples
    te_typeQ = te_typeQ.replace(" ", "")
    te_typeQ_list = te_typeQ.split(",") # Create list from numbers
    te_list = pd.DataFrame() # create dataframe to append to
    for te_t in te_typeQ_list:
        spNum = int(te_t) - 1
        te_list_temp = stemTest_vals.loc[(stemTest_vals['Label']==speciesNames[spNum])] # Collect all stems from that species
        te_list = pd.concat([te_list,te_list_temp[["ID","Label"]]],axis=0)
    # Determine Test Examples weight
    te_question = input("What is the test examples weight?")
    combinedP.at[1,teIdx+2] = str(te_question)
if te_typeQ == str(len(speciesNames)+1):
    print("Cool beans!")

# Output files
combinedP.to_csv(combinedP_file,index=False,header=None)
copyStems_DF.to_csv(os.path.join(resultsDir,resultsFolder,vaSheet + "-addList.csv"),index=False)
te_list.to_csv(os.path.join(resultsDir,resultsFolder,vaSheet + "-TestExamples.csv"),index=False)
