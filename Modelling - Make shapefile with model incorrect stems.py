# -*- coding: utf-8 -*-
"""
Create shapefile with VA stems saving only incorrectly guessed stems from a model

@author: mburnett
"""
import os
import geopandas as gpd
import pandas as pd
import xlwings as xl

# import kaleido
# import plotly.express as px
# import plotly.io as pio
# pio.renderers.default="svg"

# Get the working directory
workingDir = r"C:\--directory--\Validation\Results\Michael\--model--"
folderWithin = r"SSQ maps"

# access the independent examples CSV
files = [f for f in os.listdir(workingDir) if f.endswith('ExampleResults.csv')][0]
indEx = pd.read_csv(os.path.join(workingDir,files),header=None,skiprows=15) # skip unnecessary rows
indEx = indEx.loc[2:,:].astype(str) # get rid of two more rows and convert whole DF to string
indEx.columns = indEx.columns.astype(str)
indEx.columns.values[0] = 'UNIQUE_ID'
incorrectEx = indEx[indEx.iloc[:,3]=='0'] # Keep only incorrect stems
#incorrectList = list(incorrectEx.iloc[:,0])

# get correct stems for comparison
correctEx = indEx[indEx.iloc[:,3]=='1']

# Access all stems SHP
VAstemsFile = [f for f in os.listdir(os.path.join(workingDir,folderWithin)) if f.endswith('VAstems.shp')][0]
VAstems = gpd.read_file(os.path.join(workingDir,folderWithin,VAstemsFile))
VAstems['UNIQUE_ID'] = VAstems['UNIQUE_ID'].astype(str)
#VAstems_all = pd.merge(VAstems,indEx,on='UNIQUE_ID',how='inner') # won't work for some reason. Solution four lines later
#incorrectVAstems = VAstems.query('UNIQUE_ID in @incorrectList')
incorrectVAstems = pd.merge(VAstems,incorrectEx,on='UNIQUE_ID',how='inner')
correctVAstems = pd.merge(VAstems,correctEx,on='UNIQUE_ID',how='inner')
VAstems_all = pd.concat([incorrectVAstems,correctVAstems])

# Spatial join with DEP layer
DEP = gpd.read_file(os.path.join(workingDir,folderWithin,'DEP.shp'))
joinedData_incorrect = incorrectVAstems.sjoin(DEP,how='left')
joinedData_correct = correctVAstems.sjoin(DEP,how='left')
joinedData = VAstems_all.sjoin(DEP,how='left')
#joinedData_incorrect.to_file(os.path.join(workingDir,folderWithin,"incorrectStems_joinedWith_DEP.shp"),index=False)
joinedData.to_file(os.path.join(workingDir,folderWithin,"joinedWith_DEP.shp"),index=False)

# Examine DEP with each species
dep_cols = joinedData.columns.values[52:68].tolist() # Get list of DEP layers
unique_sp = joinedData['1'].unique().tolist()

# Create excel workbook to print results to
excelFilename = os.path.join(workingDir,folderWithin,"DEP_FrequencyTable.xlsx")
freq_wb = xl.Book()
freq_ws = freq_wb.sheets['Sheet1']

for species in unique_sp:
    temp_species_df = joinedData[joinedData['1']==species] # access each species individually
    freq_df = pd.DataFrame()
    for layer in dep_cols:
        temp_table = pd.crosstab(temp_species_df[layer], temp_species_df['3']) # Make frequency table based on correct and incorrect species
        temp_table.index = [f"{layer}_{index}" for index in temp_table.index] # change rownames
        temp_table.columns = ['Incorrect', 'Correct'] # change column names
        temp_table['Incorrect'] /= sum(temp_species_df['3'] == '0') # get proportional results for incorrect data
        temp_table['Correct'] /= sum(temp_species_df['3'] == '1')# get proportional results for correct data
        freq_df = pd.concat([freq_df,temp_table])
    
    # add results to new excel sheets
    newSheet = freq_ws.copy()
    newSheet.name = species
    newSheet.range('A1:C200').clear_contents()
    newSheet["A1"].options(pd.DataFrame, header=1, index=True, expand='table').value = freq_df  

freq_wb.sheets['Sheet1'].delete()
freq_wb.save(excelFilename)
