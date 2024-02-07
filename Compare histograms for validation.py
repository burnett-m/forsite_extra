# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 12:48:39 2023

@author: mburnett
"""

# Compare histograms of two models
import os
import pandas as pd

parentDir = r"C:\--directory--\Validation\Results\Michael"
incorrectStemsCSV = r"C:\--directory--\Validation\Results\Michael\--model--\IncorrectStems.csv"
os.chdir(parentDir)
folds = os.listdir()

####### Get histograms #####
h1 = 'M52'
h2 = 'M62'
# Histogram 1
os.chdir(os.path.join(parentDir, [f for f in folds if h1 in f][0]))
Histo1 = pd.read_csv([f for f in os.listdir() if f.endswith("initial.csv")][0])
# Histogram 2
os.chdir(os.path.join(parentDir, [f for f in folds if h2 in f][0]))
Histo2 = pd.read_csv([f for f in os.listdir() if f.endswith("initial.csv")][0])

#### Comparisons ####
# Check what's in histo1 and not histo 2 and vice versa
Histo1['Unnamed: 0'] = Histo1['Unnamed: 0'].astype(str)
Histo2['Unnamed: 0'] = Histo2['Unnamed: 0'].astype(str)
H1_diff = Histo1[~Histo1.iloc[:, 0].isin(Histo2.iloc[:, 0])]
H2_diff = Histo2[~Histo2.iloc[:, 0].isin(Histo1.iloc[:, 0])]
# What's in both models
H1_andH2 = Histo1[Histo1.iloc[:, 0].isin(Histo2.iloc[:, 0])]

H1_diff1 = H1_diff.loc[(H1_diff['Unnamed: 1'] != 'SN')&(H1_diff['Unnamed: 1']!='DP')&(H1_diff['Unnamed: 1']!='FR')]
newList = pd.concat([H1_andH2,H1_diff])

# Compare lists with incorrect stems
incorrectS = pd.read_csv(incorrectStemsCSV)
incorrectS['X'] = incorrectS['X'].astype(str)
iList = pd.concat([H1_diff[H1_diff.iloc[:, 0].isin(incorrectS.iloc[:, 0])], H2_diff[H2_diff.iloc[:, 0].isin(incorrectS.iloc[:, 0])]])
newList = pd.concat([Histo2, iList])
newList.to_csv("M68-1064.csv", index=False)

######### Addional Tasks
addList = pd.read_csv([f for f in os.listdir() if f.endswith("ddList.csv")][0])
addList['1'] = addList['1'].astype(str)
addList_diff = addList[~addList.iloc[:,0].isin(H1_diff.iloc[:,0])]
notIn_list = H1_diff[H1_diff.iloc[:,0].isin(addList.iloc[:,0])]
addList_diff.to_csv("M54-addList.csv",index=False)
