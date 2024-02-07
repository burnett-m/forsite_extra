# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 07:16:54 2023

@author: mburnett
"""

import pandas as pd
import os

def addRemove(model,outputNumber,listName,mFolder):
    os.chdir(mFolder)
    folds = os.listdir()
    os.chdir(os.path.join(mFolder, [f for f in folds if model in f][0]))
    addRemoveList = pd.read_csv(listName+"List.csv",dtype={'X':str})
    addRemoveList = addRemoveList.drop_duplicates(subset='X')
    initial = pd.read_csv([file for file in os.listdir() if file.endswith("initial.csv")][0])
    initial['Unnamed: 0'] = initial['Unnamed: 0'].astype(str).replace(" ","")
    
    addRemoveList['Species'] = addRemoveList['X'].str[-2:]
    addRemoveList['Type'] = addRemoveList['X'].str[0]
    addRemoveList['X1'] = addRemoveList['X'].str[1:-2].astype(str)
    newInitial = initial
    
    
    for i in addRemoveList.index:
        if addRemoveList['Type'][i] == "-":
            newInitial = newInitial.drop(newInitial[newInitial['Unnamed: 0']==addRemoveList['X1'][i]].index)
            #newInitial = newInitial[~newInitial['Unnamed: 0'].isin([addRemoveList['X1'][i]])]
        if addRemoveList['Type'][i] == "+" and ~newInitial['Unnamed: 0'].str.contains(addRemoveList['X1'][i]).any():
            newInitial = newInitial.append({'Unnamed: 0': addRemoveList['X1'][i]}, ignore_index=True)
    
    newInitial.to_csv("M"+outputNumber+"-" + str(newInitial.shape[0]) + ".csv", index=False)