# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 06:36:44 2024

Update CFG file and config log file

@author: mburnett
"""
import configparser
import xlwings as xw
import os
import re

cfg_version = input("Enter the seg you want to base the next seg on : ")

cfg_file = fr"C:\Users\--directory--\Scorecard\cfg_files\{str(cfg_version)}Segmentation_rakucuda10.cfg"
xlsx_file = r"C:\Users\--directory--\Scorecard\Config_log.xlsx"



# Keep this section the way it is - it's not very important though
L0 = "L0 attributes (top of canopy)"
L1 = "L1 attributes (mid heights)"
L2 = "L2 attributes (lowest heights)"
cull = "FragCull Num Lidar Points / Height"
sectionsOI = [L0,L1,L2,cull]

def update_cfg_file(cfg_file):
    # Load the existing CFG file
    config = configparser.ConfigParser(comment_prefixes='/', allow_no_value=True)
    config.optionxform = str
    config.read(cfg_file)

    # Display current configuration to the user
    print("Current Configuration:")
    for section in config.sections():
        if section in sectionsOI:
            print(f"\n[{section}]")
            for key, value in config.items(section):
                print(f"{key} = {value}")

    # Prompt user for changes
    section = input("Enter section name to update: ")
    if section == "L0":
        section = L0
    if section == "L1":
        section = L1
    if section == "L2":
        section = L2
    if section == "cull":
        section = cull
    print("\n")
    for s in config.sections():
        if s == section:
            for key, value in config.items(s):
                print("%s = %s" % (key,value))
                
    key = input("Enter key to update: ")
    value = input("Enter new value: ")

    # Update the configuration
    if not config.has_section(section):
        print("Use a section that exists!")
    config.set(section, key, value)
    
    cfg_path,cfg_filename = os.path.split(cfg_file) # get path and file name
    cfg_versionNumber = cfg_filename[0:len(cfg_filename.split("_")[0])] # Get the version number
    cfg_newFilename = re.sub(cfg_versionNumber+"_","TEMP_",cfg_filename) # Create a TEMP file to rename later
    cfg_newFilename_full = os.path.join(cfg_path, cfg_newFilename)
    
    # Save the updated configuration back to the file
    with open(cfg_newFilename_full, 'w') as configfile:
        config.write(configfile)

    return cfg_versionNumber, key, value, cfg_newFilename_full

def log_cfg_change_to_excel(xlsx_file, seg_basedOn, key, new_value):
    # Load or create the XLSX log file
    try:
        workbook = xw.Book(xlsx_file)
        sheet = workbook.sheets['Sheet1']
    except FileNotFoundError:
        print(f"file not found : {xlsx_file}")
        
    # Access right cell DONT NEED
    # sectionIdx = sectionsOI.index(section) # Access relevant section
    # row1 = []
    # for x in sheet['A1:CZ1'].value: # Get list of unique values from first row
    #     if x not in row1:
    #         row1.append(x)
    # row1_uniqueIdx = row1[sectionIdx+2] # remove first two values which aren't relevant
    # row1_idx = sheet['A1:CZ1'].value.index(row1_uniqueIdx)  
    
    # Access right key
    keyValues = sheet['J2:CZ2'].value # These are all unique values, so no need to change anything
    keyIdx = keyValues.index(key.upper())+9 # Add 9 to get true column index value
    
    # Access versions
    low_cell = sheet.range((sheet.cells.last_cell.row,1))
    if low_cell.value is None:
        low_cell = low_cell.end('up')    # go up until you hit a non-empty cell
        low_cellIdx = low_cell.row # last row index
        low_cellVal = int(low_cell.value.split(" ")[0]) # last row version number
    newRow_cell_value = int(low_cell.value.split(" ")[0])+1 # Get number for new seg
    rowToCopy_Idx = low_cellIdx-(low_cellVal-int(seg_basedOn)) # Get the section_basedOn row Idx
    sheet.range('A'+str(rowToCopy_Idx)+":CZ"+str(rowToCopy_Idx)).copy(sheet.range('A'+str(low_cellIdx+1)+":CZ"+str(low_cellIdx+1))) # copy row to new row
    sheet.range('A'+str(low_cellIdx+1)+":CZ"+str(low_cellIdx+1)).color = (255,255,255) # clear all highlights from previous row
    sheet[low_cellIdx,keyIdx].value = new_value # Change value in cell
    sheet[low_cellIdx,keyIdx].color = (255,255,0) # highlight cell
    sheet[low_cellIdx,0].value = f"{newRow_cell_value} ({seg_basedOn})" # populate first column

    # Save the updated log file
    workbook.save(xlsx_file)
    
    return newRow_cell_value

def main():
    # Update the CFG file
    versionNumber, key, new_value, cfg_newFilename_full = update_cfg_file(cfg_file)

    # Log the change to the XLSX file
    newRow_cell_value = log_cfg_change_to_excel(xlsx_file, versionNumber, key, new_value)
    
    # Change name of CFG file
    os.rename(cfg_newFilename_full,re.sub("TEMP_", str(newRow_cell_value)+"_",cfg_newFilename_full))

    print("\nConfiguration updated and change logged successfully.")

if __name__ == "__main__":
    main()