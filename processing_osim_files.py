#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 19:52:16 2020
Piece of code to process the Inverse Dynamic results
@author: nikorose
"""
import pandas as pd
import numpy as np
import os



# Finding where the dataset begins in the sto or mot file
def endheader_line(file):
    with open(file,'r') as f:
        lines = f.read().split("\n")
    
    word = 'endheader' # dummy word. you take it from input
    
    # iterate over lines, and print out line numbers which contain
    # the word of interest.
    for i,line in enumerate(lines):
        if word in line: # or word in line.split() to search for full words
            #print("Word \"{}\" found in line {}".format(word, i+1))
            break_line = i+1
    return break_line


def dyn_tables(IK, ID, _dir):
    break_line = endheader_line(os.path.join(_dir, IK))
    IK_file = pd.read_table(os.path.join(_dir, IK),
                          skiprows=break_line, delim_whitespace=True, 
                          engine='python', decimal='.', index_col=0)
    
    
    break_line = endheader_line(os.path.join(_dir, ID))
    ID_file = pd.read_table(os.path.join(_dir, ID),
                          skiprows=break_line, delim_whitespace=True, 
                          engine='python', decimal='.', index_col=0)
    
    QS_df = pd.concat([IK_file['ankle_angle_r'],ID_file['ankle_angle_r_moment']], axis=1)
    return IK_file, ID_file, QS_df

# =============================================================================
# Amputee information reading from the STO and MOT
# =============================================================================
root_dir = os.getcwd()
info_dir = '/home/nikorose/enprietop@unal.edu.co/Tesis de Doctorado/Gait Analysis Data/Downloaded/Horst et al./Nature_paper/Gait_rawdata_c3d'

# IK_S1, ID_S1, QS_S1 = dyn_tables('IK_subject001.sto', 
#                                      'inverse_dynamics.sto', S1_dir)

