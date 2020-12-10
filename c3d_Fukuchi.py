#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 17:52:54 2020
Fukushi dataset processing
@author: nikorose
"""
import pandas as pd
import numpy as np
import os
from c3d_functions import process_C3D
# =============================================================================
# Charging the folders 
# =============================================================================
root_dir = os.getcwd()
info_dir = '/home/nikorose/enprietop@unal.edu.co/Tesis de Doctorado/Gait Analysis Data/' \
            + 'Downloaded/Fukuchi/TreadmillAndOverground/Dataset'

# =============================================================================
# Anthropometric information
# =============================================================================
S1_static = "S01_0001_Static.c3d"
S1_gait = "WBDS01walkO01C.c3d"

os.chdir(info_dir)
anthro_info = pd.read_excel('WBDSinfo.xlsx')
anthro_info = anthro_info.drop(anthro_info.iloc[:, [0,-1]].columns, axis=1)
anthro_info_red = anthro_info.iloc[:,np.r_[2:6]]
anthro_info_red = anthro_info_red[['Mass','Age','Gender', 'Height']]
anthro_info_red.columns = ['mass','age','gender', 'height']
# =============================================================================
# #Loading one sample of the C3D file
# =============================================================================
data_dir = info_dir + '/WBDSc3d'
sub_plus_trial = os.path.join(data_dir, anthro_info.iloc[1,0])
S = process_C3D(sub_plus_trial, run=True, forces=True, 
                  **anthro_info_red.iloc[1].to_dict(), printing=False)

# =============================================================================
# we are going to analyze only angles and kinetic information 
# =============================================================================

processed_dir = info_dir + '/WBDSascii'

# Index wherer the dynamic information is
index_angtxt = [j for j, i in enumerate(anthro_info['FileName']) if 'ang.txt' in i]
index_knttxt = [j for j, i in enumerate(anthro_info['FileName']) if 'knt.txt' in i]

#Index labels
labels_ang = anthro_info['FileName'][index_angtxt]
labels_knt = anthro_info['FileName'][index_knttxt]


for i in range(len(index_knttxt)):
    #Reading and concatenating data
    ang_data = pd.read_table(processed_dir+'/'+labels_ang.iloc[i], index_col=[0],
                             dtype=np.float64)
    knt_data = pd.read_table(processed_dir+'/'+labels_knt.iloc[i], index_col=[0],
                             dtype=np.float64)
    comp_data = pd.concat([ang_data, knt_data], axis=1)
    #Make it in one column
    stack_data = comp_data.stack()
    #Changing levels
    stack_data = stack_data.swaplevel()
    #giving a name
    stack_data.name = labels_ang.iloc[i][:-7]
    if i == 0:
        complete_data = stack_data
    else:
        complete_data = pd.concat([complete_data, stack_data], axis=1)

#Some trials are missing
#To discover which are missing, se below:
#[sum(['S{}'.format(str(j).zfill(2)) in i for i in labels_ang]) for j in range(1,43)]

#the following are not in the df
not_listed = [('S28','T08'), ('S29','T08'), ('S32','T07'), ('S32','T08'),
              ('S36','T08'), ('S37','T08'), ('S39','T07'), ('S39','T08')]

#Creating the Multindex columns
subject = ['S{}'.format(str(sub).zfill(2)) for sub in range(1,43)]
modes = ['OC','OF','OS']
modes.extend(['T{}'.format(str(sub).zfill(2)) for sub in range(1,9)])

col_idx = pd.MultiIndex.from_product([subject, modes])
#Deleting columns not in index
not_lit_idx = []
for num, i in enumerate(col_idx):
    if i not in not_listed:
        not_lit_idx.append(num)
        
complete_data.columns = col_idx[not_lit_idx]

#Exporting

complete_data.to_csv('Fukuchi_mean.csv')
