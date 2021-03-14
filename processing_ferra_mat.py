#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 10:53:16 2021

@author: nikorose
"""
import scipy.io
import numpy as np
import pandas as pd
import os
from utilities_QS import multi_idx, create_df, change_labels

save = False
# =============================================================================
# Making a test to read a mat file
# =============================================================================

root = os.getcwd()
abs_dir = '/home/nikorose/enprietop@unal.edu.co/Tesis de Doctorado/Gait Analysis Data/Downloaded/Ferrarin2019'
os.chdir(abs_dir)
meta_data = pd.read_excel('Demo_Anthro.xlsx', skiprows=None, usecols=None, index_col=[0])


for sub_ID in range(1,meta_data.shape[0]+1):
    # sub_ID = 1
    sub = 'Subject{}'.format(sub_ID)
    mat = scipy.io.loadmat(abs_dir+'/All_Subjects/'+sub+'.mat')
    info = mat['s']
    #Extracting information from Data label
    labels_quan = ['TimeStampKin', 'TimeStampGrf','TimeStampEmg','speed','strideLength','stepWidth','cadence']
    labels_qual = ['Foot','Task']
    meta_info = {key : [item[0] for item in info['Data'][0][0][key][0]] for key in labels_qual}
    meta_info.update({key : [item[0][0] for item in info['Data'][0][0][key][0]] for key in labels_quan})
    meta_info_df = pd.DataFrame(meta_info)
    #Adding anthropometrics
    meta_info_df['subject'] = sub
    for col in meta_data.columns:
        try:
            meta_info_df[col] = meta_data.loc[sub+' ',col]
        except KeyError:
            meta_info_df[col] = meta_data.loc[sub, col]
    
    #Dynamic information
    start = 0
    for item in ['AngVarName','MomVarName','PwrVarName']: #'GrfVarName' ,
        for num, j in enumerate(meta_info_df['speed']):
            var = pd.DataFrame(np.atleast_2d([item[num] for item in info['Data'][0][0][item[:3]]])[0],
                               index = info[item][0][0])
            var = var.unstack().swaplevel().sort_index(level=0)
            var.name = sub
            var.index = pd.MultiIndex.from_product([[j],[item[:3]],info[item][0][0],np.linspace(0,1,101)])
            if start == 0:
                series = var
                start = 1
            else:
                series = pd.concat([series, var])
    df_sub = series.unstack(level=0)
    df_sub = multi_idx(sub, df_sub, idx=False)
    if sub_ID == 1:
        complete_dynamics = df_sub
        meta_complete = meta_info_df
    else:
        complete_dynamics = pd.concat([complete_dynamics, df_sub], axis=1)
        meta_complete = pd.concat([meta_complete,meta_info_df])
        
idx = pd.IndexSlice
complete_dyn_red = complete_dynamics.loc[idx[:,['AnkleFlx ', 'AnkleFlxMom', 'AnklePwr'] ,:],:]            
complete_dyn_red = complete_dyn_red.droplevel(1)        
complete_dyn_red = change_labels(complete_dyn_red, ['Ankle Angle', 'Ankle Moment', 'Ankle Power'], level=0, index=True)

#Saving to csv
if save:
    os.chdir(root)
    meta_complete.to_csv('Ferrarin2019/meta_info.csv')
    complete_dyn_red.to_csv('Ferrarin2019/dynamics_QS.csv')
    complete_dynamics.to_csv('Ferrarin2019/complete_dynamics.csv')