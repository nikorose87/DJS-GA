#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 19:52:16 2020
Piece of code to process Opensim results
@author: nikorose
"""
import pandas as pd
import numpy as np
import os
#Libraries for regression task
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error



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


def dyn_tables(IK, ID, GRF, _dir):
    break_line = endheader_line(os.path.join(_dir, IK))
    IK_file = pd.read_table(os.path.join(_dir, IK),
                          skiprows=break_line, delim_whitespace=True, 
                          engine='python', decimal='.', index_col=0)
    
    
    break_line = endheader_line(os.path.join(_dir, ID))
    ID_file = pd.read_table(os.path.join(_dir, ID),
                          skiprows=break_line, delim_whitespace=True, 
                          engine='python', decimal='.', index_col=0)
    
    break_line = endheader_line(os.path.join(_dir, GRF))
    GRF_file = pd.read_table(os.path.join(_dir, GRF),
                          skiprows=break_line, delim_whitespace=True, 
                          engine='python', decimal='.', index_col=0)
    return IK_file, ID_file, GRF_file

def interpolate(df_, samples = 100):
    #first we're gonna add columns
    time_gc = np.linspace(0,1,samples)
    # The time need to be converted to Gait cycle percentage
    df_.index = np.linspace(0,1,df_.shape[0])
    point_data_df_gc = pd.DataFrame(np.empty((time_gc.shape[0], 
                                                df_.shape[1])),
                                         index = time_gc, 
                                         columns = df_.columns)
    point_data_df_gc[:] = np.nan
    df_ = df_.append(point_data_df_gc)
    #sorting by index
    df_ = df_.sort_index(axis = 0)
    #Creating new data with interpolation
    df_ = df_.interpolate(method='polynomial', order= 3, axis=0)
    #Extracting only the 100 samples
    df_ = df_.loc[time_gc, :]
    df_.index = np.round(df_.index, 4)
    #Avoid duplicated points 
    df_ = df_[~df_.index.duplicated(keep='first')]
    return  df_
        
def arrange_df(df_, subject, begin=0, finish=None):
    df_ = df_.loc[begin:finish,:]
    df_gc = interpolate(df_)
    df_gc = df_gc.stack()
    df_gc.name = subject
    df_gc.index = df_gc.index.swaplevel()
    return df_gc
    
# =============================================================================
# Amputee information reading from the STO and MOT
# =============================================================================
root_dir = os.getcwd()
info_dir = '/home/nikorose/enprietop@unal.edu.co/Tesis de Doctorado/Gait Analysis Data/Downloaded/Horst et al./Nature_paper/Gait_rawdata_c3d'

# IK_S1, ID_S1, QS_S1 = dyn_tables('IK_subject001.sto', 
#                                      'inverse_dynamics.sto', S1_dir)

#Changing directory to process data
os.chdir(info_dir)

# Subject information 
anthro_info = pd.read_csv('../Gait_subject_info.csv', sep=";",
                          decimal=',')
anthro_info.columns = ['ID','mass','age','gender','height']     
        
        
#Loading cycle gait information 
subject_GC = pd.read_csv('Gait_cycle_bounds.csv', index_col=0)

#Activating processes 
gen_raw_data= False
gen_mean_df=True


if gen_raw_data:
    for num, subject in enumerate(subject_GC.index): # <- do not forget to choose how mny samples do you want to analyse
        subject_dir = os.path.join(info_dir, subject)
        # cut df to the HS points for the right foot
        HSr1, HSr2 = subject_GC.loc[subject][:2]
        IKr_df, IDr_df, GRFr_df = dyn_tables('{}IK.mot'.format(subject[:-4]), 
                                         '{}ID.sto'.format(subject[:-4]), 
                                         '{}Gait_forces.sto'.format(subject[:-4]),
                                         subject_dir)
        #We are building all data homogeneously
        IK_gc = arrange_df(IKr_df, subject, HSr1, HSr2)
        ID_gc = arrange_df(IDr_df, subject, HSr1, HSr2)
        GRF_gc = arrange_df(GRFr_df, subject, HSr1, HSr2)
           
        Series_subject =  IK_gc.append(ID_gc)
        Series_subject =  Series_subject.append(GRF_gc)
        
        try:
            if num == 0:
                df_complete = pd.DataFrame(Series_subject)
            else:
                df_complete = pd.concat([df_complete, Series_subject], axis=1)
        except ValueError:
            print('{} trial does not have the same row numbers, please check'.format(subject))
            continue
    #Sorting by gait cycle
    df_complete = df_complete.sort_index()
    #Removing nan columns and zero rows
    non_zero_rows = (df_complete != 0).any(axis=1)
    df_complete = df_complete.loc[non_zero_rows]
    #Removing NaN columns
    df_complete = df_complete.dropna(axis=1)
    # df_complete.to_csv('../Horst_Nature_paper.csv')

# =============================================================================
# Creating the the DF in terms of mean and SD
# =============================================================================

if gen_mean_df:
    complete_df = pd.read_csv('../Horst_Nature_paper.csv',index_col=[0,1])
    
    for sub in range(1,58):
        subject = 'S{}_'.format(str(sub).zfill(2))
        which_sub = [subject in i for i in complete_df.columns]
        how_subject = len([i for i, x in enumerate(which_sub) if x])
        mean_sub = complete_df.iloc[:,which_sub].mean(axis=1)
        std_sub = complete_df.iloc[:,which_sub].std(axis=1)
        sub_df = pd.concat([mean_sub-std_sub, mean_sub, std_sub+mean_sub], axis=1)
        #Defining the MultiIndex
        Multi_idx = [[subject[:-1]], ['-1sd', 'mean', '+1sd']]
        sub_df.columns = pd.MultiIndex.from_product(Multi_idx, 
                                                    names=['Subject', None ])
        sub_df.index.names = ['Feature', 'Gait cycle [%]']
        if sub == 1:
            complete_mean_df = sub_df
        else:
            complete_mean_df = pd.concat([complete_mean_df, sub_df], axis=1)
    

# complete_mean_df.to_excel('../Horst_mean.xlsx')    
# complete_mean_df.to_csv('../Horst_mean.csv')   
