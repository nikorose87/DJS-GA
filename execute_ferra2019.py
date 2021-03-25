#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 09:37:09 2021

@author: nikorose
"""
from DJSFunctions import extract_preprocess_data, ankle_DJS
from plot_dynamics import plot_ankle_DJS
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from utilities_QS import multi_idx, create_df, best_hyper, change_labels
import time

#stats
import scipy.stats as stats
import researchpy as rp
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
import seaborn as sns
from scipy.stats.mstats import kruskal
import scikit_posthocs as sp

sns.set_context('paper', font_scale=1.5)
sns.set_style("whitegrid")

#Reading data
meta_data = pd.read_csv('Ferrarin2019/meta_info.csv', index_col=[0])
ferra_QS = pd.read_csv('Ferrarin2019/dynamics_QS.csv', index_col=[0,1], header=[0,1])
ferra_QS_export = ferra_QS.drop('Ankle Power', axis=0, level=0)
ferra_QS_export.to_csv("Ferrarin2019/dynamic_data_Lencioni.csv")
bad_samples = pd.read_csv('Ferrarin2019/ferra_failed_walk.csv', 
                          engine='python', index_col=[0])
meta_data['ID'] = meta_data['subject'].apply(lambda x: int(x[7:]))
meta_data = meta_data.sort_values(['ID', 'speed'], ascending=[True, True])

#Modifying Meta data

def define_range_vel(x, string= True):
    if x <= 0.227:
        cel = [r'$v* < 0.227$' if string == False else 'VS'][0]
    elif 0.227 <= x <= 0.363:
        cel = [r'$0.227 < v* < 0.363$' if string == False else 'S'][0]
    elif 0.363 <= x <= 0.500:
        cel = [r'$0.363 < v* < 0.500$' if string == False else 'C'][0]
    elif 0.500 <= x <= 0.636:
        cel = [r'$0.500 < v* < 0.636$' if string == False else 'F'][0]
    elif x >= 0.636:
        cel = [r'$v* > 0.636$' if string == False else 'VF'][0]
    
    return cel

def froud(x):
    if x.Gender == 'M':
        constant = 0.5856
    else:
        constant = 0.5617
    return np.round(x.speed*100 / (x['Body Height (cm)']* constant *np.sqrt(9.81)),4)

meta_data['froude'] = meta_data.apply(froud, axis=1)
meta_data['range vel'] = meta_data['froude'].apply(define_range_vel)


dyn_midx = pd.MultiIndex.from_arrays([meta_data['subject'], meta_data['Task'],
                                      meta_data['range vel'], meta_data['speed']])
ferra_QS.columns = dyn_midx
#Samples per age groups
samples_age = meta_data['Age (years)'].value_counts()
samples_age = pd.DataFrame(samples_age)
samples_age['Age'] = samples_age.index
samples_age.columns = ['counts', 'age']

#Counting samples in group ages
age_labels = ['Children', 'Young Adults', 'Adults', 'Elderly']
age_samples = [samples_age.query("age <= 18").sum()['counts'],
                 samples_age.query("age >= 22 and age <= 35").sum()['counts'],
                 samples_age.query("age > 35 and age <= 54").sum()['counts'],
                 samples_age.query("age > 54").sum()['counts']]

idx = pd.IndexSlice

Color = [i[1] for i in mcolors.TABLEAU_COLORS.items()]*3

params_mono = {'sharex':False, 'sharey':True, 'color_DJS':['slategray']*20, 
                 'color_reg':['black']*20, 'color_symbols': ['slategray']*20, 
                 'arr_size': 13, 'left_margin': 0.1, 'DJS_linewidth': 0.2, 
                 'reg_linewidth': 1.0, 'grid': True, 'alpha_prod': 0.4,
                 'alpha_absorb': 0.1, 'text':False}

params_all = {'sharex':False, 'sharey':True, 'left_margin': 0.30, 'arr_size':5,
          'hide_labels':(False, False), 'yticks': np.arange(-0.25, 1.755, 0.5), 
          'xticks':np.arange(-0.2, 0.3, 0.1), 'alpha_prod':0.0, 'alpha_absorb':0.0, 'line_width': 1.0,
          'grid':False}

params_ind = {'sharex':False, 'sharey':True, 'left_margin': 0.05, 'arr_size':5,
          'hide_labels':(True, True), 'yticks': np.arange(-0.2, 2.0, 0.4), 
          'xticks':None, 'alpha_prod':0.3, 'alpha_absorb':0.1, 
          'line_width': 1.0,
          'grid':False}

process_ind = False
if process_ind:
    tic = time.time()
    op = 'load'
    for id_sub in np.r_[1:51]:
        sub = 'Subject{}'.format(id_sub)
        sub_df = ferra_QS.loc[:,idx[sub,'Walking',:,:]].droplevel([1,2], axis=1)
        Ferra2019_DJS = ankle_DJS(sub_df, dir_loc = 'Ferrarin2019',
                              exp_name = 'Children and adults 2019')
        
        Ferra2019_QS = Ferra2019_DJS.extract_df_QS_data(idx=[0,1])
        
        Ferra2019_QS = Ferra2019_DJS.interpolate_ankledf(replace=True)
        
        if op == 'manual':
            df_turn = Ferra2019_DJS.get_turning_points(rows=[0,1], turning_points=5, 
                                                       param_1=3, cluster_radius=25)
        elif op == 'opt':
            df_turn = best_hyper(Ferra2019_QS, save='Ferrarin2019/turn_params_{}.csv'.format(sub), 
                                 TP = [5], smooth_radius=range(1,10,2),
                                 cluster_radius=range(30,60,6), verbose=True, rows=[0,1])
        elif op == 'load':
            df_turn =  pd.read_csv('Ferrarin2019/turn_params_{}.csv'.format(sub), index_col=[0,1])
        
        # speeds_sub =list(Ferra2019_QS.columns.get_level_values(1))
        # Ferra2019_QS.columns = pd.MultiIndex.from_arrays([speeds_sub, speeds_sub])
        DJS_ind = plot_ankle_DJS(SD=False, save=True, plt_style='bmh', sep=[int(np.sqrt(sub_df.shape[1])),
                                                                            int(np.sqrt(sub_df.shape[1]))],
                                  alpha=4.0, fig_size=[8,8], params=params_ind)
        
        fig_ind = DJS_ind.plot_DJS(Ferra2019_QS, cols=None, rows= np.r_[0,1],
                            title="Ankle DJS subject comparison at {}".format(sub), 
                            legend=True, reg=df_turn.loc[idx[sub,:],:], header=None,
                            integration= True, rad = True)
        if id_sub == 1:
            reg_info_ind = pd.DataFrame(DJS_ind.reg_info_df)
            work_ind = pd.DataFrame(DJS_ind.areas)
            all_df_turn = df_turn
        else:
            reg_info_ind = pd.concat([reg_info_ind, DJS_ind.reg_info_df])
            all_df_turn = pd.concat([all_df_turn, df_turn]) 
            work_ind = pd.concat([work_ind, DJS_ind.areas])
            
    
    reg_info_ind = reg_info_ind.round(3)
    work_ind = work_ind.round(3)
    #save
    reg_info_ind.to_csv("Ferrarin2019/reg_info_ind_walk.csv")
    work_ind.to_csv("Ferrarin2019/work_ind_walk.csv")
    all_df_turn.to_csv("Ferrarin2019/turn_params_all.csv")
    toc= time.time()
else:
    reg_info_ind = pd.read_csv("Ferrarin2019/reg_info_ind_walk.csv", index_col=[0,1,2])
    work_ind = pd.read_csv("Ferrarin2019/work_ind_walk.csv", index_col=[0])
    all_df_turn = pd.read_csv("Ferrarin2019/turn_params_all.csv", index_col=[0,1])

R2 = reg_info_ind['R2'].unstack(level=2)
R2.columns = ['R2_{}'.format(i) for i in ['CP','ERP','LRP','DP']]
MSE = reg_info_ind['MSE'].unstack(level=2)
MSE.columns = ['MSE_{}'.format(i) for i in ['CP','ERP','LRP','DP']]

R2MSE = pd.concat([reg_info_ind['stiffness'].unstack(level=2), R2,MSE], axis=1)
R2MSE.index = pd.MultiIndex.from_arrays([R2MSE.index.get_level_values(0), 
                                         np.round(R2MSE.index.get_level_values(1),6)])

metrics_label = pd.MultiIndex.from_product([['R2','MSE'],['mean', 'std']])
Ferra_metrics = pd.concat([R2.mean(axis=0), R2.std(axis=0), 
                           MSE.mean(axis=0), MSE.std(axis=0)], axis=1)
Ferra_metrics.columns = metrics_label
work_ind.index = R2.index

#Building the DF
meta_data_walk = meta_data.query("Task == 'Walking'")
meta_data_walk.index = pd.MultiIndex.from_arrays([meta_data_walk.subject, 
                                                  np.round(meta_data_walk.speed,6)])

all_df_turn.index = pd.MultiIndex.from_arrays([all_df_turn.index.get_level_values(0), 
                                                  np.round(all_df_turn.index.get_level_values(1),6)])

meta_data_walk = pd.concat([meta_data_walk, all_df_turn/3, R2MSE, work_ind], 
                           axis=1, ignore_index=False)

meta_data_walk = meta_data_walk.dropna(axis=0)
meta_data_walk = meta_data_walk.rename(columns = {'Age (years)':'Age', 'range vel': 'Range'})
meta_data_walk.to_csv("Ferrarin2019/meta_info_ferra.csv")
#Deviding results per groups
vel_labels = ['Very Slow', 'Slow', 'Free', 'Fast', 'Very Fast']
meta_data_walk_ch = {vel: meta_data_walk.query("Age <= 18 and Range == '{}'".format(vel)) for vel in vel_labels}
meta_data_walk_ya = {vel: meta_data_walk.query("Age >= 22 and Age <= 35 and Range == '{}'".format(vel)) for vel in vel_labels}
meta_data_walk_a = {vel: meta_data_walk.query("Age > 35 and Age <= 54 and Range == '{}'".format(vel)) for vel in vel_labels}
meta_data_walk_old = {vel: meta_data_walk.query("Age > 54 and Range == '{}'".format(vel)) for vel in vel_labels}

# samples per group
samples_groups = {key: [metas[vel].shape[0] for vel in vel_labels] for key, metas in [('Children', meta_data_walk_ch), 
                                                                        ('Young Adults',meta_data_walk_ya),
                                                                        ('Adults', meta_data_walk_a),
                                                                        ('Elderly',meta_data_walk_old)]}
samples_groups = pd.DataFrame(samples_groups,
                              index=['Fast', 'Free', 'Slow', 'Very Fast', 'Very Slow'])

samples_groups = samples_groups.reindex(vel_labels, axis=0)








