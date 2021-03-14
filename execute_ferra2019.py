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
meta_data['ID'] = meta_data['subject'].apply(lambda x: int(x[7:]))
meta_data = meta_data.sort_values(['ID', 'speed'], ascending=[True, True])

#Modifying Meta data

def define_range_vel(x):
    if x <= 0.227:
        cel = r'$v* < 0.227$'
    elif 0.227 <= x <= 0.363:
        cel = r'$0.227 < v* < 0.363$'
    elif 0.363 <= x <= 0.500:
        cel = r'$0.363 < v* < 0.500$'
    elif 0.500 <= x <= 0.636:
        cel = r'$0.500 < v* < 0.636$'
    elif x >= 0.636:
        cel = r'$v* > 0.636$'
    
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
    
    speeds_sub =list(Ferra2019_QS.columns.get_level_values(1))
    Ferra2019_QS.columns = pd.MultiIndex.from_arrays([speeds_sub, speeds_sub])
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
    else:
        reg_info_ind = pd.concat([reg_info_ind, DJS_ind.reg_info_df])
        work_ind = pd.concat([work_ind, DJS_ind.areas])

reg_info_ind = reg_info_ind.round(3)
work_ind = work_ind.round(3)
toc= time.time()


