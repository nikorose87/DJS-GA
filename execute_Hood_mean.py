#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 09:04:57 2021

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

def label_(sub_ID, lat, df_):
    try:
        label_turn = pd.MultiIndex.from_arrays([['TF{:02d}'.format(sub_ID)]*df_.shape[0], 
                                     [lat]*df_.shape[0],list(df_.index.get_level_values(0)),
                                     list(df_.index.get_level_values(1))])
    except IndexError:
        label_turn = pd.MultiIndex.from_arrays([['TF{:02d}'.format(sub_ID)]*df_.shape[0], 
                             [lat]*df_.shape[0],list(df_.index.get_level_values(0).unique())])
    return label_turn

meta_data = pd.read_excel('Hood/Subject Information.xlsx', skiprows=np.r_[0,20:30], usecols=np.r_[1:15], index_col=[0])

idx = pd.IndexSlice

Color = [i[1] for i in mcolors.TABLEAU_COLORS.items()]*3

params_mono = {'sharex':False, 'sharey':True, 'color_DJS':['slategray']*20, 
                 'color_reg':['black']*20, 'color_symbols': ['slategray']*20, 
                 'arr_size': 13, 'left_margin': 0.1, 'DJS_linewidth': 0.2, 
                 'reg_linewidth': 1.0, 'grid': True, 'alpha_prod': 0.4,
                 'alpha_absorb': 0.1, 'text':False}

params_all = {'sharex':False, 'sharey':True, 'left_margin': 0.15, 'arr_size':5,
          'hide_labels':(False, False), 'yticks': np.arange(-0.25, 1.755, 0.5), 
          'xticks':None, 'alpha_prod':0.0, 'alpha_absorb':0.0, 'line_width': 1.0,
          'grid':False}

params_ind = {'sharex':False, 'sharey':True, 'left_margin': 0.15, 'arr_size':5,
          'hide_labels':(True, True), 'yticks': np.arange(-0.25, 1.755, 0.5), 
          'xticks':None, 'alpha_prod':0.3, 'alpha_absorb':0.1, 'line_width': 1.0,
          'grid':False}

count = 0
all_plot = False
if all_plot:
    op_ = 'load'
    for sub_ID in np.r_[1,2,5:21]: #np.r_[1,2,5:21]
        sub_df = pd.read_csv('Hood/Hood_TF{:02d}.csv'.format(sub_ID), index_col=[0,1], header=[0,1,2,3])
        for lat in ['ipsilateral', 'contralateral']: #
            sub_df_ = sub_df.loc[:,idx[:, lat,:,:]]
            sub_df_ = sub_df_.droplevel(level=[0,1], axis=1)
            Hood_sub1 = ankle_DJS(sub_df_, 
                                  dir_loc = 'Hood',
                                  exp_name = 'Above knee amputation analysis')
            
            ipsi_QS = Hood_sub1.extract_df_QS_data(idx=[0,1])
            ipsi_QS = Hood_sub1.invert_sign(idx=1)
            
            #Trial and error hyperparameters
            if op_ == 'manual':
                if lat == 'ipsilateral':
                    tp, sr, cr = [4, 4, 100]
                else:
                    tp, sr, cr = [5, 4, 150]
                
                df_turn = Hood_sub1.get_turning_points(rows=[0,1], turning_points=tp, param_1=sr,
                                                        cluster_radius=cr)
            elif op_ == 'opt':
                df_turn = best_hyper(ipsi_QS, save='Hood/turn_params_{}_TF{:02d}.csv'.format(lat, sub_ID), 
                           TP = [4 if lat == 'ipsilateral' else 5],
                           smooth_radius=range(1,10,2),
                           cluster_radius=range(100,200,25), verbose=True, rows=[0,1])
            elif op_ == 'load':
                df_turn = pd.read_csv('Hood/turn_params_{}_TF{:02d}.csv'.format(lat, sub_ID), 
                                      index_col=[0,1])
                
            
            DJS_vis = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=[1,5],
                                      alpha=1.5, fig_size=[2,6], params=params_mono)
            
            fig_vis = DJS_vis.plot_DJS(Hood_sub1.all_dfs_QS, 
                                cols=None, rows= np.r_[0,1],
                                title="Ankle DJS amputee {} TF{:02d}".format(lat, sub_ID), 
                                legend=True, reg= df_turn.loc[idx[:,'mean'],:],
                                integration= True, rad = True)
            
            df_turn.index = label_(sub_ID, lat, df_turn)
            #Saving work
            area_ = DJS_vis.areas
            area_.index = label_(sub_ID, lat, area_)
            #Regressions
            reg_info = DJS_vis.reg_info_df
            reg_info = reg_info.droplevel(1, axis=0)
            label_turn_reg = label_(sub_ID, lat, reg_info)
            reg_info.index = label_turn_reg
            
            if count == 0:
                df_turn_all = df_turn
                work_amp = area_
                reg_info_all = reg_info
                count=1
            else:
                df_turn_all = pd.concat([df_turn_all, df_turn], axis=0)
                work_amp = pd.concat([work_amp, area_], axis=0)
                reg_info_all = pd.concat([reg_info_all, reg_info], axis=0)
            
            df_turn_all.to_csv('Hood/Hood_TP.csv')
            work_amp.to_csv('Hood/hood_work_info.csv')
            reg_info_all.to_csv('Hood/Hood_regressions.csv')
else:
    df_turn_all = pd.read_csv('Hood/Hood_TP.csv', header=[0], index_col=[0,1,2,3])
    
    work_amp = pd.read_csv('Hood/hood_work_info.csv', header=[0], index_col=[0,1,2])
    reg_info_all = pd.read_csv('Hood/Hood_regressions.csv', header=[0], index_col=[0,1,2,3])
    #Changing labels in ipsilateral in second and third variables
    ipsi_reg_df = reg_info_all.loc[idx[:,'ipsilateral',:,:],:]
    ipsi_reg_df =  change_labels(ipsi_reg_df, ['CP','RP', 'PF'], level=3, index=True)
    contra_reg_df = reg_info_all.loc[idx[:,'contralateral',:,:],:]
    
# Main statistics about regression
results = {}
for et, limb in [('ipsilateral', ipsi_reg_df), ('contralateral', contra_reg_df)]:
    results.update({et+' summary': rp.summary_cont(limb, decimals=2)})
    reg_res = limb.unstack(level=3)
    # merging into one level
    reg_res.columns =  reg_res.columns.map('{0[0]}_{0[1]}'.format)
    #Adding work
    reg_res = pd.concat([reg_res, work_amp.loc[idx[:,et,:],:]], axis=1)
    #Adding points
    reg_res = pd.concat([reg_res, df_turn_all.loc[idx[:,et,:,'mean'],:].droplevel(3)], axis=1)
    for sub in meta_data.index:
        reg_res.loc[idx[sub,:,:],'subject'] = sub
        for col in meta_data.columns:
            reg_res.loc[idx[sub,:,:], col] = meta_data.loc[sub,col]
    results.update({et+' results': reg_res})
    
#Plotting brands QS

series_brands = meta_data['Foot Prosthesis']
brands = series_brands.value_counts()

# load specific subjects
for sub_ID in np.r_[5]: #np.r_[5,6,8,11,13,14,17,18,20]
    sub_df = pd.read_csv('Hood/Hood_TF{:02d}.csv'.format(sub_ID), index_col=[0,1], header=[0,1,2,3])
    for lat in ['ipsilateral', 'contralateral']: #
        sub_df_ = sub_df.loc[:,idx[:, lat,:,:]]
        sub_df_ = sub_df_.droplevel(level=[0,1], axis=1)
        Hood_sub1 = ankle_DJS(sub_df_, 
                              dir_loc = 'Hood',
                              exp_name = 'Above knee amputation analysis')
        
        ipsi_QS = Hood_sub1.extract_df_QS_data(idx=[0,1])
        ipsi_QS = Hood_sub1.invert_sign(idx=1)
        
        if lat == 'ipsilateral':
            DJS_all = plot_ankle_DJS(SD=False, save=True, plt_style='bmh', sep=False,
                                      alpha=2.0, fig_size=[2,2], params=params_all)
            
            fig_all = DJS_all.plot_DJS(Hood_sub1.all_dfs_QS, 
                                cols=None, rows= np.r_[0,1],
                                title="Ankle DJS amputee {} TF{:02d} concat".format(lat, sub_ID), 
                                legend='sep', reg= False, #df_turn_all.loc[idx['TF{:02d}'.format(sub_ID),lat,:,'mean'],:].dropna(axis=1),
                                integration= True, rad = True)
        
        DJS_limb = plot_ankle_DJS(SD=False, save=True, plt_style='bmh', sep=False,
                                  alpha=2.0, fig_size=[2,2], params=params_ind)
        
        fig_limb = DJS_limb.plot_DJS(Hood_sub1.all_dfs_QS, 
                            cols=np.r_[1], rows= np.r_[0,1],
                            title="Ankle DJS amputee {} TF{:02d} ind".format(lat, sub_ID), 
                            legend=  True, 
                            reg=df_turn_all.loc[idx['TF{:02d}'.format(sub_ID),lat,:,'mean'],:].dropna(axis=1).astype(np.int64),
                            integration= True, rad = True)
    