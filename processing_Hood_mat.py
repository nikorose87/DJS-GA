#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 19:53:45 2021

@author: nikorose
"""
import scipy.io
import numpy as np
import pandas as pd
import os
from utilities_QS import multi_idx, create_df, change_labels

# =============================================================================
# Makeing a test to read a mat file
# =============================================================================
root = os.getcwd()
abs_dir = '/home/nikorose/enprietop@unal.edu.co/Tesis de Doctorado/Gait Analysis Data/Downloaded/Hood et al/'
os.chdir(abs_dir)
meta_data = pd.read_excel('Subject Information.xlsx', skiprows=np.r_[0,20:30], usecols=np.r_[1:15], index_col=[0])
for sub_ID in np.r_[1,2,5:21]:
    sub = '/TF{:02d}'.format(sub_ID)
    mat = scipy.io.loadmat(abs_dir+sub+'/Matlab Workspace'+sub+'_DATA.mat')
    info = mat[sub[1:]]['data']
    #To know which vels exists
    vels = [item[0] for item in [(x,str(y[0])) for x,
                                 y in sorted(info[0][0].dtype.fields.items(),
                                             key=lambda k: k[1])]]
    res = []
    labels =[]
    for vel in vels:
        for side in ['ipsilateral', 'contralateral']:
            for var in ['position', 'moment']:
                for metric in ['avg', 'stdev']:
                    res.append(info[0][0][vel][0][0][side][0][0]['ankle'][0][0][var][0][0][metric][0][0])
                    labels.append((vel, side, var, metric))
    
    # Concatenating array
    res_concat = np.concatenate(res, axis=1)
    # Setting multilabels
    multilabel = pd.MultiIndex.from_tuples(labels)
    # Converting to DataFrame
    res_pd = pd.DataFrame(res_concat, columns=multilabel)
    res_pd.index = res_pd.index/1000
    res_pd_mod = res_pd.stack(level=2)
    res_pd_mod = res_pd_mod.swaplevel(axis=0)
    res_pd_mod = res_pd_mod.sort_index(level=0)
    res_pd_mod = res_pd_mod.reindex(['position', 'moment'], level=0)

    idx = pd.IndexSlice
    res_ipsi = create_df(res_pd_mod.loc[:,idx[:,'ipsilateral', 'avg']].droplevel([1,2], axis=1), 
                         res_pd_mod.loc[:,idx[:,'ipsilateral', 'stdev']].droplevel([1,2], axis=1))
    res_contra = create_df(res_pd_mod.loc[:,idx[:,'contralateral', 'avg']].droplevel([1,2], axis=1), 
                         res_pd_mod.loc[:,idx[:,'contralateral', 'stdev']].droplevel([1,2], axis=1))
    # Fancier labels
    vel_ = [float('{}.{}'.format(vel[-3],vel[-1])) for vel in vels]
    #Hood number instead
    froude = lambda v, l: v/(9.81*l)**0.5
    froude_calc = [np.round(froude(vel, meta_data.loc[sub[1:], 'Height (m)']*0.5747),3) for vel in vel_]
    # vel_label_ranges = [r'$v* < 0.227$',r'$0.227 < v* < 0.363$',r'$0.363 < v* < 0.500$',
    #                                          r'$0.500 < v* < 0.636$','$v* > 0.636$']
    complete_labels = pd.MultiIndex.from_product([[sub[1:]],
                                                  ['ipsilateral', 'contralateral'],
                                                  froude_calc, ['-1sd', 'mean', '+1sd']])
    res_total_sub = pd.concat([res_ipsi, res_contra], axis=1)
    res_total_sub.columns = complete_labels
    os.chdir(root)
    res_total_sub.to_csv('Hood/Hood_{}.csv'.format(sub[1:]))
