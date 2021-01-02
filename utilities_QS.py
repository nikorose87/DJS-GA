#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 12:24:20 2020

@author: nikorose
"""
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind_from_stats
from DJSFunctions import plot_ankle_DJS, ankle_DJS

def multi_idx(name, df_, idx=True, level=0):
    """
    

    Parameters
    ----------
    name : High level name to set. STR
    df_ : Dataframe to create the multiindex.
    idx : If true It will do over index, otherwise columns.
    level : Either to set in the first level or the second level.

    Returns
    -------
    df_ : Pandas dataframe.

    """
    if idx:
        l_ = df_.index
    else:
        l_ = df_.columns
    if level == 0:
        l_1 = [name]
        l_2 = l_
    else:
        l_1 = l_
        l_2 = [name]
    index_mi = pd.MultiIndex.from_product([l_1, l_2])
    if idx: df_.index = index_mi
    else: df_.columns = index_mi
    return df_


def ttest(df1, df2, samples=[20,20], name='T-test', 
          method='manual', equal_var=False):
    """
    

    Parameters
    ----------
    df1 : Dataframe No 1, all dfs should have a multilabeled column 
          with -1sd, mean and, +1sd.
    df2 : dataframe No 2 with the mentioned above characteristics

    samples : list with the samples of each dataset
    method : 'manual' for an empiric testing, 'scipy' for using
    ttest_ind_from_stats library.
    
    From the last one we're implementing the Welch's test 
    due to we are not assuming equal population variance.

    Returns
    -------
    t_df : T-test for each GC percent

    """
    std_1 = df1['+1sd']-df1['mean']
    std_2 = df2['+1sd']-df2['mean']
    if method == 'manual':
        diff_means = df1['mean'] - df2['mean']
        t_df = pd.Series(diff_means/(np.sqrt(std_1**2/samples[0]+std_2**2/samples[1])),
                         name=name)
        return t_df
    
    elif method == 'scipy':
        res_tt = {'t_value': [], 'p_value': []}
        for ind in std_1.index:
            stats, p_value = ttest_ind_from_stats(mean1 = df1['mean'][ind], 
                                                  std1 = std_1[ind], 
                                                  nobs1 = samples[0], 
                                                  mean2 = df2['mean'][ind], 
                                                  std2 = std_2[ind], 
                                                  nobs2 = samples[1], 
                                                  equal_var=equal_var)
            res_tt['t_value'].append(stats)
            res_tt['p_value'].append(p_value)
        
        res_tt = pd.DataFrame(res_tt, index=std_1.index)
        res_tt = multi_idx(name, res_tt, idx=False)
        return res_tt
            
    

def hyperparams(df_, smooth_radius=(4,8), c_radius=(10,14), features=
                ['Ankle Dorsi/Plantarflexion ', 'Vertical Force',
                 'Ankle Dorsi/Plantarflexion',  'Ankle'], R2=False):
    """
    Parameterization of the curvature settings to see which of them are suitable
    for all the gait instances

    Parameters
    ----------
    df_ : dataframe to analyze the best hyperparameters for Smoothing radious.
    smooth_radius : TYPE, optional
        DESCRIPTION. The default is (4,8).
    c_radius : TYPE, optional
        DESCRIPTION. The default is (10,14).
    features : TYPE, optional
        DESCRIPTION. The default is ['Ankle Dorsi/Plantarflexion ', 'Vertical Force',                 
                                     'Ankle Dorsi/Plantarflexion',  'Ankle'].

    Returns
    -------
    df_turn_instance : Dict with dataframes indicating which regression could be done

    """
    df_turn_instance = {}
    for i in range(*smooth_radius):
        for j in range(*c_radius):
            try:
                print('For {}, {} values'.format(i,j))
                _instance = ankle_DJS(df_, features= features)
                
                all_dfs_instance = _instance.extract_df_DJS_data(idx=[0,2,1,3])
                df_turn_instance.update({'sr_{}_cr_{}'.format(i,j): _instance.get_turning_points(turning_points= 6, 
                                            param_1 = i, cluster_radius= j)})
                if R2:
                    DJS_all = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=False)
                    fig4 = DJS_all.plot_DJS(df_, 
                                        cols=None, rows= np.r_[0,1],
                                        title="Individual ankle DJS C vs Y v A", 
                                        legend=True, reg=df_turn_instance['sr_{}_cr_{}'.format(i,j)],
                                        integration= True, rad = True)
                    print('The R2 for all datasets in {},{} is {}'.format(i,j,
                                            DJS_all.reg_info_df['R2'].mean()))
            except (ValueError, IndexError) as e:
                print('parameters {},{} failed'.format(i,j))
                continue
    return df_turn_instance